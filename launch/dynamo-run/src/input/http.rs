// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;
use std::pin::Pin;

use dynamo_llm::{
    backend::ExecutionContext, model_card::model::ModelDeploymentCard,
    backend::Backend,
    http::service::{discovery, service_v2},
    model_type::ModelType,
    preprocessor::OpenAIPreprocessor,
    types::{
        openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        },
        openai::completions::{CompletionRequest, CompletionResponse},
        Annotated,
    },
    protocols::common::llm_backend::{BackendInput, BackendOutput},
};
use dynamo_runtime::{
    pipeline::{
        ManyOut, 
        Operator, 
        ServiceBackend, 
        ServiceFrontend, 
        SingleIn,
        Source,
        Context,
    },
    engine::{Data, AsyncEngineStream},
    DistributedRuntime, Runtime,
};

use crate::{EngineConfig, Flags};

/// Build and run an HTTP service
pub async fn run(
    runtime: Runtime,
    flags: Flags,
    engine_config: EngineConfig,
) -> anyhow::Result<()> {
    let http_service = service_v2::HttpService::builder()
        .port(flags.http_port)
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .build()?;
    match engine_config {
        EngineConfig::Dynamic(endpoint) => {
            let distributed_runtime = DistributedRuntime::from_settings(runtime.clone()).await?;
            match distributed_runtime.etcd_client() {
                Some(etcd_client) => {
                    // This will attempt to connect to NATS and etcd

                    let component = distributed_runtime
                        .namespace(endpoint.namespace)?
                        .component(endpoint.component)?;
                    let network_prefix = component.service_name();

                    // Listen for models registering themselves in etcd, add them to HTTP service
                    let state = Arc::new(discovery::ModelWatchState {
                        prefix: network_prefix.clone(),
                        model_type: ModelType::Chat,
                        manager: http_service.model_manager().clone(),
                        drt: distributed_runtime.clone(),
                    });
                    tracing::info!("Waiting for remote model at {network_prefix}");
                    let models_watcher =
                        etcd_client.kv_get_and_watch_prefix(network_prefix).await?;
                    let (_prefix, _watcher, receiver) = models_watcher.dissolve();
                    let _watcher_task = tokio::spawn(discovery::model_watcher(state, receiver));
                }
                None => {
                    // Static endpoints don't need discovery
                }
            }
        }
        EngineConfig::StaticFull {
            service_name,
            engine,
            ..
        } => {
            let manager = http_service.model_manager();
            manager.add_chat_completions_model(&service_name, engine)?;
            manager.add_completions_model(&service_name, engine)?;
        }
        EngineConfig::StaticCore {
            service_name,
            engine: inner_engine,
            card,
        } => {
            let manager = http_service.model_manager();
            
            // Build and register chat pipeline
            let chat_pipeline = build_pipeline::<
                NvCreateChatCompletionRequest,
                NvCreateChatCompletionStreamResponse
            >(&card, inner_engine.clone()).await?;
            manager.add_chat_completions_model(&service_name, chat_pipeline)?;

            // Build and register completions pipeline
            let cmpl_pipeline = build_pipeline::<
                CompletionRequest,
                CompletionResponse
            >(&card, inner_engine).await?;
            manager.add_completions_model(&service_name, cmpl_pipeline)?;
        }
        EngineConfig::None => unreachable!(),
    }
    http_service.run(runtime.primary_token()).await
}

async fn build_pipeline<Req, Resp>(
    card: &ModelDeploymentCard,
    engine: ExecutionContext,
) -> anyhow::Result<Arc<ServiceFrontend<SingleIn<Req>, ManyOut<Annotated<Resp>>>>>
where
    Req: Data,
    Resp: Data,
    OpenAIPreprocessor: Operator<
        Context<Req>,
        Pin<Box<dyn AsyncEngineStream<Annotated<Resp>>>>,
        Context<BackendInput>,
        Pin<Box<dyn AsyncEngineStream<Annotated<BackendOutput>>>>
    >,
{
    let frontend = ServiceFrontend::<SingleIn<Req>, ManyOut<Annotated<Resp>>>::new();
    let preprocessor = OpenAIPreprocessor::new((*card).clone()).await?.into_operator();
    let backend = Backend::from_mdc((*card).clone()).await?.into_operator();
    let engine = ServiceBackend::from_engine(engine);

    Ok(frontend
        .link(preprocessor.forward_edge())?
        .link(backend.forward_edge())?
        .link(engine)?
        .link(backend.backward_edge())?
        .link(preprocessor.backward_edge())?
        .link(frontend)?)
}
