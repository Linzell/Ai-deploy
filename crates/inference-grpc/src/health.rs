//! Health service implementation.
//!
//! Standard gRPC health checking for Kubernetes.

use std::sync::Arc;

use tonic::{Request, Response, Status};
use tracing::{debug, info};

use crate::generated::health_pb::{
    health_check_response::ServingStatus, health_server::Health, HealthCheckRequest,
    HealthCheckResponse,
};
use crate::task::Task;

/// Health service implementation.
pub struct HealthServiceImpl {
    task: Arc<dyn Task>,
    service_name: String,
}

impl HealthServiceImpl {
    /// Create a new Health service.
    pub fn new(task: Arc<dyn Task>, service_name: &str) -> Self {
        Self {
            task,
            service_name: service_name.to_string(),
        }
    }
}

#[tonic::async_trait]
impl Health for HealthServiceImpl {
    async fn check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let service = &request.into_inner().service;
        info!("Health check request for service: {:?}", service);

        // Empty service name means overall health
        // Otherwise check if it matches our service
        if !service.is_empty() && service != &self.service_name && service != self.task.name() {
            debug!("Unknown service: {}", service);
            return Ok(Response::new(HealthCheckResponse {
                status: ServingStatus::ServiceUnknown as i32,
            }));
        }

        let status = if self.task.is_ready() {
            ServingStatus::Serving
        } else {
            ServingStatus::NotServing
        };

        info!("Health check response: {:?}", status);
        Ok(Response::new(HealthCheckResponse {
            status: status as i32,
        }))
    }

    type WatchStream = std::pin::Pin<
        Box<dyn tokio_stream::Stream<Item = Result<HealthCheckResponse, Status>> + Send>,
    >;

    async fn watch(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> Result<Response<Self::WatchStream>, Status> {
        let service = request.into_inner().service;
        let task = Arc::clone(&self.task);
        let service_name = self.service_name.clone();

        info!("Health watch request for service: {:?}", service);

        let stream = async_stream::stream! {
            loop {
                // Check if service matches
                if !service.is_empty() && service != service_name && service != task.name() {
                    yield Ok(HealthCheckResponse {
                        status: ServingStatus::ServiceUnknown as i32,
                    });
                } else {
                    let status = if task.is_ready() {
                        ServingStatus::Serving
                    } else {
                        ServingStatus::NotServing
                    };
                    yield Ok(HealthCheckResponse {
                        status: status as i32,
                    });
                }

                // Check every 5 seconds
                tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }
}
