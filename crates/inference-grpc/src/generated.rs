//! Generated protobuf code.
//!
//! This module includes the generated Rust code from proto files.

// Allow clippy warnings in generated code
#![allow(clippy::default_trait_access)]

// Maiia Worker Service
pub mod maiia {
    pub mod worker {
        pub mod v1 {
            tonic::include_proto!("maiia.worker.v1");
        }
    }
    pub mod common {
        pub mod v1 {
            tonic::include_proto!("maiia.common.v1");
        }
    }
}

// gRPC Health Service
pub mod grpc {
    pub mod health {
        pub mod v1 {
            tonic::include_proto!("grpc.health.v1");
        }
    }
}

// Re-exports for convenience
pub use grpc::health::v1 as health_pb;
pub use maiia::common::v1 as common_pb;
pub use maiia::worker::v1 as worker_pb;
