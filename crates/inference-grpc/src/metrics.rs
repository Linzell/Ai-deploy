//! OTel metrics instruments for the gRPC worker.
//!
//! These use `opentelemetry::global::meter()` which returns a no-op meter
//! when no `MeterProvider` has been set — zero overhead when OTel is disabled.

use opentelemetry::global;
use opentelemetry::metrics::{Counter, Histogram};
use std::sync::OnceLock;

/// Cached metric instruments — initialized once on first use.
pub struct Metrics {
    /// Total inference requests processed (success + failure).
    pub request_count: Counter<u64>,
    /// Request duration in milliseconds.
    pub request_duration_ms: Histogram<f64>,
    /// Number of failed requests.
    pub error_count: Counter<u64>,
    /// Batch sizes dispatched by the batcher.
    pub batch_size: Histogram<u64>,
}

/// Get the global metrics instance (lazily initialized).
pub fn metrics() -> &'static Metrics {
    static INSTANCE: OnceLock<Metrics> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let meter = global::meter("inference-grpc");

        Metrics {
            request_count: meter
                .u64_counter("inference.requests")
                .with_description("Total inference requests processed")
                .with_unit("requests")
                .build(),

            request_duration_ms: meter
                .f64_histogram("inference.duration")
                .with_description("Request duration")
                .with_unit("ms")
                .build(),

            error_count: meter
                .u64_counter("inference.errors")
                .with_description("Total failed inference requests")
                .with_unit("requests")
                .build(),

            batch_size: meter
                .u64_histogram("inference.batch_size")
                .with_description("Number of requests per batch dispatch")
                .with_unit("requests")
                .build(),
        }
    })
}
