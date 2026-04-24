//! OpenTelemetry initialization — env-gated, disabled by default.
//!
//! Set `OTEL_ENABLED=true` and `OTEL_ENDPOINT` (e.g. `http://192.168.1.3:4317`)
//! to enable push-based metrics and traces via OTLP/gRPC to an OpenTelemetry
//! Collector.
//!
//! When either env var is absent, all OTel setup is skipped and the application
//! runs with zero overhead from telemetry.

use inference_core::Config;
use opentelemetry::global;
use opentelemetry::KeyValue;
use opentelemetry_otlp::{MetricExporter, SpanExporter, WithExportConfig};
use opentelemetry_sdk::metrics::SdkMeterProvider;
use opentelemetry_sdk::trace::SdkTracerProvider;
use opentelemetry_sdk::Resource;
use std::env;
use tracing::info;

use crate::system_metrics;

/// Holds the OTel providers so they can be shut down gracefully.
///
/// Drop this at the end of `main()` to flush pending telemetry.
pub struct TelemetryGuard {
    tracer_provider: SdkTracerProvider,
    meter_provider: SdkMeterProvider,
}

impl TelemetryGuard {
    /// Flush and shut down all OTel providers.
    pub fn shutdown(self) {
        if let Err(e) = self.tracer_provider.shutdown() {
            tracing::warn!(error = %e, "Failed to shut down OTel tracer provider");
        }
        if let Err(e) = self.meter_provider.shutdown() {
            tracing::warn!(error = %e, "Failed to shut down OTel meter provider");
        }
    }
}

/// Build the OTel `Resource` from the service config.
fn build_resource(config: &Config) -> Resource {
    Resource::builder()
        .with_service_name(config.service_name.clone())
        .with_attributes([
            KeyValue::new("service.version", env!("CARGO_PKG_VERSION")),
            KeyValue::new("task.type", config.task_type.as_str().to_string()),
            KeyValue::new("task.name", config.effective_task_name().clone()),
            KeyValue::new("backend", format!("{:?}", config.backend)),
            KeyValue::new("device", format!("{:?}", config.device)),
            KeyValue::new(
                "model.path",
                config
                    .model_path
                    .as_deref()
                    .unwrap_or("unknown")
                    .to_string(),
            ),
        ])
        .build()
}

/// Try to initialize OpenTelemetry.
///
/// Returns `Some(TelemetryGuard)` if **both** `OTEL_ENABLED=true` (or `1`) and
/// `OTEL_ENDPOINT` are set. The guard must be kept alive for the lifetime of
/// the process; dropping or calling `.shutdown()` flushes pending telemetry.
///
/// Returns `None` if either variable is absent or disabled — telemetry is off.
pub fn try_init_otel(config: &Config) -> Option<TelemetryGuard> {
    let enabled = env::var("OTEL_ENABLED").unwrap_or_default();
    if !matches!(enabled.as_str(), "true" | "1") {
        return None;
    }

    let endpoint = env::var("OTEL_ENDPOINT").ok()?;

    if endpoint.is_empty() {
        return None;
    }

    info!(endpoint = %endpoint, "Initializing OpenTelemetry (OTLP/gRPC)");

    let resource = build_resource(config);

    // --- Traces ---
    let span_exporter = SpanExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .build()
        .expect("Failed to create OTel span exporter");

    let tracer_provider = SdkTracerProvider::builder()
        .with_batch_exporter(span_exporter)
        .with_resource(resource.clone())
        .build();

    global::set_tracer_provider(tracer_provider.clone());

    // --- Metrics ---
    let metric_exporter = MetricExporter::builder()
        .with_tonic()
        .with_endpoint(&endpoint)
        .build()
        .expect("Failed to create OTel metric exporter");

    let meter_provider = SdkMeterProvider::builder()
        .with_periodic_exporter(metric_exporter)
        .with_resource(resource)
        .build();

    global::set_meter_provider(meter_provider.clone());

    // --- System-level observable gauges ---
    let meter = global::meter("inference-system");

    let _model_memory = meter
        .f64_observable_gauge("inference.model_memory_bytes")
        .with_description("Process RSS memory in bytes (proxy for model memory)")
        .with_unit("By")
        .with_callback(|observer| {
            let rss = system_metrics::process_memory_bytes();
            observer.observe(rss as f64, &[]);
        })
        .build();

    let _gpu_usage = meter
        .f64_observable_gauge("inference.gpu_usage_percent")
        .with_description("GPU compute utilization percentage")
        .with_unit("%")
        .with_callback(|observer| {
            let (util, _, _, _) = system_metrics::gpu_stats();
            observer.observe(util, &[]);
        })
        .build();

    let _gpu_memory_used = meter
        .f64_observable_gauge("inference.gpu_memory_used_bytes")
        .with_description("GPU memory used in bytes")
        .with_unit("By")
        .with_callback(|observer| {
            let (_, used, _, _) = system_metrics::gpu_stats();
            observer.observe(used as f64, &[]);
        })
        .build();

    let _gpu_memory_total = meter
        .f64_observable_gauge("inference.gpu_memory_total_bytes")
        .with_description("GPU memory total in bytes")
        .with_unit("By")
        .with_callback(|observer| {
            let (_, _, total, _) = system_metrics::gpu_stats();
            observer.observe(total as f64, &[]);
        })
        .build();

    let _gpu_power = meter
        .f64_observable_gauge("inference.gpu_power_watts")
        .with_description("GPU power draw in watts")
        .with_unit("W")
        .with_callback(|observer| {
            let (_, _, _, watts) = system_metrics::gpu_stats();
            observer.observe(watts, &[]);
        })
        .build();

    info!("OpenTelemetry initialized — traces and metrics pushing to {endpoint}");

    Some(TelemetryGuard {
        tracer_provider,
        meter_provider,
    })
}
