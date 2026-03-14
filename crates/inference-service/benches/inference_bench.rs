//! Criterion benchmarks for inference task execution.
//!
//! Measures framework overhead and batching throughput using the lightweight
//! EchoTask (no model, no I/O). This establishes a baseline for the minimum
//! latency added by the task dispatch and gRPC layers.
//!
//! Run with:
//!   cargo bench -p inference-service
//!
//! Generate HTML reports:
//!   cargo bench -p inference-service -- --output-format bencher

use std::sync::Arc;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use inference_core::task::{Task, TaskResult};
use inference_grpc::batcher::BatchScheduler;
use inference_tasks::EchoTask;
use tokio::runtime::Runtime;

/// Simple JSON payload for benchmarks.
const ECHO_PAYLOAD: &str = r#"{"message":"hello","number":42}"#;

// ============================================================================
// 1. Direct Task::execute() — single request baseline
// ============================================================================

fn bench_echo_execute(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let task = EchoTask::new("bench.echo.v1");

    c.bench_function("echo_execute_single", |b| {
        b.to_async(&rt).iter(|| async {
            let result = task.execute(black_box(ECHO_PAYLOAD), "bench-req").await;
            black_box(result)
        });
    });
}

// ============================================================================
// 2. Direct Task::execute_batch() — sequential default impl
// ============================================================================

fn bench_echo_execute_batch(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let task = EchoTask::new("bench.echo.v1");

    let mut group = c.benchmark_group("echo_execute_batch");
    for batch_size in [1, 4, 8, 16, 32] {
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                let payloads: Vec<(&str, &str)> =
                    (0..size).map(|_| (ECHO_PAYLOAD, "bench-req")).collect();
                b.to_async(&rt).iter(|| async {
                    let results = task.execute_batch(black_box(&payloads)).await;
                    black_box(results)
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// 3. BatchScheduler throughput — measures batcher overhead
// ============================================================================

fn bench_batcher_single(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let task: Arc<dyn Task> = Arc::new(EchoTask::new("bench.echo.v1"));

    let handle = rt.block_on(async {
        BatchScheduler::spawn(Arc::clone(&task), 32, Duration::from_millis(100), 128)
    });

    c.bench_function("batcher_single_request", |b| {
        b.to_async(&rt).iter(|| {
            let h = handle.clone();
            async move {
                let result = h
                    .submit(ECHO_PAYLOAD.to_string(), "bench-req".to_string())
                    .await;
                black_box(result)
            }
        });
    });
}

fn bench_batcher_concurrent(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let task: Arc<dyn Task> = Arc::new(EchoTask::new("bench.echo.v1"));

    let mut group = c.benchmark_group("batcher_concurrent");

    for concurrency in [4, 8, 16, 32] {
        group.throughput(Throughput::Elements(concurrency as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(concurrency),
            &concurrency,
            |b, &n| {
                let handle = rt.block_on(async {
                    BatchScheduler::spawn(Arc::clone(&task), 32, Duration::from_millis(5), 128)
                });

                b.to_async(&rt).iter(|| {
                    let h = handle.clone();
                    async move {
                        let mut tasks = Vec::with_capacity(n);
                        for i in 0..n {
                            let hh = h.clone();
                            tasks.push(tokio::spawn(async move {
                                hh.submit(ECHO_PAYLOAD.to_string(), format!("bench-req-{i}"))
                                    .await
                            }));
                        }
                        let results: Vec<Option<TaskResult>> = futures::future::join_all(tasks)
                            .await
                            .into_iter()
                            .map(|r| r.unwrap())
                            .collect();
                        black_box(results)
                    }
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// 4. Payload size impact — how payload size affects overhead
// ============================================================================

fn bench_payload_sizes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let task = EchoTask::new("bench.echo.v1");

    let mut group = c.benchmark_group("echo_payload_size");

    for size_bytes in [64_usize, 1024, 16_384, 131_072, 1_048_576] {
        let label = if size_bytes >= 1_048_576 {
            format!("{}MiB", size_bytes / 1_048_576)
        } else if size_bytes >= 1024 {
            format!("{}KiB", size_bytes / 1024)
        } else {
            format!("{size_bytes}B")
        };

        // Build a payload of approximately the target size.
        let filler = "x".repeat(size_bytes.saturating_sub(30));
        let payload = format!(r#"{{"data":"{filler}"}}"#);

        group.throughput(Throughput::Bytes(payload.len() as u64));
        group.bench_with_input(BenchmarkId::new("execute", &label), &payload, |b, p| {
            b.to_async(&rt).iter(|| async {
                let result = task.execute(black_box(p), "bench-req").await;
                black_box(result)
            });
        });
    }
    group.finish();
}

// ============================================================================
// Group and main
// ============================================================================

criterion_group!(
    benches,
    bench_echo_execute,
    bench_echo_execute_batch,
    bench_batcher_single,
    bench_batcher_concurrent,
    bench_payload_sizes,
);
criterion_main!(benches);
