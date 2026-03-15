//! System-level metrics: process memory and GPU utilization.
//!
//! These are reported as OTel observable gauges — the SDK calls the callbacks
//! on each scrape interval. The values reflect current system state, not
//! per-request measurements.

use std::sync::Mutex;
use sysinfo::{get_current_pid, ProcessesToUpdate, System};

/// Collect current process RSS in bytes.
pub fn process_memory_bytes() -> u64 {
    // sysinfo::System is not Send, so we keep a thread-local-like approach
    // via a Mutex. This is only called once per scrape interval (~60s).
    static SYS: Mutex<Option<System>> = Mutex::new(None);

    let Ok(pid) = get_current_pid() else {
        return 0;
    };
    let mut guard = SYS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let sys = guard.get_or_insert_with(System::new);
    sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    sys.process(pid).map_or(0, sysinfo::Process::memory)
}

/// Collect GPU stats from NVML (CUDA) or return zeroes (CPU/Metal).
///
/// Returns `(gpu_util_percent, gpu_memory_used_bytes, gpu_memory_total_bytes, power_watts)`.
pub fn gpu_stats() -> (f64, u64, u64, f64) {
    #[cfg(feature = "cuda")]
    {
        cuda_gpu_stats()
    }
    #[cfg(not(feature = "cuda"))]
    {
        (0.0, 0, 0, 0.0)
    }
}

#[cfg(feature = "cuda")]
fn cuda_gpu_stats() -> (f64, u64, u64, f64) {
    use nvml_wrapper::Nvml;
    use std::sync::OnceLock;

    // Initialize NVML once — it's expensive to init repeatedly.
    static NVML: OnceLock<Option<Nvml>> = OnceLock::new();
    let nvml = NVML.get_or_init(|| Nvml::init().ok());

    let Some(nvml) = nvml else {
        return (0.0, 0, 0, 0.0);
    };

    // Use GPU 0 (primary device)
    let Ok(device) = nvml.device_by_index(0) else {
        return (0.0, 0, 0, 0.0);
    };

    let util = device
        .utilization_rates()
        .map(|u| f64::from(u.gpu))
        .unwrap_or(0.0);

    let (mem_used, mem_total) = device
        .memory_info()
        .map(|m| (m.used, m.total))
        .unwrap_or((0, 0));

    // power_usage() returns milliwatts — convert to watts.
    let power_watts = device
        .power_usage()
        .map(|mw| f64::from(mw) / 1000.0)
        .unwrap_or(0.0);

    (util, mem_used, mem_total, power_watts)
}
