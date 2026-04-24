//! Build script that patches llama.cpp source to accept 3-element rope.dimension_sections.
//!
//! Qwen3.5/Qwen3.6 GGUF models use a 3-element `rope.dimension_sections` array,
//! but llama.cpp's `get_key_or_arr` template requires an exact match with the
//! expected length (4). This patch changes the check to allow shorter arrays
//! (padding with 0), which is backwards-compatible.

use std::path::{Path, PathBuf};

const MARKER: &str = "/* PATCHED: arr_info.length > n */";

fn main() {
    // Tell cargo to rerun this script if the patch marker changes
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_PATCHED");

    let target_file = find_llama_model_loader();
    match target_file {
        Some(path) => {
            apply_patch(&path);
        }
        None => {
            eprintln!(
                "WARNING: Could not find llama-model-loader.cpp to patch. \
                 Qwen3.5/Qwen3.6 GGUF models may fail to load."
            );
        }
    }
}

fn find_llama_model_loader() -> Option<PathBuf> {
    // Try CARGO_HOME first
    let cargo_home = std::env::var("CARGO_HOME").ok().unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        format!("{home}/.cargo")
    });

    let git_dir = PathBuf::from(cargo_home).join("git").join("checkouts");
    if git_dir.exists() {
        if let Some(path) = search_dir(&git_dir) {
            return Some(path);
        }
    }

    None
}

fn search_dir(dir: &Path) -> Option<PathBuf> {
    for entry in std::fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        // Check if this is the llama-model-loader.cpp
        let candidate = path.join("llama-model-loader.cpp");
        if candidate.exists() {
            // Verify it's in the llama.cpp/src directory
            if candidate.to_string_lossy().contains("llama.cpp/src") {
                return Some(candidate);
            }
        }

        // Recurse into subdirectories (max depth 3)
        if let Some(found) = search_dir(&path) {
            return Some(found);
        }
    }
    None
}

fn apply_patch(path: &Path) {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("WARNING: Could not read {}: {e}", path.display());
            return;
        }
    };

    // Check if already patched
    if content.contains(MARKER) {
        return;
    }

    // The patch changes: "if (n != arr_info.length) {" → "if (arr_info.length > n) { /* PATCHED */ }"
    let original = "if (n != arr_info.length) {";
    let patched = format!("if (arr_info.length > n) {{ {MARKER}");

    if !content.contains(original) {
        // Maybe the upstream fix was applied
        if content.contains("if (arr_info.length > n)") {
            // Upstream fixed it — no need to patch
            return;
        }
        eprintln!(
            "WARNING: Could not find patch target in {}. \
             The llama.cpp source may have changed. \
             Qwen3.5/Qwen3.6 GGUF models may fail to load.",
            path.display()
        );
        return;
    }

    let patched_content = content.replace(original, &patched);
    match std::fs::write(path, &patched_content) {
        Ok(()) => {
            println!("cargo:rustc-env=LLAMA_CPP_PATCHED=1");
        }
        Err(e) => {
            eprintln!(
                "WARNING: Could not write patched file {}: {e}. \
                 Qwen3.5/Qwen3.6 GGUF models may fail to load.",
                path.display()
            );
        }
    }
}
