//! Interactive model selection using dialoguer.
//!
//! Provides a guided flow for discovering and running models:
//! 1. Select essential runtime options (server mode, device, backend)
//! 2. Select a task family (NLP, Audio, Vision, Multimodal)
//! 3. Select a specific task (text-generation, fill-mask, etc.)
//! 4. Search for models on HuggingFace and select one
//! 5. Confirm and start the server

use console::{style, Key, Term};
use dialoguer::{theme::ColorfulTheme, Select};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use crate::hf_api::{self, format_downloads, HfModelSummary, TASK_FAMILIES};

/// Debounce delay for HuggingFace API search.
const SEARCH_DEBOUNCE: Duration = Duration::from_millis(400);

/// Maximum number of model rows to display.
const MAX_VISIBLE_MODELS: usize = 20;

/// Server protocol mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerMode {
    /// HTTP server only (OpenAI-compatible REST API)
    Http,
    /// gRPC server only
    Grpc,
    /// Both HTTP and gRPC
    Both,
}

impl std::fmt::Display for ServerMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerMode::Http => write!(f, "HTTP (OpenAI-compatible REST API)"),
            ServerMode::Grpc => write!(f, "gRPC"),
            ServerMode::Both => write!(f, "Both HTTP + gRPC"),
        }
    }
}

/// Essential runtime options selected interactively.
#[derive(Debug, Clone)]
pub struct Essentials {
    /// Server protocol mode.
    pub server_mode: ServerMode,
    /// Inference device: auto, cpu, gpu, metal, cuda.
    pub device: String,
    /// Backend: auto, onnx, candle, llama.
    pub backend: String,
}

/// Result of the interactive selection flow.
pub struct Selection {
    /// Selected HuggingFace model ID (e.g., `google-bert/bert-base-uncased`)
    pub model_id: String,
    /// Essential runtime options chosen before model selection.
    pub essentials: Essentials,
}

/// Select essential deployment options before model selection.
///
/// Returns `None` if the user presses Escape at any point.
/// Server mode options are filtered based on compiled features.
pub fn select_essentials(theme: &ColorfulTheme) -> anyhow::Result<Option<Essentials>> {
    println!(
        "\n  {} {}",
        style("Step 1/2:").bold().cyan(),
        style("Configure deployment").bold()
    );

    let http_available = cfg!(feature = "http");
    let grpc_available = cfg!(feature = "grpc");

    // --- Server mode ---
    // Always show all options — unavailable ones are greyed out with a hint.
    let server_options: Vec<(String, ServerMode, bool)> = vec![
        (
            format!(
                "gRPC only  (port 50051){}",
                if grpc_available { "" } else { "  ⚠ not compiled" }
            ),
            ServerMode::Grpc,
            grpc_available,
        ),
        (
            format!(
                "HTTP only  (port 8080) {}",
                if http_available { "" } else { " ⚠ not compiled" }
            ),
            ServerMode::Http,
            http_available,
        ),
        (
            format!(
                "Both       (gRPC + HTTP){}",
                if grpc_available && http_available {
                    ""
                } else {
                    " ⚠ partially unavailable"
                }
            ),
            ServerMode::Both,
            grpc_available && http_available,
        ),
    ];

    let server_labels: Vec<String> = server_options
        .iter()
        .map(|(label, _, available)| {
            if *available {
                label.clone()
            } else {
                style(label).dim().to_string()
            }
        })
        .collect();

    // Default to first available option
    let default_idx = server_options
        .iter()
        .position(|(_, _, avail)| *avail)
        .unwrap_or(0);

    let server_idx = Select::with_theme(theme)
        .with_prompt("Select server mode")
        .items(&server_labels)
        .default(default_idx)
        .interact_opt()?;
    let Some(server_idx) = server_idx else {
        return Ok(None);
    };
    let server_mode = server_options[server_idx].1;

    // Warn if user selected an unavailable mode
    if !server_options[server_idx].2 {
        println!(
            "\n  {} {}",
            style("⚠").yellow().bold(),
            style(format!(
                "Server mode '{}' is not compiled in. \
                 Rebuild with --features {} to enable it.",
                server_mode,
                match server_mode {
                    ServerMode::Grpc => "grpc",
                    ServerMode::Http => "http",
                    ServerMode::Both => "http,grpc",
                }
            ))
            .yellow()
        );
    }

    // --- Device ---
    let devices = vec![
        ("Auto  (GPU when available, CPU fallback)", "auto"),
        ("CPU   (no GPU)", "cpu"),
        ("GPU   (Metal on macOS, CUDA on Linux/Win)", "gpu"),
    ];
    let device_labels: Vec<_> = devices.iter().map(|(l, _)| *l).collect();
    let device_idx = Select::with_theme(theme)
        .with_prompt("Select device")
        .items(&device_labels)
        .default(0)
        .interact_opt()?;
    let Some(device_idx) = device_idx else {
        return Ok(None);
    };
    let device = devices[device_idx].1.to_string();

    // --- Backend ---
    let backends = vec![
        ("Auto    (detect from model files)", "auto"),
        ("ONNX    (best for embeddings, classification)", "onnx"),
        ("Candle  (best for modern LLMs: Qwen, Llama)", "candle"),
        ("Llama   (best for GGUF models)", "llama"),
    ];
    let backend_labels: Vec<_> = backends.iter().map(|(l, _)| *l).collect();
    let backend_idx = Select::with_theme(theme)
        .with_prompt("Select backend")
        .items(&backend_labels)
        .default(0)
        .interact_opt()?;
    let Some(backend_idx) = backend_idx else {
        return Ok(None);
    };
    let backend = backends[backend_idx].1.to_string();

    Ok(Some(Essentials {
        server_mode,
        device,
        backend,
    }))
}

/// Run the full interactive flow: essentials -> family -> task -> model.
///
/// Returns `None` if the user presses Escape at any point.
pub async fn select_model() -> anyhow::Result<Option<Selection>> {
    let theme = ColorfulTheme::default();

    // Step 0: Select essential runtime options
    let essentials = select_essentials(&theme)?;
    let Some(essentials) = essentials else {
        return Ok(None);
    };

    // Step 1: Select task family
    let family_items: Vec<String> = TASK_FAMILIES
        .iter()
        .map(|f| format!("{:<12} {}", f.name, style(f.description).dim()))
        .collect();

    println!("\n{}", style("Maiia AI Inference Service").bold().cyan());
    println!(
        "{}\n",
        style("Deploy any HuggingFace model with one command.").dim()
    );

    let family_idx = Select::with_theme(&theme)
        .with_prompt("Select a task family")
        .items(&family_items)
        .default(0)
        .interact_opt()?;

    let Some(family_idx) = family_idx else {
        return Ok(None);
    };

    let family = &TASK_FAMILIES[family_idx];

    // Step 2: Select task within family
    let task = select_task(&theme, family.tasks)?;
    let Some(task) = task else {
        return Ok(None);
    };

    // Step 3: Search models and select one
    dynamic_select_model(&task, essentials).await
}

/// Run the interactive flow starting from a known task: fetch models -> select.
///
/// Returns `None` if the user presses Escape.
pub async fn select_model_for_task_str(task: &str) -> anyhow::Result<Option<Selection>> {
    let theme = ColorfulTheme::default();

    // Step 0: Select essential runtime options
    let essentials = select_essentials(&theme)?;
    let Some(essentials) = essentials else {
        return Ok(None);
    };

    dynamic_select_model(task, essentials).await
}

/// Select a task from a list of (task_id, description) pairs.
fn select_task(theme: &ColorfulTheme, tasks: &[(&str, &str)]) -> anyhow::Result<Option<String>> {
    let task_items: Vec<String> = tasks
        .iter()
        .map(|(id, desc)| format!("{id:<42} {}", style(desc).dim()))
        .collect();

    let task_idx = Select::with_theme(theme)
        .with_prompt("Select a task")
        .items(&task_items)
        .default(0)
        .interact_opt()?;

    Ok(task_idx.map(|idx| tasks[idx].0.to_string()))
}

/// Dynamic model selector with live HuggingFace search.
///
/// Uses in-place cursor movement to redraw without flickering.
/// API calls are debounced: the search fires only after the user
/// stops typing for `SEARCH_DEBOUNCE`.
///
/// Controls:
/// - Type characters to filter / search
/// - Backspace to delete
/// - ↑ / ↓ to navigate the list
/// - Enter to select the highlighted model
/// - Esc or Ctrl-C to cancel
async fn dynamic_select_model(
    task: &str,
    essentials: Essentials,
) -> anyhow::Result<Option<Selection>> {
    let term = Term::stdout();
    let mut filter = String::new();
    let mut models = match hf_api::search_models(task, MAX_VISIBLE_MODELS).await {
        Ok(m) => m,
        Err(e) => {
            println!(
                "\n  {} Failed to fetch popular models: {e}\n",
                style("!").red().bold()
            );
            return Ok(None);
        }
    };
    let mut selected = 0usize;
    let mut rendered_lines: usize;
    let mut last_searched_filter = filter.clone();

    // Channel for key events from the blocking reader thread
    let (tx, mut rx) = mpsc::channel::<Key>(32);

    // Spawn a blocking thread that reads keys and forwards them.
    std::thread::spawn(move || {
        let term = Term::stdout();
        loop {
            let key = match term.read_key() {
                Ok(k) => k,
                Err(_) => break,
            };
            if tx.blocking_send(key).is_err() {
                break;
            }
        }
    });

    // Initial render
    rendered_lines = render_model_list(&term, task, &models, selected, &filter, 0);

    // Debounce state
    let mut last_typing: Option<Instant> = None;

    loop {
        // If debounce is pending, wait for either a key or the debounce timeout
        if let Some(last) = last_typing {
            let remaining = SEARCH_DEBOUNCE.saturating_sub(last.elapsed());
            if remaining == Duration::ZERO {
                // Debounce expired — fire search
                if filter != last_searched_filter {
                    let new_models = fetch_for_filter(task, &filter).await;
                    if !new_models.is_empty() {
                        models = new_models;
                        selected = 0;
                    }
                    last_searched_filter = filter.clone();
                }
                last_typing = None;
                rendered_lines =
                    render_model_list(&term, task, &models, selected, &filter, rendered_lines);
                continue;
            }

            // Wait for either a key or the debounce timer
            tokio::select! {
                key = rx.recv() => {
                    if let Some(key) = key {
                        if let Some(result) = handle_key(
                            key,
                            &mut filter,
                            &mut selected,
                            &mut models,
                            &mut rendered_lines,
                            &mut last_typing,
                            task,
                            &term,
                        )? {
                            return Ok(Some(Selection {
                                model_id: result,
                                essentials,
                            }));
                        }
                    }
                }
                _ = tokio::time::sleep(remaining) => {
                    // Timer expired — loop back to fire the search
                    continue;
                }
            }
        } else {
            // No debounce — block on next key
            let Some(key) = rx.recv().await else {
                return Ok(None);
            };
            if let Some(result) = handle_key(
                key,
                &mut filter,
                &mut selected,
                &mut models,
                &mut rendered_lines,
                &mut last_typing,
                task,
                &term,
            )? {
                return Ok(Some(Selection {
                    model_id: result,
                    essentials,
                }));
            }
        }
    }
}

/// Handle a key press. Returns `Some(model_id)` if the user selected a model.
fn handle_key(
    key: Key,
    filter: &mut String,
    selected: &mut usize,
    models: &mut Vec<HfModelSummary>,
    rendered_lines: &mut usize,
    last_typing: &mut Option<Instant>,
    task: &str,
    term: &Term,
) -> anyhow::Result<Option<String>> {
    match key {
        Key::Char(c) => {
            filter.push(c);
            *selected = 0;
            *last_typing = Some(Instant::now());
            *rendered_lines = render_model_list(term, task, models, *selected, filter, *rendered_lines);
            Ok(None)
        }
        Key::Backspace => {
            if filter.pop().is_some() {
                *selected = 0;
                *last_typing = Some(Instant::now());
                *rendered_lines = render_model_list(term, task, models, *selected, filter, *rendered_lines);
            }
            Ok(None)
        }
        Key::ArrowUp => {
            if *selected > 0 {
                *selected -= 1;
                *rendered_lines = render_model_list(term, task, models, *selected, filter, *rendered_lines);
            }
            Ok(None)
        }
        Key::ArrowDown => {
            if *selected + 1 < models.len() {
                *selected += 1;
                *rendered_lines = render_model_list(term, task, models, *selected, filter, *rendered_lines);
            }
            Ok(None)
        }
        Key::Enter => {
            if let Some(m) = models.get(*selected) {
                let model_id = m.id.clone();
                clear_rendered_lines(term, *rendered_lines);
                term.write_line(&format!(
                    "  {} {}\n",
                    style("Selected:").green().bold(),
                    style(&model_id).cyan(),
                ))?;
                Ok(Some(model_id))
            } else {
                Ok(None)
            }
        }
        Key::Escape | Key::CtrlC => {
            clear_rendered_lines(term, *rendered_lines);
            Ok(None) // signals cancellation
        }
        _ => Ok(None),
    }
}

/// Render the model list in-place.
///
/// Clears the previously rendered lines by moving the cursor up and
/// clearing each line, then draws the new state.  Returns the number
/// of lines rendered for the next call to clear.
fn render_model_list(
    term: &Term,
    task: &str,
    models: &[HfModelSummary],
    selected: usize,
    filter: &str,
    prev_lines: usize,
) -> usize {
    // Clear previous render
    clear_rendered_lines(term, prev_lines);

    let mut lines = 0usize;

    // Header
    let _ = term.write_line(&format!(
        "  {} {}",
        style("Task:").bold(),
        style(task).yellow()
    ));
    lines += 1;

    let _ = term.write_line(""); // blank line
    lines += 1;

    if models.is_empty() {
        let _ = term.write_line("  No models found. Type to search.");
        lines += 1;
    } else {
        for (i, m) in models.iter().enumerate() {
            let prefix = if i == selected {
                style("> ").green().bold()
            } else {
                style("  ").dim()
            };
            let line = format!(
                "{prefix}{:<55} {:>8} dl",
                m.id,
                format_downloads(m.downloads)
            );
            let _ = term.write_line(&line);
            lines += 1;
        }
    }

    // Blank line + filter + help
    let _ = term.write_line("");
    lines += 1;

    let _ = term.write_line(&format!("  {} {filter}_", style("Filter:").bold()));
    lines += 1;

    let _ = term.write_line(&format!(
        "  {}",
        style("Type to search, ↑↓ navigate, Enter select, Esc cancel").dim()
    ));
    lines += 1;

    lines
}

/// Clear a previously rendered block of lines by moving the cursor up
/// and clearing each line.
fn clear_rendered_lines(term: &Term, lines: usize) {
    for _ in 0..lines {
        let _ = term.move_cursor_up(1);
        let _ = term.clear_line();
    }
}

/// Fetch models based on the current filter text.
///
/// An empty (or whitespace-only) filter returns the popular models for the
/// task; otherwise a free-text search is performed on the HuggingFace Hub.
async fn fetch_for_filter(task: &str, filter: &str) -> Vec<HfModelSummary> {
    let trimmed = filter.trim();
    let result = if trimmed.is_empty() {
        hf_api::search_models(task, MAX_VISIBLE_MODELS).await
    } else {
        hf_api::search_models_by_query(trimmed, Some(task), MAX_VISIBLE_MODELS).await
    };

    match result {
        Ok(models) => models,
        Err(e) => {
            // Keep previous results on error; log silently.
            tracing::warn!("HF search failed for filter '{filter}': {e}");
            Vec::new()
        }
    }
}
