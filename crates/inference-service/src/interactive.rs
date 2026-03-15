//! Interactive model selection using dialoguer.
//!
//! Provides a guided flow for discovering and running models:
//! 1. Select a task family (NLP, Audio, Vision, Multimodal)
//! 2. Select a specific task (text-generation, fill-mask, etc.)
//! 3. Fetch popular models from HuggingFace and select one
//! 4. Confirm and start the server

use console::style;
use dialoguer::{theme::ColorfulTheme, FuzzySelect, Select};

use crate::hf_api::{self, format_downloads, HfModelSummary, TASK_FAMILIES};

/// Result of the interactive selection flow.
pub struct Selection {
    /// Selected HuggingFace model ID (e.g., `google-bert/bert-base-uncased`)
    pub model_id: String,
}

/// Run the full interactive flow: family -> task -> model.
///
/// Returns `None` if the user presses Escape at any point.
pub async fn select_model() -> anyhow::Result<Option<Selection>> {
    let theme = ColorfulTheme::default();

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

    // Step 3: Fetch models and select one
    select_model_for_task(&theme, &task).await
}

/// Run the interactive flow starting from a known task: fetch models -> select.
///
/// Returns `None` if the user presses Escape.
pub async fn select_model_for_task_str(task: &str) -> anyhow::Result<Option<Selection>> {
    let theme = ColorfulTheme::default();
    select_model_for_task(&theme, task).await
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

/// Fetch models from HuggingFace for a task, then let user select one.
async fn select_model_for_task(
    theme: &ColorfulTheme,
    task: &str,
) -> anyhow::Result<Option<Selection>> {
    println!(
        "\n  {} models for {} ...",
        style("Fetching").dim(),
        style(task).yellow()
    );

    let models = hf_api::search_models(task, 20).await?;

    if models.is_empty() {
        anyhow::bail!("No models found for task '{task}'");
    }

    let model_items: Vec<String> = format_model_list(&models);

    let model_idx = FuzzySelect::with_theme(theme)
        .with_prompt("Select a model (type to filter)")
        .items(&model_items)
        .default(0)
        .interact_opt()?;

    let Some(model_idx) = model_idx else {
        return Ok(None);
    };

    let model_id = models[model_idx].id.clone();

    println!(
        "\n  {} {}\n",
        style("Selected:").green().bold(),
        style(&model_id).cyan(),
    );

    Ok(Some(Selection { model_id }))
}

/// Format model list for display in the selector.
fn format_model_list(models: &[HfModelSummary]) -> Vec<String> {
    // Find max model ID length for alignment
    let max_id_len = models
        .iter()
        .map(|m| m.id.len())
        .max()
        .unwrap_or(40)
        .min(55);

    models
        .iter()
        .map(|m| {
            format!(
                "{:<width$}  {:>8} dl",
                m.id,
                format_downloads(m.downloads),
                width = max_id_len,
            )
        })
        .collect()
}
