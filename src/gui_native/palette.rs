//! The command palette (Cmd/Ctrl+K).
//!
//! State lives on the [`Console`](super::Console); this module holds the
//! command catalog: what the palette offers, in what order, and under what
//! name. One list, so the palette, tooltips and tests cannot disagree about
//! what a command is called.

/// A palette command. Copy-only: each variant maps onto one console method,
/// so execution is a match in the console, not a stored closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Command {
    NewExperiment,
    LoadModel,
    RunExperiment,
    DuplicateExperiment,
    RerunExperiment,
    GoHome,
    GoExperiments,
    GoModels,
    GoRuns,
    GoSettings,
    ToggleInspector,
    ToggleSidebar,
    ToggleTheme,
    TogglePresentation,
    GoPrompt,
    GoIntervention,
    GoReview,
    EditIntervention,
}

impl Command {
    /// Catalog order: creation first, then navigation, then view toggles,
    /// then the experiment flow -- the order a new user discovers in and a
    /// regular user stops reading.
    pub(super) const ALL: [Command; 18] = [
        Command::NewExperiment,
        Command::LoadModel,
        Command::RunExperiment,
        Command::DuplicateExperiment,
        Command::RerunExperiment,
        Command::GoHome,
        Command::GoExperiments,
        Command::GoModels,
        Command::GoRuns,
        Command::GoSettings,
        Command::GoPrompt,
        Command::GoIntervention,
        Command::GoReview,
        Command::EditIntervention,
        Command::ToggleInspector,
        Command::ToggleSidebar,
        Command::ToggleTheme,
        Command::TogglePresentation,
    ];

    pub(super) fn label(self) -> &'static str {
        match self {
            Command::NewExperiment => "New experiment",
            Command::LoadModel => "Load model",
            Command::RunExperiment => "Run experiment",
            Command::DuplicateExperiment => "Duplicate experiment",
            Command::RerunExperiment => "Rerun experiment",
            Command::GoHome => "Go to Home",
            Command::GoExperiments => "Go to Experiments",
            Command::GoModels => "Go to Models",
            Command::GoRuns => "Go to Runs",
            Command::GoSettings => "Go to Settings",
            Command::GoPrompt => "Go to Prompt",
            Command::GoIntervention => "Go to Intervention",
            Command::GoReview => "Go to Review & results",
            Command::EditIntervention => "Edit intervention",
            Command::ToggleInspector => "Toggle inspector",
            Command::ToggleSidebar => "Toggle sidebar",
            Command::ToggleTheme => "Toggle theme",
            Command::TogglePresentation => "Presentation mode",
        }
    }

    /// The one-line consequence, so the palette predicts its results.
    pub(super) fn hint(self) -> &'static str {
        match self {
            Command::NewExperiment => "Start a fresh experiment",
            Command::LoadModel => "Load the selected model into memory",
            Command::RunExperiment => "Run the configured baseline + intervention pair",
            Command::DuplicateExperiment => "Branch from the last run to change one thing",
            Command::RerunExperiment => "Replay the last run exactly",
            Command::GoHome => "Recent runs and starting points",
            Command::GoExperiments => "Configure and run an experiment",
            Command::GoModels => "Local models and their state",
            Command::GoRuns => "Every run from this session",
            Command::GoSettings => "Appearance and defaults",
            Command::GoPrompt => "Step 1: model and prompt",
            Command::GoIntervention => "Step 2: internal change",
            Command::GoReview => "Step 3: evidence and results",
            Command::EditIntervention => "Change layer, strength or location",
            Command::ToggleInspector => "Show or hide the context inspector",
            Command::ToggleSidebar => "Show or hide the navigation sidebar",
            Command::ToggleTheme => "Cycle system, dark, light",
            Command::TogglePresentation => "Larger text, sidebar and inspector hidden",
        }
    }
}
