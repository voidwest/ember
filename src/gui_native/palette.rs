//! The command palette (Cmd/Ctrl+K or Cmd/Ctrl+Shift+P) and Go to run
//! (Cmd/Ctrl+P).
//!
//! State lives on the [`Console`](super::Console); this module holds the
//! command catalog: what the palette offers, in what order, and under what
//! name. One list, so the palette, tooltips and tests cannot disagree about
//! what a command is called. The keys and the matching follow code editors,
//! so a VS Code or Zed user can drive the console without a manual.

/// What the palette lists. Go to run is the editor's quick open, with saved
/// runs in place of files; a leading `>` turns it into the command list, as
/// it does in VS Code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PaletteMode {
    Commands,
    Runs,
}

impl PaletteMode {
    /// The query field's hint, which names the list it searches.
    pub(super) fn placeholder(self) -> &'static str {
        match self {
            PaletteMode::Commands => "Type a command…",
            PaletteMode::Runs => "A run number, or words from its prompt…",
        }
    }
}

/// A palette command. Copy-only: each variant maps onto one console method,
/// so execution is a match in the console, not a stored closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Command {
    NewExperiment,
    LoadModel,
    RunExperiment,
    CancelRun,
    DuplicateExperiment,
    RerunExperiment,
    GoHome,
    GoExperiments,
    GoModels,
    GoRuns,
    GoSettings,
    GoToRun,
    ShowOverview,
    ShowLayers,
    ShowTokens,
    ShowTrace,
    KeyboardShortcuts,
    ToggleSidebar,
    ToggleTheme,
    TogglePresentation,
}

impl Command {
    /// Catalog order: creation first, then navigation, then view toggles,
    /// then the experiment flow -- the order a new user discovers in and a
    /// regular user stops reading.
    pub(super) const ALL: [Command; 20] = [
        Command::NewExperiment,
        Command::LoadModel,
        Command::RunExperiment,
        Command::CancelRun,
        Command::DuplicateExperiment,
        Command::RerunExperiment,
        Command::GoHome,
        Command::GoExperiments,
        Command::GoModels,
        Command::GoRuns,
        Command::GoSettings,
        Command::GoToRun,
        Command::ShowOverview,
        Command::ShowLayers,
        Command::ShowTokens,
        Command::ShowTrace,
        Command::KeyboardShortcuts,
        Command::ToggleSidebar,
        Command::ToggleTheme,
        Command::TogglePresentation,
    ];

    pub(super) fn label(self) -> &'static str {
        match self {
            Command::NewExperiment => "New experiment",
            Command::LoadModel => "Load model",
            Command::RunExperiment => "Run experiment",
            Command::CancelRun => "Cancel run",
            Command::DuplicateExperiment => "Duplicate experiment",
            Command::RerunExperiment => "Rerun experiment",
            Command::GoHome => "Go to Home",
            Command::GoExperiments => "Go to Experiments",
            Command::GoModels => "Go to Models",
            Command::GoRuns => "Go to Runs",
            Command::GoSettings => "Go to Settings",
            Command::GoToRun => "Go to run…",
            Command::ShowOverview => "Show Overview",
            Command::ShowLayers => "Show Layers",
            Command::ShowTokens => "Show Tokens",
            Command::ShowTrace => "Show Raw trace",
            Command::KeyboardShortcuts => "Keyboard shortcuts",
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
            Command::CancelRun => "Stop the run in flight; nothing is recorded",
            Command::DuplicateExperiment => "Branch from the last run to change one thing",
            Command::RerunExperiment => "Replay the last run exactly",
            Command::GoHome => "Recent runs and starting points",
            Command::GoExperiments => "Configure and run an experiment",
            Command::GoModels => "Local models and their state",
            Command::GoRuns => "Every run from this session",
            Command::GoSettings => "Appearance and defaults",
            Command::GoToRun => "Open a saved run by number or prompt",
            Command::ShowOverview => "Result tab: outputs and landmarks",
            Command::ShowLayers => "Result tab: divergence by layer",
            Command::ShowTokens => "Result tab: token-by-token comparison",
            Command::ShowTrace => "Result tab: the raw event trace",
            Command::KeyboardShortcuts => "Every key the console answers to",
            Command::ToggleSidebar => "Show or hide the navigation sidebar",
            Command::ToggleTheme => "Cycle system, dark, light",
            Command::TogglePresentation => "Larger text, sidebar hidden",
        }
    }
}

/// How well `query` matches `text` as a fuzzy subsequence, the way editor
/// palettes match: case-insensitive, spaces in the query ignored, `None` when
/// the characters do not all appear in order. A match at the start of a word,
/// or straight after the previous match, scores more, so "tsb" puts "Toggle
/// sidebar" first and a whole-word query beats a scattered one.
pub(super) fn fuzzy_score(query: &str, text: &str) -> Option<u32> {
    let mut wanted = query
        .chars()
        .filter(|c| !c.is_whitespace())
        .flat_map(char::to_lowercase)
        .peekable();
    let mut score = 0;
    let mut previous: Option<char> = None;
    let mut adjacent = false;
    for c in text.chars() {
        let Some(&next) = wanted.peek() else {
            break;
        };
        let word_start = previous.is_none_or(|p| !p.is_alphanumeric());
        if c.to_lowercase().eq(std::iter::once(next)) {
            score += 1;
            if word_start {
                score += 8;
            }
            if adjacent {
                score += 4;
            }
            wanted.next();
            adjacent = true;
        } else {
            adjacent = false;
        }
        previous = Some(c);
    }
    wanted.peek().is_none().then_some(score)
}

/// How well `query` matches a saved run in Go to run, or `None`. The run's
/// own number, with or without `#`, ranks above everything, so `2` opens
/// run #2 and not #29. Then comes a fuzzy match on `#number prompt`, then
/// any run whose `detail` (the change and the model) holds the query.
pub(super) fn run_score(query: &str, number: u64, prompt: &str, detail: &str) -> Option<u32> {
    let query = query.trim();
    if query.is_empty() {
        return Some(0);
    }
    if query.trim_start_matches('#').parse::<u64>() == Ok(number) {
        return Some(u32::MAX);
    }
    fuzzy_score(query, &format!("#{number} {prompt}")).or_else(|| {
        detail
            .to_lowercase()
            .contains(&query.to_lowercase())
            .then_some(0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn go_to_run_puts_the_typed_run_number_first() {
        let rank = |query: &str| {
            let mut scored: Vec<(u32, u64)> = std::iter::once(2)
                .chain(20..30)
                .filter_map(|number| {
                    run_score(query, number, "The capital of France is", "Zero · Llama")
                        .map(|score| (score, number))
                })
                .collect();
            scored.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.cmp(&a.1)));
            scored
                .into_iter()
                .map(|(_, number)| number)
                .collect::<Vec<_>>()
        };
        assert_eq!(rank("2").first(), Some(&2));
        assert_eq!(rank("#2").first(), Some(&2));
        assert_eq!(rank("25").first(), Some(&25));
        assert_eq!(rank("").first(), Some(&29), "newest first");
        // The whole prompt is searched, then the change and the model.
        assert_eq!(rank("france").len(), 11);
        assert_eq!(rank("llama").len(), 11);
        assert!(rank("qwen").is_empty());
    }

    #[test]
    fn a_fuzzy_query_matches_characters_in_order() {
        assert!(fuzzy_score("tsb", "Toggle sidebar").is_some());
        assert!(fuzzy_score("go runs", "Go to Runs").is_some());
        assert!(fuzzy_score("RUN", "Rerun experiment").is_some());
        assert_eq!(fuzzy_score("bst", "Toggle sidebar"), None, "order counts");
        assert_eq!(fuzzy_score("zz", "Toggle sidebar"), None);
        assert_eq!(fuzzy_score("", "Toggle sidebar"), Some(0));
    }

    #[test]
    fn word_starts_and_adjacent_characters_rank_first() {
        let rank = |query: &str| {
            let mut scored: Vec<(u32, &str)> = Command::ALL
                .iter()
                .filter_map(|command| {
                    fuzzy_score(query, command.label()).map(|score| (score, command.label()))
                })
                .collect();
            scored.sort_by_key(|&(score, _)| std::cmp::Reverse(score));
            scored
                .into_iter()
                .map(|(_, label)| label)
                .collect::<Vec<_>>()
        };
        assert_eq!(rank("tsb").first(), Some(&"Toggle sidebar"));
        assert_eq!(rank("gtr").first(), Some(&"Go to Runs"));
        assert_eq!(rank("layers").first(), Some(&"Show Layers"));
        assert_eq!(rank("set").first(), Some(&"Go to Settings"));
    }
}
