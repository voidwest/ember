//! v0.5 token-selection primitives (contract sections 5-7).
//!
//! Token selection is a typed, fail-closed operation, never an incidental
//! integer index. Every selection produces a machine-readable record.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Which subtokens of a matched span to select.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SubtokenSelection {
    First,
    Final,
    All,
}

/// Optional text normalization for span matching.
///
/// The default (`none`) never normalizes; `nfc` is an explicit opt-in that
/// matches against an NFC-normalized copy of the input (recorded in the
/// selection record). Arabic text is never normalized silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TextNormalization {
    /// No normalization (default).
    #[default]
    None,
    Nfc,
}

/// Typed token selector (contract section 6).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum TokenSelector {
    /// The final token of the complete model input (BOS included).
    PromptFinal,
    /// The token at an explicit 0-based model-input position.
    AbsoluteToken { index: usize },
    /// `seq_len - 1 - offset`; offset 0 equals `prompt-final`.
    RelativeToken { offset_from_end: usize },
    /// The token generated at decode step `step` (1-based), observed at the
    /// decode evaluation processing it.
    GeneratedStep { step: usize },
    /// Exact text-span match with occurrence resolution.
    #[serde(rename = "matched-span")]
    MatchedTextSpan {
        text: String,
        occurrence: usize,
        #[serde(rename = "subtokens")]
        subtoken_selection: SubtokenSelection,
        #[serde(default)]
        normalization: TextNormalization,
    },
    /// Explicit byte span into the prompt text.
    ByteSpan {
        start: usize,
        end: usize,
        #[serde(rename = "subtokens")]
        subtoken_selection: SubtokenSelection,
    },
}

impl TokenSelector {
    /// Whether this selector depends on generated tokens (resolved after
    /// generation, not at prefill).
    pub const fn is_generated(&self) -> bool {
        matches!(self, TokenSelector::GeneratedStep { .. })
    }

    /// Whether this selector requires the prompt text to be present.
    pub const fn requires_text(&self) -> bool {
        matches!(
            self,
            TokenSelector::MatchedTextSpan { .. } | TokenSelector::ByteSpan { .. }
        )
    }

    /// Human-readable rule id for records.
    pub fn rule_id(&self) -> &'static str {
        match self {
            TokenSelector::PromptFinal => "prompt-final",
            TokenSelector::AbsoluteToken { .. } => "absolute-token",
            TokenSelector::RelativeToken { .. } => "relative-token",
            TokenSelector::GeneratedStep { .. } => "generated-step",
            TokenSelector::MatchedTextSpan { .. } => "matched-span",
            TokenSelector::ByteSpan { .. } => "byte-span",
        }
    }
}

impl fmt::Display for TokenSelector {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenSelector::PromptFinal => formatter.write_str("prompt-final"),
            TokenSelector::AbsoluteToken { index } => {
                write!(formatter, "absolute-token[{index}]")
            }
            TokenSelector::RelativeToken { offset_from_end } => {
                write!(formatter, "relative-token[-{offset_from_end}]")
            }
            TokenSelector::GeneratedStep { step } => write!(formatter, "generated-step[{step}]"),
            TokenSelector::MatchedTextSpan {
                text,
                occurrence,
                subtoken_selection,
                normalization,
            } => write!(
                formatter,
                "matched-span[{text:?}, occurrence {occurrence}, {subtoken_selection:?}, \
                 normalization {normalization:?}]"
            ),
            TokenSelector::ByteSpan {
                start,
                end,
                subtoken_selection,
            } => write!(
                formatter,
                "byte-span[{start}..{end}, {subtoken_selection:?}]"
            ),
        }
    }
}

/// Coverage of a matched span by selected tokens (contract section 7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CoverageKind {
    /// The union of selected token byte intervals equals the span.
    Exact,
    /// The union strictly contains the span (token boundaries exceed it).
    Enclosing,
    /// The selected token(s) intersect the span but do not cover it (a
    /// `first`/`final` subtoken selection on a multi-token span).
    Partial,
    /// The span was not covered by any token interval.
    None,
}

/// Ambiguity status of a selection (contract section 15).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AmbiguityStatus {
    /// The span text appears exactly once, or `occurrence` resolved a
    /// repeated span.
    Resolved,
    /// The span was absent, or the tokenizer produced no usable offsets.
    Failed,
}

/// Round-trip status of the selected pieces (contract section 7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RoundTripStatus {
    /// Decoding the selected pieces reproduces the span text exactly.
    Exact,
    /// Decoding does not reproduce the span text; reason recorded.
    Partial,
    /// No round trip attempted (selector does not reconstruct text).
    NotApplicable,
}

/// Tokenizer output for one input text.
///
/// The `tokenizers` crate reports byte offsets relative to the original
/// (normalized) string; they are recorded as-is.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizationInfo {
    /// Original input text.
    pub text: String,
    /// Text used for matching (equals `text` unless normalization was
    /// requested).
    pub normalized_text: String,
    /// Tokenizer output IDs (BOS included when the tokenizer defines one).
    pub token_ids: Vec<u32>,
    /// Token pieces (tokenizer vocabulary strings) when available.
    pub pieces: Vec<String>,
    /// Byte offsets `[start, end)` per token, in the `normalized_text`.
    pub byte_offsets: Vec<(usize, usize)>,
}

impl TokenizationInfo {
    /// Build tokenization info from the wrapper's output. Offsets are
    /// byte offsets relative to `normalized_text`.
    pub fn new(
        text: &str,
        normalized_text: &str,
        token_ids: Vec<u32>,
        pieces: Vec<String>,
        byte_offsets: Vec<(usize, usize)>,
    ) -> TokenizationInfo {
        TokenizationInfo {
            text: text.to_string(),
            normalized_text: normalized_text.to_string(),
            token_ids,
            pieces,
            byte_offsets,
        }
    }

    /// Total input byte length (of the normalized text).
    pub fn byte_len(&self) -> usize {
        self.normalized_text.len()
    }
}

/// Tokenize `text` for selection with the requested normalization.
///
/// When `Nfc` is requested, the NFC form of the text is what gets
/// tokenized and matched against; the record preserves both forms.
pub fn tokenize_for_selection(
    tokenizer: &crate::tokenizer::EmberTokenizer,
    text: &str,
    normalization: TextNormalization,
) -> Result<TokenizationInfo, String> {
    let normalized_text: String = match normalization {
        TextNormalization::None => text.to_string(),
        TextNormalization::Nfc => {
            use unicode_normalization::UnicodeNormalization;
            text.nfc().collect()
        }
    };
    let (token_ids, char_offsets) = tokenizer
        .encode_with_offsets(&normalized_text)
        .map_err(|error| format!("tokenizer encode failed: {error}"))?;
    let pieces = token_ids
        .iter()
        .map(|&id| {
            tokenizer
                .token_piece(id)
                .unwrap_or_else(|| format!("<{id}>"))
        })
        .collect();
    Ok(TokenizationInfo::new(
        text,
        &normalized_text,
        token_ids,
        pieces,
        char_offsets,
    ))
}

/// The full machine-readable record of one token selection (contract
/// section 6).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenSelectionRecord {
    /// The selector as resolved (normalization applied).
    pub selector: TokenSelector,
    /// Selection rule id (`prompt-final`, `absolute-token`, ...).
    pub rule: String,
    /// Original input text.
    pub input_text: String,
    /// Text the match ran against (input text when no normalization).
    pub normalized_text: String,
    /// Complete tokenizer output IDs for the input.
    pub token_ids: Vec<u32>,
    /// Token pieces where available.
    pub pieces: Vec<String>,
    /// Byte offsets per token (into `normalized_text`).
    pub byte_offsets: Vec<(usize, usize)>,
    /// The matched byte span, when the selector matches a span.
    pub matched_byte_span: Option<(usize, usize)>,
    /// Selected token indices (absolute positions in the model input).
    pub selected_indices: Vec<usize>,
    pub coverage: CoverageKind,
    /// Bytes added by token-boundary expansion, when coverage is enclosing.
    pub boundary_expansion: Option<(usize, usize)>,
    pub ambiguity: AmbiguityStatus,
    pub round_trip: RoundTripStatus,
    /// Reason when round trip is partial, or fallback used.
    pub note: Option<String>,
}

/// Select a token row within a `[rows, cols]` tensor (prefill) or a single
/// row (decode), returning the absolute token index or an error.
fn select_static(
    selector: &TokenSelector,
    info: &TokenizationInfo,
) -> Result<SelectionOutcome, String> {
    let seq_len = info.token_ids.len();
    match selector {
        TokenSelector::PromptFinal => {
            if seq_len == 0 {
                return Err("prompt-final: the prompt tokenizes to an empty sequence".into());
            }
            Ok((vec![seq_len - 1], SelectionDetail::plain()))
        }
        TokenSelector::AbsoluteToken { index } => {
            if *index >= seq_len {
                return Err(format!(
                    "absolute-token[{index}]: out of range for a {seq_len}-token input"
                ));
            }
            Ok((vec![*index], SelectionDetail::plain()))
        }
        TokenSelector::RelativeToken { offset_from_end } => {
            if *offset_from_end >= seq_len {
                return Err(format!(
                    "relative-token[-{offset_from_end}]: out of range for a {seq_len}-token input"
                ));
            }
            Ok((
                vec![seq_len - 1 - offset_from_end],
                SelectionDetail::plain(),
            ))
        }
        TokenSelector::GeneratedStep { .. } => {
            Err("generated-step requires post-generation resolution".to_string())
        }
        TokenSelector::MatchedTextSpan {
            text,
            occurrence,
            subtoken_selection,
            normalization,
        } => {
            if text.is_empty() {
                return Err("matched-span: the match text must not be empty".into());
            }
            let (haystack, needle) = match normalization {
                TextNormalization::None => (info.normalized_text.clone(), text.clone()),
                TextNormalization::Nfc => {
                    use unicode_normalization::UnicodeNormalization;
                    let haystack = info.normalized_text.nfc().collect::<String>();
                    let needle = text.nfc().collect::<String>();
                    // The info's byte offsets are relative to the NFC form,
                    // so the NFC haystack must be the info's own text.
                    if haystack != info.normalized_text {
                        return Err(
                            "matched-span: NFC normalization was requested but the tokenization \
                             was built without it"
                                .into(),
                        );
                    }
                    (haystack, needle)
                }
            };
            let matches: Vec<(usize, &str)> = haystack.match_indices(&needle).collect();
            if matches.is_empty() {
                return Err(format!(
                    "matched-span: text {text:?} is absent from the prompt"
                ));
            }
            let Some(&(span_start, span_text)) = matches.get(*occurrence) else {
                return Err(format!(
                    "matched-span: occurrence {occurrence} of {text:?} does not exist \
                     (found {} occurrence(s))",
                    matches.len()
                ));
            };
            let span_end = span_start + span_text.len();
            let (indices, coverage, expansion, note) =
                select_covering_tokens(info, span_start, span_end, *subtoken_selection);
            if coverage == CoverageKind::None {
                return Err(format!(
                    "matched-span: no token covers byte span {span_start}..{span_end} of \
                     {text:?}; tokenizer offsets may be unavailable"
                ));
            }
            Ok((
                indices,
                SelectionDetail {
                    matched_byte_span: Some((span_start, span_end)),
                    coverage,
                    boundary_expansion: expansion,
                    note,
                },
            ))
        }
        TokenSelector::ByteSpan {
            start,
            end,
            subtoken_selection,
        } => {
            if start >= end {
                return Err(format!("byte-span[{start}..{end}]: empty or reversed span"));
            }
            let total = info.byte_len();
            if *end > total {
                return Err(format!(
                    "byte-span[{start}..{end}]: end exceeds prompt byte length {total}"
                ));
            }
            let (indices, coverage, expansion, note) =
                select_covering_tokens(info, *start, *end, *subtoken_selection);
            if coverage == CoverageKind::None {
                return Err(format!(
                    "byte-span[{start}..{end}]: no token covers the span"
                ));
            }
            Ok((
                indices,
                SelectionDetail {
                    matched_byte_span: Some((*start, *end)),
                    coverage,
                    boundary_expansion: expansion,
                    note,
                },
            ))
        }
    }
}

/// Outcome of one static selection: indices plus span detail.
type SelectionOutcome = (Vec<usize>, SelectionDetail);

struct SelectionDetail {
    matched_byte_span: Option<(usize, usize)>,
    coverage: CoverageKind,
    boundary_expansion: Option<(usize, usize)>,
    note: Option<String>,
}

impl SelectionDetail {
    fn plain() -> SelectionDetail {
        SelectionDetail {
            matched_byte_span: None,
            coverage: CoverageKind::Exact,
            boundary_expansion: None,
            note: None,
        }
    }
}

/// Outcome of a covering-token selection:
/// `(selected indices, coverage, boundary expansion, note)`.
type CoveringSelection = (
    Vec<usize>,
    CoverageKind,
    Option<(usize, usize)>,
    Option<String>,
);

/// Select the token indices whose byte intervals intersect `[start, end)`,
/// per the subtoken rule, and classify coverage.
fn select_covering_tokens(
    info: &TokenizationInfo,
    start: usize,
    end: usize,
    subtoken_selection: SubtokenSelection,
) -> CoveringSelection {
    let mut intersecting = Vec::new();
    for (index, &(bs, be)) in info.byte_offsets.iter().enumerate() {
        // A token intersects the span when its interval overlaps it.
        // Zero-width tokens (e.g. BOS at (0,0)) never intersect a
        // non-empty span.
        if bs < end && be > start && be > bs {
            intersecting.push(index);
        }
    }
    if intersecting.is_empty() {
        return (Vec::new(), CoverageKind::None, None, None);
    }
    let selected = match subtoken_selection {
        SubtokenSelection::First => vec![intersecting[0]],
        SubtokenSelection::Final => vec![*intersecting.last().unwrap()],
        SubtokenSelection::All => intersecting,
    };
    // Coverage: union of selected intervals vs the span.
    let mut union_start = usize::MAX;
    let mut union_end = 0usize;
    for &index in &selected {
        let (bs, be) = info.byte_offsets[index];
        union_start = union_start.min(bs);
        union_end = union_end.max(be);
    }
    let coverage = if union_start <= start && union_end >= end {
        if union_start == start && union_end == end {
            CoverageKind::Exact
        } else {
            CoverageKind::Enclosing
        }
    } else if union_start < end && union_end > start {
        CoverageKind::Partial
    } else {
        CoverageKind::None
    };
    let expansion = if coverage == CoverageKind::Enclosing {
        Some((start - union_start, union_end - end))
    } else {
        None
    };
    (selected, coverage, expansion, None)
}

/// Resolve a static (non-generated) selector against tokenizer output.
///
/// Fails closed with a message naming the selector and the reason
/// (contract section 15).
pub fn resolve_static_selector(
    selector: &TokenSelector,
    info: &TokenizationInfo,
) -> Result<TokenSelectionRecord, String> {
    if selector.is_generated() {
        return Err("generated-step cannot be resolved before generation".into());
    }
    let (selected_indices, detail) = select_static(selector, info)?;
    let mut sorted = selected_indices.clone();
    sorted.sort_unstable();
    sorted.dedup();
    let round_trip = round_trip_status(info, &sorted, detail.matched_byte_span);
    Ok(TokenSelectionRecord {
        selector: selector.clone(),
        rule: selector.rule_id().to_string(),
        input_text: info.text.clone(),
        normalized_text: info.normalized_text.clone(),
        token_ids: info.token_ids.clone(),
        pieces: info.pieces.clone(),
        byte_offsets: info.byte_offsets.clone(),
        matched_byte_span: detail.matched_byte_span,
        selected_indices: sorted,
        coverage: detail.coverage,
        boundary_expansion: detail.boundary_expansion,
        ambiguity: AmbiguityStatus::Resolved,
        round_trip,
        note: detail.note,
    })
}

fn round_trip_status(
    info: &TokenizationInfo,
    selected: &[usize],
    span: Option<(usize, usize)>,
) -> RoundTripStatus {
    let Some((start, end)) = span else {
        return RoundTripStatus::NotApplicable;
    };
    if selected.is_empty() {
        return RoundTripStatus::NotApplicable;
    }
    // Byte-level comparison: the union of selected token byte intervals
    // must reproduce the span bytes exactly. This is the only deterministic
    // reconstruction that does not depend on tokenizer decode quirks.
    let expected = &info.normalized_text.as_bytes()[start..end];
    let mut concatenated = Vec::new();
    for &index in selected {
        let (bs, be) = info.byte_offsets[index];
        if bs >= be {
            continue;
        }
        let bytes = &info.normalized_text.as_bytes()[bs..be];
        concatenated.extend_from_slice(bytes);
    }
    if concatenated == expected {
        RoundTripStatus::Exact
    } else {
        RoundTripStatus::Partial
    }
}

/// Resolve a generated-step selector after generation.
///
/// `decoded_positions` maps decode evaluation `k` (0-based) to the
/// generated token at that position; positions are recorded during decode.
pub fn resolve_generated_step(
    step: usize,
    captured_positions: &[usize],
) -> Result<Vec<usize>, String> {
    if step == 0 {
        return Err(
            "generated-step: steps are 1-based (step 1 is the first generated token)".into(),
        );
    }
    let Some(&position) = captured_positions.get(step - 1) else {
        return Err(format!(
            "generated-step[{step}]: generation produced only {} token(s)",
            captured_positions.len()
        ));
    };
    Ok(vec![position])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info_from(
        text: &str,
        ids: Vec<u32>,
        pieces: Vec<String>,
        offsets: Vec<(usize, usize)>,
    ) -> TokenizationInfo {
        // Offsets are byte offsets (the tokenizers crate semantics).
        let byte_offsets = offsets;
        TokenizationInfo {
            text: text.to_string(),
            normalized_text: text.to_string(),
            token_ids: ids,
            pieces,
            byte_offsets,
        }
    }

    #[test]
    fn prompt_final_selects_last_token() {
        let info = info_from(
            "hello world",
            vec![1, 2, 3],
            vec!["<bos>".into(), "hello".into(), " world".into()],
            vec![(0, 0), (0, 5), (5, 11)],
        );
        let record = resolve_static_selector(&TokenSelector::PromptFinal, &info).unwrap();
        assert_eq!(record.selected_indices, vec![2]);
        assert_eq!(record.rule, "prompt-final");
    }

    #[test]
    fn absolute_and_relative_indices() {
        let info = info_from(
            "abc def",
            vec![1, 2, 3, 4],
            vec!["<bos>".into(), "abc".into(), " def".into(), ".".into()],
            vec![(0, 0), (0, 3), (3, 7), (7, 8)],
        );
        let abs =
            resolve_static_selector(&TokenSelector::AbsoluteToken { index: 1 }, &info).unwrap();
        assert_eq!(abs.selected_indices, vec![1]);
        let rel =
            resolve_static_selector(&TokenSelector::RelativeToken { offset_from_end: 1 }, &info)
                .unwrap();
        assert_eq!(rel.selected_indices, vec![2]);
        let out_of_range =
            resolve_static_selector(&TokenSelector::AbsoluteToken { index: 9 }, &info);
        assert!(out_of_range.is_err());
        let rel_out =
            resolve_static_selector(&TokenSelector::RelativeToken { offset_from_end: 9 }, &info);
        assert!(rel_out.is_err());
    }

    #[test]
    fn matched_span_first_final_all() {
        let info = info_from(
            "الكتاب على الطاولة",
            vec![1, 2, 3, 4, 5, 6],
            vec![
                "<bos>".into(),
                "ال".into(),
                "كتاب".into(),
                " على".into(),
                " الط".into(),
                "اولة".into(),
            ],
            vec![(0, 0), (0, 4), (4, 12), (12, 19), (19, 26), (26, 34)],
        );
        let selector = |sel: SubtokenSelection| TokenSelector::MatchedTextSpan {
            text: "كتاب".to_string(),
            occurrence: 0,
            subtoken_selection: sel,
            normalization: TextNormalization::None,
        };
        let first = resolve_static_selector(&selector(SubtokenSelection::First), &info).unwrap();
        assert_eq!(first.selected_indices, vec![2]);
        assert_eq!(first.coverage, CoverageKind::Exact);
        let all = resolve_static_selector(&selector(SubtokenSelection::All), &info).unwrap();
        assert_eq!(all.selected_indices, vec![2]);
        // absent span fails
        let absent = resolve_static_selector(
            &TokenSelector::MatchedTextSpan {
                text: "مفقود".to_string(),
                occurrence: 0,
                subtoken_selection: SubtokenSelection::First,
                normalization: TextNormalization::None,
            },
            &info,
        );
        assert!(absent.is_err());
    }

    #[test]
    fn repeated_span_occurrence_is_deterministic() {
        // "قطة قطة قطة": 11 chars; the target appears at chars 0..3, 4..7,
        // 8..11.
        let info = info_from(
            "قطة قطة قطة",
            vec![1, 2, 3, 4, 5],
            vec![
                "<bos>".into(),
                "قطة".into(),
                " قط".into(),
                "ة".into(),
                " قطة".into(),
            ],
            vec![(0, 0), (0, 6), (6, 11), (11, 13), (13, 20)],
        );
        let selector = |occurrence: usize| TokenSelector::MatchedTextSpan {
            text: "قطة".to_string(),
            occurrence,
            subtoken_selection: SubtokenSelection::First,
            normalization: TextNormalization::None,
        };
        let first = resolve_static_selector(&selector(0), &info).unwrap();
        assert_eq!(first.selected_indices, vec![1]);
        assert_eq!(first.coverage, CoverageKind::Exact);
        // The second occurrence crosses a token boundary: `first` selects
        // the intersecting token with partial coverage; `all` covers the
        // span (enclosing, because the first token includes the leading
        // space).
        let second = resolve_static_selector(&selector(1), &info).unwrap();
        assert_eq!(second.selected_indices, vec![2]);
        assert_eq!(second.coverage, CoverageKind::Partial);
        let second_all = resolve_static_selector(
            &TokenSelector::MatchedTextSpan {
                text: "قطة".to_string(),
                occurrence: 1,
                subtoken_selection: SubtokenSelection::All,
                normalization: TextNormalization::None,
            },
            &info,
        )
        .unwrap();
        assert_eq!(second_all.selected_indices, vec![2, 3]);
        assert_eq!(second_all.coverage, CoverageKind::Enclosing);
        let third = resolve_static_selector(&selector(2), &info).unwrap();
        assert_eq!(third.selected_indices, vec![4]);
        assert!(resolve_static_selector(&selector(3), &info).is_err());
    }

    #[test]
    fn enclosing_coverage_records_expansion() {
        // "كتاب" spans chars 2..7; the tokenizer's second token covers
        // chars 1..7 (includes a preceding char), so selection encloses.
        let info = info_from(
            "xكتابy",
            vec![1, 2, 3],
            vec!["<bos>".into(), "xكت".into(), "ابy".into()],
            vec![(0, 0), (0, 5), (5, 10)],
        );
        let record = resolve_static_selector(
            &TokenSelector::MatchedTextSpan {
                text: "كتاب".to_string(),
                occurrence: 0,
                subtoken_selection: SubtokenSelection::All,
                normalization: TextNormalization::None,
            },
            &info,
        )
        .unwrap();
        assert_eq!(record.selected_indices, vec![1, 2]);
        assert_eq!(record.coverage, CoverageKind::Enclosing);
        assert!(record.boundary_expansion.is_some());
    }

    #[test]
    fn generated_step_resolution() {
        let positions = vec![6, 7, 8];
        assert_eq!(resolve_generated_step(1, &positions).unwrap(), vec![6]);
        assert_eq!(resolve_generated_step(3, &positions).unwrap(), vec![8]);
        assert!(resolve_generated_step(4, &positions).is_err());
        assert!(resolve_generated_step(0, &positions).is_err());
    }

    #[test]
    fn byte_span_selection() {
        let info = info_from(
            "hello world",
            vec![1, 2, 3],
            vec!["<bos>".into(), "hello".into(), " world".into()],
            vec![(0, 0), (0, 5), (5, 11)],
        );
        // Exact span: the token's own interval.
        let record = resolve_static_selector(
            &TokenSelector::ByteSpan {
                start: 5,
                end: 11,
                subtoken_selection: SubtokenSelection::First,
            },
            &info,
        )
        .unwrap();
        assert_eq!(record.selected_indices, vec![2]);
        assert_eq!(record.coverage, CoverageKind::Exact);
        // Enclosing span: the token covers a larger interval.
        let record = resolve_static_selector(
            &TokenSelector::ByteSpan {
                start: 6,
                end: 11,
                subtoken_selection: SubtokenSelection::First,
            },
            &info,
        )
        .unwrap();
        assert_eq!(record.coverage, CoverageKind::Enclosing);
        assert!(record.boundary_expansion.is_some());
        assert!(resolve_static_selector(
            &TokenSelector::ByteSpan {
                start: 5,
                end: 5,
                subtoken_selection: SubtokenSelection::First,
            },
            &info,
        )
        .is_err());
        assert!(resolve_static_selector(
            &TokenSelector::ByteSpan {
                start: 0,
                end: 99,
                subtoken_selection: SubtokenSelection::First,
            },
            &info,
        )
        .is_err());
    }

    #[test]
    fn arabic_diacritics_are_not_normalized() {
        // The span contains a diacritic; matching must be exact, and the
        // span without the diacritic must NOT match.
        // "هذا كِتَاب": chars ه(0) ذ(1) ا(2) ' '(3) ك(4) ِ(5) ت(6) َ(7) ا(8) ب(9).
        let info = info_from(
            "هذا كِتَاب",
            vec![1, 2],
            vec!["<bos>".into(), " هذا".into(), " كِتَاب".into()],
            vec![(0, 0), (0, 6), (6, 19)],
        );
        let record = resolve_static_selector(
            &TokenSelector::MatchedTextSpan {
                text: "كِتَاب".to_string(),
                occurrence: 0,
                subtoken_selection: SubtokenSelection::First,
                normalization: TextNormalization::None,
            },
            &info,
        )
        .unwrap();
        assert_eq!(record.selected_indices, vec![2]);
        // The token includes the leading space, so coverage encloses and
        // the expansion is recorded — never silently repaired.
        assert_eq!(record.coverage, CoverageKind::Enclosing);
        assert!(record.boundary_expansion.is_some());
        let stripped = resolve_static_selector(
            &TokenSelector::MatchedTextSpan {
                text: "كتاب".to_string(),
                occurrence: 0,
                subtoken_selection: SubtokenSelection::First,
                normalization: TextNormalization::None,
            },
            &info,
        );
        assert!(stripped.is_err(), "silent normalization must not occur");
    }
}
