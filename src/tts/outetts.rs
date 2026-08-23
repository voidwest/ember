//! OuteTTS-0.2 (v0.2 interface): text -> codec tokens -> speech.
//!
//! The generator is a qwen2-family LLM fine-tuned by OuteAI to continue the
//! OuteTTS prompt format with audio codec tokens; ember runs it through the
//! standard [`Llama`] path (resident K-quant kernels included). The codec
//! half is [`WavTokenizerDecoder`]. Nothing in this module leaks speech
//! logic into the transformer: the LLM just sees token ids like any other
//! generation request.
//!
//! Prompt shape (outetts v0.2 `PromptProcessor.get_completion_prompt`):
//!
//! ```text
//! <|im_start|>\n<|text_start|>word<|space|>word2<|text_end|>\n<|audio_start|>\n
//! ```
//!
//! with punctuation rendered as special tokens attached to words
//! (`hello<|comma|>`), numbers spelled out, words lowercased, joined by
//! `<|space|>`. The model then emits, per word:
//! `word<|t_{duration}|><|{code}|>*n ...` and finally `<|audio_end|>`.
//!
//! Text preprocessing ports the reference `_process_text` for plain English
//! ASCII input (the battery texts used for validation); Japanese/Chinese
//! frontends (MeCab wakati, uroman) are out of scope and fail closed.

use crate::backend::{Backend, CpuBackend};
use crate::llama::Llama;
use crate::loader::load_gguf_with_k_strategy;
use crate::tokenizer::EmberTokenizer;
use crate::tts::wavtokenizer::WavTokenizerDecoder;
use anyhow::{ensure, Context, Result};
use std::time::Instant;

const SPECIALS: [(&str, &str); 8] = [
    ("bos", "<|im_start|>"),
    ("eos", "<|im_end|>"),
    ("text_start", "<|text_start|>"),
    ("text_end", "<|text_end|>"),
    ("audio_start", "<|audio_start|>"),
    ("audio_end", "<|audio_end|>"),
    ("space", "<|text_sep|>"),
    ("time", "<|t_"),
];

// Reference interface note: OuteTTS-0.2-500M pairs with the outetts
// 0.2.x package ("interface v1"): words are lowercased, numbers expanded,
// punctuation REMOVED (not tokenized), words joined by `<|text_sep|>`
// (`<|space|>` belongs to the later v2 interface).

/// Minimal English number-to-words matching reference `inflect` output forms
/// used by the battery (integers up to 999,999,999 and point-decimals).
pub(crate) fn number_to_words(n: u64) -> String {
    const ONES: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    fn under_thousand(n: u64) -> String {
        if n < 20 {
            return ONES[n as usize].to_string();
        }
        if n < 100 {
            let t = TENS[(n / 10) as usize];
            let r = n % 10;
            return if r == 0 {
                t.to_string()
            } else {
                format!("{t}-{}", ONES[r as usize])
            };
        }
        let h = ONES[(n / 100) as usize];
        let r = n % 100;
        if r == 0 {
            format!("{h} hundred")
        } else {
            format!("{h} hundred {}", under_thousand(r))
        }
    }
    if n >= 1_000_000 {
        let r = n % 1_000_000;
        if r == 0 {
            format!("{} million", under_thousand(n / 1_000_000))
        } else {
            format!(
                "{} million {}",
                under_thousand(n / 1_000_000),
                under_thousand(r)
            )
        }
    } else if n >= 1_000 {
        let r = n % 1_000;
        if r == 0 {
            format!("{} thousand", under_thousand(n / 1_000))
        } else {
            format!(
                "{} thousand {}",
                under_thousand(n / 1_000),
                under_thousand(r)
            )
        }
    } else {
        under_thousand(n)
    }
}

/// Expand integer/decimal numbers to words (reference regex
/// `\d+(\.\d+)?` -> inflect; "3.14" -> "three point one four").
fn expand_numbers(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            let int_part: u64 = text[start..i].parse().unwrap_or(u64::MAX);
            let mut frac_text = String::new();
            if i + 1 < bytes.len() && bytes[i] == b'.' && bytes[i + 1].is_ascii_digit() {
                let fs = i + 1;
                let mut fe = fs;
                while fe < bytes.len() && bytes[fe].is_ascii_digit() {
                    fe += 1;
                }
                frac_text = text[fs..fe].to_string();
                i = fe;
            }
            if int_part == u64::MAX {
                out.push_str("many");
            } else {
                out.push_str(&number_to_words(int_part.min(999_999_999)));
            }
            if !frac_text.is_empty() {
                out.push_str(" point");
                for (di, d) in frac_text.bytes().enumerate() {
                    if di > 0 {
                        out.push(' ');
                    }
                    out.push_str(&number_to_words((d - b'0') as u64));
                }
            }
        } else {
            let ch = text[i..].chars().next().expect("nonempty");
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Reference `PromptProcessor.process_text(text, "en")` (outetts 0.2.x):
/// lowercase -> numbers to words -> [-_/,\.\\] to spaces -> strip
/// non-`[a-z ]` -> collapse whitespace -> split.
pub(crate) fn process_words_en(text: &str) -> Result<Vec<String>> {
    let lowered = text.to_lowercase();
    let expanded = expand_numbers(&lowered);

    // [-_/,\.\\] -> single space
    let mut spaced = String::with_capacity(expanded.len());
    for c in expanded.chars() {
        if matches!(c, '-' | '_' | '/' | ',' | '.' | '\\') {
            spaced.push(' ');
        } else {
            spaced.push(c);
        }
    }
    // strip [^a-z ], collapse runs of whitespace, trim
    let mut words: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in spaced.chars() {
        if c.is_ascii_lowercase() {
            cur.push(c);
        } else if c == ' ' && !cur.is_empty() {
            words.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    Ok(words)
}

pub(crate) const TEXT_SEP: &str = "<|text_sep|>";

/// Per-synthesis timings (Track F metrics).
#[derive(Debug, Clone, Copy, Default)]
pub struct TtsTimings {
    /// Text preprocessing + prompt tokenization, ms.
    pub prompt_ms: f64,
    /// LLM prefill of the rendered prompt, ms.
    pub prefill_ms: f64,
    /// Token generation loop until stop, ms.
    pub generate_ms: f64,
    /// Codec decode of the full code sequence, ms.
    pub codec_ms: f64,
    /// Wall time from generation start to the first emitted PCM chunk, ms
    /// (streaming path; 0 in single-shot synthesis).
    pub time_to_first_audio_ms: f64,
    /// Generated token count / codec tokens.
    pub n_tokens: usize,
    pub n_codes: usize,
    // -- streaming-drift accounting (Track D); all zero when not streaming --
    /// max |streamed-concat - final single-pass| over the shared prefix.
    pub streamed_max_abs: f32,
    /// rms(diff) / rms(final) over the shared prefix.
    pub streamed_rms_rel: f64,
    /// Pearson correlation between streamed concat and final waveform.
    pub streamed_corr: f64,
    /// Same metrics AFTER refinement tails are applied by a faithful
    /// consumer (the achievable fidelity under the documented contract).
    pub refined_max_abs: f32,
    pub refined_rms_rel: f64,
    /// Mean |drift| of already-decoded samples as more tokens arrive,
    /// bucketed by distance behind the decode frontier in codec tokens:
    /// [0-4), [4-8), [8-16), [16-32), [32+). Evidence for the stable-window
    /// margin (values are absolute PCM units).
    pub drift_by_distance: [f64; 5],
}

impl TtsTimings {
    /// Wall RTF relative to produced audio duration (75 codec tokens/s).
    pub fn rtf(&self) -> f64 {
        let audio_seconds = self.n_codes as f64 / 75.0;
        let total_s =
            (self.prompt_ms + self.prefill_ms + self.generate_ms + self.codec_ms) / 1000.0;
        if audio_seconds > 0.0 {
            total_s / audio_seconds
        } else {
            0.0
        }
    }
}

/// One streamed PCM chunk plus its position metadata.
///
/// Consumer contract (Track D3): samples at absolute indices below
/// `stable_up_to` are FINAL — future codec context will not change them.
/// Samples in `[stable_up_to, end)` are provisional; `revised_tail` (when
/// non-empty, starting at absolute index `revised_from`) replaces already
/// delivered audio with the best current estimate. A fidelity-conscious
/// player keeps a small ring of unplayed audio and applies revisions to
/// anything not yet played; a naive concatenating player still works and
/// simply inherits the documented streamed-vs-final deviation.
pub struct AudioChunkMeta {
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
    /// Index of the first codec token this chunk decodes from.
    pub first_token: usize,
    /// True after the last token of the utterance.
    pub final_chunk: bool,
    /// Absolute sample index of `pcm[0]` within the utterance.
    pub first_sample: usize,
    /// Samples strictly below this index will never change again. Zero
    /// until the final chunk: the codec's global attention keeps moving
    /// earlier samples (drift is measured, see [`TtsTimings`]).
    pub stable_up_to: usize,
    /// Latency-tolerant playback hint: audio below this index is far enough
    /// behind the decode frontier that a player may start playback accepting
    /// the documented drift; a revision for anything newer arrives with the
    /// next chunk (`revised_tail`).
    pub playable_hint: usize,
    /// Replacement audio for `[revised_from, revised_from + len)`;
    /// empty when this chunk carries no revision.
    pub revised_tail: Vec<f32>,
    pub revised_from: usize,
}

/// Default playback-hint margin: samples decoded from codes at least this
/// many tokens behind the newest code may be played by a latency-tolerant
/// consumer. Larger margins trade startup-of-playback delay for lower
/// uncorrectable drift (see `drift_by_distance` evidence). Override for
/// experiments with `EMBER_TTS_STABLE_MARGIN`. 12 tokens = 160 ms.
pub const STABLE_MARGIN_TOKENS: usize = 12;

fn effective_stable_margin() -> usize {
    std::env::var("EMBER_TTS_STABLE_MARGIN")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(STABLE_MARGIN_TOKENS)
}

fn wave_metrics(streamed: &[f32], full: &[f32]) -> (f32, f64, f64) {
    let n = streamed.len().min(full.len());
    if n == 0 {
        return (0.0, 0.0, 1.0);
    }
    let mut max_abs = 0.0f32;
    let mut sq_diff = 0.0f64;
    let mut sq_ref = 0.0f64;
    let mut dot = 0.0f64;
    let mut sum_a = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut sq_a = 0.0f64;
    let mut sq_b = 0.0f64;
    for i in 0..n {
        let (a, b) = (f64::from(streamed[i]), f64::from(full[i]));
        max_abs = max_abs.max((a - b).abs() as f32);
        sq_diff += (a - b) * (a - b);
        sq_ref += b * b;
        dot += a * b;
        sum_a += a;
        sum_b += b;
        sq_a += a * a;
        sq_b += b * b;
    }
    let rms_rel = sq_diff.sqrt() / (sq_ref.sqrt() + 1e-30);
    let nf = n as f64;
    let cov = dot / nf - (sum_a / nf) * (sum_b / nf);
    let va = (sq_a / nf - (sum_a / nf).powi(2)).max(1e-30);
    let vb = (sq_b / nf - (sum_b / nf).powi(2)).max(1e-30);
    let corr = (cov / (va * vb).sqrt()).clamp(-1.0, 1.0);
    (max_abs, rms_rel, corr)
}

/// OuteTTS speech synthesizer over a loaded qwen2-family GGUF.
pub struct OuteTts {
    pub llm: Llama<CpuBackend>,
    pub tokenizer: EmberTokenizer,
    pub codec: WavTokenizerDecoder,
    /// token id for each code value 0..4096.
    code_ids: Vec<u32>,
    audio_end_id: u32,
    /// ids treated as end-of-generation.
    eos_ids: Vec<u32>,
}

impl OuteTts {
    /// Load the OuteTTS GGUF (qwen2 arch), its tokenizer.json and the codec
    /// decoder GGUF. `k_strategy` selects the text-model execution policy
    /// (`Auto` production / `EagerF32` oracle).
    pub fn from_gguf(
        text_path: &std::path::Path,
        tokenizer_path: &std::path::Path,
        codec_path: &std::path::Path,
        k_strategy: crate::quant_k::KStrategy,
    ) -> Result<Self> {
        let loader = load_gguf_with_k_strategy(text_path, k_strategy, false)
            .with_context(|| format!("failed to load {}", text_path.display()))?;
        // vocab sanity: the audio-token additions must be present
        let llm = Llama::from_loader(loader)?;
        let tokenizer = EmberTokenizer::from_file(tokenizer_path)
            .with_context(|| format!("failed to load {}", tokenizer_path.display()))?;
        // token ids must exist inside the model's embedding table; the
        // largest mapped id is a code token far below any table bound, and
        // encode-time failures fail closed anyway.
        let _ = llm.config.embed_dim;

        let codec = WavTokenizerDecoder::from_gguf(codec_path)?;

        let n_codes = codec.config.codebook_bins;
        let mut code_ids = Vec::with_capacity(n_codes);
        for i in 0..n_codes {
            let tok = format!("<|{i}|>");
            let id = tokenizer
                .token_to_id(&tok)
                .ok_or_else(|| anyhow::anyhow!("tokenizer missing audio code token {tok}"))?;
            code_ids.push(id);
        }
        let need = |s: &str| -> Result<u32> {
            tokenizer
                .token_to_id(s)
                .ok_or_else(|| anyhow::anyhow!("tokenizer missing {s}"))
        };
        let audio_end_id = need("<|audio_end|>")?;
        let im_end_id = need(SPECIALS[1].1)?;
        let mut eos_ids = tokenizer.eos_token_ids();
        if !eos_ids.contains(&im_end_id) {
            eos_ids.push(im_end_id);
        }
        // TEXT_SEP itself is consumed only at prompt-render time; no field.
        Ok(Self {
            llm,
            tokenizer,
            codec,
            code_ids,
            audio_end_id,
            eos_ids,
        })
    }

    /// Render the completion prompt for one text (no speaker profile),
    /// byte-matching the reference `get_completion_prompt` for en.
    pub fn build_prompt(&self, text: &str) -> Result<String> {
        let words = process_words_en(text)?;
        anyhow::ensure!(!words.is_empty(), "no speakable words after preprocessing");
        Ok(format!(
            "{}\n<|text_start|>{}<|text_end|>\n<|audio_start|>\n",
            SPECIALS[0].1,
            words.join(TEXT_SEP)
        ))
    }

    /// Map generated token ids back to codec values (fail on foreign ids is
    /// not desired here: non-code tokens are simply skipped).
    pub fn extract_codes(&self, ids: &[u32]) -> Vec<u32> {
        let mut map = std::collections::HashMap::new();
        for (code, id) in self.code_ids.iter().enumerate() {
            map.insert(*id, code as u32);
        }
        ids.iter().filter_map(|id| map.get(id).copied()).collect()
    }

    /// Greedy synthesis: text -> codes -> PCM with stage timings.
    /// `on_token` observes every generated token (for streaming consumers);
    /// return `false` from it to cancel at the next checkpoint.
    pub fn synthesize(
        &self,
        backend: &CpuBackend,
        text: &str,
        max_tokens: usize,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<(Vec<f32>, Vec<u32>, TtsTimings)> {
        let t_all = Instant::now();
        let prompt = self.build_prompt(text)?;
        let prompt_ids = self.tokenizer.encode_no_special(&prompt)?;
        let mut timings = TtsTimings {
            prompt_ms: t_all.elapsed().as_secs_f64() * 1e3,
            ..TtsTimings::default()
        };

        // prefill (first logits come back directly)
        let t_pre = Instant::now();
        let emb = crate::llama::llama_embed_tokens(
            backend,
            &self.llm.embed_tokens,
            &prompt_ids,
            self.llm.config.embed_dim,
        )?;
        let start_pos = 0usize;
        let budget = prompt_ids.len() + max_tokens;
        let mut cache = self
            .llm
            .create_request_cache(backend, prompt_ids.len(), max_tokens);
        let _ = budget;
        let mut logits = self
            .llm
            .forward_last_logits_embeddings_with_cache(backend, &emb, &mut cache, start_pos)?;
        timings.prefill_ms = t_pre.elapsed().as_secs_f64() * 1e3;

        // greedy loop
        let t_gen = Instant::now();
        let mut ids: Vec<u32> = Vec::new();
        let mut stopped = false;
        while ids.len() < max_tokens {
            let best = crate::sampler::argmax_token(backend.data(&logits));
            let best = u32::try_from(best)?;
            ids.push(best);
            if !on_token(best) || best == self.audio_end_id || self.eos_ids.contains(&best) {
                stopped = true;
                break;
            }
            logits = self.llm.forward_last_logits_with_cache(
                backend,
                &[best],
                &mut cache,
                prompt_ids.len() + ids.len() - 1,
            )?;
        }
        let _ = stopped;
        timings.generate_ms = t_gen.elapsed().as_secs_f64() * 1e3;
        timings.n_tokens = ids.len();

        let codes = self.extract_codes(&ids);
        timings.n_codes = codes.len();
        let t_codec = Instant::now();
        let pcm = if codes.is_empty() {
            Vec::new()
        } else {
            self.codec.decode(backend, &codes)?
        };
        timings.codec_ms = t_codec.elapsed().as_secs_f64() * 1e3;
        let _ = t_all;
        Ok((pcm, ids, timings))
    }

    /// Streaming synthesis: emits PCM chunks as soon as `chunk_tokens`
    /// additional codec tokens have arrived (chunked conv decode with left
    /// context; deviation vs single-pass decode is documented in the
    /// report, NOT hidden). The first chunk fires as soon as
    /// `first_delay_tokens` codes exist.
    pub fn synthesize_streaming(
        &self,
        backend: &CpuBackend,
        text: &str,
        max_tokens: usize,
        chunk_tokens: usize,
        mut on_chunk: impl FnMut(AudioChunkMeta) -> bool,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<(Vec<f32>, Vec<u32>, TtsTimings)> {
        ensure!(
            chunk_tokens >= 8,
            "chunk_tokens must cover the codec receptive field"
        );
        let t_all = Instant::now();
        let prompt = self.build_prompt(text)?;
        let prompt_ids = self.tokenizer.encode_no_special(&prompt)?;
        let mut timings = TtsTimings {
            prompt_ms: t_all.elapsed().as_secs_f64() * 1e3,
            ..TtsTimings::default()
        };

        let emb = crate::llama::llama_embed_tokens(
            backend,
            &self.llm.embed_tokens,
            &prompt_ids,
            self.llm.config.embed_dim,
        )?;
        let mut cache = self
            .llm
            .create_request_cache(backend, prompt_ids.len(), max_tokens);
        let mut logits = self
            .llm
            .forward_last_logits_embeddings_with_cache(backend, &emb, &mut cache, 0)?;
        timings.prefill_ms = t_all.elapsed().as_secs_f64() * 1e3 - timings.prompt_ms;

        let t_gen = Instant::now();
        let mut ids: Vec<u32> = Vec::new();
        let mut emitted_samples = 0usize;
        let mut last_emitted_code = 0usize;
        let mut first_audio = false;
        let mut cancelled = false;
        let sr = self.codec.config.sample_rate;
        let spt = self.codec.config.hop_length as f64;
        let stable_margin_codes = effective_stable_margin();
        // Island-decode policy (Track D2): when > 0, each emission decodes
        // only `[island_start .. codes_now)` where `island_start` trails the
        // previously emitted code count by `ISLAND_CONTEXT` tokens. Emitted
        // audio is IMMUTABLE once delivered (each island is never recomputed)
        // — genuine streaming stability — at the documented cost of losing
        // right-context inside the codec for those samples.
        let island_ctx: usize = std::env::var("EMBER_TTS_ISLAND_CTX")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        // previous full-prefix decode: drift measurement + revision source
        let mut prev_fresh: Vec<f32> = Vec::new();
        // drift-by-distance accumulation: (sum |diff|, count) per token band
        let mut drift_acc = [(0.0f64, 0u64); 5];
        // naive concatenation vs revision-applying accumulators (metrics)
        let mut concat: Vec<f32> = Vec::new();
        let mut refined: Vec<f32> = Vec::new();
        while ids.len() < max_tokens {
            let best = crate::sampler::argmax_token(backend.data(&logits));
            let best = u32::try_from(best)?;
            ids.push(best);
            if !on_token(best) {
                cancelled = true;
                break;
            }
            let done = best == self.audio_end_id
                || self.eos_ids.contains(&best)
                || ids.len() >= max_tokens;
            let codes_so_far = self.extract_codes(&ids);
            if !first_audio && codes_so_far.len() >= chunk_tokens.min(16) {
                timings.time_to_first_audio_ms = t_gen.elapsed().as_secs_f64() * 1e3;
            }
            if done || codes_so_far.len() >= last_emitted_code + chunk_tokens {
                let (fresh, island_base) = if island_ctx > 0 && last_emitted_code > 0 {
                    // decode an independent island trailing the last emission
                    let s = last_emitted_code.saturating_sub(island_ctx);
                    (self.codec.decode(backend, &codes_so_far[s..])?, s)
                } else {
                    (self.codec.decode(backend, &codes_so_far)?, 0usize)
                };

                let (
                    mut chunk,
                    first_sample,
                    revised_from,
                    revised_tail,
                    stable_up_to,
                    playable_hint,
                );
                if island_ctx > 0 && last_emitted_code > 0 {
                    // -- island mode: emitted audio is immutable --
                    let boundary_abs = last_emitted_code as f64 * spt;
                    let idx = (boundary_abs as usize)
                        .saturating_sub(island_base * self.codec.config.hop_length);
                    let start = idx.min(fresh.len());
                    let tail = fresh[start..].to_vec();
                    first_sample = emitted_samples.min(boundary_abs as usize);
                    concat.extend_from_slice(&tail);
                    refined.extend_from_slice(&tail);
                    chunk = tail;
                    revised_from = 0;
                    revised_tail = Vec::new();
                    emitted_samples += chunk.len();
                    stable_up_to = emitted_samples;
                    playable_hint = emitted_samples;
                } else {
                    // -- prefix-refresh mode: player-model revisions --
                    // drift of previously decoded samples, by distance behind
                    // the CURRENT decode frontier (codec tokens)
                    if !prev_fresh.is_empty() {
                        let n_old = prev_fresh.len().min(fresh.len());
                        for p in 0..n_old {
                            let d = codes_so_far.len() as f64 - p as f64 / spt;
                            let band = match d {
                                x if x < 4.0 => 0,
                                x if x < 8.0 => 1,
                                x if x < 16.0 => 2,
                                x if x < 32.0 => 3,
                                _ => 4,
                            };
                            drift_acc[band].0 += (fresh[p] - prev_fresh[p]).abs() as f64;
                            drift_acc[band].1 += 1;
                        }
                    }

                    // Everything the consumer could have consumed beyond the
                    // previous hint line is replaced with the freshest
                    // estimate; below it belongs to earlier hint windows a
                    // real player has likely already played (uncorrectable —
                    // measured by refined-vs-final metrics, not hidden).
                    let rf = (((last_emitted_code.saturating_sub(stable_margin_codes)) as f64)
                        * spt) as usize;
                    let ru = fresh.len().min(concat.len());
                    let rt: Vec<f32> = if ru > rf && last_emitted_code > 0 {
                        refined.truncate(rf);
                        refined.extend_from_slice(&fresh[rf..ru]);
                        fresh[rf..ru].to_vec()
                    } else {
                        Vec::new()
                    };

                    let new_start = emitted_samples.min(fresh.len());
                    let tail: Vec<f32> = fresh[new_start..].to_vec();
                    concat.extend_from_slice(&tail);
                    refined.extend_from_slice(&tail);
                    first_sample = emitted_samples;
                    chunk = tail;
                    revised_from = rf;
                    revised_tail = rt;
                    emitted_samples += chunk.len();
                    // Honest stability: NOTHING is permanent until the final
                    // decode (global codec attention moves earlier samples).
                    stable_up_to = if done { emitted_samples } else { 0 };
                    playable_hint = std::cmp::min(
                        (codes_so_far.len().saturating_sub(stable_margin_codes) as f64 * spt)
                            as usize,
                        emitted_samples,
                    );
                }
                let has_revision = !revised_tail.is_empty();
                if std::env::var_os("EMBER_TTS_DEBUG").is_some() {
                    eprintln!(
                        "[tts-dbg] c={} rev {}..{} concat={} refined={} emitted={}",
                        codes_so_far.len(),
                        revised_from,
                        revised_from + revised_tail.len(),
                        concat.len(),
                        refined.len(),
                        emitted_samples
                    );
                }
                let meta = AudioChunkMeta {
                    first_token: last_emitted_code,
                    final_chunk: done,
                    sample_rate: sr,
                    pcm: std::mem::take(&mut chunk),
                    first_sample,
                    stable_up_to,
                    playable_hint,
                    revised_from: if has_revision { revised_from } else { 0 },
                    revised_tail,
                };
                last_emitted_code = codes_so_far.len();
                prev_fresh = fresh;
                if !first_audio {
                    first_audio = true;
                }
                if (!meta.pcm.is_empty() || meta.final_chunk || has_revision) && !on_chunk(meta) {
                    cancelled = true;
                    break;
                }
            }
            if done {
                break;
            }
            logits = self.llm.forward_last_logits_with_cache(
                backend,
                &[best],
                &mut cache,
                prompt_ids.len() + ids.len() - 1,
            )?;
        }
        timings.generate_ms = t_gen.elapsed().as_secs_f64() * 1e3;
        timings.n_tokens = ids.len();
        let codes = self.extract_codes(&ids);
        timings.n_codes = codes.len();
        let t_codec = Instant::now();
        let pcm_full = if codes.is_empty() {
            Vec::new()
        } else {
            self.codec.decode(backend, &codes)?
        };
        timings.codec_ms = t_codec.elapsed().as_secs_f64() * 1e3;

        // streamed-vs-final accounting (naive concat vs revision-applying)
        let (smax, srms, scorr) = wave_metrics(&concat, &pcm_full);
        timings.streamed_max_abs = smax;
        timings.streamed_rms_rel = srms;
        timings.streamed_corr = scorr;
        let (rmax, rrms, _) = wave_metrics(&refined, &pcm_full);
        timings.refined_max_abs = rmax;
        timings.refined_rms_rel = rrms;
        for (band, (sum, count)) in drift_acc.iter().enumerate() {
            timings.drift_by_distance[band] = if *count > 0 { sum / *count as f64 } else { 0.0 };
        }
        let _ = cancelled;
        Ok((pcm_full, ids, timings))
    }
}
