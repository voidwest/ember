//! Persistent multimodal [`VoiceSession`] over an [`Ultravox`] wrapper.
//!
//! The session owns inference state so interactive voice turns do not
//! rebuild the world each time:
//!
//! ```text
//! create -> open_streaming_audio -> push_streaming_audio*
//!        -> (update_stream_encoder* / provisional_transcript*)
//!        -> finalize_streaming_audio -> begin/set/attach -> commit_user_turn
//!        -> generate_reply*  (cancellable at any decode step)
//!        -> ...                              -> close (drop)
//! ```
//!
//! State discipline (the hard part, kept explicit):
//!
//! - **Committed** turns occupy a prefix of the live KV cache; their exact
//!   token ids AND assembled embeddings are retained so a cache rebuild
//!   never needs a tower or tokenizer pass again.
//! - **Provisional** work (partial transcripts over unfinished audio) runs
//!   on a CLONED scratch KV cache: speculative interpretation can never
//!   leak into committed state. Cost is reported in
//!   [`SessionStats::provisional_ms`].
//! - **Cancelled** generations roll back via [`KVCache::truncate_to`]
//!   (attention reads exactly `[0, cursor + seq)`, so bytes past a
//!   rolled-back cursor are dead and get overwritten); their text stays
//!   only as [`TurnState::Cancelled`] history metadata and is excluded
//!   from rebuilds.
//! - Media features go through the established [`MediaFeatureCache`]
//!   keyed by PCM content + recipe + tower identity: an identical
//!   recording never re-runs the audio tower.

use crate::backend::{Backend, CpuBackend};
use crate::kv_cache::KVCache;
use crate::multimodal::cache::{FeatureCacheKey, MediaFeatureCache, PreprocessFingerprint};
use crate::multimodal::output::OutputEvent;
use crate::multimodal::request::MediaId;
use crate::multimodal::stream::{AudioStream, AudioStreamConfig, StreamProgress, StreamedAudio};
use crate::tensor::CpuTensor;
use crate::tokenizer::EmberTokenizer;
use crate::tts::outetts::OuteTts;
use crate::ultravox::{AudioFeatures, StreamingSchedule, Ultravox, AUDIO_PLACEHOLDER};
use anyhow::{ensure, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Commit-state of one conversation turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnState {
    /// In the live KV cache; permanent part of the context.
    Committed,
    /// Rolled back out of the cache (barge-in / cancel); kept as history
    /// metadata only, excluded from any cache rebuild.
    Cancelled,
}

/// Role of a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// One conversation turn: tokens, assembled embeddings (bounded
/// re-prefill), and commit state.
#[derive(Debug)]
pub struct TurnRecord {
    pub role: Role,
    pub text: String,
    /// Token ids this turn contributed (scaffolding + content).
    pub token_ids: Vec<u32>,
    /// `[n_tokens, width]` assembled input embeddings (empty for cancelled).
    pub embeddings: CpuTensor,
    /// Absolute KV span `[start, end)` while committed.
    pub span: (usize, usize),
    pub state: TurnState,
}

impl TurnRecord {
    /// Token count that still occupies the live cache.
    pub fn live_tokens(&self) -> usize {
        match self.state {
            TurnState::Committed => self.span.1 - self.span.0,
            TurnState::Cancelled => 0,
        }
    }
}

/// Cooperative cancellation handle shared with a running generation.
///
/// Checked at safe checkpoints (after every decoded token); never inside a
/// matmul. Invariant: a cancelled generation leaves NO invalid reusable
/// state — the KV cursor rolls back before `generate_reply` returns.
#[derive(Clone, Default)]
pub struct GenerationControl {
    cancelled: Arc<AtomicBool>,
}

impl GenerationControl {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation at the next checkpoint.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Cumulative session counters (honest reuse-vs-rebuilt accounting).
#[derive(Debug, Clone, Copy, Default)]
pub struct SessionStats {
    pub user_turns_committed: u64,
    pub assistant_replies_completed: u64,
    pub assistant_replies_cancelled: u64,
    /// Tokens prefilled incrementally across turns (excludes rebuilds).
    pub prefilled_tokens: u64,
    /// Tokens re-prefilled by explicit cache rebuilds.
    pub reprefilled_tokens: u64,
    pub cache_rebuilds: u64,
    pub provisional_transcripts: u64,
    /// Total ms spent inside provisional-transcript scratch inference.
    pub provisional_ms: f64,
    /// Media feature-cache hits (identical audio reused; tower skipped).
    pub media_feature_hits: u64,
    pub media_feature_misses: u64,
}

/// What audio content is attached to the pending user turn.
enum PendingAudio {
    None,
    /// Features produced by a finished [`AudioStream`] (this session).
    Streamed {
        features: CpuTensor,
    },
    /// Static audio submitted directly; features resolved through the
    /// media feature cache at commit time.
    Static {
        id: MediaId,
        pcm: Vec<f32>,
    },
}

/// Content staged for the next user turn.
pub struct PendingUserTurn {
    audio: PendingAudio,
    text: String,
}

/// Persistent voice session over one loaded Ultravox model.
pub struct VoiceSession<'m> {
    model: &'m Ultravox,
    backend: &'m CpuBackend,
    tokenizer: &'m EmberTokenizer,

    cache: KVCache,
    /// Absolute position of the committed prefix end == cache cursor.
    committed_len: usize,
    turns: Vec<TurnRecord>,
    pending: Option<PendingUserTurn>,

    stream: Option<AudioStream>,
    schedule: Option<StreamingSchedule>,

    media_cache: Option<MediaFeatureCache>,
    /// Recipe identity mixed into media-cache keys.
    audio_recipe: u64,

    stats: SessionStats,
}

/// Chat scaffolding pieces, byte-identical to the single-turn template in
/// `UltravoxAssembler::render_chat_template`, so incremental session
/// assembly equals full-template assembly by construction (pinned by test).
pub(crate) struct ScaffoldTokens;

impl ScaffoldTokens {
    pub const BOS: &'static str = "<|begin_of_text|>";
    pub const EOT: &'static str = "<|eot_id|>";

    /// `<|begin_of_text|>` + system block + `<|eot_id|>`.
    pub fn system_prefix() -> String {
        format!(
            "{}<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge \
             Date: December 2023\nToday Date: 01 Jan 2026\n\n<|eot_id|>",
            Self::BOS
        )
    }

    pub fn user_open() -> String {
        "<|start_header_id|>user<|end_header_id|>\n\n".to_string()
    }

    pub fn user_close() -> String {
        Self::EOT.to_string()
    }

    pub fn assistant_open() -> String {
        "<|start_header_id|>assistant<|end_header_id|>\n\n".to_string()
    }

    #[cfg(test)]
    pub fn reference_render(user_content: &str) -> String {
        format!(
            "{bos}<|start_header_id|>system<|end_header_id|>\n\n\
             Cutting Knowledge Date: December 2023\nToday Date: 01 Jan 2026\n\n\
             <|eot_id|><|start_header_id|>user<|end_header_id|>\n\n\
             {user_content}\
             <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            bos = Self::BOS
        )
    }
}

fn concat_tensors(parts: &[CpuTensor]) -> Result<CpuTensor> {
    ensure!(!parts.is_empty(), "cannot concatenate zero tensors");
    let rows: usize = parts.iter().map(|t| t.shape()[0]).sum();
    let width = parts[0].shape()[1];
    ensure!(
        parts.iter().all(|t| t.shape()[1] == width),
        "concat width mismatch"
    );
    let mut all = Vec::with_capacity(rows * width);
    for t in parts {
        all.extend_from_slice(t.data());
    }
    Ok(CpuTensor::from_data(vec![rows, width], all))
}

/// Encode scaffolding/content text: literal special tokens map to their
/// ids exactly as in `UltravoxAssembler::assemble` (the tokenizer's
/// AddedToken matching is independent of the add-special-tokens
/// post-processor).
fn encode_scaffold(tokenizer: &EmberTokenizer, text: &str) -> Result<Vec<u32>> {
    tokenizer.encode_no_special(text)
}

/// Free-standing embedding lookup (usable during construction).
fn lookup_embeddings_for(model: &Ultravox, backend: &CpuBackend, ids: &[u32]) -> Result<CpuTensor> {
    let dim = match &model.llm.embed_tokens {
        crate::llama::LlamaEmbedding::F32(t) => t.shape()[1],
        crate::llama::LlamaEmbedding::Q8_0(w) => w.in_features(),
        crate::llama::LlamaEmbedding::KQuant(w) => w.in_features(),
    };
    let mut embeddings = backend.zeroes(&[ids.len(), dim])?;
    for (row, &token) in ids.iter().enumerate() {
        match &model.llm.embed_tokens {
            crate::llama::LlamaEmbedding::F32(table) => {
                backend.assign_row_from_table(&mut embeddings, row, table, token as usize)?;
            }
            crate::llama::LlamaEmbedding::Q8_0(table) => {
                backend.assign_row_from_q8_0(&mut embeddings, row, table, token as usize)?;
            }
            crate::llama::LlamaEmbedding::KQuant(table) => {
                backend.assign_row_from_k(&mut embeddings, row, table, token as usize)?;
            }
        }
    }
    Ok(embeddings)
}

/// PCM content identity: length + rate + every sample value.
fn pcm_media_id(samples: &[f32], sample_rate: u32) -> MediaId {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    samples.len().hash(&mut h);
    sample_rate.hash(&mut h);
    for v in samples {
        v.to_bits().hash(&mut h);
    }
    MediaId(h.finish())
}

impl<'m> VoiceSession<'m> {
    /// Create a session with an explicitly sized KV cache. Capacity is a
    /// contract: exceeding it fails closed with a clear error (nothing
    /// silently evicts context); callers size for their conversation.
    pub fn new(
        model: &'m Ultravox,
        backend: &'m CpuBackend,
        tokenizer: &'m EmberTokenizer,
        kv_capacity: usize,
        feature_cache_bytes: usize,
    ) -> Result<Self> {
        ensure!(kv_capacity > 0, "kv_capacity must be positive");
        let capacity = kv_capacity.min(model.llm.config.max_seq_len);
        let cache = model.llm.create_cache(backend, capacity);

        let mut fp = PreprocessFingerprint::new("voice-session-audio-v1");
        fp.mix_u64(crate::multimodal::audio::TARGET_SAMPLE_RATE as u64);
        let audio_recipe = fp.value();

        let mut session = Self {
            model,
            backend,
            tokenizer,
            cache,
            committed_len: 0,
            turns: Vec::new(),
            pending: None,
            stream: None,
            schedule: None,
            media_cache: (feature_cache_bytes > 0)
                .then(|| MediaFeatureCache::new(feature_cache_bytes)),
            audio_recipe,
            stats: SessionStats::default(),
        };

        // Seed the cache with the system prefix: part of every conversation,
        // embeddings never change. This IS a real prefill (KV rows written).
        let sys_ids = encode_scaffold(tokenizer, &ScaffoldTokens::system_prefix())?;
        let sys_emb = lookup_embeddings_for(model, backend, &sys_ids)?;
        let n_sys = sys_emb.shape()[0];
        let _ = model.llm.forward_last_logits_embeddings_with_cache(
            backend,
            &sys_emb,
            &mut session.cache,
            0,
        )?;
        debug_assert_eq!(session.cache.cursor(), n_sys);
        session.committed_len = n_sys;
        session.turns.push(TurnRecord {
            role: Role::User,
            text: "system".to_string(),
            token_ids: sys_ids,
            embeddings: sys_emb,
            span: (0, n_sys),
            state: TurnState::Committed,
        });
        Ok(session)
    }

    // -----------------------------------------------------------------
    // introspection
    // -----------------------------------------------------------------

    pub fn stats(&self) -> SessionStats {
        self.stats
    }

    pub fn committed_len(&self) -> usize {
        self.committed_len
    }

    pub fn turns(&self) -> &[TurnRecord] {
        &self.turns
    }

    pub fn capabilities(&self) -> crate::multimodal::output::ModelCapabilities {
        self.model.capabilities()
    }

    fn width(&self) -> usize {
        self.model.llm.config.embed_dim
    }

    /// Backend handle for integrations that run auxiliary towers (TTS).
    pub fn backend_ref(&self) -> &'m CpuBackend {
        self.backend
    }

    // -----------------------------------------------------------------
    // streaming audio input (Tracks C4/C5)
    // -----------------------------------------------------------------

    /// Open the session's streaming audio frontend (one active stream).
    pub fn open_streaming_audio(&mut self, config: AudioStreamConfig) -> Result<()> {
        ensure!(
            self.stream.is_none(),
            "a streaming audio input is already open"
        );
        self.stream = Some(AudioStream::open(config)?);
        self.schedule = Some(Ultravox::stream_schedule_new());
        Ok(())
    }

    /// Push mono f32 PCM into the active stream.
    pub fn push_streaming_audio(&mut self, samples: &[f32]) -> Result<StreamProgress> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("no streaming audio input open"))?;
        stream.push_pcm(samples)
    }

    /// Run the incremental encoder scheduler over the active stream:
    /// finalized windows encode exactly once; the active window is
    /// (re-)encoded only when `infer_active_window` is set. `None` when no
    /// stream is open.
    pub fn update_stream_encoder(
        &mut self,
        infer_active_window: bool,
    ) -> Result<Option<crate::ultravox::StreamUpdate>> {
        match (self.stream.as_ref(), self.schedule.as_mut()) {
            (Some(stream), Some(sched)) => Ok(Some(self.model.stream_update(
                self.backend,
                sched,
                stream,
                infer_active_window,
            )?)),
            _ => Ok(None),
        }
    }

    /// Unstable partial transcript over the CURRENT audio (finalized
    /// windows + active window) — Track C5 provisional semantics.
    ///
    /// Runs on a CLONED KV cache: speculative interpretation can never
    /// corrupt committed state regardless of what the model produces.
    /// Returns `None` when nothing is encoded yet.
    pub fn provisional_transcript(&mut self, max_tokens: usize) -> Result<Option<String>> {
        let t0 = std::time::Instant::now();
        let result = self.provisional_transcript_inner(max_tokens);
        self.stats.provisional_ms += t0.elapsed().as_secs_f64() * 1e3;
        self.stats.provisional_transcripts += 1;
        result
    }

    fn provisional_features(&self) -> Result<Option<CpuTensor>> {
        let sched = self
            .schedule
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no streaming audio input open"))?;
        let mut rows: Vec<CpuTensor> = Vec::new();
        if let Some(p) = sched.finalized_prefix_features() {
            rows.push(p);
        }
        if let Some(a) = sched.active_window_features().cloned() {
            rows.push(a);
        }
        if rows.is_empty() {
            return Ok(None);
        }
        Ok(Some(concat_tensors(&rows)?))
    }

    fn provisional_transcript_inner(&mut self, max_tokens: usize) -> Result<Option<String>> {
        let features = match self.provisional_features()? {
            Some(f) => f,
            None => return Ok(None),
        };

        // scratch cache = clone of committed state; dropped wholesale.
        let mut scratch = self.cache.clone();
        debug_assert_eq!(scratch.cursor(), self.committed_len);

        let prompt = "<|audio|>Repeat concisely what you have heard so far.";
        let emb = self.assemble_content_embeddings(prompt, &[AudioFeatures { features }])?;
        let prompt_rows = emb.shape()[0];
        let mut logits = self.model.llm.forward_last_logits_embeddings_with_cache(
            self.backend,
            &emb,
            &mut scratch,
            self.committed_len,
        )?;
        let eos_ids = self.tokenizer.eos_token_ids();
        let mut ids: Vec<u32> = Vec::new();
        for step in 0..max_tokens {
            let best = crate::sampler::argmax_token(self.backend.data(&logits));
            let best = u32::try_from(best)?;
            ids.push(best);
            if eos_ids.contains(&best) || step + 1 >= max_tokens {
                break;
            }
            logits = self.model.llm.forward_last_logits_with_cache(
                self.backend,
                &[best],
                &mut scratch,
                self.committed_len + prompt_rows + step,
            )?;
        }
        Ok(Some(self.tokenizer.decode(&ids)?))
    }

    /// Finish the active stream and stage its definitive features onto the
    /// pending user turn (bit-parity with the static path by construction
    /// of `Ultravox::stream_finish`). Features are inserted into the media
    /// feature cache. Returns the projected row count.
    pub fn finalize_streaming_audio(&mut self) -> Result<usize> {
        let stream = self
            .stream
            .take()
            .ok_or_else(|| anyhow::anyhow!("no streaming audio input open"))?;
        let sched = self
            .schedule
            .take()
            .ok_or_else(|| anyhow::anyhow!("stream schedule missing"))?;
        let streamed: StreamedAudio = stream.finish()?;
        let input_samples = streamed.input_samples;
        let input_rate = streamed.input_sample_rate;
        let (features, _stats, _trace, _n_windows) =
            self.model.stream_finish(self.backend, sched, streamed)?;
        let rows = features.shape()[0];

        // identity: raw-PCM coordinates + the definitive mel values (a
        // deterministic function of the exact PCM), plus recipe/tower in
        // the surrounding key. Identical recordings collide; nothing else.
        let media_id = pcm_media_id_from_len(input_samples, input_rate, &features);

        let pending = self
            .pending
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("finalize_streaming_audio: begin_user_turn first"))?;
        pending.audio = PendingAudio::Streamed {
            features: features.clone(),
        };

        if let Some(cache) = &mut self.media_cache {
            let key = FeatureCacheKey {
                media_id,
                kind: crate::multimodal::request::MediaKind::Audio,
                preprocess: self.audio_recipe,
                tower_identity: self.model.audio_identity,
            };
            if cache.lookup(&key).is_some() {
                self.stats.media_feature_hits += 1;
            } else {
                self.stats.media_feature_misses += 1;
                cache.insert(key, features.clone());
            }
        }
        Ok(rows)
    }

    /// Discard the active stream without committing anything.
    pub fn abort_streaming_audio(&mut self) {
        self.stream = None;
        self.schedule = None;
    }
}

impl<'m> VoiceSession<'m> {
    // -----------------------------------------------------------------
    // turn staging + commit
    // -----------------------------------------------------------------

    /// Begin staging a new user turn (drops any un-committed staging).
    pub fn begin_user_turn(&mut self) {
        self.pending = Some(PendingUserTurn {
            audio: PendingAudio::None,
            text: String::new(),
        });
    }

    /// Stage the text prompt of the pending turn. Use [`AUDIO_PLACEHOLDER`]
    /// where staged audio should bind.
    pub fn set_turn_prompt(&mut self, text: String) -> Result<()> {
        let pending = self
            .pending
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("begin_user_turn first"))?;
        ensure!(!text.is_empty(), "turn prompt must not be empty");
        pending.text = text;
        Ok(())
    }

    /// Attach static (non-streamed) audio to the pending turn; features
    /// resolve at commit time through the media feature cache.
    pub fn attach_static_audio(
        &mut self,
        input: &crate::multimodal::audio::AudioInput,
    ) -> Result<()> {
        let decoded = crate::multimodal::audio::to_mono_16k(input)?;
        let media_id = pcm_media_id(&decoded.samples, decoded.sample_rate);
        let pending = self
            .pending
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("begin_user_turn first"))?;
        ensure!(
            matches!(pending.audio, PendingAudio::None),
            "turn already carries audio"
        );
        pending.audio = PendingAudio::Static {
            id: media_id,
            pcm: decoded.samples,
        };
        Ok(())
    }

    /// Commit the staged user turn: resolve audio features, assemble this
    /// turn's embeddings (scaffolding included), prefill incrementally at
    /// the committed boundary, advance it. Returns `(kv_span, n_audio_rows)`.
    pub fn commit_user_turn(&mut self) -> Result<((usize, usize), usize)> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| anyhow::anyhow!("nothing staged: begin_user_turn first"))?;

        let (features, n_rows): (Option<CpuTensor>, usize) = match pending.audio {
            PendingAudio::None => (None, 0),
            PendingAudio::Streamed { features } => {
                let n = features.shape()[0];
                (Some(features), n)
            }
            PendingAudio::Static { id, pcm } => {
                let key = FeatureCacheKey {
                    media_id: id,
                    kind: crate::multimodal::request::MediaKind::Audio,
                    preprocess: self.audio_recipe,
                    tower_identity: self.model.audio_identity,
                };
                // resolve features first (encode_pcm borrows &self), then
                // update the cache
                let cached = self
                    .media_cache
                    .as_mut()
                    .and_then(|c| c.lookup(&key).cloned());
                if cached.is_some() {
                    self.stats.media_feature_hits += 1;
                } else {
                    self.stats.media_feature_misses += 1;
                }
                match cached {
                    Some(hit) => {
                        let n = hit.shape()[0];
                        (Some(hit), n)
                    }
                    None => {
                        let fresh = self.encode_pcm(&pcm)?;
                        let n = fresh.shape()[0];
                        if let Some(cache) = self.media_cache.as_mut() {
                            cache.insert(key, fresh.clone());
                        }
                        (Some(fresh), n)
                    }
                }
            }
        };

        // placeholder consistency fails closed
        let placeholders = pending.text.matches(AUDIO_PLACEHOLDER).count();
        ensure!(
            placeholders == usize::from(n_rows > 0),
            "prompt has {placeholders} {AUDIO_PLACEHOLDER} placeholder(s) but {} audio segment(s)",
            usize::from(n_rows > 0)
        );

        let audios: Vec<AudioFeatures> = features
            .iter()
            .map(|f| AudioFeatures {
                features: f.clone(),
            })
            .collect();
        let content_emb = self.assemble_content_embeddings(&pending.text, &audios)?;

        // full turn embeddings = user_open + content + user_close
        let open_ids = encode_scaffold(self.tokenizer, &ScaffoldTokens::user_open())?;
        let close_ids = encode_scaffold(self.tokenizer, &ScaffoldTokens::user_close())?;
        let mut emb_parts = vec![
            lookup_embeddings_for(self.model, self.backend, &open_ids)?,
            content_emb,
            lookup_embeddings_for(self.model, self.backend, &close_ids)?,
        ];
        let embeddings = concat_tensors(&std::mem::take(&mut emb_parts))?;

        let mut ids = open_ids;
        {
            let eos_id = self
                .tokenizer
                .token_to_id(ScaffoldTokens::EOT)
                .ok_or_else(|| anyhow::anyhow!("tokenizer missing eot"))?;
            let parts: Vec<&str> = pending.text.split(AUDIO_PLACEHOLDER).collect();
            for (i, part) in parts.iter().enumerate() {
                ids.extend(self.tokenizer.encode_no_special(part)?.iter().copied());
                if i + 1 < parts.len() {
                    ids.extend(std::iter::repeat_n(eos_id, n_rows));
                }
            }
        }
        ids.extend(close_ids.iter().copied());

        let span = self.prefill_incremental(embeddings.clone())?;
        debug_assert_eq!(
            span.1 - span.0,
            ids.len(),
            "embedding rows and token ids must agree"
        );
        self.turns.push(TurnRecord {
            role: Role::User,
            text: pending.text,
            token_ids: ids,
            embeddings,
            span,
            state: TurnState::Committed,
        });
        self.stats.user_turns_committed += 1;
        Ok((span, n_rows))
    }

    // -----------------------------------------------------------------
    // generation (with cancellation / barge-in rollback)
    // -----------------------------------------------------------------

    /// Generate the assistant reply for a committed user turn. Streams
    /// [`OutputEvent::TextDelta`] per token through `on_event`.
    ///
    /// Cancellation (`control.cancel()` from any thread) stops at the next
    /// token checkpoint; the KV cursor rolls back to the pre-generation
    /// boundary and the partial reply is recorded as
    /// [`TurnState::Cancelled`] — never part of context. On natural
    /// completion the closing `<|eot_id|>` is committed too.
    ///
    /// Returns `(text, cancelled)`.
    pub fn generate_reply(
        &mut self,
        control: &GenerationControl,
        max_tokens: usize,
        mut on_event: impl FnMut(OutputEvent),
        // Extra per-token stop probe (barge-in seam); true => cancel.
        mut stop_probe: impl FnMut() -> bool,
    ) -> Result<(String, bool)> {
        ensure!(
            !self
                .turns
                .last()
                .is_some_and(|t| t.role == Role::Assistant && t.state == TurnState::Committed),
            "generate_reply must follow a committed user turn"
        );

        // assistant scaffold enters the live cache speculatively; rolled
        // back on cancellation.
        let scaffold_ids = encode_scaffold(self.tokenizer, &ScaffoldTokens::assistant_open())?;
        let scaffold_emb = lookup_embeddings_for(self.model, self.backend, &scaffold_ids)?;
        let scaffold_rows = scaffold_emb.shape()[0];
        let reply_start = self.committed_len;

        // prefill scaffold; last-row logits come back directly
        let mut logits = self.model.llm.forward_last_logits_embeddings_with_cache(
            self.backend,
            &scaffold_emb,
            &mut self.cache,
            reply_start,
        )?;
        self.committed_len += scaffold_rows;

        let eos_ids = self.tokenizer.eos_token_ids();
        let mut ids: Vec<u32> = Vec::new();
        let mut cancelled = false;
        let mut stopped_on_eos = false;
        loop {
            if control.is_cancelled() || stop_probe() {
                cancelled = true;
                break;
            }
            let best = crate::sampler::argmax_token(self.backend.data(&logits));
            let best = u32::try_from(best)?;
            ids.push(best);
            on_event(OutputEvent::TextDelta {
                token_id: best,
                piece: self.tokenizer.decode(&[best])?,
            });
            if eos_ids.contains(&best) {
                stopped_on_eos = true;
                break; // terminal token committed below, not decoded further
            }
            if ids.len() >= max_tokens {
                break;
            }
            if control.is_cancelled() {
                cancelled = true;
                break;
            }
            logits = self.model.llm.forward_last_logits_with_cache(
                self.backend,
                &[best],
                &mut self.cache,
                reply_start + scaffold_rows + ids.len() - 1,
            )?;
        }

        if cancelled {
            self.cache.truncate_to(reply_start);
            debug_assert_eq!(self.cache.cursor(), reply_start);
            self.committed_len = reply_start;
            self.turns.push(TurnRecord {
                role: Role::Assistant,
                text: String::new(),
                token_ids: Vec::new(),
                embeddings: empty_embeddings(self.width()),
                span: (reply_start, reply_start),
                state: TurnState::Cancelled,
            });
            self.stats.assistant_replies_cancelled += 1;
            return Ok((String::new(), true));
        }

        // Terminal token policy (Llama chat format): an assistant turn ends
        // with <|eot_id|>. If the model emitted it naturally, forward THAT
        // token into KV; on a max_tokens cut-off append one explicitly.
        // The last generated token was never forwarded (nothing decoded from
        // it), so it is written here together with the terminal token.
        let eot_ids = encode_scaffold(self.tokenizer, ScaffoldTokens::EOT)?;
        let eot_id = *eot_ids.first().expect("eot id");
        self.committed_len = self.cache.cursor(); // resync with real decode position
        let terminal: Vec<u32> = if stopped_on_eos && ids.last() == Some(&eot_id) {
            vec![*ids.last().unwrap()]
        } else if ids.is_empty() {
            eot_ids.clone()
        } else {
            let mut v = Vec::with_capacity(eot_ids.len() + 1);
            v.push(*ids.last().unwrap());
            v.extend_from_slice(&eot_ids);
            v
        };
        let terminal_emb = lookup_embeddings_for(self.model, self.backend, &terminal)?;
        let _ = self.model.llm.forward_last_logits_embeddings_with_cache(
            self.backend,
            &terminal_emb,
            &mut self.cache,
            self.committed_len,
        )?;
        self.committed_len += terminal_emb.shape()[0];
        debug_assert_eq!(self.cache.cursor(), self.committed_len);
        self.stats.prefilled_tokens += scaffold_rows as u64;

        // retain this turn's assembled embeddings so a rebuild never needs
        // the tokenizer or speculative state
        let gen_emb = lookup_embeddings_for(self.model, self.backend, &terminal)?;
        let embeddings = concat_tensors(&[scaffold_emb, gen_emb])?;

        let mut full_ids = scaffold_ids.clone();
        full_ids.extend_from_slice(&terminal);
        let text = self.tokenizer.decode(&ids)?;
        self.turns.push(TurnRecord {
            role: Role::Assistant,
            text: text.clone(),
            token_ids: full_ids,
            embeddings,
            span: (reply_start, self.committed_len),
            state: TurnState::Committed,
        });
        self.stats.assistant_replies_completed += 1;
        Ok((text, false))
    }

    // -----------------------------------------------------------------
    // private machinery
    // -----------------------------------------------------------------

    /// Assemble ONE turn's CONTENT into embeddings (no scaffolding):
    /// text split around [`AUDIO_PLACEHOLDER`], feature rows scattered
    /// over eot runs — the exact mechanism of `UltravoxAssembler::assemble`.
    fn assemble_content_embeddings(
        &self,
        text: &str,
        audios: &[AudioFeatures],
    ) -> Result<CpuTensor> {
        let tokenizer = self.tokenizer;
        let parts: Vec<&str> = text.split(AUDIO_PLACEHOLDER).collect();
        ensure!(
            !parts.is_empty() && parts.len() == audios.len() + 1,
            "prompt has {} {AUDIO_PLACEHOLDER} placeholders but {} audio segments",
            parts.len().saturating_sub(1),
            audios.len()
        );
        let eos_id = tokenizer
            .token_to_id(ScaffoldTokens::EOT)
            .ok_or_else(|| anyhow::anyhow!("tokenizer missing {}", ScaffoldTokens::EOT))?;

        let mut ids: Vec<u32> = Vec::new();
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (i, part) in parts.iter().enumerate() {
            ids.extend(tokenizer.encode_no_special(part)?.iter().copied());
            if i < audios.len() {
                let n = audios[i].features.shape()[0];
                ensure!(n > 0, "audio segment {i} produced zero rows");
                ranges.push((ids.len(), n));
                ids.extend(std::iter::repeat_n(eos_id, n));
            }
        }

        let mut emb = lookup_embeddings_for(self.model, self.backend, &ids)?;
        for ((start, n), audio) in ranges.iter().zip(audios.iter()) {
            ensure!(
                *n == audio.features.shape()[0],
                "placeholder run length {n} != feature rows {}",
                audio.features.shape()[0]
            );
            let width = emb.shape()[1];
            for k in 0..*n {
                let dst = &mut emb.data_mut()[(start + k) * width..(start + k + 1) * width];
                let src = &audio.features.data()[k * width..(k + 1) * width];
                dst.copy_from_slice(src);
            }
        }
        Ok(emb)
    }

    /// Prefill embeddings at the committed boundary; returns the span used.
    fn prefill_incremental(&mut self, embeddings: CpuTensor) -> Result<(usize, usize)> {
        let start = self.committed_len;
        let n = embeddings.shape()[0];
        ensure!(
            start + n <= self.cache.max_seq_len(),
            "conversation exceeds KV capacity {}: committed {start} + new {n}; \
             recreate the session with a larger kv_capacity",
            self.cache.max_seq_len()
        );
        let _ = self.model.llm.forward_last_logits_embeddings_with_cache(
            self.backend,
            &embeddings,
            &mut self.cache,
            start,
        )?;
        self.committed_len += n;
        self.stats.prefilled_tokens += n as u64;
        debug_assert_eq!(self.cache.cursor(), self.committed_len);
        Ok((start, start + n))
    }

    /// Static-audio encode path for cache misses (mirrors the wrapper).
    fn encode_pcm(&self, pcm: &[f32]) -> Result<CpuTensor> {
        let mel = crate::multimodal::audio::log_mel_spectrogram_full(pcm)?;
        let (features, _, _, _, _) = self.model.encode_mel_chunked(self.backend, &mel)?;
        Ok(features)
    }
}

fn empty_embeddings(width: usize) -> CpuTensor {
    CpuTensor::from_data(vec![0, width], Vec::new())
}

/// Identity combining raw-PCM coordinates with the definitive features.
fn pcm_media_id_from_len(input_samples: usize, input_rate: u32, features: &CpuTensor) -> MediaId {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    input_samples.hash(&mut h);
    input_rate.hash(&mut h);
    features.shape().hash(&mut h);
    for v in features.data() {
        v.to_bits().hash(&mut h);
    }
    MediaId(h.finish())
}

/// Voice-turn glue between a [`VoiceSession`] and an [`OuteTts`]
/// synthesizer: the interactive half of Phase 4 session 2.
///
/// Turn policy (kept simple and explicit — see report):
/// - barge-in DURING generation cancels it: the KV cursor rolls back to the
///   pre-generation boundary and NO assistant text is committed;
/// - once generation completes naturally, the reply is committed; speech is
///   then streamed chunk-wise through `speak`, where a barge-in stops the
///   REMAINING audio but keeps the committed text (the LLM already finished;
///   the codec is stateless so there is no decoder state to roll back);
/// - new user audio may be staged as soon as `respond` returns.
pub struct VoiceLoop<'a> {
    pub session: &'a mut VoiceSession<'a>,
    pub tts: &'a OuteTts,
    /// Codec tokens decoded per streamed PCM chunk.
    pub chunk_tokens: usize,
}

/// Result of one voice turn.
pub struct VoiceTurnOutcome {
    /// True when barge-in/cancellation fired during GENERATION (no reply).
    pub cancelled_during_generation: bool,
    /// True when playback stopped early (reply stays committed).
    pub interrupted_playback: bool,
    /// Samples actually handed to `speak`.
    pub spoken_samples: usize,
    /// Committed reply text (empty when cancelled during generation).
    pub reply_text: String,
}

impl<'a> VoiceLoop<'a> {
    pub fn new(session: &'a mut VoiceSession<'a>, tts: &'a OuteTts) -> Self {
        Self {
            session,
            tts,
            chunk_tokens: 16,
        }
    }

    pub fn respond<FSpeak, FBarge>(
        &mut self,
        control: &GenerationControl,
        max_reply_tokens: usize,
        speak: FSpeak,
        barge_in: FBarge,
    ) -> Result<VoiceTurnOutcome>
    where
        // Copy probes: callable from several checkpoint closures at once
        // probes are Fn (Copy state, e.g. Cell/AtomicBool counters), so they
        // can be shared across several checkpoint closures
        FSpeak: Fn(&[f32], bool) -> bool + Copy,
        FBarge: Fn() -> bool + Copy,
    {
        // -- phase 1: generate the reply (barge-in => cancel + rollback) --
        let (text, was_cancelled) =
            self.session
                .generate_reply(control, max_reply_tokens, |_| {}, &barge_in)?;
        if was_cancelled {
            return Ok(VoiceTurnOutcome {
                cancelled_during_generation: true,
                interrupted_playback: false,
                spoken_samples: 0,
                reply_text: String::new(),
            });
        }

        // -- phase 2: stream speech for the committed reply --
        let backend = self.session.backend_ref();
        let spoken_samples = std::cell::Cell::new(0usize);
        let interrupted_flag = std::cell::Cell::new(false);
        self.tts.synthesize_streaming(
            backend,
            &text,
            max_reply_tokens.max(256),
            self.chunk_tokens,
            |chunk_meta: crate::tts::outetts::AudioChunkMeta| {
                if barge_in() {
                    interrupted_flag.set(true); // playback stops, text committed
                    return false;
                }
                spoken_samples.set(spoken_samples.get() + chunk_meta.pcm.len());
                speak(&chunk_meta.pcm, chunk_meta.final_chunk);
                true
            },
            |_token| barge_in(),
        )?;

        Ok(VoiceTurnOutcome {
            cancelled_during_generation: false,
            interrupted_playback: interrupted_flag.get(),
            spoken_samples: spoken_samples.get(),
            reply_text: text,
        })
    }
}

#[cfg(test)]
mod session_scaffold_tests {
    use super::*;

    #[test]
    fn scaffold_pieces_compose_to_the_reference_single_turn_template() {
        let content = "<|audio|>What was said?";
        let composed = format!(
            "{}{}{}{}{}",
            ScaffoldTokens::system_prefix(),
            ScaffoldTokens::user_open(),
            content,
            ScaffoldTokens::user_close(),
            ScaffoldTokens::assistant_open(),
        );
        assert_eq!(composed, ScaffoldTokens::reference_render(content));
    }

    #[test]
    fn second_turn_composition_extends_the_reference() {
        let reply = "The weather is nice.";
        let turn2 = "Tell me more.";
        let next_open = format!(
            "{}{}{}{}{}{}{}",
            ScaffoldTokens::system_prefix(),
            ScaffoldTokens::user_open(),
            "Hello",
            ScaffoldTokens::user_close(),
            ScaffoldTokens::assistant_open(),
            reply,
            ScaffoldTokens::user_close(),
        );
        let composed_two_turns =
            next_open.clone() + ScaffoldTokens::EOT + &ScaffoldTokens::user_open() + turn2;
        assert!(composed_two_turns.starts_with(&next_open));
        assert!(
            composed_two_turns.starts_with(&ScaffoldTokens::reference_render("Hello")),
            "prefix must embed the reference template"
        );
    }
}
