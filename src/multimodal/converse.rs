//! Model-in-the-loop live voice conversation (Phase 5 Session 2, Track A).
//!
//! This module composes the proven parts into ONE executable conversation
//! path — the piece Phase 5 session 1 left unwired:
//!
//! ```text
//! capture ring (device rate) ──> TurnDetector ──> AudioStream (resample)
//!      ──> Ultravox streaming scheduler ──> VoiceSession commit
//!      ──> LLM reply ──> OuteTTS ──> WavTokenizer PCM ──> playback ring
//! ```
//!
//! Design split (so the transition graph is testable without weights):
//!
//! * [`ConversationMachine`] — the explicit `(state, event) → (state,
//!   action)` graph. No inference, no devices; pinned by hermetic tests.
//! * [`VoiceConversation`] — the model-backed driver executing those
//!   actions against a real `VoiceSession`
//!   (`crate::multimodal::session::VoiceSession`) and
//!   `OuteTts` (`crate::tts::outetts::OuteTts`). The CLI
//!   (`ember voice --converse`) is a thin wrapper around exactly this;
//!   applications call the same API.
//!
//! Barge-in semantics follow the established VoiceLoop policy, now fed from
//! the CONCURRENT detector instead of single-threaded checkpoints:
//!
//! * interrupt during GENERATION → cancel at the next token checkpoint + KV
//!   rollback; nothing committed;
//! * interrupt during SPEECH OUTPUT → queued/remaining audio dropped within
//!   one callback block; committed reply text stays;
//! * capture never stops: every phase pumps the rings between checkpoints.
//!
//! Ownership and state-transition discipline (explicit by design):
//!
//! * The [`DuplexController`] lives behind `Rc<RefCell<…>>`: exactly one
//!   owner (the driver) plus short-lived borrows inside generation/synthesis
//!   probe closures. Single-threaded; no lock contention is possible, and a
//!   re-entrant borrow would panic immediately instead of deadlocking.
//! * Detector events may only be consumed where they are applied to the
//!   machine. Probes pump the controller and RECORD a barge-in onset, but
//!   the state transition itself is DEFERRED to the phase boundary — an
//!   in-flight decode cannot open a stream mid-matmul. The controller's own
//!   utterance collection keeps every sample of the new turn safe until the
//!   driver catches up (`take_utterance` seeds the fresh stream), so no
//!   audio is lost across the deferral.

use crate::duplex::{DuplexController, TurnEvent};
use crate::multimodal::output::OutputEvent;
use crate::multimodal::session::GenerationControl;
use crate::multimodal::stream::AudioStreamConfig;
use crate::multimodal::VoiceSession;
use crate::tts::SpeechOut;
use anyhow::Result;
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

/// Explicit conversation states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationState {
    /// Listening for the user; assistant fully idle.
    Idle,
    /// User speech open: live audio flows into the active stream frontend.
    CapturingUser,
    /// Speech ended; finalizing features and committing the user turn.
    FinalizingUser,
    /// LLM decoding the assistant reply (barge-in cancels + rolls back).
    GeneratingAssistant,
    /// Reply committed; synthesis/playback active or draining. Barge-in
    /// drops remaining audio, keeps the committed text.
    SpeakingAssistant,
}

/// Actions the machine asks its driver to execute. Kept as data so tests pin
/// the graph without models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationAction {
    OpenUserStream,
    FinalizeAndCommit,
}

/// Why a turn's assistant output ended the way it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssistantEnd {
    /// Reply spoken to completion.
    Completed,
    /// Cancelled during generation: nothing committed, KV rolled back.
    InterruptedDuringGeneration,
    /// Generation had committed; remaining audio was dropped on barge-in.
    InterruptedDuringPlayback,
}

/// Per-turn latency breakdown (relative to speech end unless stated).
#[derive(Debug, Clone, Copy, Default)]
pub struct TurnTimings {
    /// Device-rate samples captured for this utterance / device rate.
    pub utterance_seconds: f64,
    /// SpeechEnded → first reply text event (0 when interrupted pre-token).
    pub end_to_first_token_ms: f64,
    /// SpeechEnded → first PCM chunk queued for playback (0 when none).
    pub end_to_first_audio_ms: f64,
    /// SpeechEnded → turn resolution (cancel / playback queued).
    pub end_to_turn_done_ms: f64,
    /// Reply codec tokens produced (75/s ⇒ audio seconds).
    pub reply_codes: usize,
}

/// Events an application observes during a conversation.
#[derive(Debug, Clone)]
pub enum ConverseEvent {
    SpeechStarted,
    /// Unstable partial transcript over the open stream (pulses enabled).
    PartialTranscript {
        text: String,
    },
    /// The user turn is now part of the context.
    UserCommitted {
        audio_seconds: f64,
    },
    ReplyTextDelta {
        piece: String,
    },
    /// First PCM chunk of the reply was queued for playback.
    AssistantAudioStart {
        samples: usize,
    },
    /// Barge-in latched while the assistant was active.
    BargeIn {
        during_generation: bool,
    },
    TurnComplete {
        reply_text: String,
        end: AssistantEnd,
        timings: TurnTimings,
    },
}

/// Conversation knobs.
#[derive(Debug, Clone)]
pub struct ConverseConfig {
    /// User-turn prompt; `<|audio|>` binds the streamed features.
    pub prompt: String,
    /// Max reply tokens per turn.
    pub max_reply_tokens: usize,
    /// Codec tokens per streamed TTS chunk.
    pub chunk_tokens: usize,
    /// Minimum ms between provisional transcript pulses while capturing
    /// (0 disables — pulses cost an active-window re-encode each).
    pub partial_every_ms: u64,
    /// Max provisional tokens per pulse.
    pub partial_max_tokens: usize,
}

impl Default for ConverseConfig {
    fn default() -> Self {
        Self {
            prompt: "<|audio|>".to_string(),
            max_reply_tokens: 128,
            chunk_tokens: 24,
            partial_every_ms: 0,
            partial_max_tokens: 24,
        }
    }
}

/// The explicit transition graph (A3). Complete mapping; any `(state,
/// event)` pair not listed is a no-op returning the same state.
#[derive(Debug, Default)]
pub struct ConversationMachine;

impl ConversationMachine {
    /// Apply one detector event.
    ///
    /// ```text
    /// Idle                --SpeechStarted--> Capturing [OpenUserStream]
    /// Capturing           --SpeechEnded-----> Finalizing [FinalizeAndCommit]
    /// Generating/Speaking --SpeechStarted---> Capturing [OpenUserStream]
    ///                                      (barge-in side effects are the
    ///                                       DRIVER's job: cancel probe /
    ///                                       request_clear)
    /// ```
    ///
    /// `SpeechContinues` never changes state. A hangover `SpeechEnded`
    /// arriving outside Capturing is ignored (a barge-in switch must not
    /// resurrect stale endpoints). `SpeechStarted` inside Capturing is
    /// idempotent (the detector should not fire it twice, but the graph is
    /// total regardless).
    pub fn apply(
        &self,
        state: ConversationState,
        event: TurnEvent,
    ) -> (ConversationState, Vec<ConversationAction>) {
        use ConversationAction as A;
        use ConversationState as S;
        match (state, event) {
            (S::Idle, TurnEvent::SpeechStarted) => (S::CapturingUser, vec![A::OpenUserStream]),
            (S::GeneratingAssistant | S::SpeakingAssistant, TurnEvent::SpeechStarted) => {
                (S::CapturingUser, vec![A::OpenUserStream])
            }
            (S::CapturingUser, TurnEvent::SpeechEnded) => {
                (S::FinalizingUser, vec![A::FinalizeAndCommit])
            }
            (_, _) => (state, vec![]),
        }
    }

    /// Driver completion edge: commit finished → generate.
    pub fn after_commit(state: ConversationState) -> ConversationState {
        match state {
            ConversationState::FinalizingUser => ConversationState::GeneratingAssistant,
            other => other,
        }
    }

    /// Driver completion edge: generation finished.
    pub fn after_generation(state: ConversationState, cancelled: bool) -> ConversationState {
        match (state, cancelled) {
            (ConversationState::GeneratingAssistant, false) => ConversationState::SpeakingAssistant,
            (ConversationState::GeneratingAssistant, true) => ConversationState::Idle,
            (other, _) => other,
        }
    }

    /// Driver completion edge: speech output finished → Idle.
    pub fn after_speech(state: ConversationState) -> ConversationState {
        match state {
            ConversationState::SpeakingAssistant => ConversationState::Idle,
            other => other,
        }
    }
}

/// Bookkeeping for one in-flight turn.
struct TurnBookkeeping {
    device_rate: u32,
    captured_samples: u64,
    speech_end_at: Instant,
    first_token_ms: Option<f64>,
    first_audio_ms: Option<f64>,
    reply_codes: usize,
}

/// Model-backed conversation driver over a duplex controller.
///
/// Pure Rust (no cpal here): hermetic tests drive it with synthetic ring
/// producers; the CLI plugs cpal's `LiveDuplex` in front of the same rings.
pub struct VoiceConversation<'m> {
    pub session: VoiceSession<'m>,
    tts: &'m dyn SpeechOut,
    /// Shared handle: the driver plus at most ONE probe closure borrow at a
    /// time (generation loop / synthesis callbacks never overlap each other
    /// in time — they run inside their respective sequential loops).
    pub duplex: Rc<RefCell<DuplexController>>,
    config: ConverseConfig,

    state: ConversationState,
    /// Barge-in onset observed while generating/speaking whose state
    /// transition is pending until the phase boundary.
    deferred_capture_open: bool,
    last_partial: Option<Instant>,
    bk: Option<TurnBookkeeping>,
    turns_completed: usize,
    events: Vec<ConverseEvent>,
    /// Set once the first reply PCM chunk is queued — lets host threads
    /// synchronize on time-to-first-audio without polling session internals.
    pub audio_started: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl<'m> VoiceConversation<'m> {
    pub fn new(
        session: VoiceSession<'m>,
        tts: &'m dyn SpeechOut,
        duplex: DuplexController,
        config: ConverseConfig,
    ) -> Self {
        Self {
            session,
            tts,
            duplex: Rc::new(RefCell::new(duplex)),
            config,
            state: ConversationState::Idle,
            deferred_capture_open: false,
            last_partial: None,
            bk: None,
            turns_completed: 0,
            events: Vec::new(),
            audio_started: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    pub fn state(&self) -> ConversationState {
        self.state
    }

    pub fn turns_completed(&self) -> usize {
        self.turns_completed
    }

    /// Drain observed events since the last call.
    pub fn take_events(&mut self) -> Vec<ConverseEvent> {
        std::mem::take(&mut self.events)
    }

    /// One pump step: drain captured audio through the detector, feed live
    /// PCM into the stream frontend while capturing, run periodic partial
    /// transcripts, execute machine actions. Returns emitted events.
    pub fn pump(&mut self) -> Vec<ConverseEvent> {
        self.events.clear();
        let capturing = self.state == ConversationState::CapturingUser;

        // Phase 1: drain captured audio; mirror live PCM into the stream
        // frontend while capturing. Split borrows keep session/bookkeeping
        // reachable alongside the shared controller.
        let events_in = {
            let this = &mut *self;
            let bk = &mut this.bk;
            let session = &mut this.session;
            this.duplex
                .borrow_mut()
                .pump_with_chunk_cb(|samples, _rate| {
                    if capturing && let Err(e) = session.push_streaming_audio(samples) {
                        eprintln!("converse: stream feed failed: {e}");
                    }
                    if let Some(bk) = bk.as_mut() {
                        bk.captured_samples += samples.len() as u64;
                    }
                })
        };

        // Phase 2: apply detector transitions.
        for event in events_in {
            self.apply_event(event);
        }

        // Phase 3: periodic partial transcript while capturing.
        if self.state == ConversationState::CapturingUser && self.config.partial_every_ms > 0 {
            let due = self
                .last_partial
                .map(|t| t.elapsed().as_millis() as u64 >= self.config.partial_every_ms)
                .unwrap_or(true);
            if due {
                self.last_partial = Some(Instant::now());
                match self
                    .session
                    .provisional_transcript(self.config.partial_max_tokens)
                {
                    Ok(Some(text)) if !text.trim().is_empty() => {
                        self.events.push(ConverseEvent::PartialTranscript { text });
                    }
                    _ => {}
                }
            }
        }
        std::mem::take(&mut self.events)
    }

    fn apply_event(&mut self, event: TurnEvent) {
        use ConversationAction as A;
        if matches!(event, TurnEvent::SpeechStarted) && self.assistant_active() {
            // ---- barge-in edge (state transition deferred) ----
            self.duplex.borrow_mut().clear_playback();
            self.deferred_capture_open = true;
            self.events.push(ConverseEvent::BargeIn {
                during_generation: self.state == ConversationState::GeneratingAssistant,
            });
            return;
        }
        let (next, actions) = ConversationMachine.apply(self.state, event);
        for action in actions {
            match action {
                A::OpenUserStream => self.open_user_stream(),
                A::FinalizeAndCommit => {
                    self.state = next;
                    self.run_turn_pipeline();
                    return;
                }
            }
        }
        self.state = next;
        if matches!(event, TurnEvent::SpeechStarted)
            && self.state == ConversationState::CapturingUser
        {
            self.events.push(ConverseEvent::SpeechStarted);
        }
    }

    fn assistant_active(&self) -> bool {
        matches!(
            self.state,
            ConversationState::GeneratingAssistant | ConversationState::SpeakingAssistant
        )
    }

    /// Open the user stream frontend, seeding it with everything the
    /// controller collected so far. This covers the samples of the very
    /// chunk that triggered SpeechStarted — they were consumed by the
    /// controller's queue before the transition could fire, and without the
    /// seed the utterance head would be truncated out of the features.
    fn open_user_stream(&mut self) {
        let (seed, rate) = {
            let mut d = self.duplex.borrow_mut();
            let (pcm, r, _off) = d.take_utterance();
            (pcm, r)
        };
        if let Err(e) = self
            .session
            .open_streaming_audio(AudioStreamConfig { sample_rate: rate })
        {
            eprintln!("converse: open stream failed: {e}");
        }
        if !seed.is_empty()
            && let Err(e) = self.session.push_streaming_audio(&seed)
        {
            eprintln!("converse: seed stream failed: {e}");
        }
        self.bk = Some(TurnBookkeeping {
            device_rate: rate,
            captured_samples: 0,
            speech_end_at: Instant::now(),
            first_token_ms: None,
            first_audio_ms: None,
            reply_codes: 0,
        });
        self.last_partial = None;
    }

    /// Execute the whole assistant half of the turn: finalize → commit →
    /// generate → speak (or cancel). Synchronous on the caller's thread;
    /// probes keep the mic judged throughout via the shared controller.
    fn run_turn_pipeline(&mut self) {
        if let Err(e) = self.finalize_and_commit() {
            eprintln!("converse: user-turn commit failed: {e}");
            self.session.abort_streaming_audio();
            let _ = self.duplex.borrow_mut().take_utterance();
            self.bk = None;
            self.state = ConversationState::Idle;
            return;
        }
        self.generate_reply_phase();
    }

    fn finalize_and_commit(&mut self) -> Result<()> {
        // Trailing hangover silence after SpeechEnded lives in the
        // controller's buffer; it is intentionally discarded here — the
        // streamed features already cover everything up to the endpoint.
        let _ = self.duplex.borrow_mut().take_utterance();
        self.session.begin_user_turn();
        self.session.set_turn_prompt(self.config.prompt.clone())?;
        let rows = self.session.finalize_streaming_audio()?;
        anyhow::ensure!(
            rows > 0,
            "streamed features empty ({rows} rows); utterance too short"
        );
        self.session.commit_user_turn()?;
        if let Some(bk) = &mut self.bk {
            bk.speech_end_at = Instant::now();
        }
        let audio_seconds = self
            .bk
            .as_ref()
            .map(|bk| bk.captured_samples as f64 / bk.device_rate as f64)
            .unwrap_or(0.0);
        self.events
            .push(ConverseEvent::UserCommitted { audio_seconds });
        self.state = ConversationMachine::after_commit(self.state);
        Ok(())
    }

    fn generate_reply_phase(&mut self) {
        struct Probe<'a> {
            duplex: &'a RefCell<DuplexController>,
            hit: bool,
        }
        impl Probe<'_> {
            /// Pump the detector between tokens. Returns true once a barge-in
            /// onset has fired (latched for every later checkpoint).
            fn poll(&mut self) -> bool {
                for ev in self.duplex.borrow_mut().pump_events() {
                    if matches!(ev, TurnEvent::SpeechStarted) {
                        // Controller armed collection for the new utterance;
                        // the driver performs the state transition after the
                        // phase returns (deferred-capture contract).
                        self.hit = true;
                    }
                }
                self.hit
            }
        }
        let mut probe = Probe {
            duplex: &self.duplex,
            hit: false,
        };

        let control = GenerationControl::new();
        let mut first_token_seen = false;
        let mut text_events: Vec<ConverseEvent> = Vec::new();

        let reply_res = self.session.generate_reply(
            &control,
            self.config.max_reply_tokens,
            |ev| {
                if let OutputEvent::TextDelta { piece, .. } = ev
                    && !piece.is_empty()
                {
                    first_token_seen = true;
                    text_events.push(ConverseEvent::ReplyTextDelta { piece });
                }
            },
            || probe.poll(),
        );

        let barge_in_during_generation = probe.hit;
        let (reply_text, was_cancelled) = match reply_res {
            Ok(v) => v,
            Err(e) => {
                eprintln!("converse: generation failed: {e}");
                self.bk = None;
                self.state = ConversationState::Idle;
                return;
            }
        };
        if first_token_seen
            && let Some(bk) = &mut self.bk
            && bk.first_token_ms.is_none()
        {
            bk.first_token_ms = Some(bk.speech_end_at.elapsed().as_secs_f64() * 1e3);
        }
        self.events.append(&mut text_events);

        // Barge-in observed but the model finished naturally first: the
        // reply is committed (established policy — generation completed),
        // synthesis is skipped, and the new user turn takes over.
        if barge_in_during_generation && !was_cancelled {
            self.duplex.borrow_mut().clear_playback();
            self.state = ConversationMachine::after_speech(ConversationState::SpeakingAssistant);
            self.finish_turn(reply_text, AssistantEnd::InterruptedDuringPlayback);
            self.enter_deferred_capture();
            return;
        }

        self.state = ConversationMachine::after_generation(self.state, was_cancelled);
        if was_cancelled {
            self.finish_turn(String::new(), AssistantEnd::InterruptedDuringGeneration);
            if barge_in_during_generation {
                self.enter_deferred_capture();
            } else {
                // cancelled by max_tokens/external control without new user
                // speech: nothing to defer
                let _ = self.duplex.borrow_mut().take_utterance();
            }
        } else {
            self.speak_phase_with_reply(reply_text);
        }
    }

    fn speak_phase_with_reply(&mut self, reply_text: String) {
        self.duplex.borrow_mut().set_assistant_active(true);
        // A barge-in detected inside this phase arms deferred capture so
        // enter_deferred_capture reopens the user stream afterwards.
        self.deferred_capture_open = true;
        let interrupt_flag = Rc::new(std::cell::Cell::new(false));
        let interrupt_probe = interrupt_flag.clone();

        struct SpeakProbe<'a> {
            duplex: &'a RefCell<DuplexController>,
            hit: bool,
        }
        impl SpeakProbe<'_> {
            fn poll(&mut self) -> bool {
                for ev in self.duplex.borrow_mut().pump_events() {
                    if matches!(ev, TurnEvent::SpeechStarted) {
                        self.hit = true;
                    }
                }
                self.hit
            }
        }
        // Shared between BOTH closures sequentially (they never overlap):
        // wrapped so each closure gets its own copy of the handle.
        let probe_cell = Rc::new(RefCell::new(SpeakProbe {
            duplex: &self.duplex,
            hit: false,
        }));
        let probe_for_chunk = probe_cell.clone();
        let probe_for_token = probe_cell.clone();

        let tts = self.tts;
        let backend = self.session.backend_ref();
        let max_codes = self.config.max_reply_tokens * 3 + 64;
        let chunk_tokens = self.config.chunk_tokens;

        let mut spoken_samples = 0usize;
        let mut first_audio_at: Option<Instant> = None;
        let audio_started_flag = self.audio_started.clone();
        let audio_started = std::cell::Cell::new(false);
        let mut synth_err: Option<String> = None;
        let _t_debug = std::cell::Cell::new(0u32);

        let synth = tts.stream_speech(
            backend,
            &reply_text,
            max_codes,
            chunk_tokens,
            &mut |chunk_meta| {
                let hit = probe_for_chunk.borrow_mut().poll();
                if hit {
                    interrupt_probe.set(true);
                    return false; // stop synthesis; remainder dropped
                }
                if !audio_started.get() {
                    audio_started.set(true);
                    audio_started_flag.store(true, std::sync::atomic::Ordering::Release);
                    first_audio_at = Some(Instant::now());
                }
                spoken_samples += chunk_meta.pcm.len();
                let rejected = probe_for_chunk
                    .borrow()
                    .duplex
                    .borrow_mut()
                    .play_audio(&chunk_meta.pcm);
                if rejected > 0 {
                    eprintln!("converse: playback ring full, dropped {rejected} samples");
                }
                true
            },
            &mut |_tok| {
                // Contract: return FALSE to cancel. Normal tokens CONTINUE
                // (true); a barge-in onset (probe.hit latched) cancels.
                let hit = probe_for_token.borrow_mut().poll();
                !hit
            },
        );
        let barge_in_during_speech = interrupt_flag.get() || probe_cell.borrow().hit;
        match synth {
            Ok((_, _, timings)) => {
                if std::env::var("EMBER_CONVERSE_DBG").is_ok() {
                    eprintln!(
                        "SPEAKDBG codes={} n_tok={} gen_ms={:.0} codec_ms={:.0} err=None",
                        timings.n_codes, timings.n_tokens, timings.generate_ms, timings.codec_ms
                    );
                }
                if let Some(bk) = &mut self.bk {
                    if let Some(t) = first_audio_at {
                        bk.first_audio_ms =
                            Some(t.duration_since(bk.speech_end_at).as_secs_f64() * 1e3);
                    }
                    bk.reply_codes = timings.n_codes;
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                if std::env::var("EMBER_CONVERSE_DBG").is_ok() {
                    eprintln!("SPEAKERR {msg}");
                }
                if msg.contains("no speakable words") || msg.contains("speakable") {
                    // Model replied with something the engine cannot voice
                    // (non-Latin/empty after preprocessing). Honest skip: the
                    // turn completes with zero audio rather than failing.
                    eprintln!("converse: nothing speakable to synthesize");
                    synth_err = None;
                    if let Some(bk) = &mut self.bk {
                        bk.reply_codes = 0;
                    }
                } else {
                    synth_err = Some(msg);
                }
            }
        }

        if audio_started.get() {
            self.events.push(ConverseEvent::AssistantAudioStart {
                samples: spoken_samples,
            });
        }
        if let Some(err) = synth_err {
            eprintln!("converse: synthesis failed: {err}");
        }

        let end = if barge_in_during_speech {
            self.duplex.borrow_mut().clear_playback();
            AssistantEnd::InterruptedDuringPlayback
        } else {
            AssistantEnd::Completed
        };
        self.duplex.borrow_mut().set_assistant_active(false);
        self.state = ConversationMachine::after_speech(self.state);
        self.finish_turn(reply_text, end);
        if end == AssistantEnd::InterruptedDuringPlayback {
            self.enter_deferred_capture();
        }
    }

    /// Perform a barge-in state transition deferred from a phase boundary:
    /// open the fresh user stream seeded with everything the controller
    /// collected since the onset (no audio lost across the gap).
    fn enter_deferred_capture(&mut self) {
        if !self.deferred_capture_open {
            return;
        }
        self.deferred_capture_open = false;
        self.state = ConversationState::CapturingUser;
        // open_user_stream seeds from the controller buffer, which holds
        // every sample since the deferred onset (no audio lost).
        self.open_user_stream();
        self.events.push(ConverseEvent::SpeechStarted);
    }

    fn finish_turn(&mut self, reply_text: String, end: AssistantEnd) {
        let timings = self
            .bk
            .take()
            .map(|bk| TurnTimings {
                utterance_seconds: bk.captured_samples as f64 / bk.device_rate as f64,
                end_to_first_token_ms: bk.first_token_ms.unwrap_or(0.0),
                end_to_first_audio_ms: bk.first_audio_ms.unwrap_or(0.0),
                end_to_turn_done_ms: bk.speech_end_at.elapsed().as_secs_f64() * 1e3,
                reply_codes: bk.reply_codes,
            })
            .unwrap_or_default();
        self.turns_completed += 1;
        self.events.push(ConverseEvent::TurnComplete {
            reply_text,
            end,
            timings,
        });
    }
}
