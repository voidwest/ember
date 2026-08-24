//! Hermetic contracts for the live conversation layer (Phase 5 Session 2,
//! Track A). No models, no devices: the transition graph is pinned exactly,
//! and the ring→controller→stream wiring is proven with synthetic producers
//! at multiple device rates.

use ember::duplex::{
    capture_ring, playback_ring, DuplexController, EnergyVad, TurnEvent, CAPTURE_QUEUE_SAMPLES,
};
use ember::multimodal::converse::{ConversationAction, ConversationMachine, ConversationState};

// ---------------------------------------------------------------------------
// A3: the complete (state, event) -> (state, actions) graph
// ---------------------------------------------------------------------------

#[test]
fn transition_graph_is_total_and_pinned() {
    use ConversationState as S;
    use TurnEvent as E;
    let m = ConversationMachine;
    let states = [
        S::Idle,
        S::CapturingUser,
        S::FinalizingUser,
        S::GeneratingAssistant,
        S::SpeakingAssistant,
    ];
    let events = [E::SpeechStarted, E::SpeechContinues, E::SpeechEnded];

    for &state in &states {
        for &event in &events {
            let (next, actions) = m.apply(state, event);
            match (state, event) {
                // the four live edges
                (S::Idle, E::SpeechStarted) => {
                    assert_eq!(next, S::CapturingUser);
                    assert_eq!(actions, vec![ConversationAction::OpenUserStream]);
                }
                (S::GeneratingAssistant | S::SpeakingAssistant, E::SpeechStarted) => {
                    assert_eq!(next, S::CapturingUser);
                    assert_eq!(actions, vec![ConversationAction::OpenUserStream]);
                }
                (S::CapturingUser, E::SpeechEnded) => {
                    assert_eq!(next, S::FinalizingUser);
                    assert_eq!(actions, vec![ConversationAction::FinalizeAndCommit]);
                }
                // everything else: no-op
                _ => {
                    assert_eq!(next, state, "{state:?} + {event:?} must be a no-op");
                    assert!(actions.is_empty());
                }
            }
        }
    }
}

#[test]
fn driver_completion_edges_compose_the_turn_pipeline() {
    use ConversationState as S;
    // commit edge only from Finalizing
    assert_eq!(
        ConversationMachine::after_commit(S::FinalizingUser),
        S::GeneratingAssistant
    );
    assert_eq!(ConversationMachine::after_commit(S::Idle), S::Idle);
    // generation edge: cancel vs complete
    assert_eq!(
        ConversationMachine::after_generation(S::GeneratingAssistant, true),
        S::Idle
    );
    assert_eq!(
        ConversationMachine::after_generation(S::GeneratingAssistant, false),
        S::SpeakingAssistant
    );
    // speech edge back to Idle; foreign states pass through
    assert_eq!(
        ConversationMachine::after_speech(S::SpeakingAssistant),
        S::Idle
    );
    assert_eq!(ConversationMachine::after_speech(S::Idle), S::Idle);
}

// ---------------------------------------------------------------------------
// controller-level wiring with synthetic device callbacks
// ---------------------------------------------------------------------------

struct Harness {
    producer: ember::duplex::CaptureProducer,
    ctl: DuplexController,
}

fn harness(rate: u32) -> Harness {
    let (producer, consumer) = capture_ring(CAPTURE_QUEUE_SAMPLES);
    // The controller only needs SOME playback sink for these hermetic
    // tests; its reader half is never pulled.
    let (writer, _reader) = playback_ring(16_000);
    let detector = Box::new(EnergyVad::new(rate));
    let ctl = DuplexController::new_with_sample_rate(consumer, writer, detector, rate);
    Harness { producer, ctl }
}

fn speech_burst(rate: u32, seconds: f32) -> Vec<f32> {
    let n = (rate as f32 * seconds) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / rate as f32;
            0.8 * (2.0 * std::f32::consts::PI * 300.0 * t).sin()
        })
        .collect()
}

#[test]
fn utterance_collection_rate_and_offset_coherence_at_device_rates() {
    for rate in [16_000u32, 44_100, 48_000] {
        let mut h = harness(rate);
        // quiet warm-up so the noise floor adapts below the burst level
        let quiet = vec![0.0f32; rate as usize / 4]; // 250 ms silence
        h.producer.push(&quiet);
        assert!(h.ctl.pump_events().is_empty(), "no events in silence");

        let burst = speech_burst(rate, 1.0);
        h.producer.push(&burst);
        let events = h.ctl.pump_events();
        assert!(
            events.contains(&TurnEvent::SpeechStarted),
            "onset must fire @ {rate}"
        );

        h.producer.push(&vec![0.0f32; rate as usize / 2]); // 500 ms hangover+
        let events = h.ctl.pump_events();
        assert!(
            events.contains(&TurnEvent::SpeechEnded),
            "endpoint must fire @ {rate}"
        );

        let (utt, utt_rate, offset) = h.ctl.take_utterance();
        assert_eq!(utt_rate, rate);
        // utterance spans onset..end plus hangover tail: >= burst length
        assert!(
            utt.len() >= burst.len(),
            "utterance {} < burst {} @ {rate}",
            utt.len(),
            burst.len()
        );
        assert_eq!(offset, 0, "no drops occurred");
    }
}

#[test]
fn chunk_tap_sees_every_chunk_and_both_transitions_in_one_pop() {
    let mut h = harness(16_000);
    let mut taps: Vec<usize> = Vec::new();
    // Feed a single LARGE chunk containing quiet + loud + quiet: the tap
    // fires once for the whole pop while BOTH transitions are reported.
    let mut big = vec![0.0f32; 16_000 / 10];
    big.extend(speech_burst(16_000, 0.4));
    big.extend(vec![0.0f32; 16_000]); // > hangover
    h.producer.push(&big);
    let events = h.ctl.pump_with_chunk_cb(|samples, rate| {
        taps.push(samples.len());
        assert_eq!(rate, 16_000);
    });
    assert_eq!(taps, vec![big.len()], "one tap per drained pop");
    assert!(events.contains(&TurnEvent::SpeechStarted));
    assert!(
        events.contains(&TurnEvent::SpeechEnded),
        "both transitions in one pump must survive"
    );
}

#[test]
fn barge_in_latch_still_fires_during_assistant_activity() {
    let mut h = harness(16_000);
    h.ctl.set_assistant_active(true);
    h.producer.push(&speech_burst(16_000, 0.5));
    let events = h.ctl.pump_events();
    assert!(events.contains(&TurnEvent::SpeechStarted));
    assert!(h.ctl.is_barge_in(), "latch set during assistant activity");
    assert!(h.ctl.stop_probe());
    h.ctl.set_assistant_active(false);
    assert!(
        !h.ctl.is_barge_in(),
        "latch clears when assistant deactivates"
    );
}

#[test]
fn capture_overrun_policy_keeps_offsets_visible() {
    let mut h = harness(16_000);
    // overflow the 32k ring without pumping
    let flood = vec![0.1f32; CAPTURE_QUEUE_SAMPLES + 8_000];
    h.producer.push(&flood);
    assert!(h.ctl.dropped_samples() >= 8_000, "drop-newest counted");
    // drain everything queued; the next chunk's offset must reflect the gap
    while h.ctl.capture.queued() > 0 {
        let _ = h.ctl.capture.pop_chunk(16_000).expect("nonempty queue");
    }
    let before = h.ctl.captured_total();
    h.producer.push(&vec![0.1f32; 100]);
    let chunk = h.ctl.capture.pop_chunk(16_000).expect("chunk");
    assert_eq!(chunk.samples.len(), 100);
    assert_eq!(h.ctl.captured_total(), before + 100);
    assert!(
        chunk.first_sample_offset >= CAPTURE_QUEUE_SAMPLES as u64,
        "offset advanced past the dropped head (got {})",
        chunk.first_sample_offset
    );
}

// ---------------------------------------------------------------------------
// Track K additions: explicit failure-behavior contracts
// ---------------------------------------------------------------------------

#[test]
fn degenerate_inputs_fail_closed_without_panic() {
    // empty capture queue: pump is a no-op
    let mut h = harness(16_000);
    assert!(h.ctl.pump_events().is_empty());

    // pure-digital silence at high volume never triggers VAD
    h.producer.push(&vec![0.0f32; 32_000]);
    assert!(h.ctl.pump_events().is_empty());

    // take_utterance on empty buffer is clean
    let (utt, _, _) = h.ctl.take_utterance();
    assert!(utt.is_empty());
}

#[test]
fn playback_ring_full_rejects_rather_than_blocks() {
    let (_p, _consumer) = capture_ring(64);
    let (mut writer, mut reader) = playback_ring(64);
    // fill the 64-sample ring
    let rejected = writer.push(&vec![0.5f32; 100]);
    assert_eq!(rejected, 36, "over-push rejected, not blocked");
    let mut out = vec![0.0f32; 64];
    reader.pull(&mut out);
    assert!(out.iter().take(64).all(|&v| v == 0.5));
}
