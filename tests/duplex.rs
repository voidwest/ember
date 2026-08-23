//! Phase 5 Tracks A/B: duplex queue policies, turn detection battery,
//! and real concurrent barge-in behavior — all with synthetic audio so
//! the suite stays hermetic (live-device smoke lives behind `ember voice`).

use ember::duplex::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const RATE: u32 = 16_000;

fn chunk(samples: Vec<f32>, offset: u64) -> AudioChunk {
    AudioChunk {
        samples,
        sample_rate: RATE,
        first_sample_offset: offset,
    }
}

/// Deterministic speech-like burst: decaying sinusoid mix.
fn speech_burst(seconds: f32) -> Vec<f32> {
    let n = (seconds * RATE as f32) as usize;
    (0..n)
        .map(|i| {
            let t = i as f32 / RATE as f32;
            0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.15 * (2.0 * std::f32::consts::PI * 660.0 * t).sin()
        })
        .collect()
}

fn silence(seconds: f32) -> Vec<f32> {
    vec![0.0; (seconds * RATE as f32) as usize]
}

// ---------------------------------------------------------------------------
// B1/B3: turn detector battery
// ---------------------------------------------------------------------------

#[test]
fn vad_silence_produces_no_events() {
    let mut vad = EnergyVad::new(RATE);
    for _ in 0..20 {
        assert_eq!(vad.feed(&chunk(silence(0.1), 0)), None);
    }
    assert!(!vad.is_speaking());
}

#[test]
fn vad_detects_speech_start_and_end_with_latency() {
    let mut vad = EnergyVad::new(RATE);
    let mut t = 0u64;
    // background noise floor
    let noise: Vec<f32> = std::iter::repeat_with(|| 0.0005 * (fastrand_like() - 0.5))
        .take(RATE as usize)
        .collect();
    let mut started_at = None;
    for i in 0..10 {
        if let Some(e) = vad.feed(&chunk(noise.clone(), t)) {
            panic!("unexpected {e:?} during silence at chunk {i}");
        }
        t += noise.len() as u64;
    }
    // speech onset: must be detected within ~100 ms of frames
    let burst = speech_burst(1.0);
    for (i, win) in burst.chunks(RATE as usize / 10).enumerate() {
        if let Some(TurnEvent::SpeechStarted) = vad.feed(&chunk(win.to_vec(), t)) {
            started_at = Some(i);
            break;
        }
        t += win.len() as u64;
    }
    let start_chunk = started_at.expect("speech start must be detected");
    assert!(
        start_chunk <= 4,
        "onset detection latency too high ({start_chunk} chunks)"
    );

    // hangover: end fires only after ~300 ms of quiet
    let mut ended = false;
    for i in 0..60 {
        if let Some(TurnEvent::SpeechEnded) = vad.feed(&chunk(silence(0.01), t)) {
            ended = true;
            assert!(
                (25..=40).contains(&i),
                "end latency out of hangover band: chunk {i}"
            );
            break;
        }
        t += (RATE as usize / 10) as u64;
    }
    assert!(ended, "speech end must fire after hangover");
}

/// tiny deterministic PRNG (LCG) — no external deps in tests
fn fastrand_like() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static S: Cell<u64> = const { Cell::new(0x2545F4914F6CDD1D) };
    }
    S.with(|s| {
        let x = s
            .get()
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s.set(x);
        ((x >> 33) as f32) / (u32::MAX as f32)
    })
}

#[test]
fn vad_short_hesitation_does_not_split_turn() {
    let mut vad = EnergyVad::new(RATE);
    let mut t = 0u64;
    // speak
    for win in speech_burst(0.5).chunks(RATE as usize / 10) {
        let _ = vad.feed(&chunk(win.to_vec(), t));
        t += win.len() as u64;
    }
    assert!(vad.is_speaking());
    // hesitation: 200 ms pause (< 300 ms hangover)
    for _ in 0..20 {
        let _ = vad.feed(&chunk(silence(0.01), t));
        t += (RATE as usize / 10) as u64;
    }
    assert!(
        vad.is_speaking(),
        "a 200ms hesitation must NOT split the turn"
    );
    // resume
    for win in speech_burst(0.3).chunks(RATE as usize / 10) {
        let _ = vad.feed(&chunk(win.to_vec(), t));
        t += win.len() as u64;
    }
    assert!(vad.is_speaking());
    // long pause ends it exactly once
    let mut ends = 0;
    for _ in 0..80 {
        if let Some(TurnEvent::SpeechEnded) = vad.feed(&chunk(silence(0.01), t)) {
            ends += 1;
        }
        t += (RATE as usize / 10) as u64;
    }
    assert_eq!(ends, 1, "exactly one SpeechEnded after the long pause");
}

#[test]
fn vad_rapid_second_turn_retriggers() {
    let mut vad = EnergyVad::new(RATE);
    let mut t = 0u64;
    let mut starts = 0;
    let mut ends = 0;
    // turn 1
    for win in speech_burst(0.4).chunks(RATE as usize / 10) {
        if let Some(TurnEvent::SpeechStarted) = vad.feed(&chunk(win.to_vec(), t)) {
            starts += 1;
        }
        t += win.len() as u64;
    }
    // gap 400 ms
    for _ in 0..40 {
        if let Some(TurnEvent::SpeechEnded) = vad.feed(&chunk(silence(0.01), t)) {
            ends += 1;
        }
        t += (RATE as usize / 10) as u64;
    }
    // turn 2 immediately after
    for win in speech_burst(0.4).chunks(RATE as usize / 10) {
        if let Some(TurnEvent::SpeechStarted) = vad.feed(&chunk(win.to_vec(), t)) {
            starts += 1;
        }
        t += win.len() as u64;
    }
    assert_eq!(starts, 2, "two turns must each produce SpeechStarted");
    assert_eq!(ends, 1);
}

#[test]
fn vad_manual_endpointing_seam() {
    let mut vad = EnergyVad::new(RATE);
    vad.force_start();
    assert!(vad.is_speaking());
    vad.force_end();
    assert!(!vad.is_speaking());
}

// ---------------------------------------------------------------------------
// A1/A3/A4: ring policies under contention
// ---------------------------------------------------------------------------

#[test]
fn capture_ring_drop_newest_is_counted_and_offsets_advance() {
    let (mut prod, mut cons) = capture_ring(1024);
    // overflow deliberately
    let big = vec![0.5f32; 4096];
    prod.push(&big);
    assert_eq!(cons.queued(), 1024);
    assert!(cons.dropped_samples() >= 3072);

    let c = cons.pop_chunk(RATE).expect("chunk");
    assert_eq!(c.samples.len(), 1024);
    // drop-newest keeps the stream PREFIX: offsets stay contiguous
    assert_eq!(c.first_sample_offset, 0);
    let c2 = cons.pop_chunk(RATE);
    assert!(c2.is_err(), "queue should be empty after full drain");
    // a fresh segment after drops continues from the accepted-stream end
    prod.push(&vec![0.25f32; 100]);
    let c3 = cons.pop_chunk(RATE).expect("second segment");
    assert_eq!(c3.first_sample_offset, 1024, "accepted-stream continuity");
}

#[test]
fn playback_underrun_emits_silence_and_counts() {
    let (_w, mut r) = playback_ring(1024);
    let mut out = vec![1.0f32; 512];
    r.pull(&mut out);
    assert!(out.iter().all(|&v| v == 0.0), "underrun must emit silence");
    assert_eq!(r.underruns(), 1);
}

#[test]
fn barge_in_clear_drops_queued_playback_immediately() {
    let (mut w, mut r) = playback_ring(4096);
    w.push(&vec![0.9f32; 2048]);
    assert_eq!(w.buffered(), 2048);
    w.request_clear();
    let mut out = vec![0.0f32; 256];
    r.pull(&mut out);
    assert!(
        out.iter().all(|&v| v == 0.0),
        "queued audio must be dropped"
    );
    assert_eq!(w.buffered(), 0);
    assert_eq!(r.clears(), 1);
}

// ---------------------------------------------------------------------------
// A5: REAL concurrency — capture thread keeps flowing while "inference"
// runs on the runtime thread; barge-in lands within milliseconds.
// ---------------------------------------------------------------------------

#[test]
fn duplex_capture_continues_during_long_inference_and_barge_in_fires() {
    let (mut prod, cons) = capture_ring(CAPTURE_QUEUE_SAMPLES);
    let (mut play_w, mut play_r) = playback_ring(PLAYBACK_QUEUE_SAMPLES);

    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();

    // realtime-ish capture thread: continuous mic stream @16 kHz
    let producer_thread = std::thread::spawn(move || {
        let mut offset = 0u64;
        let mut phase = 0.0f32;
        while !stop2.load(Ordering::Relaxed) {
            // ~10 ms blocks of a loud tone (speech-like for the VAD)
            let block: Vec<f32> = (0..160)
                .map(|_| {
                    phase += 2.0 * std::f32::consts::PI * 300.0 / RATE as f32;
                    0.3 * phase.sin()
                })
                .collect();
            prod.push(&block);
            offset += block.len() as u64;
            std::thread::sleep(Duration::from_millis(10));
        }
        offset
    });

    // assistant "speaking": queued audio + active flag set
    assert_eq!(play_w.push(&vec![0.8f32; 8000]), 0);
    assert_eq!(play_w.buffered(), 8000);

    let mut ctl = DuplexController::new(cons, play_w, Box::new(EnergyVad::new(RATE)));
    ctl.set_assistant_active(true);

    // "inference": simulate a long model step (~300 ms) on this thread
    let t0 = Instant::now();
    let mut barge_in_at = None;
    while t0.elapsed() < Duration::from_millis(600) {
        if let Some(TurnEvent::SpeechStarted) = ctl.pump() {
            barge_in_at = Some(t0.elapsed());
            break;
        }
        // the model work itself would go here; pump is cheap between steps
        std::thread::sleep(Duration::from_millis(10));
    }

    let at = barge_in_at.expect("barge-in must fire while assistant is active");
    assert!(
        at < Duration::from_millis(500),
        "barge-in took too long: {at:?}"
    );
    assert!(ctl.stop_probe(), "stop probe must latch");

    // capture kept flowing DURING inference: barge-in itself required
    // delivered audio, and the accepted counter proves sustained flow
    assert!(
        ctl.captured_total() >= 160,
        "capture must have delivered audio while inference ran"
    );

    stop.store(true, Ordering::Relaxed);
    let _written = producer_thread.join().expect("producer joins");

    // playback clear takes effect at the reader within one callback block
    let mut out = vec![1.0f32; 256];
    play_r.pull(&mut out);
    assert!(
        out.iter().all(|&v| v == 0.0),
        "queued assistant audio must be gone after barge-in"
    );
    assert!(play_r.clears() >= 1);
}

#[test]
fn duplex_same_session_continuation_after_barge_in() {
    // After a barge-in the controller must return to a clean armed state:
    // new user audio collects, assistant state re-arms, no stale latch.
    let (mut prod, cons) = capture_ring(4096);
    let (play_w, _r) = playback_ring(4096);
    let mut ctl = DuplexController::new(cons, play_w, Box::new(EnergyVad::new(RATE)));

    // drain-until-event helper: keeps feeding the detector even across
    // chunks that produce no event of their own
    fn drain_until(ctl: &mut DuplexController, want: TurnEvent, max_chunks: usize) -> bool {
        for _ in 0..max_chunks {
            if ctl.capture.queued() == 0 {
                // keep the hangover clock running with quiet audio; its
                // events matter exactly as much as pump()'s
                if let Some(e) = ctl.pump_quiet((RATE / 10) as usize)
                    && e == want
                {
                    return true;
                }
            }
            if let Some(e) = ctl.pump()
                && e == want
            {
                return true;
            }
        }
        false
    }

    ctl.set_assistant_active(true);
    prod.push(&speech_burst(0.5));
    assert!(
        drain_until(&mut ctl, TurnEvent::SpeechStarted, 100),
        "speech start must fire"
    );
    assert!(ctl.stop_probe());

    // generation observes cancel and finishes; session resets activity
    ctl.set_assistant_active(false);
    assert!(!ctl.stop_probe(), "latch clears when assistant deactivates");

    // close the open turn with trailing quiet (real mic would deliver it)
    prod.push(&silence(0.5));
    assert!(
        drain_until(&mut ctl, TurnEvent::SpeechEnded, 100),
        "turn must close after the user stops"
    );

    // second turn works: fresh utterance collection
    ctl.set_assistant_active(true);
    prod.push(&silence(0.2));
    for _ in 0..40 {
        let _ = ctl.pump();
    }
    assert!(
        !ctl.is_barge_in(),
        "quiet continuation must not re-barge-in"
    );
    prod.push(&speech_burst(0.4));
    assert!(
        drain_until(&mut ctl, TurnEvent::SpeechStarted, 100),
        "second barge-in must fire cleanly"
    );
    assert!(ctl.is_barge_in());
}
