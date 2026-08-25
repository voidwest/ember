//! Deterministic scripted model (Track T).
//!
//! A `ChatModelEngine` whose behavior comes entirely from a script — no
//! GGUF, no tokenizer, no randomness. This is what makes the agent loop's
//! correctness tests hermetic and exact: tests pin WHICH tool was called,
//! with WHICH arguments, WHAT was reinjected, and WHAT the final answer
//! is, without a live model anywhere on the path.
//!
//! The script also simulates failure modes honestly:
//! - mid-turn cancellation (`cancel_after_tokens`) exercises the engine
//!   transaction contract from the loop's side;
//! - `fail_with` produces a generation error;
//! - every committed message is recorded verbatim so tests can assert
//!   exactly what the protocol reinjected into the conversation.

use anyhow::{bail, Result};

use super::model::{ChatModelEngine, GeneratedTurn, GenerationParams, ModelIdentity, TurnStop};
use super::tool::CancelFlag;

/// One scripted model turn.
#[derive(Clone)]
pub struct ScriptedTurn {
    /// Raw generated text (the protocol parses this like real output).
    pub output: String,
    /// Simulate cooperative cancellation after N observed tokens.
    pub cancel_after_tokens: Option<usize>,
    /// Simulate an infrastructure generation failure.
    pub fail_with: Option<String>,
}

impl ScriptedTurn {
    pub fn output(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            cancel_after_tokens: None,
            fail_with: None,
        }
    }

    pub fn cancel_after(mut self, tokens: usize) -> Self {
        self.cancel_after_tokens = Some(tokens);
        self
    }

    pub fn failing(message: impl Into<String>) -> Self {
        Self {
            output: String::new(),
            cancel_after_tokens: None,
            fail_with: Some(message.into()),
        }
    }
}

/// Scripted engine with full observation of what the loop commits.
pub struct ScriptedModel {
    identity: ModelIdentity,
    turns: Vec<ScriptedTurn>,
    turn_taken: usize,
    cursor: usize,

    // observations (tests assert on these)
    pub committed_messages: Vec<(String, String)>,
    pub generate_calls: usize,
    pub saw_cancellation_probe: bool,
}

impl ScriptedModel {
    pub fn new(turns: Vec<ScriptedTurn>) -> Self {
        Self {
            identity: ModelIdentity {
                model_path: "scripted://test-model".to_string(),
                model_sha256: None,
                architecture: "scripted".to_string(),
                quantization: None,
                n_layers: 0,
                embed_dim: 0,
                vocab_size: 0,
                tokenizer_sha256: None,
                context_len: 1 << 20,
            },
            turns,
            turn_taken: 0,
            cursor: 0,
            committed_messages: Vec::new(),
            generate_calls: 0,
            saw_cancellation_probe: false,
        }
    }

    /// Convenience: alternate tool-call JSON / final-text script.
    pub fn call_then_answer(call_json: &str, final_text: &str) -> Self {
        Self::new(vec![
            ScriptedTurn::output(call_json),
            ScriptedTurn::output(final_text),
        ])
    }

    fn fake_span(&mut self, rendered: &str) -> (usize, usize) {
        let n = (rendered.len() / 8).max(1);
        let span = (self.cursor, self.cursor + n);
        self.cursor += n;
        span
    }
}

impl ChatModelEngine for ScriptedModel {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }

    fn committed_len(&self) -> usize {
        self.cursor
    }

    fn commit_message(&mut self, rendered: &str) -> Result<(usize, usize)> {
        let span = self.fake_span(rendered);
        self.committed_messages
            .push(("message".to_string(), rendered.to_string()));
        Ok(span)
    }

    fn generate_turn(
        &mut self,
        prefix_rendered: &str,
        suffix_rendered: &str,
        _params: &GenerationParams,
        control: &CancelFlag,
        on_token: &mut dyn FnMut(u32, &str),
    ) -> Result<GeneratedTurn> {
        self.generate_calls += 1;
        self.committed_messages
            .push(("assistant_prefix".to_string(), prefix_rendered.to_string()));

        if self.turn_taken >= self.turns.len() {
            bail!("script exhausted after {} turns", self.generate_calls);
        }
        let script = self.turns[self.turn_taken].clone();
        let turn_index = self.turn_taken;
        self.turn_taken += 1;

        if let Some(message) = script.fail_with {
            return Err(anyhow::anyhow!("scripted generation failure: {message}"));
        }

        if let Some(after) = script.cancel_after_tokens {
            // emit `after` pseudo tokens, then observe cancellation
            for i in 0..after {
                on_token(1000 + i as u32, "x");
            }
            if control.is_cancelled() || after == 0 {
                self.saw_cancellation_probe = true;
                // transaction contract: nothing committed, caller rolls back
                return Ok(GeneratedTurn {
                    text: String::new(),
                    committed_ids: Vec::new(),
                    stop: None,
                    cancelled: true,
                    prompt_tokens_prefilled: 2,
                    decode_evaluations: after,
                    prefill_ms: 0.01,
                    decode_ms: 0.02,
                });
            }
        }

        // stream the scripted text in small chunks like a real detokenizer
        let pieces: Vec<char> = script.output.chars().collect();
        for (i, chunk) in pieces.chunks(3).enumerate() {
            let piece: String = chunk.iter().copied().collect();
            on_token((turn_index as u32) * 100 + i as u32, &piece);
            if control.is_cancelled() {
                self.saw_cancellation_probe = true;
                return Ok(GeneratedTurn {
                    text: String::new(),
                    committed_ids: Vec::new(),
                    stop: None,
                    cancelled: true,
                    prompt_tokens_prefilled: 2,
                    decode_evaluations: i,
                    prefill_ms: 0.01,
                    decode_ms: 0.02 * (i as f64),
                });
            }
        }

        let _ = self.fake_span(&format!(
            "{}x{}",
            prefix_rendered.len(),
            suffix_rendered.len()
        ));
        self.committed_messages.push((
            "assistant_turn".to_string(),
            format!("{}{}{}", prefix_rendered, script.output, suffix_rendered),
        ));

        Ok(GeneratedTurn {
            text: script.output.clone(),
            committed_ids: vec![turn_index as u32],
            stop: Some(TurnStop::MaxTokens),
            cancelled: false,
            prompt_tokens_prefilled: 2,
            decode_evaluations: pieces.len().div_ceil(3),
            prefill_ms: 0.01,
            decode_ms: 0.02,
        })
    }

    fn truncate_to(&mut self, len: usize) -> Result<()> {
        self.cursor = len;
        Ok(())
    }
}
