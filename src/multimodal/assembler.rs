//! Embedding assembly: turns text + image features into a model-ready
//! [`crate::embedding::EmbeddingSequence`].
//!
//! The generic transformer must never see model-specific formatting rules.
//! This module owns them: the SmolVLM assembler renders the chat template,
//! expands image placeholders into the tile structure (`<fake_token_around_image>`
//! `<row_i_col_j>` + image tokens, global tile last), tokenizes, looks up
//! text embeddings, and scatters vision features over the `<image>` token
//! positions — exactly the reference pipeline (HuggingFace `Idefics3Processor`
//! + `Idefics3ForConditionalGeneration.inputs_merger`).

use crate::backend::{Backend, CpuBackend};
use crate::llama::LlamaEmbedding;
use crate::tensor::CpuTensor;
use crate::tokenizer::EmberTokenizer;
use anyhow::{anyhow, ensure, Result};

/// The assembled result: token IDs (for decode/provenance) and the merged
/// embedding rows ready for the LLM prefill path.
#[derive(Debug)]
pub struct AssembledSequence {
    /// Full token sequence after chat-template rendering and image-token
    /// expansion (image tokens included; decode continues from the last id).
    pub input_ids: Vec<u32>,
    /// `[seq_len, llm_width]` merged embeddings: text lookup rows with the
    /// vision features scattered over `<image>` positions, in order.
    pub embeddings: CpuTensor,
}

/// A model-specific embedding assembler.
pub trait EmbeddingAssembler {
    /// Assemble one multimodal request into an [`AssembledSequence`].
    ///
    /// `text` is the user prompt (may contain one `<image>` placeholder),
    /// `image_features` is `[n_image_tokens, llm_width]` from the vision
    /// encoder + connector, `tile_grid` is the processed tile layout
    /// `(rows, cols)` (0,0 = no splitting).
    fn assemble(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        text: &str,
        image_features: &CpuTensor,
        tile_grid: (usize, usize),
        embed_table: &LlamaEmbedding<CpuBackend>,
    ) -> Result<AssembledSequence>;
}

/// SmolVLM/Idefics3 assembler (chat template + tile expansion + scatter).
pub struct SmolVlmAssembler {
    /// Number of `<image>` tokens per tile (= image_seq_len).
    pub image_seq_len: usize,
    pub image_token: String,
    pub fake_token: String,
    pub global_image_tag: String,
    pub end_of_utterance: String,
    pub im_start: String,
    pub im_end: String,
}

impl Default for SmolVlmAssembler {
    fn default() -> Self {
        Self {
            image_seq_len: 64,
            image_token: "<image>".into(),
            fake_token: "<fake_token_around_image>".into(),
            global_image_tag: "<global-img>".into(),
            end_of_utterance: "<end_of_utterance>".into(),
            im_start: "<|im_start|>".into(),
            im_end: "<|im_end|>".into(),
        }
    }
}

impl SmolVlmAssembler {
    /// Resolve all special-token IDs from the tokenizer.
    fn token_ids(&self, tokenizer: &EmberTokenizer) -> Result<SmolVlmTokenIds> {
        let need = |name: &str, tok: &str| {
            tokenizer
                .token_to_id(tok)
                .ok_or_else(|| anyhow!("tokenizer is missing {name} token {tok:?}"))
        };
        // Validate every special token the expansion can emit; the
        // assembler only needs the image id itself, but a tokenizer missing
        // any structural token must fail closed before the model runs.
        need("image", &self.image_token)?;
        need("fake-token", &self.fake_token)?;
        need("global-image", &self.global_image_tag)?;
        need("end-of-utterance", &self.end_of_utterance)?;
        need("im_start", &self.im_start)?;
        need("im_end", &self.im_end)?;
        for i in 0..36 {
            let (r, c) = (i / 6 + 1, i % 6 + 1);
            need(&format!("row {r} col {c}"), &format!("<row_{r}_col_{c}>"))?;
        }
        Ok(SmolVlmTokenIds {
            image: need("image", &self.image_token)?,
        })
    }

    /// Render the reference chat template for one user message plus the
    /// generation prompt: `<|im_start|>User:<sep><content><end_of_utterance>\nAssistant:`.
    ///
    /// The separator is `":"` when the message's first content element is an
    /// image and `": "` otherwise (exactly the reference Jinja template).
    pub fn render_chat_template(&self, user_text: &str, image_first: bool) -> String {
        let sep = if image_first { ":" } else { ": " };
        format!(
            "{}User{sep}{}{}\nAssistant:",
            self.im_start, user_text, self.end_of_utterance
        )
    }

    /// Expand the `<image>` placeholder into the tile token string, exactly
    /// like the reference `replace_image_token` (tiles row-major, newline
    /// after each row, global tile last).
    pub fn expand_image_placeholder(&self, tile_grid: (usize, usize)) -> String {
        let mut out = String::new();
        let (rows, cols) = tile_grid;
        if rows == 0 || cols == 0 {
            // no splitting: fake + global + images + fake
            out.push_str(&self.fake_token);
            out.push_str(&self.global_image_tag);
            out.push_str(&self.image_token.repeat(self.image_seq_len));
            out.push_str(&self.fake_token);
            return out;
        }
        for r in 0..rows {
            for c in 0..cols {
                out.push_str(&self.fake_token);
                out.push_str(&format!("<row_{}_col_{}>", r + 1, c + 1));
                out.push_str(&self.image_token.repeat(self.image_seq_len));
            }
            out.push('\n');
        }
        out.push('\n');
        out.push_str(&self.fake_token);
        out.push_str(&self.global_image_tag);
        out.push_str(&self.image_token.repeat(self.image_seq_len));
        out.push_str(&self.fake_token);
        out
    }
}

struct SmolVlmTokenIds {
    image: u32,
}

impl EmbeddingAssembler for SmolVlmAssembler {
    fn assemble(
        &self,
        backend: &CpuBackend,
        tokenizer: &EmberTokenizer,
        text: &str,
        image_features: &CpuTensor,
        tile_grid: (usize, usize),
        embed_table: &LlamaEmbedding<CpuBackend>,
    ) -> Result<AssembledSequence> {
        let ids = self.token_ids(tokenizer)?;

        // 1. render template with the placeholder expanded
        let placeholder_count = text.matches(&self.image_token).count();
        ensure!(
            placeholder_count <= 1,
            "SmolVLM assembler supports at most one <image> placeholder per request"
        );
        ensure!(
            placeholder_count > 0 || image_features.shape()[0] == 0,
            "image provided but prompt has no <image> placeholder              (the reference processor rejects this too)"
        );
        let content = text.replace(&self.image_token, &self.expand_image_placeholder(tile_grid));
        let image_first = text.trim_start().starts_with(&self.image_token);
        let rendered = self.render_chat_template(&content, image_first);

        // 2. tokenize (no BOS for SmolLM2: bos_token_id() is None)
        let input_ids = tokenizer.encode(&rendered)?;

        // 3. text embeddings via the normal lookup (same row-copy ops the
        //    LLM token path uses, so quantized tables work identically)
        let embed_dim = match embed_table {
            LlamaEmbedding::F32(t) => t.shape()[1],
            LlamaEmbedding::Q8_0(w) => w.in_features(),
            LlamaEmbedding::KQuant(w) => w.in_features(),
        };
        let mut embeddings = backend.zeroes(&[input_ids.len(), embed_dim])?;
        for (row, &token) in input_ids.iter().enumerate() {
            match embed_table {
                LlamaEmbedding::F32(table) => {
                    backend.assign_row_from_table(&mut embeddings, row, table, token as usize)?;
                }
                LlamaEmbedding::Q8_0(table) => {
                    backend.assign_row_from_q8_0(&mut embeddings, row, table, token as usize)?;
                }
                LlamaEmbedding::KQuant(table) => {
                    backend.assign_row_from_k(&mut embeddings, row, table, token as usize)?;
                }
            }
        }

        // 4. scatter vision features over <image> positions, in order
        let n_image_tokens = input_ids.iter().filter(|&&t| t == ids.image).count();
        ensure!(
            n_image_tokens == image_features.shape()[0],
            "assembler found {n_image_tokens} <image> tokens but vision encoder produced {} rows",
            image_features.shape()[0]
        );
        let mut feat_row = 0usize;
        for (row, &token) in input_ids.iter().enumerate() {
            if token == ids.image {
                let dst = &mut embeddings.data_mut()[row * embed_dim..(row + 1) * embed_dim];
                let src = &image_features.data()[feat_row * embed_dim..(feat_row + 1) * embed_dim];
                dst.copy_from_slice(src);
                feat_row += 1;
            }
        }
        ensure!(
            feat_row == image_features.shape()[0],
            "scatter count mismatch"
        );

        Ok(AssembledSequence {
            input_ids,
            embeddings,
        })
    }
}
