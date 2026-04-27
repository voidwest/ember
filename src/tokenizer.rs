use anyhow::{Context, Result};
use tokenizers::Tokenizer;

pub struct EmberTokenizer {
    inner: Tokenizer,
}

impl EmberTokenizer {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {}
}
