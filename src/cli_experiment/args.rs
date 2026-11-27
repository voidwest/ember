use super::*;

/// `ember experiment ...` subcommands.
#[derive(ClapArgs)]
pub(crate) struct ExperimentCommand {
    #[command(subcommand)]
    pub command: ExperimentSubcommand,
}

#[derive(Subcommand)]
pub(crate) enum ExperimentSubcommand {
    /// Validate an experiment specification without inference.
    Validate(ValidateArgs),
    /// Resolve, execute, and bundle an experiment.
    Run(RunArgs),
    /// Summarize a bundle's contents.
    Inspect(InspectArgs),
    /// Verify a bundle offline (optionally deep against a model file).
    Verify(VerifyArgs),
    /// Compare two bundles semantically and numerically.
    Compare(CompareArgs),
    /// Re-run a bundle's experiment and classify reproduction.
    Reproduce(ReproduceArgs),
    /// Inspect tokenization and span matching.
    Tokenize(TokenizeArgs),
    /// Logit lens: project each captured residual-stream row through the
    /// model's final norm and LM head.
    Lens(crate::cli_experiment_lens::LensArgs),
}

#[derive(ClapArgs)]
pub(crate) struct ValidateArgs {
    /// Path to the experiment specification (TOML).
    pub spec: PathBuf,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

#[derive(ClapArgs)]
pub(crate) struct RunArgs {
    /// Path to the experiment specification (TOML).
    pub spec: PathBuf,
    /// Override the specification's execution mode.
    #[arg(long, value_name = "reference|planned|planned-fused")]
    pub execution: Option<String>,
    /// Override the specification's thread count.
    #[arg(long)]
    pub threads: Option<usize>,
    /// Override the specification's output directory.
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Keep the staging directory on failure (clearly marked incomplete).
    #[arg(long)]
    pub retain_incomplete: bool,
    /// Also run this intervention spec (repeatable), starting its prefill
    /// from the prompt prefix this run computes instead of recomputing it.
    /// Each variant writes its own bundle (its spec's output directory),
    /// bit-identical to running it alone; runtime.json records the path.
    #[arg(long = "variant", value_name = "spec.toml")]
    pub variants: Vec<PathBuf>,
    /// Sign the bundle's manifest.json with this private key (from
    /// `ember evidence init`) and write the signed-evidence-v2 envelope next
    /// to the bundle as `<bundle>.evidence.json`. Defaults to the
    /// `EMBER_SIGN_KEY` environment variable when set.
    #[arg(long, value_name = "key", conflicts_with = "no_sign")]
    pub sign_key: Option<PathBuf>,
    /// Do not sign, even when `EMBER_SIGN_KEY` is set.
    #[arg(long)]
    pub no_sign: bool,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

/// Environment variable naming the default signing key for
/// `experiment run`.
pub(crate) const SIGN_KEY_ENV: &str = "EMBER_SIGN_KEY";

/// Where `experiment run --sign-key` writes a bundle's evidence envelope,
/// and where `--trusted-key` looks for one without `--expect-evidence`:
/// `<bundle>.evidence.json`, a sibling of the bundle directory. It lives
/// outside the bundle because a file inside would change the bundle's
/// inventory (and so fail verification of the very bundle it signs).
pub(crate) fn bundle_evidence_path(bundle: &std::path::Path) -> PathBuf {
    let named = bundle
        .file_name()
        .map(|_| bundle.to_path_buf())
        .or_else(|| bundle.canonicalize().ok())
        .unwrap_or_else(|| bundle.to_path_buf());
    let mut name = named
        .file_name()
        .map(|name| name.to_os_string())
        .unwrap_or_else(|| "bundle".into());
    name.push(".evidence.json");
    named.with_file_name(name)
}

#[derive(ClapArgs)]
pub(crate) struct InspectArgs {
    /// Bundle directory.
    pub bundle: PathBuf,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

#[derive(ClapArgs)]
pub(crate) struct VerifyArgs {
    /// Bundle directory.
    pub bundle: PathBuf,
    /// Deep verification against a model file.
    #[arg(long, value_name = "model.gguf")]
    pub model: Option<PathBuf>,
    /// Deep tokenizer verification.
    #[arg(long, value_name = "tokenizer.json")]
    pub tokenizer: Option<PathBuf>,
    #[command(flatten)]
    pub anchor: AnchorArgs,
    /// Also write the JSON report to this path (never inside the bundle;
    /// verification does not modify the bundle it checks).
    #[arg(long, value_name = "report.json")]
    pub write_report: Option<PathBuf>,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

/// External anchors for a bundle's identity. Every hash a bundle carries is
/// written by the bundle's producer, so verification alone shows only that
/// the bundle is self-consistent: an edited bundle can be fully resealed.
/// An anchor compares the recomputed identity with a value from outside.
#[derive(ClapArgs, Clone, Default)]
pub(crate) struct AnchorArgs {
    /// Require this semantic hash (64 hex chars, obtained from a trusted
    /// record such as a paper or a lab notebook).
    #[arg(long, value_name = "hex", value_parser = parse_sha256_hex)]
    pub expect_semantic_hash: Option<String>,
    /// Require a signed evidence envelope over the bundle's manifest.json
    /// (`ember evidence sign --manifest <bundle>/manifest.json`) whose signed
    /// semantic and payload hashes equal the bundle's. `experiment run
    /// --sign-key` writes one as `<bundle>.evidence.json`, the default when
    /// only `--trusted-key` is given.
    #[arg(long, value_name = "envelope.json", requires = "trusted_key")]
    pub expect_evidence: Option<PathBuf>,
    /// The envelope signer's public key (`.pub` file or hex fingerprint).
    /// Without `--expect-evidence` the envelope is looked up next to the
    /// bundle as `<bundle>.evidence.json`.
    #[arg(long, value_name = "key.pub")]
    pub trusted_key: Option<String>,
}

impl AnchorArgs {
    pub(crate) fn is_anchored(&self) -> bool {
        self.expect_semantic_hash.is_some() || self.trusted_key.is_some()
    }

    /// The envelope to check for `bundle` and its trusted key: the explicit
    /// `--expect-evidence`, else the sibling `<bundle>.evidence.json`.
    pub(super) fn evidence_for(&self, bundle: &std::path::Path) -> Option<(PathBuf, &str)> {
        let trusted_key = self.trusted_key.as_deref()?;
        let envelope = self
            .expect_evidence
            .clone()
            .unwrap_or_else(|| bundle_evidence_path(bundle));
        Some((envelope, trusted_key))
    }
}

#[derive(ClapArgs)]
pub(crate) struct CompareArgs {
    /// First bundle directory.
    pub a: PathBuf,
    /// Second bundle directory.
    pub b: PathBuf,
    /// Require the first bundle to have this semantic hash.
    #[arg(long, value_name = "hex", value_parser = parse_sha256_hex)]
    pub expect_a_semantic_hash: Option<String>,
    /// Require the second bundle to have this semantic hash.
    #[arg(long, value_name = "hex", value_parser = parse_sha256_hex)]
    pub expect_b_semantic_hash: Option<String>,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

#[derive(ClapArgs)]
pub(crate) struct ReproduceArgs {
    /// Bundle directory to reproduce.
    pub bundle: PathBuf,
    /// Model file to re-run with (validated against the bundle hash).
    #[arg(long, value_name = "model.gguf")]
    pub model: PathBuf,
    /// Tokenizer file to re-run with (default: the path the bundle's spec
    /// names). Either way it must match the bundle's recorded SHA-256.
    #[arg(long, value_name = "tokenizer.json")]
    pub tokenizer: Option<PathBuf>,
    /// Anchors for the original bundle, checked before anything runs.
    #[command(flatten)]
    pub anchor: AnchorArgs,
    /// Output directory for the new bundle (default:
    /// `<bundle>-reproduced`).
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Keep the staging directory on failure.
    #[arg(long)]
    pub retain_incomplete: bool,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}

#[derive(ClapArgs)]
pub(crate) struct TokenizeArgs {
    /// GGUF model file.
    #[arg(long)]
    pub model: PathBuf,
    /// Architecture override (`auto`, `gpt2`, `llama`, `qwen3`, `gemma4`).
    #[arg(long, default_value = "auto")]
    pub arch: String,
    /// tokenizer.json path.
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// Text to tokenize.
    #[arg(long)]
    pub text: String,
    /// Optional span to match (matched-span selection, occurrence 0,
    /// all subtokens).
    #[arg(long)]
    pub match_span: Option<String>,
    /// Machine-readable output.
    #[arg(long)]
    pub json: bool,
}
