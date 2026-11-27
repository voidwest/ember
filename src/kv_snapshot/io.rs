use super::*;

pub(super) fn execution_fingerprint(plan: &ExecutionPlan) -> anyhow::Result<String> {
    let mut value = serde_json::to_value(plan)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("execution plan did not serialize as an object"))?;
    object.remove("plan_hash");
    // Scratch offsets and KV allocation strides/capacity are runtime sizing,
    // not numerical execution semantics. The operation graph still records
    // where KV storage and attention occur.
    object.remove("scratch");
    if let Some(kv) = object
        .get_mut("kv")
        .and_then(serde_json::Value::as_object_mut)
    {
        for field in ["layer_stride", "head_stride", "pos_stride", "max_seq"] {
            kv.insert(field.into(), serde_json::json!(0));
        }
    }
    if let Some(gguf) = object
        .get_mut("gguf")
        .and_then(serde_json::Value::as_object_mut)
    {
        // The loaded runtime may deliberately cap the same model's table and
        // cache below its GGUF context. Positions within the imported prefix
        // use identical tables.
        gguf.insert("context_length".into(), serde_json::json!(0));
    }
    if let Some(provenance) = object
        .get_mut("provenance")
        .and_then(serde_json::Value::as_object_mut)
    {
        provenance.insert(
            "plan_build_time".into(),
            serde_json::Value::String(String::new()),
        );
    }
    Ok(sha256_hex(&serde_json::to_vec(&value)?))
}

pub(super) fn manifest_hash(manifest: &KvSnapshotManifest) -> anyhow::Result<String> {
    let mut value = manifest.clone();
    value.snapshot_hash.clear();
    Ok(sha256_hex(&serde_json::to_vec(&value)?))
}

/// Domain-separated SHA-256 of a token-ID prefix (little-endian u32 per token).
///
/// The fixed domain separator ties the hash to this exact schema so a prefix
/// hashed by one producer (e.g. [`KvSnapshot::export_native`]) compares
/// correctly against a prefix hashed elsewhere. Do not change the separator
/// without a schema bump.
///
/// Internal implementation helper: `#[doc(hidden)]` keeps it source-visible to
/// the binary target without a compatibility promise (see
/// `docs/api-stability.md`, "Internal implementation").
#[doc(hidden)]
pub fn hash_token_ids(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"ember.kv-prefix-token-ids.v1\0");
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    hex(&hasher.finalize())
}

pub(super) fn ensure_finite_f16(file: &str, values: &[f16]) -> anyhow::Result<()> {
    for (index, value) in values.iter().enumerate() {
        anyhow::ensure!(
            value.to_f32().is_finite(),
            "{file} contains a non-finite f16 payload value at index {index}"
        );
    }
    Ok(())
}

pub(super) fn write_f16_new_file(path: &Path, values: &[f16]) -> anyhow::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut bytes = [0u8; PAYLOAD_READ_CHUNK_BYTES];
    for chunk in values.chunks(bytes.len() / 2) {
        for (index, value) in chunk.iter().enumerate() {
            let offset = index * 2;
            bytes[offset..offset + 2].copy_from_slice(&value.to_bits().to_le_bytes());
        }
        file.write_all(&bytes[..chunk.len() * 2])?;
    }
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
pub(super) fn f16_to_le_bytes(values: &[f16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(2));
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    bytes
}

pub(super) const PAYLOAD_READ_CHUNK_BYTES: usize = 64 * 1024;

/// Read a serialized f16 payload into its final representation without first
/// materializing a second full-size byte buffer. The bounded chunk also keeps
/// this path independent of the payload size supplied by an untrusted
/// manifest.
pub(super) fn read_f16_file(path: &Path, length: u64) -> anyhow::Result<Vec<f16>> {
    let length_usize = usize::try_from(length)
        .map_err(|_| anyhow::anyhow!("file '{}' is too large for this platform", path.display()))?;
    anyhow::ensure!(
        length_usize.is_multiple_of(2),
        "f16 payload byte length is odd"
    );
    let mut file = open_snapshot_file(path)?;
    let initial_metadata = file.metadata()?;
    anyhow::ensure!(
        initial_metadata.file_type().is_file(),
        "'{}' is not a regular file",
        path.display()
    );
    let mut values = Vec::new();
    values
        .try_reserve_exact(length_usize / 2)
        .map_err(|error| anyhow::anyhow!("cannot allocate decoded f16 payload: {error}"))?;
    let mut bytes = [0u8; PAYLOAD_READ_CHUNK_BYTES];
    let mut remaining = length_usize;
    while remaining > 0 {
        // The payload length is even and the chunk size is even, so every
        // iteration can be decoded as complete little-endian f16 pairs.
        let chunk_length = remaining.min(bytes.len());
        file.read_exact(&mut bytes[..chunk_length])?;
        for pair in bytes[..chunk_length].chunks_exact(2) {
            let value = f16::from_bits(u16::from_le_bytes([pair[0], pair[1]]));
            anyhow::ensure!(
                value.to_f32().is_finite(),
                "'{}' contains a non-finite f16 payload value",
                path.display()
            );
            values.push(value);
        }
        remaining -= chunk_length;
    }
    let mut trailing = [0u8; 1];
    anyhow::ensure!(
        file.read(&mut trailing)? == 0,
        "'{}' has unexpected extra bytes",
        path.display()
    );
    ensure_snapshot_file_unchanged(path, &file, &initial_metadata)?;
    Ok(values)
}

/// Hash f16 values in the canonical little-endian representation without
/// allocating a byte vector the size of the payload.
pub(super) fn hash_f16_le(values: &[f16]) -> String {
    let mut hasher = Sha256::new();
    let mut bytes = [0u8; PAYLOAD_READ_CHUNK_BYTES];
    for chunk in values.chunks(bytes.len() / 2) {
        for (index, value) in chunk.iter().enumerate() {
            let offset = index * 2;
            bytes[offset..offset + 2].copy_from_slice(&value.to_bits().to_le_bytes());
        }
        hasher.update(&bytes[..chunk.len() * 2]);
    }
    hex(&hasher.finalize())
}

pub(super) fn validate_sha256(name: &str, value: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "{name} SHA-256 must be 64 lowercase hexadecimal characters"
    );
    Ok(())
}

pub(super) fn nonempty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

pub(super) fn validate_snapshot_directory(path: &Path) -> anyhow::Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    anyhow::ensure!(
        metadata.file_type().is_dir(),
        "snapshot '{}' is not a real directory",
        path.display()
    );
    let mut names = Vec::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        anyhow::ensure!(
            file_type.is_file(),
            "snapshot entry '{}' is not a regular file",
            entry.path().display()
        );
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| anyhow::anyhow!("snapshot contains a non-UTF-8 file name"))?;
        names.push(name);
    }
    names.sort();
    let mut expected = vec![
        KV_KEY_FILE.to_string(),
        KV_MANIFEST_FILE.to_string(),
        KV_VALUE_FILE.to_string(),
    ];
    expected.sort();
    anyhow::ensure!(
        names == expected,
        "snapshot directory must contain exactly {expected:?}; found {names:?}"
    );
    Ok(())
}

pub(super) fn open_snapshot_file(path: &Path) -> anyhow::Result<File> {
    // Reject symlinks before opening and compare the opened descriptor with
    // that lstat result. This closes the common replacement race without
    // following an attacker-controlled snapshot path outside its directory.
    let path_metadata = std::fs::symlink_metadata(path)?;
    anyhow::ensure!(
        path_metadata.file_type().is_file(),
        "'{}' is not a regular file",
        path.display()
    );
    let file = File::open(path)?;
    let file_metadata = file.metadata()?;
    anyhow::ensure!(
        file_metadata.file_type().is_file()
            && file_metadata.len() == path_metadata.len()
            && file_metadata.modified().ok() == path_metadata.modified().ok(),
        "file changed while opening '{}'",
        path.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        anyhow::ensure!(
            file_metadata.dev() == path_metadata.dev()
                && file_metadata.ino() == path_metadata.ino(),
            "file changed while opening '{}'",
            path.display()
        );
    }
    Ok(file)
}

pub(super) fn read_exact_file(path: &Path, length: u64) -> anyhow::Result<Vec<u8>> {
    let length_usize = usize::try_from(length)
        .map_err(|_| anyhow::anyhow!("file '{}' is too large for this platform", path.display()))?;
    let mut file = open_snapshot_file(path)?;
    let initial_metadata = file.metadata()?;
    anyhow::ensure!(
        initial_metadata.file_type().is_file(),
        "'{}' is not a regular file",
        path.display()
    );
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length_usize).map_err(|error| {
        anyhow::anyhow!(
            "cannot allocate {length} bytes for '{}': {error}",
            path.display()
        )
    })?;
    bytes.resize(length_usize, 0);
    file.read_exact(&mut bytes)?;
    let mut trailing = [0u8; 1];
    anyhow::ensure!(
        file.read(&mut trailing)? == 0,
        "'{}' has unexpected extra bytes",
        path.display()
    );
    ensure_snapshot_file_unchanged(path, &file, &initial_metadata)?;
    Ok(bytes)
}

pub(super) fn ensure_snapshot_file_unchanged(
    path: &Path,
    file: &std::fs::File,
    initial_metadata: &std::fs::Metadata,
) -> anyhow::Result<()> {
    let final_metadata = file.metadata()?;
    let final_path_metadata = std::fs::symlink_metadata(path)?;
    anyhow::ensure!(
        final_metadata.len() == initial_metadata.len()
            && final_metadata.modified().ok() == initial_metadata.modified().ok()
            && final_path_metadata.file_type().is_file()
            && final_path_metadata.len() == initial_metadata.len()
            && final_path_metadata.modified().ok() == initial_metadata.modified().ok(),
        "file changed while reading '{}'",
        path.display()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        anyhow::ensure!(
            final_path_metadata.dev() == initial_metadata.dev()
                && final_path_metadata.ino() == initial_metadata.ino(),
            "file changed while reading '{}'",
            path.display()
        );
    }
    Ok(())
}

pub(super) fn create_staging_directory(parent: &Path, name: &str) -> anyhow::Result<PathBuf> {
    for _ in 0..128 {
        let staging = parent.join(format!(
            ".{name}.tmp-{}-{}",
            std::process::id(),
            STAGING_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        match std::fs::create_dir(&staging) {
            Ok(()) => return Ok(staging),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    anyhow::bail!("could not allocate a unique snapshot staging directory")
}

/// Normalize a user-supplied path to an absolute, component-collapsed form.
///
/// Resolves against the working directory for relative inputs and collapses
/// `.`/`..` without touching the filesystem (no symlink resolution). This is
/// the shared primitive behind snapshot-output placement checks and
/// trace-output path resolution — keep it pure so both consumers stay
/// consistent.
///
/// Internal implementation helper: `#[doc(hidden)]` keeps it source-visible to
/// the binary target without a compatibility promise (see
/// `docs/api-stability.md`, "Internal implementation").
#[doc(hidden)]
pub fn normalize_path(path: &Path) -> anyhow::Result<PathBuf> {
    use std::path::Component;
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    Ok(normalized)
}

pub(super) fn reject_dangerous_snapshot_output(path: &Path) -> anyhow::Result<()> {
    let normalized = normalize_path(path)?;
    let cwd = std::env::current_dir()?.canonicalize()?;
    anyhow::ensure!(
        !cwd.starts_with(&normalized),
        "refusing snapshot output '{}' because it is the working directory or one of its ancestors",
        path.display()
    );
    Ok(())
}

pub(super) fn write_new_file(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

pub(super) fn reject_symlink_if_present(path: &Path) -> anyhow::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) => anyhow::ensure!(
            !metadata.file_type().is_symlink(),
            "refusing snapshot output symlink '{}'",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

pub(super) struct StagingGuard {
    path: PathBuf,
    pub(super) published: bool,
}

impl StagingGuard {
    pub(super) fn new(path: PathBuf) -> Self {
        Self {
            path,
            published: false,
        }
    }
}

impl Drop for StagingGuard {
    fn drop(&mut self) {
        if !self.published {
            std::fs::remove_dir_all(&self.path).ok();
        }
    }
}
