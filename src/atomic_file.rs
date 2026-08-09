//! Small, dependency-free helpers for transactional file replacement.
//!
//! Writers create a sibling temporary file, flush it, and rename it over the
//! destination only after the complete payload has been persisted. Keeping
//! the temporary file in the destination directory also guarantees that the
//! rename does not cross filesystems.

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMP_FILE: AtomicU64 = AtomicU64::new(0);

fn create_sibling_temp(path: &Path) -> io::Result<(PathBuf, File)> {
    let filename = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("output path has no filename: {}", path.display()),
        )
    })?;
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));

    // create_new makes collisions harmless, including across processes.
    for _ in 0..128 {
        let sequence = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
        let mut temporary_name = OsString::from(".");
        temporary_name.push(filename);
        temporary_name.push(format!(".ember-tmp-{}-{sequence}", std::process::id()));
        let temporary_path = parent.join(temporary_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((temporary_path, file)),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }

    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!(
            "could not allocate a unique temporary file next to {}",
            path.display()
        ),
    ))
}

/// Atomically replace `path` with `bytes` without exposing a partial payload.
///
/// The destination's parent directory must already exist. On any failure
/// before the rename, the temporary file is removed and the prior destination
/// remains untouched.
pub fn atomic_write(path: impl AsRef<Path>, bytes: &[u8]) -> io::Result<()> {
    atomic_write_with_policy(path.as_ref(), bytes, true)
}

/// Atomically publish a new file and fail if the destination already exists.
///
/// The hard-link publication step is atomic and cannot replace a file that
/// appeared after a caller's preflight check.
pub fn atomic_write_new(path: impl AsRef<Path>, bytes: &[u8]) -> io::Result<()> {
    atomic_write_with_policy(path.as_ref(), bytes, false)
}

fn atomic_write_with_policy(path: &Path, bytes: &[u8], overwrite: bool) -> io::Result<()> {
    let (temporary_path, mut file) = create_sibling_temp(path)?;

    let write_result = (|| {
        file.write_all(bytes)?;
        file.flush()?;
        file.sync_all()?;
        Ok::<(), io::Error>(())
    })();
    drop(file);

    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary_path);
        return Err(error);
    }
    let publish_result = if overwrite {
        fs::rename(&temporary_path, path)
    } else {
        fs::hard_link(&temporary_path, path)
    };
    if let Err(error) = publish_result {
        let _ = fs::remove_file(&temporary_path);
        return Err(error);
    }
    if !overwrite {
        let _ = fs::remove_file(&temporary_path);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let sequence = NEXT_TEMP_FILE.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "ember-atomic-file-test-{}-{sequence}",
            std::process::id()
        ))
    }

    #[test]
    fn replaces_complete_payload_and_leaves_no_temporary_file() {
        let directory = temp_dir();
        fs::create_dir(&directory).unwrap();
        let output = directory.join("artifact.json");
        fs::write(&output, b"old").unwrap();

        atomic_write(&output, b"new payload\n").unwrap();

        assert_eq!(fs::read(&output).unwrap(), b"new payload\n");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn new_publication_never_replaces_existing_file() {
        let directory = temp_dir();
        fs::create_dir(&directory).unwrap();
        let output = directory.join("artifact.json");
        fs::write(&output, b"concurrent").unwrap();

        assert!(atomic_write_new(&output, b"new payload").is_err());
        assert_eq!(fs::read(&output).unwrap(), b"concurrent");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn missing_parent_does_not_create_directories() {
        let directory = temp_dir();
        let output = directory.join("missing").join("artifact.json");
        assert!(atomic_write(&output, b"payload").is_err());
        assert!(!directory.exists());
    }
}
