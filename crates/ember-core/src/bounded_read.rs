//! Reading user-supplied files without trusting them to be small, finite or
//! regular: a path can name a FIFO (blocks on open), a device such as
//! `/dev/zero` (never ends) or a file far larger than any real input.

use std::io::Read as _;
use std::path::Path;

/// Read a whole regular file of at most `cap` bytes, or `None` if it does
/// not exist.
///
/// The file is opened without blocking (a FIFO fails instead of hanging),
/// must be a regular file (a symlink to one is followed), and at most
/// `cap + 1` bytes are read, so a file that grows or never ends cannot
/// exhaust memory.
pub fn read_regular_file(path: &Path, cap: u64) -> std::io::Result<Option<Vec<u8>>> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(rustix::fs::OFlags::NONBLOCK.bits() as i32);
    }
    let file = match options.open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "not a regular file",
        ));
    }
    let too_large = || {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("larger than the {cap}-byte limit"),
        )
    };
    if metadata.len() > cap {
        return Err(too_large());
    }
    let mut bytes = Vec::new();
    file.take(cap + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > cap {
        return Err(too_large());
    }
    Ok(Some(bytes))
}

/// [`read_regular_file`] for a file that must exist.
pub fn read_existing_regular_file(path: &Path, cap: u64) -> std::io::Result<Vec<u8>> {
    read_regular_file(path, cap)?
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no such file"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn devices_fifos_and_oversized_files_fail_without_hanging() {
        let dir = std::env::temp_dir().join(format!("ember-bounded-read-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let small = dir.join("small");
        std::fs::write(&small, b"12345").unwrap();
        assert_eq!(read_regular_file(&small, 5).unwrap().unwrap(), b"12345");
        assert!(read_regular_file(&small, 4).is_err());
        assert!(read_regular_file(&dir.join("absent"), 4).unwrap().is_none());
        assert!(read_existing_regular_file(&dir.join("absent"), 4).is_err());
        assert!(read_regular_file(&dir, 4).is_err(), "a directory");
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            assert!(read_regular_file(Path::new("/dev/zero"), 4).is_err());
            let fifo = dir.join("fifo");
            assert!(std::process::Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success());
            assert!(read_regular_file(&fifo, 4).is_err());
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
