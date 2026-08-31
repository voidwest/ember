use anyhow::Context;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Maximum `.npy` payload accepted by the path and in-memory readers.
///
/// This is a trust boundary, not a format limitation. Research callers that
/// need larger activation matrices should use a bounded/chunked reader rather
/// than handing an untrusted path to `read_npy_2d`.
pub const MAX_NPY_BYTES: usize = 256 * 1024 * 1024;

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub fn write_npy_2d(path: &str, data: &[f32], shape: &[usize; 2]) -> anyhow::Result<()> {
    write_npy_shape(path, data, shape)
}

fn write_npy_shape(path: &str, data: &[f32], shape: &[usize]) -> anyhow::Result<()> {
    let expected = shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .context("npy shape product overflow")?;
    if data.len() != expected {
        anyhow::bail!(
            "npy data length {} does not match shape {:?} ({expected} elements)",
            data.len(),
            shape
        );
    }
    let mut writer = NpyStreamWriter::create(path, shape)?;
    writer.write_f32s(data)?;
    writer.finish()?;
    Ok(())
}

fn write_npy_header(w: &mut impl Write, shape: &[usize]) -> anyhow::Result<()> {
    let shape_text = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let dims = shape
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        format!("({dims})")
    };
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
        shape_text
    );
    let mut header_bytes = header.into_bytes();
    // NPY v1 headers end in a newline and the complete preamble+header is
    // padded to a 64-byte boundary. NumPy accepts older 16-byte padding, but
    // emitting the current canonical layout improves interoperability.
    let unpadded_len = 10usize
        .checked_add(header_bytes.len())
        .and_then(|len| len.checked_add(1))
        .context("npy header length overflow")?;
    let padding = (64 - (unpadded_len % 64)) % 64;
    header_bytes.extend(std::iter::repeat_n(b' ', padding));
    header_bytes.push(b'\n');
    let header_len = u16::try_from(header_bytes.len()).context("npy v1 header exceeds u16")?;

    w.write_all(b"\x93NUMPY")?;
    w.write_all(&[1u8, 0u8])?;
    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(&header_bytes)?;
    Ok(())
}

fn write_f32_slice(w: &mut impl Write, data: &[f32]) -> anyhow::Result<()> {
    const FLOATS_PER_CHUNK: usize = 1024;
    let mut bytes = [0u8; FLOATS_PER_CHUNK * std::mem::size_of::<f32>()];
    for values in data.chunks(FLOATS_PER_CHUNK) {
        for (slot, value) in bytes
            .chunks_exact_mut(std::mem::size_of::<f32>())
            .zip(values)
        {
            slot.copy_from_slice(&value.to_le_bytes());
        }
        w.write_all(&bytes[..std::mem::size_of_val(values)])?;
    }
    Ok(())
}

/// Read an in-memory little-endian f32 `.npy` file.
///
/// This is the same strict parser used by [`read_npy_2d`], without a
/// filesystem dependency, so callers handling untrusted bytes can validate
/// them before deciding where (or whether) to persist them.
pub fn read_npy_2d_bytes(bytes: &[u8]) -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
    read_npy_2d_bytes_named(bytes, "<memory>")
}

/// Read a little-endian f32 `.npy` file written by [`write_npy_2d`].
///
/// Returns `(shape, values)` with values in row-major order. Rejects
/// non-f32 dtypes, fortran order, and truncated or oversized payloads.
pub fn read_npy_2d(path: &str) -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
    use std::io::Read;

    let path_metadata =
        fs::symlink_metadata(path).with_context(|| format!("failed to stat npy '{path}'"))?;
    anyhow::ensure!(
        path_metadata.file_type().is_file(),
        "'{path}' is not a regular file"
    );
    let mut file = fs::File::open(path).with_context(|| format!("failed to read npy '{path}'"))?;
    let initial_metadata = file
        .metadata()
        .with_context(|| format!("failed to stat npy '{path}'"))?;
    anyhow::ensure!(
        initial_metadata.file_type().is_file()
            && initial_metadata.len() == path_metadata.len()
            && initial_metadata.modified().ok() == path_metadata.modified().ok(),
        "npy file changed while opening '{path}'"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        anyhow::ensure!(
            initial_metadata.dev() == path_metadata.dev()
                && initial_metadata.ino() == path_metadata.ino(),
            "npy file changed while opening '{path}'"
        );
    }
    let length = initial_metadata.len();
    anyhow::ensure!(
        length <= MAX_NPY_BYTES as u64,
        "'{path}' is {length} bytes, exceeding the {MAX_NPY_BYTES} byte limit"
    );
    // Read through the already-open handle and one byte beyond the limit. A
    // replacement/growth race cannot make fs::read allocate without bound.
    let capacity = usize::try_from(length).context("NPY file length exceeds address space")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|error| anyhow::anyhow!("cannot reserve NPY buffer: {error}"))?;
    let mut chunk = [0u8; 64 * 1024];
    loop {
        let remaining = MAX_NPY_BYTES.saturating_sub(bytes.len()).saturating_add(1);
        if remaining == 0 {
            break;
        }
        let count = remaining.min(chunk.len());
        let read = file
            .read(&mut chunk[..count])
            .with_context(|| format!("failed to read npy '{path}'"))?;
        if read == 0 {
            break;
        }
        anyhow::ensure!(
            bytes.len() <= MAX_NPY_BYTES.saturating_sub(read),
            "'{path}' grew beyond the {MAX_NPY_BYTES} byte limit while reading"
        );
        bytes
            .try_reserve_exact(read)
            .map_err(|error| anyhow::anyhow!("cannot grow NPY buffer: {error}"))?;
        bytes.extend_from_slice(&chunk[..read]);
    }
    let final_metadata = file
        .metadata()
        .with_context(|| format!("failed to stat npy '{path}' after reading"))?;
    let final_path_metadata = fs::symlink_metadata(path)
        .with_context(|| format!("failed to stat npy '{path}' after reading"))?;
    anyhow::ensure!(
        final_metadata.len() == length
            && final_metadata.modified().ok() == initial_metadata.modified().ok()
            && final_path_metadata.file_type().is_file()
            && final_path_metadata.len() == initial_metadata.len()
            && final_path_metadata.modified().ok() == initial_metadata.modified().ok(),
        "npy file changed while reading '{path}'"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        anyhow::ensure!(
            final_path_metadata.dev() == initial_metadata.dev()
                && final_path_metadata.ino() == initial_metadata.ino(),
            "npy file changed while reading '{path}'"
        );
    }
    read_npy_2d_bytes_named(&bytes, path)
}

fn read_npy_2d_bytes_named(bytes: &[u8], source: &str) -> anyhow::Result<(Vec<usize>, Vec<f32>)> {
    anyhow::ensure!(
        bytes.len() <= MAX_NPY_BYTES,
        "'{source}' is {} bytes, exceeding the {MAX_NPY_BYTES} byte limit",
        bytes.len()
    );
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        anyhow::bail!("'{source}' is not a valid npy file (bad magic)");
    }
    if bytes[6] != 1 {
        anyhow::bail!("'{source}' uses unsupported npy major version {}", bytes[6]);
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header_start = 10;
    let header_end = header_start + header_len;
    if header_end > bytes.len() {
        anyhow::bail!("'{source}' has a truncated npy header");
    }
    let header = String::from_utf8_lossy(&bytes[header_start..header_end]);

    // Parse the Python-dict header: {'descr': '<f4', 'fortran_order': False, 'shape': (..), }
    let descr = parse_header_value(&header, "descr")
        .with_context(|| format!("'{source}' header has no descr"))?;
    let descr = descr.trim().trim_matches('\'');
    if descr != "<f4" {
        anyhow::bail!("'{source}' has unsupported dtype '{descr}' (expected '<f4')");
    }
    let fortran = parse_header_value(&header, "fortran_order")
        .with_context(|| format!("'{source}' header has no fortran_order"))?;
    if fortran.trim() != "False" {
        anyhow::bail!("'{source}' is fortran-ordered; expected row-major");
    }
    let shape_text =
        parse_shape_value(&header).with_context(|| format!("'{source}' header has no shape"))?;
    let shape_text = shape_text.trim();
    let shape = if shape_text == "()" {
        Vec::new()
    } else {
        let inner = shape_text
            .trim_start_matches('(')
            .trim_end_matches(')')
            .trim_end_matches(',');
        if inner.trim().is_empty() {
            Vec::new()
        } else {
            inner
                .split(',')
                .map(|dim| {
                    dim.trim()
                        .parse::<usize>()
                        .with_context(|| format!("'{source}' has an invalid shape '{shape_text}'"))
                })
                .collect::<anyhow::Result<Vec<_>>>()?
        }
    };
    if shape.len() != 2 {
        anyhow::bail!("'{source}' has rank {}, expected a 2D tensor", shape.len());
    }
    let element_count = shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .with_context(|| format!("'{source}' shape overflows: {shape:?}"))?;
    let payload = &bytes[header_end..];
    let expected = element_count
        .checked_mul(4)
        .with_context(|| format!("'{source}' payload size overflows"))?;
    if payload.len() != expected {
        anyhow::bail!(
            "'{source}' payload is {} bytes, expected {expected} for shape {shape:?}",
            payload.len()
        );
    }
    let values = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<f32>>();
    Ok((shape, values))
}

fn parse_header_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let marker = format!("'{key}':");
    let start = header.find(&marker)?;
    let rest = &header[start + marker.len()..];
    let end = rest.find(',').unwrap_or(rest.len());
    Some(&rest[..end])
}

/// Parse the shape value `(d0, d1, ...)` or `()`; the naive
/// first-comma cut in [`parse_header_value`] would split inside the parens.
fn parse_shape_value(header: &str) -> Option<&str> {
    let marker = "'shape':";
    let start = header.find(marker)? + marker.len();
    let rest = &header[start..];
    let open = rest.find('(')?;
    let close = rest[open..].find(')')? + open;
    Some(&rest[open..=close])
}

pub struct NpyStreamWriter {
    writer: Option<BufWriter<fs::File>>,
    final_path: PathBuf,
    temporary_path: PathBuf,
    expected_floats: usize,
    written_floats: usize,
    committed: bool,
}

impl NpyStreamWriter {
    pub fn create(path: &str, shape: &[usize]) -> anyhow::Result<Self> {
        let expected_floats = shape
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            .context("npy stream shape product overflow")?;
        let final_path = PathBuf::from(path);
        let (file, temporary_path) = create_temporary_output(&final_path)?;
        let mut writer = BufWriter::new(file);
        if let Err(error) = write_npy_header(&mut writer, shape) {
            drop(writer);
            let _ = fs::remove_file(&temporary_path);
            return Err(error);
        }
        Ok(Self {
            writer: Some(writer),
            final_path,
            temporary_path,
            expected_floats,
            written_floats: 0,
            committed: false,
        })
    }

    pub fn write_f32s(&mut self, data: &[f32]) -> anyhow::Result<()> {
        let next = self
            .written_floats
            .checked_add(data.len())
            .context("npy stream write length overflowed usize")?;
        if next > self.expected_floats {
            anyhow::bail!(
                "npy stream overflow: wrote {} floats, next chunk {} exceeds expected {}",
                self.written_floats,
                data.len(),
                self.expected_floats
            );
        }
        let writer = self
            .writer
            .as_mut()
            .context("cannot write to a finished npy stream")?;
        write_f32_slice(writer, data)?;
        self.written_floats = next;
        Ok(())
    }

    pub fn finish(&mut self) -> anyhow::Result<()> {
        self.finish_with_policy(true)
    }

    /// Publish only if the destination is still absent. This closes the race
    /// between a caller's preflight existence check and final publication.
    pub fn finish_no_replace(&mut self) -> anyhow::Result<()> {
        self.finish_with_policy(false)
    }

    fn finish_with_policy(&mut self, overwrite: bool) -> anyhow::Result<()> {
        if self.committed {
            return Ok(());
        }
        if self.written_floats != self.expected_floats {
            anyhow::bail!(
                "npy stream length mismatch: wrote {} floats, expected {}",
                self.written_floats,
                self.expected_floats
            );
        }
        let mut writer = self
            .writer
            .take()
            .context("npy stream has already been closed")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        if overwrite {
            fs::rename(&self.temporary_path, &self.final_path).with_context(|| {
                format!(
                    "failed to publish npy '{}' from temporary file '{}'",
                    self.final_path.display(),
                    self.temporary_path.display()
                )
            })?;
        } else {
            fs::hard_link(&self.temporary_path, &self.final_path).with_context(|| {
                format!(
                    "failed to publish new npy '{}' without replacement",
                    self.final_path.display()
                )
            })?;
            let _ = fs::remove_file(&self.temporary_path);
        }
        self.committed = true;
        Ok(())
    }
}

impl Drop for NpyStreamWriter {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.temporary_path);
        }
    }
}

fn create_temporary_output(final_path: &Path) -> anyhow::Result<(fs::File, PathBuf)> {
    let filename = final_path
        .file_name()
        .context("npy output path must include a filename")?
        .to_string_lossy();
    let parent = final_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    for _ in 0..128 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temporary_path = parent.join(format!(
            ".{filename}.ember-tmp-{}-{sequence}",
            std::process::id()
        ));
        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
        {
            Ok(file) => return Ok((file, temporary_path)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to create temporary npy next to '{}'",
                        final_path.display()
                    )
                });
            }
        }
    }
    anyhow::bail!(
        "could not allocate a unique temporary npy next to '{}'",
        final_path.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_round_trip_2d() {
        let dir = std::env::temp_dir().join(format!("ember_npy_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("roundtrip.npy");
        let path_str = path.to_str().unwrap();
        let data: Vec<f32> = (0..24).map(|i| i as f32 * 0.5).collect();
        write_npy_2d(path_str, &data, &[4, 6]).unwrap();
        let (shape, values) = read_npy_2d(path_str).unwrap();
        assert_eq!(shape, vec![4, 6]);
        assert_eq!(values, data);
        let bytes = std::fs::read(path_str).unwrap();
        let (bytes_shape, bytes_values) = read_npy_2d_bytes(&bytes).unwrap();
        assert_eq!(bytes_shape, shape);
        assert_eq!(bytes_values, values);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn npy_writer_rejects_shape_data_mismatch() {
        let path = std::env::temp_dir().join(format!(
            "ember_npy_test_shape_mismatch_{}.npy",
            std::process::id()
        ));
        let error = write_npy_2d(path.to_str().unwrap(), &[1.0], &[1, 2]).unwrap_err();
        assert!(error.to_string().contains("does not match shape"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn npy_header_is_newline_terminated_and_aligned() {
        let path =
            std::env::temp_dir().join(format!("ember_npy_test_header_{}.npy", std::process::id()));
        write_npy_2d(path.to_str().unwrap(), &[1.0, 2.0], &[1, 2]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
        assert_eq!(bytes[10 + header_len - 1], b'\n');
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn npy_reader_rejects_oversized_file_before_reading() {
        let path =
            std::env::temp_dir().join(format!("ember_npy_test_oversized_{}", std::process::id()));
        let file = fs::File::create(&path).unwrap();
        file.set_len((MAX_NPY_BYTES as u64) + 1).unwrap();
        let error = read_npy_2d(path.to_str().unwrap()).unwrap_err();
        assert!(error.to_string().contains("limit"));
        fs::remove_file(path).ok();
    }

    #[test]
    fn npy_reader_rejects_bad_dtype_and_truncation() {
        let dir = std::env::temp_dir().join(format!("ember_npy_test_bad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.npy");
        let path_str = path.to_str().unwrap();
        write_npy_2d(path_str, &[1.0, 2.0], &[1, 2]).unwrap();
        // corrupt the descr marker
        let mut bytes = std::fs::read(path_str).unwrap();
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let descr_pos = 10
            + bytes[10..10 + header_len]
                .windows(7)
                .position(|w| w == b"'descr'")
                .unwrap()
            + 8;
        bytes[descr_pos] = b'X';
        std::fs::write(path_str, &bytes).unwrap();
        assert!(read_npy_2d(path_str).is_err());
        // truncated payload
        std::fs::write(path_str, &bytes[..bytes.len() - 2]).unwrap();
        assert!(read_npy_2d(path_str).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn npy_stream_writer_writes_expected_shape_and_payload() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ember_npy_stream_{}_{}.npy",
            std::process::id(),
            unique
        ));
        let path_str = path.to_str().expect("temp path should be utf-8");

        let mut writer = NpyStreamWriter::create(path_str, &[2, 2, 2]).expect("create npy stream");
        writer
            .write_f32s(&[1.0, 2.0, 3.0, 4.0])
            .expect("write first row");
        writer
            .write_f32s(&[5.0, 6.0, 7.0, 8.0])
            .expect("write second row");
        writer.finish().expect("finish npy stream");

        let bytes = fs::read(&path).expect("read streamed npy file");
        assert!(bytes.starts_with(b"\x93NUMPY"));
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..10 + header_len]).expect("utf-8 header");
        assert!(header.contains("'shape': (2, 2, 2)"));

        let payload = &bytes[10 + header_len..];
        assert_eq!(payload.len(), 8 * 4);
        let values = payload
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn no_replace_stream_preserves_a_concurrent_destination() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ember_npy_no_replace_{}_{}.npy",
            std::process::id(),
            unique
        ));
        let mut writer =
            NpyStreamWriter::create(path.to_str().expect("temp path should be utf-8"), &[1, 1])
                .expect("create npy stream");
        writer.write_f32s(&[1.0]).expect("write value");
        fs::write(&path, b"concurrent").expect("create concurrent destination");
        assert!(writer.finish_no_replace().is_err());
        assert_eq!(fs::read(&path).unwrap(), b"concurrent");
        fs::remove_file(path).ok();
    }

    #[test]
    fn incomplete_stream_never_publishes_final_path() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ember_npy_incomplete_{}_{}.npy",
            std::process::id(),
            unique
        ));
        {
            let mut writer =
                NpyStreamWriter::create(path.to_str().expect("temp path should be utf-8"), &[2, 2])
                    .expect("create npy stream");
            writer.write_f32s(&[1.0, 2.0]).expect("partial write");
            assert!(!path.exists(), "partial output must remain staged");
            assert!(writer.finish().is_err());
        }
        assert!(
            !path.exists(),
            "dropping a failed stream must clean staging"
        );
    }
}
