use anyhow::Context;
use std::fs;
use std::io::{BufWriter, Write};

pub fn write_npy_2d(path: &str, data: &[f32], shape: &[usize; 2]) -> anyhow::Result<()> {
    write_npy_shape(path, data, shape)
}

fn write_npy_shape(path: &str, data: &[f32], shape: &[usize]) -> anyhow::Result<()> {
    let file = fs::File::create(path)?;
    let mut w = BufWriter::new(file);
    write_npy_header(&mut w, shape)?;
    write_f32_slice(&mut w, data)?;
    w.flush()?;
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
    let total_prefix = 10 + header_bytes.len();
    let padding = (16 - (total_prefix % 16)) % 16;
    if padding > 0 {
        header_bytes.extend(std::iter::repeat_n(b' ', padding));
    }
    let header_len = header_bytes.len() as u16;

    w.write_all(b"\x93NUMPY")?;
    w.write_all(&[1u8, 0u8])?;
    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(&header_bytes)?;
    Ok(())
}

fn write_f32_slice(w: &mut impl Write, data: &[f32]) -> anyhow::Result<()> {
    let data_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    w.write_all(data_bytes)?;
    Ok(())
}

pub struct NpyStreamWriter {
    writer: BufWriter<fs::File>,
    expected_floats: usize,
    written_floats: usize,
}

impl NpyStreamWriter {
    pub fn create(path: &str, shape: &[usize]) -> anyhow::Result<Self> {
        let file = fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        write_npy_header(&mut writer, shape)?;
        Ok(Self {
            writer,
            expected_floats: shape.iter().product(),
            written_floats: 0,
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
        write_f32_slice(&mut self.writer, data)?;
        self.written_floats = next;
        Ok(())
    }

    pub fn finish(&mut self) -> anyhow::Result<()> {
        if self.written_floats != self.expected_floats {
            anyhow::bail!(
                "npy stream length mismatch: wrote {} floats, expected {}",
                self.written_floats,
                self.expected_floats
            );
        }
        self.writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
