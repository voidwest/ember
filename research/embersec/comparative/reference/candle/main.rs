// EmberSEC comparative harness for Candle's GGUF parser
// (candle_core::quantized::gguf_file). Parser-level only: Content::read
// parses header + metadata + tensor infos; no model construction, no
// tensor data loading (TensorInfo::read), so semantic/config cases are
// expected to parse. Exit 0 = parsed, 1 = structured reject, panic = 101.
use std::fs::File;
use std::io::BufReader;

fn main() {
    let path = std::env::args().nth(1).expect("fixture path");
    let file = File::open(&path).expect("open fixture");
    let mut reader = BufReader::new(file);
    match candle_core::quantized::gguf_file::Content::read(&mut reader) {
        Ok(content) => {
            eprintln!(
                "HARNESS: GGUF_OK tensors={} kv={}",
                content.tensor_infos.len(),
                content.metadata.len()
            );
        }
        Err(err) => {
            eprintln!("HARNESS: GGUF_REJECT: {err}");
            std::process::exit(1);
        }
    }
}
