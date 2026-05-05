use ember::backend::Backend;
use ember::backend::CpuBackend;
use ember::loader::load_gguf;
use ember::model::Gpt2;

fn main() -> anyhow::Result<()> {
    let loader = load_gguf("gpt2.Q8_0.gguf")?;
    println!("loaded {} tensors", loader.tensors.len());

    let model = Gpt2::from_loader(loader)?;
    println!("model built");

    let backend = CpuBackend;
    println!("wte shape: {:?}", backend.shape(&model.wte));

    let tokens = &[15496u32];
    let logits = model.forward(&backend, tokens)?;

    println!("output shape: {:?}", backend.shape(&logits));
    let logit_data = backend.data(&logits);
    let next_token = logit_data
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("logits should not be empty"))
        .map(|(i, _)| i)
        .expect("logits should not be empty");
    println!("predicted next token: {}", next_token);
    let tokenizer = ember::tokenizer::EmberTokenizer::from_file("tokenizer.json")?;
    let decoded = tokenizer.decode(&[next_token as u32])?;
    println!("predicted next word: {}", decoded);
    Ok(())
}
