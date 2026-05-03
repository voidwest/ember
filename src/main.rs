use ember::backend::Backend;
use ember::backend::CpuBackend;
use ember::loader::load_gguf;
use ember::model::Gpt2;

fn main() {
    let loader = load_gguf("gpt2.Q8_0.gguf").expect("failed to load");
    println!("loaded {} tensors", loader.tensors.len());

    let model = Gpt2::from_loader(loader).expect("failed to build model");
    println!("model built");

    let backend = CpuBackend;
    println!("wte shape: {:?}", backend.shape(&model.wte));

    let tokens = &[15496u32]; // "Hello" in gpt2
    let logits = model
        .forward(&backend, tokens)
        .expect("forward pass failed");

    println!("output shape: {:?}", backend.shape(&logits));
}
