use ember::loader::load_gguf;

fn main() {
    let loader = load_gguf("gpt2.gguf").expect("failed to load");

    println!("=== metadata ===");
    for key in loader.metadata.keys() {
        println!("{}", key);
    }

    println!("\n=== tensors ===");
    for name in loader.tensors.keys() {
        println!("{}", name);
    }
}
