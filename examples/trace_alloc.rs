use ember::backend::CpuBackend;
use ember::loader::load_gguf_with_k_strategy;
use ember::model::ForwardModel;
use ember::plan::ExecutionMode;
use ember::quant_k::KStrategy;
use ember::tokenizer::EmberTokenizer;
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let loader = load_gguf_with_k_strategy(&args[1], KStrategy::Auto, false)?;
    let model = ember::llama::Llama::from_loader_with_max_seq_len(loader, Some(2048))?;
    let tokenizer = EmberTokenizer::from_file("tokenizer.json")?;
    let backend = CpuBackend;
    model.set_execution_mode(ExecutionMode::Planned);
    let ids = tokenizer.encode("The capital of France is")?;
    let mut cache = model.create_cache(&backend, 2048);
    ForwardModel::forward_last_logits_with_cache(&model, &backend, &ids, &mut cache, 0)?;
    for step in 0..4 {
        ForwardModel::forward_last_logits_with_cache(
            &model,
            &backend,
            &[ids[0]],
            &mut cache,
            ids.len() + step,
        )
        .unwrap();
    }
    eprintln!("decode done");
    Ok(())
}
