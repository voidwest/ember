from __future__ import annotations

from pathlib import Path

from .artifacts import BackendDescriptor, SarfArtifactManifest
from .ember_config import write_ember_config
from .io import write_json, write_jsonl
from .prompts import make_prompts
from .splits import lemma_heldout_split
from .toy import toy_records


DEFAULT_TEMPLATE = "حلل صرفيا الكلمة العربية: {word}"


def write_example_workflow(out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    records = toy_records()
    split_records, split_metadata = lemma_heldout_split(records)
    prompts = make_prompts(split_records, DEFAULT_TEMPLATE)

    dataset_path = out / "toy_morphology.jsonl"
    prompts_path = out / "prompts.jsonl"
    split_path = out / "split_metadata.json"
    config_path = out / "ember_native_logits.placeholder.toml"
    workflow_path = out / "workflow.json"
    artifact_manifest_path = out / "sarf_artifact_manifest.placeholder.json"

    write_jsonl(dataset_path, split_records)
    write_jsonl(prompts_path, prompts)
    write_json(split_path, split_metadata)
    write_ember_config(
        config_path,
        run_id="sarf-v01-toy-native-logits",
        model_path="/path/to/model.gguf",
        tokenizer_path="/path/to/tokenizer.json",
        prompts_path=str(prompts_path),
        output_dir=str(out / "runs"),
        backend="native",
        architecture="qwen3",
        write_logits=True,
    )
    artifact_manifest = SarfArtifactManifest(
        run_id="sarf-v01-toy-native-logits",
        backend=BackendDescriptor(name="native", adapter="sarf.backends.ember"),
        prompts_path=str(prompts_path),
        tokenization_path="<ember-run>/tokenization.jsonl",
        positions_path="<ember-run>/positions.jsonl",
        logits_path="<ember-run>/logits.npy",
        report_path="<ember-run>/report.json",
        not_research_output=True,
    )
    write_json(artifact_manifest_path, artifact_manifest.to_dict())
    manifest = {
        "kind": "sarf_v0_1_example_workflow",
        "toy": True,
        "not_research_output": True,
        "architecture": "Sarf organizes. Backends extract. Auditors compare.",
        "dataset": str(dataset_path),
        "prompts": str(prompts_path),
        "split_metadata": str(split_path),
        "ember_config": str(config_path),
        "sarf_artifact_manifest": str(artifact_manifest_path),
        "validate_command": "cargo run -- validate-run <run-dir>",
        "notes": [
            "Replace placeholder model/tokenizer paths before running Ember extraction.",
            "Use adapters to import Ember, file-based, or future backend outputs into Sarf artifacts.",
            "This workflow demonstrates scaffolding only; it is not Paper 1 reproduction.",
        ],
    }
    write_json(workflow_path, manifest)
    return {key: str(value) for key, value in manifest.items() if isinstance(value, str)}
