from __future__ import annotations

from sarf.artifacts import BackendDescriptor, SarfArtifactManifest


def import_files(
    *,
    run_id: str,
    prompts_path: str,
    tokenization_path: str | None = None,
    positions_path: str | None = None,
    logits_path: str | None = None,
    hidden_states_path: str | None = None,
    report_path: str | None = None,
) -> SarfArtifactManifest:
    return SarfArtifactManifest(
        run_id=run_id,
        backend=BackendDescriptor(name="files", adapter="sarf.backends.files"),
        prompts_path=prompts_path,
        tokenization_path=tokenization_path,
        positions_path=positions_path,
        logits_path=logits_path,
        hidden_states_path=hidden_states_path,
        report_path=report_path,
        not_research_output=True,
    )
