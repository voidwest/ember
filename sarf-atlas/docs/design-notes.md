# Design Notes

Sarf Atlas exists to keep the Arabic morphology research workspace separate
from Ember's engineering responsibilities.

Ember should stay an instrumentation and tooling layer: extraction code,
dataset pipelines, probe runners, reports, validation hooks, and artifact
generation belong there. Sarf Atlas should hold research plans, paper-specific
configuration, experiment manifests, meeting notes, and paper-series planning.

This separation reduces the risk of mixing engine refactors with paper
artifacts. Research claims, figures, tables, and manuscript planning can remain
organized without encouraging incidental backend changes.

The prototype is intentionally small and isolated under `sarf-atlas/` so a
future split into a separate repository should be easy.

## Smoke-Test Boundary

The intended smoke-test flow is:

```text
sarf-atlas config/manifest
  -> Ember CLI
  -> Ember backend abstraction
  -> llama.cpp external backend
  -> Ember run artifacts
  -> Ember artifact-contract validation or validate-backends
  -> gguf-parity-tools optional validation
```

Sarf Atlas should not duplicate extraction logic. It should prepare research
inputs and then call Ember. If a llama.cpp-backed path is used, it should enter
through Ember's backend abstraction or CLI so the same artifact contract is
preserved.
