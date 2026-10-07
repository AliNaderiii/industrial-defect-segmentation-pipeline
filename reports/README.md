# Generated and recorded experiment artifacts

`reports/` is ignored by Git because it is the local scratch location for outputs from the current checkpoint. A temporary dashboard or metric JSON must not be committed on its own.

The recorded, protocol-compliant DeepCrack CPU baseline is the exception because its complete safe-to-publish evidence bundle has been deliberately copied to:

- `assets/experiments/deepcrack_cpu_baseline_dashboard.png`;
- `experiments/runs/deepcrack-cpu-baseline/`;
- `docs/experiments/deepcrack-cpu-baseline.md`.

That bundle contains no raw DeepCrack data and no model weights. It includes hashes that identify the local selected checkpoint, configuration, and dashboard. For a new run, keep using this ignored directory and publish a new named evidence bundle only after its protocol, provenance, and held-out evaluation have been reviewed.
