# Visual evidence policy

This repository has three deliberately separate visual layers.

1. `assets/pipeline-protocol.svg` documents the current code and experiment protocol. It does not claim performance, so it is safe to show before a trained checkpoint exists.
2. `assets/legacy-v1/` holds selected genuine visuals retained from the pre-v2 repository. These are clearly marked archival qualitative context. Their numerical annotations must not be represented as v2 results.
3. `assets/experiments/deepcrack_cpu_baseline_dashboard.png` is a reportable v2 visual. It was generated from the recorded training history and the one-time official held-out evaluation documented in [`docs/experiments/deepcrack-cpu-baseline.md`](experiments/deepcrack-cpu-baseline.md). Its headline crack IoU is 0.665591; the matching configuration, split manifest, metric JSON, run metadata, and SHA-256 artifact manifest are versioned alongside it.

This separation preserves visual storytelling without implying that a legacy graphic validates the current protocol. New score-bearing visuals may be committed only with an equivalently complete, real experiment evidence bundle; raw data and checkpoint weights remain excluded.
