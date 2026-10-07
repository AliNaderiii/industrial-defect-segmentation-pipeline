# Generated experiment artifacts

Git intentionally contains no metric dashboards, predictions, or headline scores. After a complete recorded experiment, run:

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt
python -m src.reporting
```

This creates `test_metrics.json` and `experiment_dashboard.png` from the current checkpoint, split manifest, and held-out test set. Do not commit a report without its checkpoint metadata, configuration, commit SHA, seed, hardware details, and dataset provenance.
