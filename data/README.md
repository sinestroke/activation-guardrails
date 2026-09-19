# Data

The repository does not commit dataset payloads. Run:

```bash
python scripts/download_datasets.py --dataset all
```

This creates:

- `data/high-stakes/` from `Arrrlex/models-under-pressure`;
- `data/scheduled-chemical-harm-detection/` from `Qiyuan-Sine-F/scheduled-chemical-harm-detection`.

Both downloads are pinned to the revisions in `manifest.json`. The chemical repository uses automatic gating, so authenticate with `HF_TOKEN` and accept its access terms on Hugging Face before downloading.

Do not rename downloaded subdirectories without updating all task configurations. Run `python scripts/validate_release.py` to check the expected files.

