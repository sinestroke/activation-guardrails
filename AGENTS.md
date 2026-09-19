# AI Agent Onboarding

This repository is the collaborator-facing activation-guardrail codebase. It contains the reusable vanilla probe and classifier pipeline, not the full dissertation experiment archive.

## Scope

Work within these supported tasks:

- high-stakes interaction detection;
- scheduled chemical-harm intent detection;
- training and evaluation of the six probes in `src/probes.py`;
- LoRA classifier training with `src/classifier_train.py`;
- unified probe/classifier evaluation with `src/eval.py`.

Do not add dissertation results, checkpoints, SAE experiments, ablations, cross-model transfer sweeps, cluster-specific launch files, or dataset-generation intermediates unless a maintainer explicitly expands the repository scope.

## Start here

1. Read `README.md`.
2. Install `requirements.txt` and the `dev` extra.
3. Run `python scripts/download_datasets.py --dataset all`.
4. Run `python -m pytest`.
5. Run `python scripts/validate_release.py` after data is present.

## Repository map

- `src/train.py`: activation-probe training CLI.
- `src/classifier_train.py`: LoRA classifier training CLI.
- `src/eval.py`: unified evaluation CLI.
- `src/probes.py`: Mean, Softmax, Attention, RMAttn, SWiM, and SC-TopK implementations.
- `src/models.py` and `src/extraction.py`: model adapters and activation extraction.
- `src/data.py` and `src/classifier_data.py`: labelled-text loading and classifier feature construction.
- `configs/probes/`: shared probe defaults and task-specific training configs.
- `configs/classifiers/`: shared LoRA defaults and task-specific Llama 3.2 1B configs.
- `configs/evaluation/`: task evaluation protocols.
- `constitutions/`: exact classifier prompts retained from the research repository.
- `data/manifest.json`: pinned Hugging Face repositories and required paths.

## Configuration rules

- Keep model identity, output directories, checkpoint directories, and dataset paths explicit in task configs.
- Keep `base_config` paths relative to the YAML file that declares them.
- Do not introduce machine-specific absolute paths.
- New evaluation splits must exist in the inherited `data.files` mapping.
- Keep generated artifacts under `results/` or `checkpoints/`.
- Preserve checkpoint provenance checks unless a documented migration requires otherwise.

## Data rules

Datasets are downloaded from Hugging Face and are not committed to this repository. Preserve the Hub directory layout because the YAML files refer to it directly. Update `data/manifest.json`, the download script, configs, and release validation together when a dataset revision changes.

The chemical dataset contains safety-evaluation material. Do not print prompt bodies in tests, logs, issues, or pull-request descriptions. Aggregate counts, hashes, schemas, and metric summaries are appropriate.

## Change validation

For Python or configuration changes:

```bash
python -m pytest
python scripts/validate_release.py  # requires downloaded data
python -m compileall -q src scripts tests
```

Use the smoke commands in `README.md` when a change affects model loading, activation extraction, training, checkpoint layout, or evaluation. GPU smoke runs are not expected in CPU-only CI.

Before committing, confirm that `git status` contains no `results/`, `checkpoints/`, downloaded data, caches, tokens, or local paths.

