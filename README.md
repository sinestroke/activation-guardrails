# Activation Guardrails

This repository contains the reusable training and evaluation pipeline for activation probes and LoRA text classifiers used in two binary safety-detection tasks:

- high-stakes interaction detection;
- scheduled chemical-harm intent detection.

It intentionally excludes dissertation results, learned checkpoints, dataset-generation machinery, ablations, transfer sweeps, SAE experiments, cluster job files, and analysis-specific plotting scripts.

## Included models and methods

The probe pipeline trains Mean, Softmax, Attention, RMAttn, SWiM, and SC-TopK probes over residual, MLP, and attention activations. The classifier pipeline fine-tunes `meta-llama/Llama-3.2-1B-Instruct` with LoRA. The provided probe configurations use `google/gemma-3-12b-it`.

## Environment

The recorded environment targets Python 3.12, PyTorch 2.11, CUDA 12.8, and a Linux host with an NVIDIA GPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e '.[dev]'
```

Gemma 3 and Llama 3.2 may require accepted model terms and Hugging Face authentication:

```bash
export HF_TOKEN=YOUR_TOKEN
```

## Data

Download both pinned datasets into the paths expected by the YAML files:

```bash
python scripts/download_datasets.py --dataset all
```

The chemical dataset is hosted at <https://huggingface.co/datasets/Qiyuan-Sine-F/scheduled-chemical-harm-detection>. Its local mirror preserves the Hub layout:

```text
data/scheduled-chemical-harm-detection/
├── train/
├── dev/
└── evals/
    ├── bengali/
    ├── hindi/
    └── everyday_chemical_generalization/
```

The high-stakes dataset is hosted at <https://huggingface.co/datasets/Arrrlex/models-under-pressure> and downloads to `data/high-stakes/`. Exact repository revisions and required files are recorded in `data/manifest.json`.

## Probe training

Run a one-step smoke job before a full training run:

```bash
python -m src.train \
  --config configs/probes/high-stakes-gemma3-12b-vanilla.yaml \
  --smoke --no-wandb

python -m src.train \
  --config configs/probes/chemical-harm-gemma3-12b-vanilla.yaml \
  --smoke --no-wandb
```

Remove `--smoke` for the configured five-seed runs.

## Classifier training

```bash
python -m src.classifier_train \
  --config configs/classifiers/high-stakes-llama3.2-1b-vanilla.yaml \
  --no-wandb

python -m src.classifier_train \
  --config configs/classifiers/chemical-harm-llama3.2-1b-instruct.yaml \
  --no-wandb
```

For a small classifier smoke run, append `--n-seeds 1 --max-train-examples 8 --max-dev-examples 8`.

## Evaluation

Each evaluation configuration scores both the Gemma probes and the Llama classifier by default:

```bash
python -m src.eval \
  --config configs/evaluation/high-stakes.yaml \
  --classifier-config configs/classifiers/high-stakes-llama3.2-1b-vanilla.yaml

python -m src.eval \
  --config configs/evaluation/chemical-harm.yaml \
  --classifier-config configs/classifiers/chemical-harm-llama3.2-1b-instruct.yaml
```

The chemical evaluation includes internal, chemical, persona, persuasion, Bengali, Hindi, everyday-chemistry, HarmBench, Chemistry-GPQA, and XSTest splits. Use `--skip-classifier` or `--skip-probes` to score only one family.

## Tests

The unit tests do not download models or datasets:

```bash
python -m pytest
```

After downloading data, validate the release configs and file layout with:

```bash
python scripts/validate_release.py
```

Generated outputs belong under `results/` and `checkpoints/`; both directories are ignored by Git.
