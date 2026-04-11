# Tab (VisionTab)

A compact end-to-end layout for training a **multimodal tab-completion** model: **Gemma 4 E4B** as the base, **AST-aware Fill-in-the-Middle (FIM)** data, **QLoRA** fine-tuning, and optional **screenshot-conditioned** completion. Repository structure follows the same “staged pipeline + `scripts/` + `configs/`” pattern as [OpenComposer](https://github.com/KenWuqianghao/OpenComposer), adapted for the Tab model instead of Composer 2 / GLM-4.

## Pipeline overview

| Stage | Description | Entry point |
|-------|-------------|-------------|
| 0 | Data preparation — text FIM corpus + multimodal (synthetic IDE screenshots + FIM text) | `scripts/prepare_data.py` |
| 1 | Text-only FIM QLoRA fine-tuning | `scripts/train_text_fim.py` → `notebooks/text_fim_training.ipynb` |
| 2 | Multimodal fine-tuning (image + text) | `scripts/train_multimodal.py` → `notebooks/multimodal_training.ipynb` |
| 3 | Inference and smoke checks | `scripts/evaluate.py` → `run_multimodal_inference.py`; optional `notebooks/inference_smoke_test.ipynb` |

## Quick start

```bash
pip install -r requirements.txt

# Stage 0: Prepare data (FIM only, multimodal only, or both)
python scripts/prepare_data.py --stage all --config configs/training_config.yaml

# Stage 1 & 2: Open the printed notebook paths (Kaggle T4 x2 or local CUDA)
python scripts/train_text_fim.py
python scripts/train_multimodal.py

# Stage 3: Evaluate / run inference (same flags as run_multimodal_inference.py)
python scripts/evaluate.py --adapter path/to/adapter --image path/to.png --prompt "..."
```

Module equivalents for Stage 0:

```bash
python -m tab --config configs/training_config.yaml
python -m tab.multimodal_dataset --config configs/training_config.yaml
```

## Hardware notes

- **Data prep:** CPU-friendly (streaming datasets; optional GPU for faster iteration).
- **Training:** Documented around **2× NVIDIA T4** (e.g. Kaggle); requires a recent PyTorch build with supported GPU architectures (see `notebooks/text_fim_training.ipynb`).
- **Optional:** `configs/deepspeed_zero2.json` is included if you later move training from notebooks to DeepSpeed multi-GPU scripts.

## Architecture (conceptual)

```
Gemma 4 E4B (base)
    |
    v
Stage 0 — AST-aware FIM dataset (+ optional multimodal pairs)
    |
    v
Stage 1 — Text FIM QLoRA (tab completion in the middle)
    |
    v
Stage 2 — Multimodal LoRA (IDE screenshot + FIM text)
    |
    v
Stage 3 — Inference / smoke evaluation
```

## Layout

| Path | Role |
|------|------|
| `tab/` | Core library: FIM transforms, dataset builders, screenshot rendering |
| `scripts/` | Thin CLI entry points for each pipeline stage |
| `configs/` | `training_config.yaml` + optional DeepSpeed JSON |
| `notebooks/` | Kaggle-oriented training and smoke-test notebooks |
| `evaluation/` | Placeholder package for future eval harnesses |
| `run_multimodal_inference.py` | Full multimodal inference CLI (HTTP server or one-shot) |

### Model details

- **Base model:** `google/gemma-4-E4B` (see `configs/training_config.yaml`).
- **Techniques:** LoRA / QLoRA, FIM, multimodal image+text pairs.
- **References:** AST-FIM, curriculum FIM, and multimodal code papers cited in the notebooks and module docstrings.
