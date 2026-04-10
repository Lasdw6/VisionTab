"""
Multimodal dataset builder for Gemma 4 E4B vision+code completion training.

Takes the text-only FIM dataset from Phase 1, renders code context as synthetic
IDE screenshots, and produces image+text pairs for multimodal fine-tuning.

Each sample contains:
  - image: JPEG bytes of the IDE screenshot (context the developer sees)
  - text:  FIM-formatted code (what the model should complete)
  - language: source language

The screenshot shows the FULL code, while the text field contains the FIM-masked
version. This teaches the model to use visual context for predicting completions.

References:
    [VisCodex] Unified Multimodal Code Generation (ICLR 2026)
    [VinciCoder] Multimodal Code via Visual RL (arXiv:2511.00391)
    [Design2Code] Benchmarking Screenshot-to-Code (NAACL 2025)

Usage:
    python -m data_prep.multimodal_dataset --config configs/training_config.yaml
"""

import argparse
import io
import json
import os
import random
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict, Features, Image, Value, load_from_disk
from PIL import Image as PILImage
from tqdm import tqdm

from .fim_transform import FIMConfig, FIMTransformer
from .screenshot_renderer import RenderConfig, ScreenshotRenderer


_EXT_MAP = {
    "python": "py", "javascript": "js", "typescript": "ts",
    "rust": "rs", "go": "go", "java": "java",
}


def _make_filename(language: str, idx: int) -> str:
    ext = _EXT_MAP.get(language, "txt")
    prefixes = ["main", "utils", "helpers", "service", "handler", "app",
                "index", "lib", "core", "api", "models", "types", "config"]
    name = random.choice(prefixes)
    return f"{name}.{ext}"


def _extract_original_code(text: str) -> str | None:
    """
    Recover the original code from a FIM-formatted sample.
    FIM format is: <fim_prefix>PREFIX<fim_suffix>SUFFIX<fim_middle>MIDDLE
    Original code = PREFIX + MIDDLE + SUFFIX
    For PSM: prefix, suffix, middle order
    For SPM: suffix, prefix, middle order
    """
    fp, fs, fm = "<fim_prefix>", "<fim_suffix>", "<fim_middle>"

    if fm not in text:
        return text

    mid_idx = text.index(fm)
    before_mid = text[:mid_idx]
    middle = text[mid_idx + len(fm):]

    if fp in before_mid and fs in before_mid:
        fp_idx = before_mid.index(fp)
        fs_idx = before_mid.index(fs)

        if fp_idx < fs_idx:
            prefix = before_mid[fp_idx + len(fp):fs_idx]
            suffix = before_mid[fs_idx + len(fs):]
        else:
            suffix = before_mid[fs_idx + len(fs):fp_idx]
            prefix = before_mid[fp_idx + len(fp):]

        return prefix + middle + suffix

    return None


def build_multimodal_sample(
    text: str,
    language: str,
    renderer: ScreenshotRenderer,
    idx: int,
) -> dict | None:
    """
    Build a single multimodal sample from a FIM text sample.

    The screenshot shows the original (un-masked) code so the model
    learns to leverage visual context. The text field keeps FIM masking.
    """
    original_code = _extract_original_code(text)
    if not original_code or len(original_code.strip()) < 30:
        return None

    lines = original_code.strip().split("\n")
    if len(lines) < 3:
        return None

    filename = _make_filename(language, idx)
    try:
        img = renderer.render(original_code.strip(), language, filename)
        buf = io.BytesIO()
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=85)
        img_bytes = buf.getvalue()
    except Exception as e:
        return None

    return {
        "image": {"bytes": img_bytes, "path": None},
        "text": text,
        "language": language,
    }


class MultimodalPipeline:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.mm_config = self.config.get("multimodal", {})
        self.data_config = self.config.get("data", {})

        self.renderer = ScreenshotRenderer(RenderConfig(
            width=800,
            min_height=400,
            max_height=900,
            font_size=13,
            jpeg_quality=85,
        ))

        fim_cfg = self.config.get("fim", {})
        self.transformer = FIMTransformer(FIMConfig(
            prefix_token=fim_cfg.get("prefix_token", "<fim_prefix>"),
            suffix_token=fim_cfg.get("suffix_token", "<fim_suffix>"),
            middle_token=fim_cfg.get("middle_token", "<fim_middle>"),
            fim_rate=1.0,  # 100% — all multimodal samples should be FIM
            psm_ratio=fim_cfg.get("psm_ratio", 0.5),
            use_ast_aware=fim_cfg.get("use_ast_aware", True),
            min_mask_bytes=fim_cfg.get("min_mask_bytes", 10),
            max_mask_bytes=fim_cfg.get("max_mask_bytes", 500),
        ))

        self.max_mm_samples = self.mm_config.get("max_samples", 20000)
        self.output_dir = self.data_config.get(
            "multimodal_output_dir", "./data/multimodal_dataset"
        )
        self.fim_dataset_dir = self.data_config.get("output_dir", "./data/fim_dataset")

    def run(self):
        """Build multimodal dataset from the Phase 1 FIM dataset."""
        print("=" * 60)
        print("Phase 3: Multimodal Dataset Construction")
        print("=" * 60)

        print(f"\nLoading FIM dataset from {self.fim_dataset_dir}...")
        if not os.path.exists(self.fim_dataset_dir):
            raise FileNotFoundError(
                f"FIM dataset not found at {self.fim_dataset_dir}. "
                "Run Phase 1 first: python -m data_prep --config configs/training_config.yaml"
            )

        fim_ds = load_from_disk(self.fim_dataset_dir)
        train_ds = fim_ds["train"]
        print(f"Loaded {len(train_ds)} training samples")

        fim_samples = [
            (row["text"], row["language"])
            for row in train_ds
            if row.get("is_fim", False)
        ]
        print(f"FIM-masked samples available: {len(fim_samples)}")

        random.shuffle(fim_samples)
        target_count = min(self.max_mm_samples, len(fim_samples))
        print(f"Target multimodal samples: {target_count}")

        print("\nRendering IDE screenshots...")
        samples = []
        errors = 0
        for i, (text, language) in enumerate(tqdm(
            fim_samples[:target_count * 2],  # over-sample to compensate for skips
            desc="Rendering screenshots",
        )):
            if len(samples) >= target_count:
                break

            result = build_multimodal_sample(text, language, self.renderer, i)
            if result:
                samples.append(result)
            else:
                errors += 1

        print(f"\nGenerated {len(samples)} multimodal samples ({errors} skipped)")

        random.shuffle(samples)
        val_split = self.data_config.get("validation_split", 0.05)
        split_idx = int(len(samples) * (1 - val_split))
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        features = Features({
            "image": Image(),
            "text": Value("string"),
            "language": Value("string"),
        })

        print("\nBuilding HuggingFace Dataset...")
        train_dataset = Dataset.from_list(train_samples, features=features)
        val_dataset = Dataset.from_list(val_samples, features=features)
        ds_dict = DatasetDict({"train": train_dataset, "validation": val_dataset})

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        ds_dict.save_to_disk(str(output_path))

        meta = {
            "total_samples": len(samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "errors_skipped": errors,
            "source": "synthetic_ide_screenshots",
            "renderer": "pygments+pillow",
            "image_format": "JPEG",
            "languages": list(set(s["language"] for s in samples)),
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nMultimodal dataset saved to {output_path}")
        print(f"  Train: {len(train_samples)}")
        print(f"  Validation: {len(val_samples)}")

        lang_counts = {}
        for s in samples:
            lang_counts[s["language"]] = lang_counts.get(s["language"], 0) + 1
        print("\nSamples per language:")
        for lang, count in sorted(lang_counts.items()):
            print(f"  {lang}: {count}")

        print(f"\nZip this folder and upload to Kaggle as 'gemma4-multimodal-dataset'")
        return ds_dict


def main():
    parser = argparse.ArgumentParser(
        description="Build multimodal dataset for Gemma 4 E4B tab completion"
    )
    parser.add_argument(
        "--config", type=str, default="configs/training_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Override max number of multimodal samples to generate",
    )
    args = parser.parse_args()

    pipeline = MultimodalPipeline(args.config)
    if args.max_samples:
        pipeline.max_mm_samples = args.max_samples
    pipeline.run()


if __name__ == "__main__":
    main()
