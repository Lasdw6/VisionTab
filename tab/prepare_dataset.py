"""
Dataset preparation pipeline for Gemma 4 E4B tab completion training.

Downloads a subset of The Stack v2 from HuggingFace, applies AST-aware FIM
transformation, and saves as a HuggingFace Dataset ready for upload to Kaggle.

Usage:
    python -m tab.prepare_dataset --config configs/training_config.yaml

References:
    [AST-FIM] Structure-Aware Fill-in-the-Middle Pretraining for Code
    [Curriculum-FIM] Improving FIM Code Completions via Context & Curriculum
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from .fim_transform import FIMConfig, FIMTransformer


class DatasetPipeline:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        fim_cfg = self.config.get("fim", {})
        self.transformer = FIMTransformer(
            FIMConfig(
                prefix_token=fim_cfg.get("prefix_token", "<fim_prefix>"),
                suffix_token=fim_cfg.get("suffix_token", "<fim_suffix>"),
                middle_token=fim_cfg.get("middle_token", "<fim_middle>"),
                fim_rate=fim_cfg.get("fim_rate", 0.5),
                psm_ratio=fim_cfg.get("psm_ratio", 0.5),
                use_ast_aware=fim_cfg.get("use_ast_aware", True),
                min_mask_bytes=fim_cfg.get("min_mask_bytes", 10),
                max_mask_bytes=fim_cfg.get("max_mask_bytes", 500),
            )
        )

        self.data_config = self.config.get("data", {})
        self.languages = self.data_config.get("languages", ["python"])
        self.max_samples = self.data_config.get("max_samples_per_language", 50000)
        self.chunk_size = self.data_config.get("chunk_size_tokens", 2048)
        self.val_split = self.data_config.get("validation_split", 0.05)
        self.output_dir = self.data_config.get("output_dir", "./data/fim_dataset")

    # code_search_net supports: python, javascript, go, java, ruby, php
    # For languages it doesn't cover, we fall back to other open datasets.
    _CSN_LANGUAGES = {"python", "javascript", "go", "java", "ruby", "php"}

    _DATASET_SOURCES = [
        {
            "name": "code_search_net",
            "loader": lambda lang: load_dataset(
                "code_search_net", lang, split="train", streaming=True, trust_remote_code=True,
            ),
            "content_field": "whole_func_string",
            "languages": {"python", "javascript", "go", "java", "ruby", "php"},
        },
        {
            "name": "codeparrot/github-code",
            "loader": lambda lang: load_dataset(
                "codeparrot/github-code", streaming=True, split="train",
                languages=[lang], licenses=["mit", "apache-2.0"],
            ),
            "content_field": "code",
            "languages": None,  # supports all
        },
        {
            "name": "bigcode/starcoderdata",
            "loader": lambda lang: load_dataset(
                "bigcode/starcoderdata", data_dir=lang, split="train", streaming=True,
            ),
            "content_field": "content",
            "languages": None,
        },
    ]

    def download_language(self, language: str) -> list[str]:
        """Download code samples for a given language, trying multiple open sources."""
        print(f"\nDownloading {language}...")

        for source in self._DATASET_SOURCES:
            supported = source["languages"]
            if supported is not None and language not in supported:
                continue

            name = source["name"]
            content_field = source["content_field"]
            print(f"  Trying {name}...")

            try:
                ds = source["loader"](language)
            except Exception as e:
                print(f"  {name} failed: {e}")
                continue

            samples = []
            try:
                for i, item in enumerate(
                    tqdm(ds, desc=f"  {language} ({name})", total=self.max_samples)
                ):
                    if i >= self.max_samples:
                        break

                    content = item.get(content_field, "") or item.get("content", "")
                    if not content or len(content) < 50:
                        continue
                    if len(content) > 100_000:
                        continue

                    samples.append(content)
            except Exception as e:
                print(f"  Error during iteration: {e}")
                if samples:
                    print(f"  Keeping {len(samples)} samples collected before error.")

            if samples:
                print(f"  Collected {len(samples)} samples for {language} from {name}")
                return samples

            print(f"  No samples from {name}, trying next source...")

        print(f"  WARNING: Could not load any data for {language}")
        return []

    def chunk_code(self, code: str, approx_chars_per_chunk: int = 8192) -> list[str]:
        """
        Split long code files into chunks.
        Uses ~4 chars per token as a rough estimate for chunk_size_tokens.
        """
        approx_chars = self.chunk_size * 4

        if len(code) <= approx_chars:
            return [code]

        chunks = []
        lines = code.split("\n")
        current_chunk = []
        current_len = 0

        for line in lines:
            line_len = len(line) + 1
            if current_len + line_len > approx_chars and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = [line]
                current_len = line_len
            else:
                current_chunk.append(line)
                current_len += line_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks

    def process_language(self, language: str) -> list[dict]:
        """Download, chunk, and FIM-transform code for one language."""
        raw_samples = self.download_language(language)

        all_chunks = []
        for code in raw_samples:
            all_chunks.extend(self.chunk_code(code))

        print(f"  {language}: {len(raw_samples)} files -> {len(all_chunks)} chunks")

        fim_samples = []
        for chunk in tqdm(all_chunks, desc=f"  FIM transform ({language})"):
            transformed = self.transformer.transform(chunk, language)
            fim_samples.append({
                "text": transformed,
                "language": language,
                "is_fim": transformed != chunk,
            })

        return fim_samples

    def run(self):
        """Run the full pipeline: download -> chunk -> FIM -> save."""
        print("=" * 60)
        print("Gemma 4 E4B Tab Complete - Dataset Preparation")
        print("=" * 60)

        all_samples = []
        for language in self.languages:
            samples = self.process_language(language)
            all_samples.extend(samples)

        print(f"\nTotal samples: {len(all_samples)}")

        fim_count = sum(1 for s in all_samples if s["is_fim"])
        ltr_count = len(all_samples) - fim_count
        print(f"  FIM samples: {fim_count}")
        print(f"  Left-to-right samples: {ltr_count}")

        import random as _random
        _random.shuffle(all_samples)

        split_idx = int(len(all_samples) * (1 - self.val_split))
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]

        train_ds = Dataset.from_list(train_samples)
        val_ds = Dataset.from_list(val_samples)
        ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        ds_dict.save_to_disk(str(output_path))

        meta = {
            "total_samples": len(all_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "languages": self.languages,
            "fim_rate": self.transformer.config.fim_rate,
            "ast_aware": self.transformer.config.use_ast_aware,
            "fim_count": fim_count,
            "ltr_count": ltr_count,
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nDataset saved to {output_path}")
        print(f"  Train: {len(train_samples)}")
        print(f"  Validation: {len(val_samples)}")
        print(f"\nUpload this folder to Kaggle as a Dataset for training.")

        return ds_dict


def main():
    parser = argparse.ArgumentParser(
        description="Prepare FIM dataset for Gemma 4 E4B tab completion"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    pipeline = DatasetPipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
