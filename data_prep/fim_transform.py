"""
AST-aware Fill-in-the-Middle (FIM) transformation engine.

Implements the approach from [AST-FIM] (arXiv:2506.00204): instead of
randomly splitting code at character boundaries, we parse with Tree-sitter
and mask complete syntactic structures (function bodies, if-blocks, etc.).
This aligns training with real editing patterns and yields +5 pts over
random FIM on standard benchmarks.

Also supports:
- PSM/SPM ratio mixing per the StarCoder recipe
- Random fallback when AST parsing fails or yields no candidates
- Configurable mask size bounds
"""

import importlib
import random
from dataclasses import dataclass, field

from tree_sitter import Language, Parser

from .languages import LANGUAGE_CONFIGS


@dataclass
class FIMConfig:
    prefix_token: str = "<fim_prefix>"
    suffix_token: str = "<fim_suffix>"
    middle_token: str = "<fim_middle>"
    fim_rate: float = 0.5
    psm_ratio: float = 0.5
    use_ast_aware: bool = True
    min_mask_bytes: int = 10
    max_mask_bytes: int = 500
    fallback_to_random: bool = True


class FIMTransformer:
    """Transforms code into FIM training samples using AST-aware masking."""

    def __init__(self, config: FIMConfig | None = None):
        self.config = config or FIMConfig()
        self._parsers: dict[str, Parser] = {}
        self._maskable_types: dict[str, set[str]] = {}

    def _get_parser(self, language: str) -> Parser | None:
        if language in self._parsers:
            return self._parsers[language]

        lang_config = LANGUAGE_CONFIGS.get(language)
        if not lang_config:
            return None

        try:
            mod = importlib.import_module(lang_config["module"])
            lang_attr = lang_config.get("language_attr", "language")
            lang_fn = getattr(mod, lang_attr)
            ts_language = Language(lang_fn())
            parser = Parser(ts_language)
            self._parsers[language] = parser
            self._maskable_types[language] = set(
                lang_config["maskable_node_types"]
            )
            return parser
        except (ImportError, AttributeError) as e:
            print(f"Warning: could not load tree-sitter for {language}: {e}")
            return None

    def _get_maskable_nodes(self, root_node, language: str) -> list:
        """Walk AST and collect nodes suitable for masking [AST-FIM]."""
        maskable_types = self._maskable_types.get(language, set())
        min_b = self.config.min_mask_bytes
        max_b = self.config.max_mask_bytes
        candidates = []

        stack = [root_node]
        while stack:
            node = stack.pop()
            size = node.end_byte - node.start_byte
            if (
                min_b <= size <= max_b
                and node.type in maskable_types
            ):
                candidates.append(node)
            for child in node.children:
                stack.append(child)

        return candidates

    def _random_split(self, code: str) -> tuple[str, str, str]:
        """Fallback: random character-boundary split."""
        if len(code) < 10:
            return code, "", ""

        start = random.randint(1, max(1, len(code) - 2))
        end = random.randint(start, len(code))
        return code[:start], code[start:end], code[end:]

    def _format_fim(self, prefix: str, middle: str, suffix: str) -> str:
        """Format as PSM or SPM based on configured ratio."""
        cfg = self.config
        if random.random() < cfg.psm_ratio:
            return (
                f"{cfg.prefix_token}{prefix}"
                f"{cfg.suffix_token}{suffix}"
                f"{cfg.middle_token}{middle}"
            )
        else:
            return (
                f"{cfg.suffix_token}{suffix}"
                f"{cfg.prefix_token}{prefix}"
                f"{cfg.middle_token}{middle}"
            )

    def transform(self, code: str, language: str) -> str:
        """
        Transform a code sample into a FIM training sample.

        With probability (1 - fim_rate), returns the original code unchanged
        for standard left-to-right language modeling.
        """
        if random.random() > self.config.fim_rate:
            return code

        if not self.config.use_ast_aware:
            prefix, middle, suffix = self._random_split(code)
            return self._format_fim(prefix, middle, suffix)

        parser = self._get_parser(language)
        if parser is None:
            prefix, middle, suffix = self._random_split(code)
            return self._format_fim(prefix, middle, suffix)

        tree = parser.parse(bytes(code, "utf-8"))
        candidates = self._get_maskable_nodes(tree.root_node, language)

        if not candidates:
            if self.config.fallback_to_random:
                prefix, middle, suffix = self._random_split(code)
                return self._format_fim(prefix, middle, suffix)
            return code

        node = random.choice(candidates)
        prefix = code[: node.start_byte]
        middle = code[node.start_byte : node.end_byte]
        suffix = code[node.end_byte :]

        return self._format_fim(prefix, middle, suffix)

    def transform_batch(
        self, codes: list[str], language: str
    ) -> list[str]:
        return [self.transform(code, language) for code in codes]
