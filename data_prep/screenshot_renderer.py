"""
Synthetic IDE screenshot renderer for multimodal code completion training.

Renders code snippets as dark-themed IDE-like images using Pygments for syntax
highlighting and Pillow for image generation. Produces realistic editor visuals
with line numbers, cursor indicators, and a VS Code-inspired color scheme.

References:
    [VisCodex] Unified Multimodal Code Generation (ICLR 2026)
    [Design2Code] Benchmarking Screenshot-to-Code (NAACL 2025)
"""

import io
import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import get_lexer_by_name
from pygments.styles import get_style_by_name

_LANGUAGE_LEXER_MAP = {
    "python": "python3",
    "javascript": "javascript",
    "typescript": "typescript",
    "rust": "rust",
    "go": "go",
    "java": "java",
    "ruby": "ruby",
    "php": "php",
}

# VS Code dark+ inspired palette
_BG_COLOR = "#1e1e1e"
_GUTTER_COLOR = "#252526"
_GUTTER_TEXT = "#858585"
_CURSOR_LINE_BG = "#2a2d2e"
_TAB_BG = "#2d2d2d"
_TAB_ACTIVE_BG = "#1e1e1e"
_TAB_TEXT = "#cccccc"
_STATUSBAR_BG = "#007acc"
_STATUSBAR_TEXT = "#ffffff"


@dataclass
class RenderConfig:
    width: int = 800
    min_height: int = 400
    max_height: int = 900
    font_size: int = 13
    line_height: int = 20
    gutter_width: int = 50
    padding_left: int = 12
    padding_top: int = 8
    tab_bar_height: int = 35
    status_bar_height: int = 25
    style: str = "monokai"
    show_cursor_line: bool = True
    show_tab_bar: bool = True
    show_status_bar: bool = True
    jpeg_quality: int = 85


class ScreenshotRenderer:
    """Renders code as synthetic IDE screenshots."""

    def __init__(self, config: RenderConfig | None = None):
        self.config = config or RenderConfig()
        self._font = None
        self._font_bold = None

    def _get_font(self, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Get a monospace font, falling back to the default bitmap font."""
        if bold and self._font_bold:
            return self._font_bold
        if not bold and self._font:
            return self._font

        candidates = [
            "Consolas", "Cascadia Code", "Cascadia Mono",
            "Courier New", "DejaVu Sans Mono", "Liberation Mono",
            "Menlo", "Monaco", "monospace",
        ]
        for name in candidates:
            try:
                font = ImageFont.truetype(name, self.config.font_size)
                if bold:
                    self._font_bold = font
                else:
                    self._font = font
                return font
            except (OSError, IOError):
                continue

        font = ImageFont.load_default()
        self._font = font
        self._font_bold = font
        return font

    def _render_via_pygments(self, code: str, language: str) -> Image.Image:
        """Use Pygments' ImageFormatter for accurate syntax-highlighted rendering."""
        lexer_name = _LANGUAGE_LEXER_MAP.get(language, language)
        try:
            lexer = get_lexer_by_name(lexer_name)
        except Exception:
            lexer = get_lexer_by_name("text")

        font_name = "Consolas"
        formatter = ImageFormatter(
            style=self.config.style,
            font_name=font_name,
            font_size=self.config.font_size,
            line_numbers=True,
            line_number_fg="#858585",
            line_number_bg="#252526",
            image_pad=10,
            line_pad=3,
        )

        img_bytes = highlight(code, lexer, formatter)
        return Image.open(io.BytesIO(img_bytes))

    def render(
        self,
        code: str,
        language: str = "python",
        filename: str | None = None,
        cursor_line: int | None = None,
    ) -> Image.Image:
        """
        Render code as an IDE-like screenshot.

        Returns a PIL Image (RGB) suitable for saving as JPEG/PNG.
        """
        cfg = self.config

        code_img = self._render_via_pygments(code, language)

        extra_h = 0
        if cfg.show_tab_bar:
            extra_h += cfg.tab_bar_height
        if cfg.show_status_bar:
            extra_h += cfg.status_bar_height

        target_w = max(cfg.width, code_img.width + 20)
        target_h = code_img.height + extra_h + 10
        target_h = max(cfg.min_height, min(cfg.max_height, target_h))

        canvas = Image.new("RGB", (target_w, target_h), _BG_COLOR)
        draw = ImageDraw.Draw(canvas)
        font = self._get_font()

        y_offset = 0

        if cfg.show_tab_bar:
            draw.rectangle([0, 0, target_w, cfg.tab_bar_height], fill=_TAB_BG)
            tab_text = filename or f"main.{_ext_for_lang(language)}"
            tab_w = min(180, len(tab_text) * 8 + 30)
            draw.rectangle([0, 0, tab_w, cfg.tab_bar_height], fill=_TAB_ACTIVE_BG)
            draw.line([0, cfg.tab_bar_height, target_w, cfg.tab_bar_height], fill="#333333")
            try:
                draw.text((12, 9), tab_text, fill=_TAB_TEXT, font=font)
            except Exception:
                draw.text((12, 9), tab_text, fill=_TAB_TEXT)
            y_offset += cfg.tab_bar_height

        paste_y = y_offset + 5
        paste_x = 5
        visible_h = target_h - y_offset - (cfg.status_bar_height if cfg.show_status_bar else 0) - 10
        crop_box = (0, 0, min(code_img.width, target_w - 10), min(code_img.height, visible_h))
        cropped = code_img.crop(crop_box)
        canvas.paste(cropped, (paste_x, paste_y))

        if cfg.show_status_bar:
            sb_y = target_h - cfg.status_bar_height
            draw.rectangle([0, sb_y, target_w, target_h], fill=_STATUSBAR_BG)
            lines = code.count("\n") + 1
            col = random.randint(1, 80)
            cur_line = cursor_line or random.randint(1, lines)
            status_left = f"  Ln {cur_line}, Col {col}"
            status_right = f"{language.capitalize()}   UTF-8   LF  "
            try:
                draw.text((8, sb_y + 5), status_left, fill=_STATUSBAR_TEXT, font=font)
                rw = len(status_right) * 7
                draw.text((target_w - rw - 8, sb_y + 5), status_right, fill=_STATUSBAR_TEXT, font=font)
            except Exception:
                draw.text((8, sb_y + 5), status_left, fill=_STATUSBAR_TEXT)

        return canvas

    def render_to_bytes(
        self,
        code: str,
        language: str = "python",
        filename: str | None = None,
        cursor_line: int | None = None,
        fmt: str = "JPEG",
    ) -> bytes:
        """Render and return raw image bytes."""
        img = self.render(code, language, filename, cursor_line)
        buf = io.BytesIO()
        if fmt.upper() == "JPEG":
            img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=self.config.jpeg_quality)
        else:
            img.save(buf, format=fmt)
        return buf.getvalue()

    def render_to_file(
        self,
        code: str,
        output_path: str | Path,
        language: str = "python",
        filename: str | None = None,
        cursor_line: int | None = None,
    ) -> Path:
        """Render and save to a file."""
        output_path = Path(output_path)
        img = self.render(code, language, filename, cursor_line)
        fmt = "JPEG" if output_path.suffix.lower() in (".jpg", ".jpeg") else "PNG"
        if fmt == "JPEG":
            img = img.convert("RGB")
        img.save(output_path, format=fmt, quality=self.config.jpeg_quality)
        return output_path


def _ext_for_lang(language: str) -> str:
    exts = {
        "python": "py", "javascript": "js", "typescript": "ts",
        "rust": "rs", "go": "go", "java": "java",
        "ruby": "rb", "php": "php",
    }
    return exts.get(language, "txt")
