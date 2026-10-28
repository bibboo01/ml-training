"""Message language: English by default, Thai with MLT_LANG=th (run.ps1 sets it from .lang)."""

from __future__ import annotations

import os


def lang() -> str:
    return os.environ.get("MLT_LANG", "en").strip().lower()


def tr(th: str, en: str) -> str:
    return th if lang() == "th" else en
