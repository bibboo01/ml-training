"""Message language: English by default, Thai with MLT_LANG=th (run.ps1 / run.sh set it from .lang).

Messages are written with the Windows spelling (.\\run.ps1 setup, data\\raw\\<name>); local() turns
the helper into the short `run` command (installed by `run setup`) and paths into this OS's form.
"""

from __future__ import annotations

import os

RUN = "run"


def lang() -> str:
    return os.environ.get("MLT_LANG", "en").strip().lower()


def local(text: str) -> str:
    """'.\\run.ps1 setup' -> 'run setup'; on Linux also data\\raw\\x -> data/raw/x."""
    text = text.replace(".\\run.ps1", RUN)
    if os.name != "nt":
        for win, posix in (("data\\raw\\", "data/raw/"), ("cat\\  dog\\", "cat/  dog/")):
            text = text.replace(win, posix)
    return text


def tr(th: str, en: str) -> str:
    return local(th if lang() == "th" else en)
