"""Check this computer: Python, GPU, installed parts, free disk.

python -m ml_training.doctor          (menu: More tools > Check this computer)
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
import sys

from ml_training.i18n import tr
from ml_training.utils import PROJECT_ROOT

PARTS = [
    (
        "ultralytics",
        "YOLO (ตรวจจับ / แยกวัตถุ / pose / จัดหมวดภาพ)",
        "YOLO (detect / segment / pose / classify)",
        "setup",
    ),
    (
        "transformers",
        "ข้อความ และเอกสาร (HuggingFace)",
        "text and documents (HuggingFace)",
        "setup",
    ),
    ("pythainlp", "ตัดคำภาษาไทย", "Thai word segmentation", "setup"),
    (
        "timm",
        "จัดหมวดภาพ ResNet / EfficientNet / ConvNeXt / ViT",
        "image models ResNet / EfficientNet / ConvNeXt / ViT",
        "setup",
    ),
    ("lightgbm", "LightGBM สำหรับข้อมูลตาราง", "LightGBM for tables", "setup"),
    ("xgboost", "XGBoost สำหรับข้อมูลตาราง", "XGBoost for tables", "setup"),
    (
        "easyocr",
        "OCR อ่านเอกสารสแกน / รูปเอกสาร (ไม่บังคับ)",
        "OCR for scanned documents (optional)",
        "setup-ocr",
    ),
]


def linux_checks(ok: str, no: str) -> None:
    """Ubuntu server: NVIDIA driver and the system libraries OpenCV / YOLO load."""
    import ctypes.util

    if shutil.which("nvidia-smi"):
        q = "--query-gpu=name,driver_version,memory.total"
        try:
            out = subprocess.run(
                ["nvidia-smi", q, "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out = ""
        for line in out.splitlines():
            print(f"{ok}nvidia-smi: {line.strip()}")
        if not out:
            print(
                no
                + tr(
                    "nvidia-smi ใช้งานไม่ได้ – ตรวจไดรเวอร์ NVIDIA",
                    "nvidia-smi failed – check the NVIDIA driver",
                )
            )
    else:
        print(
            no
            + tr(
                "ไม่พบ nvidia-smi – ถ้ามีการ์ดจอให้ติดตั้งไดรเวอร์: sudo ubuntu-drivers install",
                "nvidia-smi not found – if there is a GPU, install the driver: "
                "sudo ubuntu-drivers install",
            )
        )
    if ctypes.util.find_library("GL") and ctypes.util.find_library("glib-2.0"):
        print(
            ok
            + tr(
                "ไลบรารีระบบสำหรับ OpenCV (libGL, glib)", "system libraries for OpenCV (libGL, glib)"
            )
        )
    else:
        print(
            no
            + tr(
                "ยังไม่มี libGL / glib ที่ YOLO ใช้ – พิมพ์ run setup-system",
                "libGL / glib that YOLO needs are missing – type run setup-system",
            )
        )


def main() -> int:
    ok, no = "  [OK] ", "  [--] "
    print(tr("ตรวจเครื่อง", "Computer check"))
    print(f"{ok}Python {platform.python_version()} - {platform.system()} {platform.release()}")
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            vram = p.total_memory / 2**30
            print(f"{ok}GPU {p.name} ({vram:.1f} GB) – PyTorch {torch.__version__}")
            if vram < 6:
                print(
                    tr(
                        "       VRAM น้อย: เลือกโมเดลเล็ก "
                        "(YOLO n/s, efficientnet, seg-b0, text small)",
                        "       Small VRAM: pick small models "
                        "(YOLO n/s, efficientnet, seg-b0, text small)",
                    )
                )
        else:
            print(
                no
                + tr(
                    "ไม่พบการ์ดจอ NVIDIA ที่ใช้ได้ – ใช้ CPU (ช้ากว่า 10-50 เท่า)",
                    "No usable NVIDIA GPU – training runs on the CPU (10-50x slower)",
                )
            )
            if "+cpu" in torch.__version__ or not torch.version.cuda:
                print(
                    tr(
                        "       ถ้ามีการ์ดจอ NVIDIA ให้พิมพ์ .\\run.ps1 setup",
                        "       If this computer has an NVIDIA GPU, type .\\run.ps1 setup",
                    )
                )
    except Exception:  # noqa: BLE001
        print(
            no
            + tr(
                "ยังไม่มี PyTorch – พิมพ์ .\\run.ps1 setup",
                "PyTorch is missing – type .\\run.ps1 setup",
            )
        )
    if sys.platform.startswith("linux"):
        linux_checks(ok, no)
    for mod, th, en, cmd in PARTS:
        have = importlib.util.find_spec(mod) is not None
        how = tr(f" – ติดตั้งด้วย .\\run.ps1 {cmd}", f" – install with .\\run.ps1 {cmd}")
        print(f"{ok if have else no}{tr(th, en)}" + ("" if have else how))
    free = shutil.disk_usage(PROJECT_ROOT).free / 2**30
    warn = tr(" – ควรมีอย่างน้อย 20 GB", " – at least 20 GB is recommended")
    print(
        f"{ok if free > 20 else no}"
        + tr(f"พื้นที่ว่างในไดรฟ์โปรเจกต์ {free:.0f} GB", f"Free disk space {free:.0f} GB")
        + ("" if free > 20 else warn)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
