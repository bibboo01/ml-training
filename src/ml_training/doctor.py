"""Check this computer: Python, GPU, installed parts, free disk.

python -m ml_training.doctor          (menu: More tools > Check this computer)
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys

from ml_training.i18n import tr
from ml_training.utils import PROJECT_ROOT

PARTS = [
    (
        "ultralytics",
        "YOLO (ตรวจจับ / แยกวัตถุ / pose / จัดหมวดภาพ)",
        "YOLO (detect / segment / pose / classify)",
        "setup-gpu",
    ),
    (
        "transformers",
        "ข้อความ และเอกสาร (HuggingFace)",
        "text and documents (HuggingFace)",
        "setup-gpu",
    ),
    ("pythainlp", "ตัดคำภาษาไทย", "Thai word segmentation", "setup-gpu"),
    (
        "timm",
        "จัดหมวดภาพ ResNet / EfficientNet / ConvNeXt / ViT",
        "image models ResNet / EfficientNet / ConvNeXt / ViT",
        "setup-gpu",
    ),
    ("lightgbm", "LightGBM สำหรับข้อมูลตาราง", "LightGBM for tables", "setup-gpu"),
    ("xgboost", "XGBoost สำหรับข้อมูลตาราง", "XGBoost for tables", "setup-gpu"),
    (
        "easyocr",
        "OCR อ่านเอกสารสแกน / รูปเอกสาร (ไม่บังคับ)",
        "OCR for scanned documents (optional)",
        "setup-ocr",
    ),
]


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
                        "       ถ้ามีการ์ดจอ NVIDIA ให้รัน .\\run.ps1 setup-gpu",
                        "       If this PC has an NVIDIA GPU, run .\\run.ps1 setup-gpu",
                    )
                )
    except Exception:  # noqa: BLE001
        print(
            no
            + tr(
                "ยังไม่มี PyTorch – รัน .\\run.ps1 setup-gpu หรือ setup-yolo",
                "PyTorch is missing – run .\\run.ps1 setup-gpu or setup-yolo",
            )
        )
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
