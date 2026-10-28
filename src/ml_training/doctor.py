"""Check this computer: Python, GPU, installed parts, free disk – in plain Thai.

python -m ml_training.doctor          (menu: เครื่องมือเพิ่มเติม > ตรวจเครื่อง)
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys

from ml_training.utils import PROJECT_ROOT

PARTS = [
    ("ultralytics", "YOLO (ตรวจจับ / แยกวัตถุ / pose / จัดหมวดภาพ)", "setup-gpu"),
    ("transformers", "ข้อความ และเอกสาร (HuggingFace)", "setup-gpu"),
    ("pythainlp", "ตัดคำภาษาไทย", "setup-gpu"),
    ("timm", "จัดหมวดภาพ ResNet / EfficientNet / ConvNeXt / ViT", "setup-gpu"),
    ("lightgbm", "LightGBM สำหรับข้อมูลตาราง", "setup-gpu"),
    ("xgboost", "XGBoost สำหรับข้อมูลตาราง", "setup-gpu"),
    ("easyocr", "OCR อ่านเอกสารสแกน / รูปเอกสาร (ไม่บังคับ)", "setup-ocr"),
]


def main() -> int:
    ok = "  [OK] "
    no = "  [--] "
    print("ตรวจเครื่อง")
    print(f"{ok}Python {platform.python_version()} บน {platform.system()} {platform.release()}")
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            vram = p.total_memory / 2**30
            print(f"{ok}การ์ดจอ {p.name} ({vram:.1f} GB) – PyTorch {torch.__version__}")
            if vram < 6:
                print("       VRAM น้อย: เลือกโมเดลเล็ก (YOLO n/s, efficientnet, seg-b0, text small)")
        else:
            print(f"{no}ไม่พบการ์ดจอ NVIDIA ที่ใช้ได้ – ใช้ CPU (ช้ากว่า 10-50 เท่า)")
            if "+cpu" in torch.__version__ or not torch.version.cuda:
                print("       ถ้ามีการ์ดจอ NVIDIA ให้รัน .\\run.ps1 setup-gpu")
    except Exception:  # noqa: BLE001
        print(f"{no}ยังไม่มี PyTorch – รัน .\\run.ps1 setup-gpu หรือ setup-yolo")
    for mod, what, cmd in PARTS:
        have = importlib.util.find_spec(mod) is not None
        print(f"{ok if have else no}{what}" + ("" if have else f" – ติดตั้งด้วย .\\run.ps1 {cmd}"))
    free = shutil.disk_usage(PROJECT_ROOT).free / 2**30
    print(
        f"{ok if free > 20 else no}พื้นที่ว่างในไดรฟ์โปรเจกต์ {free:.0f} GB"
        + ("" if free > 20 else " – ควรมีอย่างน้อย 20 GB")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
