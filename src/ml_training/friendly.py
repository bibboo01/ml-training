"""Plain-language (Thai) explanation of an error + what to do, printed before the traceback.

Every mlt-* command is wrapped with @friendly_errors, so a beginner sees for example
"ปิดไฟล์ Excel ที่เปิดค้างไว้แล้วลองใหม่" instead of only a PermissionError traceback.
"""

from __future__ import annotations

import functools
import sys

SETUP = "รัน .\\run.ps1 setup-gpu (มีการ์ดจอ NVIDIA) หรือ .\\run.ps1 setup-yolo (CPU) แล้วลองใหม่"
PACKAGES = {
    "ultralytics": "YOLO",
    "transformers": "โมเดลข้อความ (HuggingFace)",
    "timm": "โมเดลภาพ (timm)",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "easyocr": "OCR อ่านเอกสารสแกน",
    "pythainlp": "PyThaiNLP",
    "torch": "PyTorch",
    "docx": "อ่านไฟล์ Word",
    "pypdf": "อ่านไฟล์ PDF",
}


def explain(e: BaseException) -> str:
    """Thai hint for common errors ('' when there is nothing better than the message itself)."""
    msg, name = str(e), type(e).__name__
    low = msg.lower()
    if isinstance(e, ModuleNotFoundError):
        mod = (e.name or "").split(".")[0]
        if mod == "easyocr":
            return "ยังไม่ได้ติดตั้ง OCR – รัน .\\run.ps1 setup-ocr แล้วลองใหม่"
        return f"ยังไม่ได้ติดตั้งส่วน {PACKAGES.get(mod, mod)} – {SETUP}"
    if "out of memory" in low or "cuda error: out of memory" in low:
        return (
            "หน่วยความจำการ์ดจอ (VRAM) ไม่พอ – ลดขนาด batch เช่น model.params.batch_size=8 "
            "หรือเลือกโมเดลที่เล็กลง (n / small / efficientnet / seg-b0)"
        )
    if isinstance(e, MemoryError) or "unable to allocate" in low:
        return "หน่วยความจำ (RAM) ไม่พอ – ปิดโปรแกรมอื่น ลดขนาดภาพ / batch หรือใช้ข้อมูลน้อยลงก่อน"
    if isinstance(e, PermissionError):
        return (
            "เขียนไฟล์ไม่ได้ – ถ้าเปิดไฟล์ Excel / CSV ของผลลัพธ์ค้างไว้ ให้ปิดก่อนแล้วลองใหม่ "
            "(หรือโปรแกรมแอนตี้ไวรัสกำลังล็อกไฟล์)"
        )
    if "no space left" in low or "disk full" in low:
        return "พื้นที่ดิสก์เต็ม – ลบผลเก่าด้วย .\\run.ps1 clean <ชื่อชุดข้อมูล> หรือย้ายไปไดรฟ์ที่ว่างกว่า"
    if any(
        k in low
        for k in ("connection", "timed out", "name resolution", "huggingface.co", "max retries")
    ):
        return (
            "ดาวน์โหลดโมเดล / ข้อมูลไม่ได้ – ตรวจอินเทอร์เน็ต (ครั้งแรกต้องดาวน์โหลด weight) "
            "หรือใช้โมเดล tiny ที่ไม่ต้องดาวน์โหลด"
        )
    if "cuda" in low and ("driver" in low or "no kernel image" in low or "not compiled" in low):
        return (
            "การ์ดจอ / ไดรเวอร์ไม่ตรงกับ PyTorch – อัปเดตไดรเวอร์ NVIDIA แล้วรัน .\\run.ps1 setup-gpu อีกครั้ง"
        )
    if isinstance(e, FileNotFoundError):
        if "models" in low or ".pt" in low or ".joblib" in low or "trained model" in low:
            return "ยังไม่มีโมเดลที่เทรนแล้วสำหรับชุดข้อมูลนี้ – เทรนก่อนจากเมนู 1"
        return "ไม่พบไฟล์ / โฟลเดอร์ – ตรวจชื่อหรือที่อยู่ไฟล์ หรือวางข้อมูลไว้ใน data\\raw\\<ชื่อ>"
    if isinstance(e, KeyError) and ("column" in low or "not found" in low or "target" in low):
        return (
            "ไม่พบคอลัมน์ที่ระบุ – ดูรายชื่อคอลัมน์ในข้อความด้านบน "
            "แล้วใส่ target=<ชื่อคอลัมน์> ให้ตรง (ตัวพิมพ์เล็ก-ใหญ่มีผล)"
        )
    if "could not convert string to float" in low:
        return "มีข้อความปนในคอลัมน์ที่ควรเป็นตัวเลข – ตรวจคอลัมน์ target หรือลบแถวที่พิมพ์ผิด"
    if (
        "least populated class" in low
        or "should be greater or equal to the number of classes" in low
    ):
        return "บางหมวดมีข้อมูลน้อยเกินไป (1-2 ตัวอย่าง) – เพิ่มข้อมูลหมวดนั้น หรือรวม / ลบหมวดที่มีน้อย"
    if name in ("ValueError", "KeyError") and any("฀" <= ch <= "๿" for ch in msg):
        return msg  # already a Thai message from our own checks
    return ""


def show(e: BaseException) -> None:
    hint = explain(e)
    line = "=" * 70
    text = f"\n{line}\nเกิดข้อผิดพลาด: {type(e).__name__}: {str(e)[:400]}\n"
    if hint and hint != str(e):
        text += f"วิธีแก้: {hint}\n"
    text += f"(รายละเอียดสำหรับนักพัฒนาอยู่ด้านล่าง){chr(10)}{line}\n"
    try:
        print(text, file=sys.stderr, flush=True)
    except UnicodeEncodeError:  # console without UTF-8
        print(text.encode("ascii", "replace").decode(), file=sys.stderr, flush=True)


def friendly_errors(fn):
    """Decorator for Hydra mains: print the Thai explanation, then let the error continue."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            print("\nหยุดแล้ว (Ctrl+C) – เทรนต่อได้ด้วยเมนู 'เทรนต่อจากที่ค้าง'", file=sys.stderr)
            raise
        except Exception as e:
            show(e)
            raise

    return wrapper
