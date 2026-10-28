"""Plain-language explanation of an error + what to do, printed before the traceback.

Every mlt-* command is wrapped with @friendly_errors, so a beginner sees for example
"Close the results Excel file and try again" instead of only a PermissionError traceback.
English by default; Thai when MLT_LANG=th (menu: More tools > Language).
"""

from __future__ import annotations

import functools
import sys

from ml_training.i18n import tr

PACKAGES = {
    "ultralytics": ("YOLO", "YOLO"),
    "transformers": ("โมเดลข้อความ (HuggingFace)", "text models (HuggingFace)"),
    "timm": ("โมเดลภาพ (timm)", "image models (timm)"),
    "lightgbm": ("LightGBM", "LightGBM"),
    "xgboost": ("XGBoost", "XGBoost"),
    "easyocr": ("OCR อ่านเอกสารสแกน", "OCR for scanned documents"),
    "pythainlp": ("PyThaiNLP", "PyThaiNLP"),
    "torch": ("PyTorch", "PyTorch"),
    "docx": ("อ่านไฟล์ Word", "Word file reader"),
    "pypdf": ("อ่านไฟล์ PDF", "PDF file reader"),
}


def _setup() -> str:
    return tr(
        "รัน .\\run.ps1 setup-gpu (มีการ์ดจอ NVIDIA) หรือ .\\run.ps1 setup-yolo (CPU) แล้วลองใหม่",
        "run .\\run.ps1 setup-gpu (NVIDIA GPU) or .\\run.ps1 setup-yolo (CPU), then try again",
    )


def explain(e: BaseException) -> str:
    """Hint for common errors ('' when there is nothing better than the message itself)."""
    low = str(e).lower()
    if isinstance(e, ModuleNotFoundError):
        mod = (e.name or "").split(".")[0]
        if mod == "easyocr":
            return tr(
                "ยังไม่ได้ติดตั้ง OCR – รัน .\\run.ps1 setup-ocr แล้วลองใหม่",
                "OCR is not installed – run .\\run.ps1 setup-ocr, then try again",
            )
        th, en = PACKAGES.get(mod, (mod, mod))
        return tr(f"ยังไม่ได้ติดตั้งส่วน {th} – {_setup()}", f"{en} is not installed – {_setup()}")
    if "out of memory" in low:
        return tr(
            "หน่วยความจำการ์ดจอ (VRAM) ไม่พอ – ลดขนาด batch เช่น model.params.batch_size=8 "
            "หรือเลือกโมเดลที่เล็กลง (n / small / efficientnet / seg-b0)",
            "Not enough GPU memory (VRAM) – lower the batch size, e.g. model.params.batch_size=8, "
            "or pick a smaller model (n / small / efficientnet / seg-b0)",
        )
    if isinstance(e, MemoryError) or "unable to allocate" in low:
        return tr(
            "หน่วยความจำ (RAM) ไม่พอ – ปิดโปรแกรมอื่น ลดขนาดภาพ / batch หรือใช้ข้อมูลน้อยลงก่อน",
            "Not enough RAM – close other programs, lower the image size / batch, or use less data",
        )
    if isinstance(e, PermissionError):
        return tr(
            "เขียนไฟล์ไม่ได้ – ถ้าเปิดไฟล์ Excel / CSV ของผลลัพธ์ค้างไว้ ให้ปิดก่อนแล้วลองใหม่ "
            "(หรือโปรแกรมแอนตี้ไวรัสกำลังล็อกไฟล์)",
            "Cannot write a file – if a results Excel / CSV file is open, close it and try again "
            "(or an antivirus program is locking the file)",
        )
    if "no space left" in low or "disk full" in low:
        return tr(
            "พื้นที่ดิสก์เต็ม – ลบผลเก่าด้วย .\\run.ps1 clean <ชื่อชุดข้อมูล> หรือย้ายไปไดรฟ์ที่ว่างกว่า",
            "The disk is full – remove old results with .\\run.ps1 clean <dataset> "
            "or move the project to a drive with more space",
        )
    net = ("connection", "timed out", "name resolution", "huggingface.co", "max retries")
    if any(k in low for k in net):
        return tr(
            "ดาวน์โหลดโมเดล / ข้อมูลไม่ได้ – ตรวจอินเทอร์เน็ต (ครั้งแรกต้องดาวน์โหลด weight) "
            "หรือใช้โมเดล tiny ที่ไม่ต้องดาวน์โหลด",
            "Cannot download the model / data – check the internet connection (the first run "
            "downloads weights) or use a tiny model that needs no download",
        )
    if "cuda" in low and ("driver" in low or "no kernel image" in low or "not compiled" in low):
        return tr(
            "การ์ดจอ / ไดรเวอร์ไม่ตรงกับ PyTorch – "
            "อัปเดตไดรเวอร์ NVIDIA แล้วรัน .\\run.ps1 setup-gpu อีกครั้ง",
            "GPU / driver does not match PyTorch – update the NVIDIA driver, "
            "then run .\\run.ps1 setup-gpu again",
        )
    if isinstance(e, FileNotFoundError):
        if any(k in low for k in ("models", ".pt", ".joblib", "trained model")):
            return tr(
                "ยังไม่มีโมเดลที่เทรนแล้วสำหรับชุดข้อมูลนี้ – เทรนก่อนจากเมนู 1",
                "No trained model for this dataset yet – train one first (menu 1)",
            )
        return tr(
            "ไม่พบไฟล์ / โฟลเดอร์ – ตรวจชื่อหรือที่อยู่ไฟล์ หรือวางข้อมูลไว้ใน data\\raw\\<ชื่อ>",
            "File / folder not found – check the name or path, "
            "or put the data in data\\raw\\<name>",
        )
    if isinstance(e, KeyError) and ("column" in low or "not found" in low or "target" in low):
        return tr(
            "ไม่พบคอลัมน์ที่ระบุ – ดูรายชื่อคอลัมน์ในข้อความด้านบน "
            "แล้วใส่ target=<ชื่อคอลัมน์> ให้ตรง (ตัวพิมพ์เล็ก-ใหญ่มีผล)",
            "Column not found – see the column list in the message above "
            "and set target=<column name> exactly (case matters)",
        )
    if "could not convert string to float" in low:
        return tr(
            "มีข้อความปนในคอลัมน์ที่ควรเป็นตัวเลข – ตรวจคอลัมน์ target หรือลบแถวที่พิมพ์ผิด",
            "A numeric column contains text – check the target column or fix the mistyped rows",
        )
    if (
        "least populated class" in low
        or "should be greater or equal to the number of classes" in low
    ):
        return tr(
            "บางหมวดมีข้อมูลน้อยเกินไป (1-2 ตัวอย่าง) – เพิ่มข้อมูลหมวดนั้น หรือรวม / ลบหมวดที่มีน้อย",
            "Some classes have too few examples (1-2) – "
            "add more, or merge / remove the rare classes",
        )
    return ""


def show(e: BaseException) -> None:
    hint = explain(e)
    line = "=" * 70
    text = f"\n{line}\n{tr('เกิดข้อผิดพลาด', 'Error')}: {type(e).__name__}: {str(e)[:400]}\n"
    if hint and hint != str(e):
        text += f"{tr('วิธีแก้', 'What to do')}: {hint}\n"
    text += tr("(รายละเอียดสำหรับนักพัฒนาอยู่ด้านล่าง)", "(details for developers below)")
    text += f"\n{line}\n"
    try:
        print(text, file=sys.stderr, flush=True)
    except UnicodeEncodeError:  # console without UTF-8
        print(text.encode("ascii", "replace").decode(), file=sys.stderr, flush=True)


def friendly_errors(fn):
    """Decorator for Hydra mains: print the explanation, then let the error continue."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            print(
                tr(
                    "\nหยุดแล้ว (Ctrl+C) – เทรนต่อได้ด้วยเมนู 'เทรนต่อจากที่ค้าง'",
                    "\nStopped (Ctrl+C) – continue later with the menu 'Resume training'",
                ),
                file=sys.stderr,
            )
            raise
        except Exception as e:
            show(e)
            raise

    return wrapper
