"""Small built-in Thai datasets to try the text backend without downloading anything.

thai-sentiment  reviews.csv: text + label (positive / negative / neutral)
thai-ner        data.jsonl: tokens + BIO tags (PER, ORG, LOC, DATE, MONEY) – invoice-like sentences
thai-docs       <class>/*.txt|*.docx: invoice / receipt / contract / quotation documents
Generated from templates with a fixed seed, so they are always the same.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from ml_training.utils import ensure_dir, log

PRODUCTS = [
    "โทรศัพท์",
    "หูฟัง",
    "รองเท้า",
    "กระเป๋า",
    "ร้านอาหาร",
    "โรงแรม",
    "แอปนี้",
    "เสื้อ",
    "คอมพิวเตอร์",
    "บริการส่งของ",
]
POS = [
    "ดีมาก",
    "ประทับใจสุดๆ",
    "คุ้มค่ากับราคา",
    "ใช้งานง่าย",
    "ส่งไว ของถึงเร็ว",
    "คุณภาพดีเกินคาด",
    "พนักงานบริการดี",
    "แนะนำเลยครับ",
    "ชอบมากค่ะ",
    "จะกลับมาซื้ออีก",
]
NEG = [
    "แย่มาก",
    "ไม่คุ้มเงิน",
    "พังภายในสัปดาห์",
    "ส่งช้ามาก",
    "บริการแย่",
    "ผิดหวังมาก",
    "ของไม่ตรงปก",
    "ใช้ไม่ได้เลย",
    "ไม่แนะนำ",
    "จะไม่ซื้ออีกแล้ว",
]
NEU = [
    "ก็โอเค",
    "ธรรมดา ไม่มีอะไรพิเศษ",
    "ราคาพอๆ กับที่อื่น",
    "ได้รับของแล้ว",
    "ตามที่บรรยายไว้",
    "ใช้ได้ตามปกติ",
    "เฉยๆ",
    "ยังไม่ได้ลองใช้จริง",
    "กล่องมาครบ",
    "สีตามรูป",
]
ENG = {
    "positive": ["great", "love it", "excellent"],
    "negative": ["terrible", "broken", "awful"],
    "neutral": ["okay", "as described", "average"],
}

FIRST = ["สมชาย", "สมหญิง", "วิชัย", "มาลี", "ประยุทธ", "อรุณี", "กิตติ", "นภา", "ธนา", "ศิริพร"]
LAST = ["ใจดี", "รักไทย", "สุขสันต์", "ทองคำ", "ศรีสุข", "บุญมา", "แก้วใส", "พงษ์ไทย"]
ORGS = [
    ["บริษัท", "เอบีซี", "จำกัด"],
    ["บริษัท", "ไทยเทค", "จำกัด"],
    ["ธนาคาร", "กรุงเทพ"],
    ["ห้างหุ้นส่วน", "สยามพาณิชย์"],
    ["บริษัท", "สมาร์ทโลจิสติกส์", "จำกัด"],
]
LOCS = [["กรุงเทพมหานคร"], ["เชียงใหม่"], ["ขอนแก่น"], ["ภูเก็ต"], ["นนทบุรี"], ["ชลบุรี"]]
MONTHS = [
    "มกราคม",
    "กุมภาพันธ์",
    "มีนาคม",
    "เมษายน",
    "พฤษภาคม",
    "มิถุนายน",
    "กรกฎาคม",
    "สิงหาคม",
    "กันยายน",
    "ตุลาคม",
    "พฤศจิกายน",
    "ธันวาคม",
]


def _sentiment(dst: Path, rng: random.Random) -> None:
    rows = []
    for label, words in (("positive", POS), ("negative", NEG), ("neutral", NEU)):
        for _ in range(60):
            p, w = rng.choice(PRODUCTS), rng.choice(words)
            extra = rng.choice(
                ["", " " + rng.choice(words), " " + rng.choice(ENG[label]), "!!", " 555"]
            )
            rows.append({"text": f"{p}{rng.choice(['', ' ', 'ร้านนี้ '])}{w}{extra}", "label": label})
    rng.shuffle(rows)
    with (ensure_dir(dst) / "reviews.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["text", "label"])
        w.writeheader()
        w.writerows(rows)


def _ent(tokens: list[str], tags: list[str], words: list[str], ent: str | None) -> None:
    for i, word in enumerate(words):
        tokens.append(word)
        tags.append("O" if ent is None else ("B-" if i == 0 else "I-") + ent)


def _ner(dst: Path, rng: random.Random) -> None:
    templates = [
        [
            ("ลูกค้า", None),
            ("PER",),
            ("ชำระเงิน", None),
            ("MONEY",),
            ("ให้", None),
            ("ORG",),
            ("เมื่อวันที่", None),
            ("DATE",),
        ],
        [("ORG",), ("ออกใบแจ้งหนี้", None), ("ถึง", None), ("PER",), ("ยอดรวม", None), ("MONEY",)],
        [("PER",), ("เดินทางไป", None), ("LOC",), ("วันที่", None), ("DATE",)],
        [("ส่งสินค้าจาก", None), ("LOC",), ("โดย", None), ("ORG",), ("ค่าขนส่ง", None), ("MONEY",)],
        [("สัญญาลงนามโดย", None), ("PER",), ("และ", None), ("PER",), ("ที่", None), ("LOC",)],
        [("ใบเสร็จ", None), ("ORG",), ("วันที่", None), ("DATE",), ("จำนวนเงิน", None), ("MONEY",)],
    ]
    with (ensure_dir(dst) / "data.jsonl").open("w", encoding="utf-8") as fh:
        for _ in range(240):
            tokens: list[str] = []
            tags: list[str] = []
            for part in rng.choice(templates):
                if part[-1] is None and len(part) == 2:
                    _ent(tokens, tags, [part[0]], None)
                    continue
                ent = part[0]
                if ent == "PER":
                    words = [rng.choice(["นาย", "นาง", "นางสาว"])] if rng.random() < 0.5 else []
                    words += [rng.choice(FIRST), rng.choice(LAST)]
                elif ent == "ORG":
                    words = rng.choice(ORGS)
                elif ent == "LOC":
                    words = rng.choice(LOCS)
                elif ent == "DATE":
                    words = [
                        str(rng.randint(1, 28)),
                        rng.choice(MONTHS),
                        str(rng.randint(2560, 2569)),
                    ]
                else:
                    words = [f"{rng.randint(1, 99) * 100:,}", "บาท"]
                _ent(tokens, tags, words, ent)
            fh.write(json.dumps({"tokens": tokens, "tags": tags}, ensure_ascii=False) + "\n")


DOCS = {
    "invoice": [
        "ใบแจ้งหนี้ / INVOICE",
        "เลขที่ใบแจ้งหนี้ INV-{n}",
        "ครบกำหนดชำระ {d}",
        "กรุณาชำระเงินภายใน 30 วัน",
        "ยอดค้างชำระ {m} บาท",
    ],
    "receipt": [
        "ใบเสร็จรับเงิน / RECEIPT",
        "เลขที่ RC-{n}",
        "ได้รับเงินเรียบร้อยแล้ว {m} บาท",
        "ชำระด้วยเงินสด",
        "ขอบคุณที่ใช้บริการ",
    ],
    "contract": [
        "สัญญาจ้างงาน",
        "สัญญาฉบับนี้ทำขึ้นเมื่อ {d}",
        "ผู้ว่าจ้างและผู้รับจ้างตกลงกันดังนี้",
        "ข้อ 1 ระยะเวลาสัญญา 12 เดือน",
        "ลงชื่อ ผู้ว่าจ้าง ลงชื่อ ผู้รับจ้าง พยาน",
    ],
    "quotation": [
        "ใบเสนอราคา / QUOTATION",
        "เลขที่ QT-{n}",
        "ราคานี้ยืนยันได้ 30 วัน",
        "ราคารวม {m} บาท ยังไม่รวมภาษีมูลค่าเพิ่ม",
        "หากตกลงกรุณาเซ็นอนุมัติ",
    ],
}


def _docs(dst: Path, rng: random.Random) -> None:
    import docx

    for label, lines in DOCS.items():
        folder = ensure_dir(dst / label)
        for i in range(24):
            fill = {
                "n": rng.randint(1000, 9999),
                "m": f"{rng.randint(5, 900) * 100:,}",
                "d": f"{rng.randint(1, 28)} {rng.choice(MONTHS)} {rng.randint(2560, 2569)}",
            }
            body = [ln.format(**fill) for ln in lines if rng.random() > 0.15]
            body.insert(1, " ".join(rng.choice(ORGS)))
            body.append(f"ติดต่อ {rng.choice(FIRST)} {rng.choice(LAST)}")
            if i % 3 == 0:
                d = docx.Document()
                for ln in body:
                    d.add_paragraph(ln)
                d.save(str(folder / f"{label}_{i:02d}.docx"))
            else:
                (folder / f"{label}_{i:02d}.txt").write_text("\n".join(body), encoding="utf-8")


SAMPLES = {"thai-sentiment": _sentiment, "thai-ner": _ner, "thai-docs": _docs}


def make(name: str, dst: Path) -> Path:
    if name not in SAMPLES:
        raise KeyError(f"Unknown text sample {name!r}: {list(SAMPLES)}")
    SAMPLES[name](Path(dst), random.Random(42))
    log.info("Sample dataset %s written to %s", name, dst)
    return Path(dst)
