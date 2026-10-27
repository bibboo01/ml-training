"""XAI workbook: heatmap evidence + error analysis + statistics, with Thai explanations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GLOSSARY = [
    (
        "EigenCAM",
        "Heatmap ภาพรวม: ส่วนประกอบหลัก (PCA) ของ feature map ก่อนเข้า head – บอกว่าโมเดล 'สนใจ' ตรงไหนของภาพ ไม่ขึ้นกับ class",
    ),
    (
        "Grad-CAM (รายกรอบ)",
        "Heatmap ของกรอบเดียว: gradient ของคะแนน class ของกรอบนั้นย้อนกลับไปที่ feature map ของ scale ที่สร้างกรอบ – จุดร้อน = หลักฐานที่ทำให้โมเดลตัดสินว่าเป็น class นี้",
    ),
    ("attention inside", "สัดส่วนความร้อนทั้งหมดที่อยู่ในกรอบ (0-1)"),
    (
        "attention lift",
        "inside ÷ สัดส่วนพื้นที่กรอบ  · 1 = ไม่ได้โฟกัสเป็นพิเศษ · >1 = โมเดลมองที่วัตถุ · <1 = ตัดสินจากบริบทรอบๆ",
    ),
    ("pointing game", "จุดที่ร้อนที่สุดของ heatmap อยู่ในกรอบหรือไม่ (1/0) – ค่าเฉลี่ย = % ที่โมเดล 'ชี้ถูกที่'"),
    ("heat spread", "ความกระจายของ heatmap (entropy 0-1): ต่ำ = จ้องจุดเดียว, สูง = มองกว้างทั้งภาพ"),
    ("TP", "กรอบที่ถูก class และ IoU ≥ เกณฑ์ กับวัตถุจริงที่ยังไม่ถูกจับคู่"),
    ("FP background", "ทายว่ามีวัตถุในที่ที่ไม่มีอะไรเลย (IoU < 0.1 กับทุกวัตถุ) – โมเดลหลงกับพื้นหลัง"),
    ("FP localization", "ถูก class แต่กรอบเพี้ยน (0.1 ≤ IoU < เกณฑ์) – ปรับ box loss / label ไม่แม่น"),
    ("FP class", "ตำแหน่งถูกแต่ผิด class – class ที่หน้าตาคล้ายกัน"),
    ("FP class+loc", "ผิดทั้ง class และตำแหน่ง"),
    ("FP duplicate", "จับวัตถุเดิมซ้ำ – NMS IoU อาจต่ำ/สูงเกินไป"),
    ("FN missed", "ไม่มีกรอบใดใกล้วัตถุเลย – วัตถุเล็ก/มืด/ถูกบัง หรือข้อมูล train ไม่พอ"),
    ("FN wrong class", "หาเจอแต่ใส่ class ผิด"),
    ("FN poor box", "ถูก class แต่กรอบซ้อนน้อยเกินเกณฑ์"),
    (
        "Wilson 95% CI",
        "ช่วงความเชื่อมั่น 95% ของ precision / recall ต่อ class – ช่วงกว้าง = ข้อมูลน้อย ยังสรุปไม่ได้",
    ),
    ("Bootstrap 95% CI", "สุ่มภาพซ้ำ 1,000 รอบเพื่อหาช่วงความเชื่อมั่นของ P / R / F1 รวม"),
    (
        "ECE",
        "Expected Calibration Error: ค่าเฉลี่ยของ |ความแม่นจริง − conf| ถ่วงตามจำนวน · ยิ่งต่ำยิ่งเชื่อ conf ได้",
    ),
    ("MCE", "ช่องที่ conf กับความแม่นจริงต่างกันมากที่สุด"),
    ("Brier score", "ค่าเฉลี่ย (conf − ถูก/ผิด)² · 0 = สมบูรณ์แบบ"),
    ("p-value", "< 0.05 = ความต่างนี้ไม่น่าเกิดจากความบังเอิญ (มีนัยสำคัญ)"),
    ("rank-biserial", "ขนาดผลของ Mann-Whitney (-1..1): |r| 0.1 เล็ก · 0.3 กลาง · 0.5 ใหญ่"),
    ("Cramer's V", "ขนาดความสัมพันธ์ของ chi-square (0..1): 0.1 เล็ก · 0.3 กลาง · 0.5 ใหญ่"),
    ("size small/medium/large", "ตามเกณฑ์ COCO: พื้นที่ < 32² / 32²-96² / > 96² พิกเซล (ในภาพจริง)"),
]


def _fmt(wb) -> dict[str, Any]:
    return {
        "title": wb.add_format({"bold": True, "font_size": 16, "font_color": "#1E3A8A"}),
        "h": wb.add_format({"bold": True, "bg_color": "#1E3A8A", "font_color": "white", "border": 1,
                            "text_wrap": True, "valign": "top"}),
        "key": wb.add_format({"bold": True, "bg_color": "#E0E7FF", "border": 1}),
        "cell": wb.add_format({"border": 1}),
        "num": wb.add_format({"border": 1, "num_format": "0.000"}),
        "pct": wb.add_format({"border": 1, "num_format": "0.0%"}),
        "wrap": wb.add_format({"border": 1, "text_wrap": True, "valign": "top"}),
        "note": wb.add_format({"italic": True, "font_color": "#475569", "text_wrap": True}),
        "good": wb.add_format({"border": 1, "bg_color": "#DCFCE7"}),
        "bad": wb.add_format({"border": 1, "bg_color": "#FEE2E2"}),
    }  # fmt: skip


def _val(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, list | tuple | dict):
        return str(v)
    return v


def _table(ws, df: pd.DataFrame, f, row: int = 0, col: int = 0, widths: dict | None = None) -> int:
    for j, c in enumerate(df.columns):
        ws.write(row, col + j, str(c), f["h"])
        ws.set_column(col + j, col + j, (widths or {}).get(c, max(10, min(40, len(str(c)) + 2))))
    for i, rec in enumerate(df.itertuples(index=False), start=1):
        for j, v in enumerate(rec):
            v = _val(v)
            if v is None:
                ws.write_blank(row + i, col + j, None, f["cell"])
            elif isinstance(v, float):
                ws.write_number(row + i, col + j, v, f["num"])
            elif isinstance(v, bool):
                ws.write(row + i, col + j, "yes" if v else "no", f["cell"])
            else:
                ws.write(row + i, col + j, v, f["cell"])
    if len(df.columns):
        ws.autofilter(row, col, row + max(len(df), 1), col + len(df.columns) - 1)
        ws.freeze_panes(row + 1, 0)
    return row + len(df) + 1


def _links(ws, df: pd.DataFrame, column: str, row0: int, base: Path) -> None:
    if column not in df:
        return
    j = list(df.columns).index(column)
    for i, v in enumerate(df[column], start=1):
        if isinstance(v, str) and v:
            ws.write_url(row0 + i, j, f"external:{v}", string=Path(v).name)


def write_xai_report(
    path: Path,
    info: dict[str, Any],
    headline: list[tuple[str, Any, str]],
    tests: pd.DataFrame,
    sheets: list[tuple[str, pd.DataFrame, str]],
    calib: pd.DataFrame | None,
    sweep: pd.DataFrame | None,
    errors: pd.DataFrame | None,
    size: pd.DataFrame | None,
    gallery: list[tuple[str, Path]],
) -> Path:
    import xlsxwriter
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(str(path), {"nan_inf_to_errors": True})
    f = _fmt(wb)

    # ---------------- Summary
    ws = wb.add_worksheet("Summary")
    ws.write(0, 0, "XAI report – why the model decides what it decides", f["title"])
    r = 2
    for k, v in info.items():
        ws.write(r, 0, k, f["key"])
        ws.write(r, 1, _val(v), f["cell"])
        r += 1
    r += 1
    ws.write_row(r, 0, ["key result", "value", "ความหมาย"], f["h"])
    for k, v, note in headline:
        r += 1
        v = _val(v)
        ws.write(r, 0, k, f["key"])
        if isinstance(v, float):
            ws.write_number(r, 1, v, f["num"])
        else:
            ws.write(r, 1, v if v is not None else "-", f["cell"])
        ws.write(r, 2, note, f["wrap"])
    r += 2
    ws.write(r, 0, "อ่านต่อ: Stats = คำตอบจากการทดสอบสถิติ · PerClass = รายคลาส · Errors = ผิดแบบไหน · "
             "Gallery = ภาพ heatmap · Glossary = ความหมายทุกคำ", f["note"])  # fmt: skip
    ws.set_column(0, 0, 34)
    ws.set_column(1, 1, 28)
    ws.set_column(2, 2, 90)

    # ---------------- Stats
    ws = wb.add_worksheet("Stats")
    if tests is not None and len(tests):
        end = _table(
            ws, tests, f, widths={"question (ภาษาไทย)": 55, "interpretation": 80, "test": 24}
        )
        for i, sig in enumerate(tests["significant (p<0.05)"], start=1):
            ws.write(i, list(tests.columns).index("significant (p<0.05)"), sig,
                     f["good"] if sig == "yes" else f["cell"])  # fmt: skip
        ws.write(
            end + 1,
            0,
            "p < 0.05 = มีนัยสำคัญ · effect size บอกว่าต่างกันมากแค่ไหน (ดู Glossary)",
            f["note"],
        )
    else:
        ws.write(0, 0, "Not enough data for statistical tests (need more images / errors).")

    # ---------------- data sheets
    for name, df, link_col in sheets:
        if df is None:
            continue
        ws = wb.add_worksheet(name[:31])
        _table(ws, df, f)
        _links(ws, df, link_col, 0, path.parent)

    # ---------------- Errors (+ chart)
    if errors is not None and len(errors):
        ws = wb.add_worksheet("Errors")
        _table(ws, errors, f)
        n = len(errors)
        ch = wb.add_chart({"type": "bar", "subtype": "stacked"})
        palette = [
            "#DC2626",
            "#F59E0B",
            "#7C3AED",
            "#DB2777",
            "#64748B",
            "#0EA5E9",
            "#14B8A6",
            "#84CC16",
        ]
        for j in range(1, len(errors.columns)):
            ch.add_series({"name": ["Errors", 0, j], "categories": ["Errors", 1, 0, n, 0],
                           "values": ["Errors", 1, j, n, j],
                           "fill": {"color": palette[(j - 1) % len(palette)]}})  # fmt: skip
        ch.set_title({"name": "Error types per class"})
        ch.set_size({"width": 820, "height": 60 + 28 * max(4, n)})
        ws.insert_chart(n + 3, 0, ch)

    # ---------------- Size (+ chart)
    if size is not None and len(size):
        ws = wb.add_worksheet("Size")
        _table(ws, size, f)
        n = len(size)
        rc = [j for j, c in enumerate(size.columns) if str(c).startswith("recall")]
        if rc:
            ch = wb.add_chart({"type": "column"})
            for j in rc:
                ch.add_series({"name": ["Size", 0, j], "categories": ["Size", 1, 0, n, 0],
                               "values": ["Size", 1, j, n, j]})  # fmt: skip
            ch.set_title({"name": "Recall by object size"})
            ch.set_y_axis({"min": 0, "max": 1})
            ws.insert_chart(n + 3, 0, ch)

    # ---------------- Calibration (+ chart)
    if calib is not None and len(calib):
        ws = wb.add_worksheet("Calibration")
        _table(ws, calib, f)
        n = len(calib)
        ch = wb.add_chart({"type": "scatter", "subtype": "straight_with_markers"})
        cols = list(calib.columns)
        ch.add_series({"name": "model", "categories": ["Calibration", 1, cols.index("mean_conf"), n, cols.index("mean_conf")],
                       "values": ["Calibration", 1, cols.index("accuracy"), n, cols.index("accuracy")]})  # fmt: skip
        ws.write_row(n + 2, 0, ["perfect", 0, 1])
        ch.add_series({"name": "perfect calibration", "categories": ["Calibration", n + 2, 1, n + 2, 2],
                       "values": ["Calibration", n + 2, 1, n + 2, 2],
                       "line": {"dash_type": "dash", "color": "#94A3B8"}, "marker": {"type": "none"}})  # fmt: skip
        ch.set_title({"name": "Reliability: confidence vs real accuracy"})
        ch.set_x_axis({"name": "mean confidence", "min": 0, "max": 1})
        ch.set_y_axis({"name": "fraction correct", "min": 0, "max": 1})
        ws.insert_chart(n + 4, 0, ch)
        ws.write(
            n + 22,
            0,
            "เส้นต่ำกว่าเส้นประ = มั่นใจเกินจริง (overconfident) · สูงกว่า = มั่นใจน้อยกว่าที่ควร",
            f["note"],
        )

    # ---------------- Threshold sweep (+ chart)
    if sweep is not None and len(sweep):
        ws = wb.add_worksheet("Threshold")
        allrows = sweep[sweep["class"] == "ALL"] if "class" in sweep else sweep
        rest = sweep[sweep["class"] != "ALL"] if "class" in sweep else sweep.iloc[0:0]
        sweep_sorted = pd.concat([allrows, rest])
        _table(ws, sweep_sorted, f)
        n = len(allrows)
        cols = list(sweep_sorted.columns)
        xcol = cols.index("conf")
        ch = wb.add_chart({"type": "scatter", "subtype": "straight"})
        for m in [c for c in ("precision", "recall", "f1", "accuracy", "coverage") if c in cols]:
            ch.add_series({"name": m, "categories": ["Threshold", 1, xcol, n, xcol],
                           "values": ["Threshold", 1, cols.index(m), n, cols.index(m)]})  # fmt: skip
        ch.set_title({"name": "All classes: metric vs confidence threshold"})
        ch.set_x_axis({"name": "confidence threshold", "min": 0, "max": 1})
        ch.set_y_axis({"min": 0, "max": 1})
        ch.set_size({"width": 760, "height": 380})
        ws.insert_chart(1, len(cols) + 1, ch)

    # ---------------- Gallery
    if gallery:
        ws = wb.add_worksheet("Gallery")
        ws.write(
            0,
            0,
            "ซ้าย: ภาพ + กรอบ (เขียว TP · แดง FP · ส้ม FN) · กลาง: EigenCAM · ขวา: Grad-CAM ของแต่ละกรอบรวมกัน",
            f["note"],
        )
        r = 2
        for title, img_path in gallery:
            if not Path(img_path).exists():
                continue
            ws.write(r, 0, title, f["key"])
            with Image.open(img_path) as im:
                w, h = im.size
            scale = min(1.0, 1100 / w)
            ws.insert_image(
                r + 1, 0, str(img_path), {"x_scale": scale, "y_scale": scale, "object_position": 1}
            )
            r += int(h * scale / 20) + 3
        ws.set_column(0, 0, 60)

    # ---------------- Glossary
    ws = wb.add_worksheet("Glossary")
    ws.write_row(0, 0, ["term", "ความหมาย"], f["h"])
    for i, (k, v) in enumerate(GLOSSARY, start=1):
        ws.write(i, 0, k, f["key"])
        ws.write(i, 1, v, f["wrap"])
    ws.set_column(0, 0, 26)
    ws.set_column(1, 1, 120)
    wb.close()
    return path
