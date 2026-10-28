import json

import pandas as pd
import pytest
from omegaconf import OmegaConf

from ml_training import text_data as T
from ml_training.backends import data_backend, model_backend
from ml_training.modeling import text as M


def _cfg(tmp_path, **data):
    return OmegaConf.create({
        "seed": 0,
        "paths": {"root": str(tmp_path), "raw": str(tmp_path / "raw"),
                  "processed": str(tmp_path / "processed" / data.get("name", "ds"))},
        "data": {"name": "ds", "kind": "text", "path": None, "task": "auto", "text_column": "auto",
                 "label_column": "auto", "val_size": 0.2, "test_size": 0.2, "stratify": True,
                 **data},
        "model": {"backend": "transformers", "pretrained": None},
    })  # fmt: skip


def test_backend_dispatch(tmp_path):
    cfg = _cfg(tmp_path)
    assert data_backend(cfg).__name__.endswith("backends.text")
    assert model_backend(cfg).__name__.endswith("backends.text")
    cfg.data.kind, cfg.model.backend = None, "sklearn"
    assert data_backend(cfg) is None and model_backend(cfg) is None


def test_entity_spans_and_ner_eval():
    tags = ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-LOC", "O"]
    assert M.entity_spans(tags) == {("PER", 0, 1), ("ORG", 3, 4), ("LOC", 5, 5)}
    true = [["B-PER", "I-PER", "O", "B-MONEY"]]
    pred = [["B-PER", "O", "O", "B-MONEY"]]  # PER span cut short = FP + FN
    r = M.ner_eval(true, pred, ["O", "B-PER", "I-PER", "B-MONEY"])
    pc = r.per_class.set_index("class")
    assert (pc.loc["MONEY", "TP"], pc.loc["PER", "FP"], pc.loc["PER", "FN"]) == (1, 1, 1)
    assert r.summary["TP"] == 1 and abs(r.summary["f1"] - 0.5) < 1e-9


def test_table_autodetect_and_splits(tmp_path):
    df = pd.DataFrame({"id": range(40), "review": [f"ข้อความ {i} " * 3 for i in range(40)],
                       "sentiment": ["pos", "neg"] * 20})  # fmt: skip
    task, recs = T.table_records(df, _cfg(tmp_path).copy())
    assert task == "classify" and recs[0] == {"text": df.review[0], "label": "pos"}
    s = T.split_records(recs, 0.2, 0.2, 0)
    assert {k: len(v) for k, v in s.items()} == {"train": 24, "val": 8, "test": 8}
    assert sum(r["label"] == "pos" for r in s["test"]) == 4  # stratified


def test_conll_and_ner_labels(tmp_path):
    f = tmp_path / "train.txt"
    f.write_text("สมชาย B-PER\nใจดี I-PER\nจ่าย O\n\n500 B-MONEY\nบาท I-MONEY\n", encoding="utf-8")
    assert T._is_conll(f)
    recs = T.read_conll(f)
    assert len(recs) == 2 and recs[1]["tags"] == ["B-MONEY", "I-MONEY"]
    assert T.ner_labels(recs) == ["O", "B-MONEY", "I-MONEY", "B-PER", "I-PER"]


def test_documents_folder_featurize(tmp_path):
    import docx

    for label in ("invoice", "receipt"):
        d = tmp_path / "raw" / "ds" / label
        d.mkdir(parents=True)
        for i in range(5):
            (d / f"{i}.txt").write_text(f"{label} เลขที่ {i}", encoding="utf-8")
        doc = docx.Document()
        doc.add_paragraph(f"{label} word document")
        doc.save(str(d / "w.docx"))
    cfg = _cfg(tmp_path)
    out = T.featurize(cfg)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["task"] == "classify" and meta["class_names"] == ["invoice", "receipt"]
    assert sum(meta["n_rows"].values()) == 12
    rows = T.load_split(out, "train")
    assert any(r["file"].endswith(".docx") for r in rows + T.load_split(out, "test")
               + T.load_split(out, "val"))  # fmt: skip
    assert (tmp_path / "data" / "interim" / "ds" / "texts").exists()  # text cache


@pytest.mark.parametrize("task", ["classify", "ner"])
def test_tiny_model_trains_and_predicts(tmp_path, task):
    if task == "ner":
        recs = [{"tokens": ["นาย", "สมชาย", "จ่าย", "100", "บาท"],
                 "tags": ["B-PER", "I-PER", "O", "B-MONEY", "I-MONEY"]}] * 16  # fmt: skip
        labels = T.ner_labels(recs)
        texts = [" ".join(r["tokens"]) for r in recs]
    else:
        recs = [{"text": "ดีมาก ชอบ", "label": "pos"}, {"text": "แย่มาก เกลียด", "label": "neg"}] * 8
        labels, texts = ["neg", "pos"], [r["text"] for r in recs]
    cfg = _cfg(tmp_path)
    cfg.model.scratch = {"hidden": 32, "layers": 1, "heads": 1, "vocab_size": 60}
    tok = M.scratch_tokenizer(texts, 60, 32)
    l2i = {v: i for i, v in enumerate(labels)}
    ds = M.encode(tok, recs, task, l2i, 32)
    p = M._params(cfg, {"epochs": 1, "batch_size": 8, "lr": 1e-3})
    tr = M.fit(cfg, task, tok, labels, ds, ds, tmp_path / "out", p, save=False)
    tr.save_model(str(tmp_path / "m"))
    tok.save_pretrained(str(tmp_path / "m"))
    tok2, model, t = M.load_trained(tmp_path / "m")
    assert t == task
    inputs = [r["tokens"] for r in recs[:2]] if task == "ner" else texts[:2]
    out = M.infer(model, tok2, task, inputs, 32)
    assert len(out) == 2
    if task == "ner":
        assert len(out[0]) == 5 and set(out[0]) <= set(labels)
    else:
        assert out.shape == (2, 2) and abs(out[0].sum() - 1) < 1e-5


def test_split_words_thai():
    words = M.split_words("ลูกค้าชำระเงิน 500 บาท")
    assert "500" in words and "บาท" in words
