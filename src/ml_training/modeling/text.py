"""HuggingFace transformers backend: text classification and NER (also documents -> text).

Same workflow as YOLO and the tabular models:
Optuna search -> final training -> evaluation (TP/FP/FN per class or per entity) -> MLflow ->
Excel, with lineage, history and resume.

    uv run mlt-train data=thai-sentiment model=text_base
    uv run mlt-train data=thai-ner model=text_thai hpo.n_trials=10
    uv run mlt-predict data=thai-docs model=text_base +input=D:/scans
    uv run mlt-predict data=thai-sentiment model=text_base "+text='ของดีมาก ส่งไว'"

model.pretrained = a HuggingFace model id (downloaded once into models/pretrained/hf) or null =
a tiny BERT trained from scratch with its own tokenizer (offline smoke tests, model=text_tiny).
"""

from __future__ import annotations

import math
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ml_training import lineage, runstate
from ml_training.hpo import param_importances, run_study, suggest_params, trials_dataframe
from ml_training.i18n import RUN
from ml_training.metrics import EvalResult, evaluate
from ml_training.reporting.excel import write_report
from ml_training.text_data import load_meta, load_split
from ml_training.tracking import log_params, setup_mlflow
from ml_training.utils import ensure_dir, flatten_dict, log, set_seed, write_json

PARAMS = {  # model.params understood by this backend (other keys, e.g. from YOLO presets, ignored)
    "epochs": 5,
    "batch_size": 16,
    "lr": 3e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "grad_accum": 1,
    "max_length": 256,
    "patience": 3,
    "fp16": True,
    "label_smoothing": 0.0,
}
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _params(cfg: DictConfig, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    given = cfg.model.get("params")
    given = OmegaConf.to_container(given, resolve=True) if given is not None else {}
    out = {k: given.get(k, v) if given.get(k) is not None else v for k, v in PARAMS.items()}
    out.update({k: v for k, v in (overrides or {}).items() if k in PARAMS})
    return out


def _cache_dir(cfg: DictConfig) -> str:
    return str(ensure_dir(Path(cfg.paths.models) / "pretrained" / "hf"))


# --------------------------------------------------------------------------- tokenizer / model
def scratch_tokenizer(texts: list[str], vocab_size: int = 4000, max_length: int = 256):
    """Unigram (sentencepiece-style) tokenizer trained on the data – works for Thai (no spaces)."""
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
    from tokenizers.trainers import UnigramTrainer
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.Unigram())
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = UnigramTrainer(vocab_size=vocab_size, special_tokens=SPECIAL, unk_token="[UNK]")
    tok.train_from_iterator(texts, trainer)
    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", cls), ("[SEP]", sep)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=max_length,
    )


def build_tokenizer(cfg: DictConfig, texts: list[str]):
    from transformers import AutoTokenizer

    pre = cfg.model.get("pretrained")
    if pre:
        return AutoTokenizer.from_pretrained(pre, cache_dir=_cache_dir(cfg))
    sc = cfg.model.get("scratch") or {}
    return scratch_tokenizer(texts, int(sc.get("vocab_size", 4000)), _params(cfg)["max_length"])


def build_model(cfg: DictConfig, tok, task: str, labels: list[str]):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        BertConfig,
        BertForSequenceClassification,
        BertForTokenClassification,
    )

    id2label = dict(enumerate(labels))
    label2id = {v: k for k, v in id2label.items()}
    pre = cfg.model.get("pretrained")
    if pre:
        auto = (
            AutoModelForTokenClassification if task == "ner" else AutoModelForSequenceClassification
        )
        return auto.from_pretrained(
            pre,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
            cache_dir=_cache_dir(cfg),
        )
    sc = cfg.model.get("scratch") or {}
    conf = BertConfig(
        vocab_size=len(tok),
        hidden_size=int(sc.get("hidden", 128)),
        num_hidden_layers=int(sc.get("layers", 2)),
        num_attention_heads=int(sc.get("heads", 2)),
        intermediate_size=int(sc.get("hidden", 128)) * 2,
        max_position_embeddings=512,
        pad_token_id=tok.pad_token_id,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    return (BertForTokenClassification if task == "ner" else BertForSequenceClassification)(conf)


# --------------------------------------------------------------------------- encoding
class EncodedDataset:
    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.items[i]


def encode(tok, recs: list[dict], task: str, label2id: dict[str, int], max_length: int):
    if task == "ner":
        enc = tok(
            [r["tokens"] for r in recs],
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
        )
        items = []
        for i, r in enumerate(recs):
            prev, labels = None, []
            for w in enc.word_ids(i):
                labels.append(-100 if w is None or w == prev else label2id.get(r["tags"][w], 0))
                prev = w
            items.append(
                {
                    "input_ids": enc["input_ids"][i],
                    "attention_mask": enc["attention_mask"][i],
                    "labels": labels,
                }
            )
        return EncodedDataset(items)
    enc = tok([r["text"] for r in recs], truncation=True, max_length=max_length)
    return EncodedDataset(
        [
            {
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
                "labels": label2id[r["label"]],
            }
            for i, r in enumerate(recs)
        ]
    )


# --------------------------------------------------------------------------- metrics
def entity_spans(tags: list[str]) -> set[tuple[str, int, int]]:
    """BIO / BIOES tags -> {(type, start, end)}; a stray I- starts a new entity (like seqeval)."""
    spans, start, kind = set(), None, None
    for i, t in enumerate([*tags, "O"]):
        prefix, _, ent = t.partition("-")
        continues = prefix in ("I", "E") and ent == kind and start is not None
        if start is not None and not continues:
            spans.add((kind, start, i - 1))
            start = kind = None
        if prefix in ("B", "S", "U") or (prefix in ("I", "E") and start is None):
            start, kind = i, ent
        if prefix in ("E", "S", "U") and start is not None:
            spans.add((kind, start, i))
            start = kind = None
    return spans


def ner_eval(
    true_tags: list[list[str]], pred_tags: list[list[str]], labels: list[str]
) -> EvalResult:
    """Entity-level TP / FP / FN per type (exact span + type match) + token confusion matrix."""
    ents = sorted({t.split("-", 1)[1] for t in labels if "-" in t})
    tp = dict.fromkeys(ents, 0)
    fp, fn = dict(tp), dict(tp)
    for t, p in zip(true_tags, pred_tags, strict=True):
        ts, ps = entity_spans(t), entity_spans(p)
        for e, *_ in ps & ts:
            tp[e] = tp.get(e, 0) + 1
        for e, *_ in ps - ts:
            fp[e] = fp.get(e, 0) + 1
        for e, *_ in ts - ps:
            fn[e] = fn.get(e, 0) + 1
    rows = []
    for e in sorted(set(tp) | set(fp) | set(fn)):
        p = tp[e] / (tp[e] + fp[e]) if tp[e] + fp[e] else 0.0
        r = tp[e] / (tp[e] + fn[e]) if tp[e] + fn[e] else 0.0
        rows.append(
            {
                "class": e,
                "TP": tp[e],
                "FP": fp[e],
                "FN": fn[e],
                "support": tp[e] + fn[e],
                "precision": p,
                "recall": r,
                "f1": 2 * p * r / (p + r) if p + r else 0.0,
            }
        )
    pc = pd.DataFrame(rows)
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    flat_t = [x for s in true_tags for x in s]
    flat_p = [x for s in pred_tags for x in s]
    cm = pd.crosstab(pd.Series(flat_t, name="true"), pd.Series(flat_p, name="pred"))
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    summary = {
        "precision": P,
        "recall": R,
        "f1": 2 * P * R / (P + R) if P + R else 0.0,
        "f1_macro": float(pc.f1.mean()) if len(pc) else 0.0,
        "token_accuracy": float(np.mean([a == b for a, b in zip(flat_t, flat_p, strict=True)]))
        if flat_t
        else 0.0,
        "TP": TP,
        "FP": FP,
        "FN": FN,
    }
    return EvalResult(summary=summary, per_class=pc, confusion=cm)


def _compute_metrics(task: str, labels: list[str]):
    def fn(eval_pred) -> dict[str, float]:
        logits, gold = eval_pred
        pred = np.asarray(logits).argmax(-1)
        if task == "ner":
            t, p = [], []
            for ps, gs in zip(pred, gold, strict=True):
                keep = gs != -100
                t.append([labels[g] for g in gs[keep]])
                p.append([labels[x] for x in ps[keep]])
            s = ner_eval(t, p, labels).summary
            return {"f1": s["f1"], "precision": s["precision"], "recall": s["recall"]}
        from sklearn.metrics import accuracy_score, f1_score

        return {
            "f1": f1_score(gold, pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(gold, pred),
        }

    return fn


# --------------------------------------------------------------------------- training
def _device_fp16(p: dict) -> bool:
    import torch

    return bool(p.get("fp16")) and torch.cuda.is_available()


def fit(
    cfg,
    task,
    tok,
    labels,
    train_ds,
    val_ds,
    out_dir: Path,
    p: dict,
    save: bool,
    resume_from: str | None = None,
):
    from transformers import (
        DataCollatorForTokenClassification,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model = build_model(cfg, tok, task, labels)
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(p["epochs"]),
        per_device_train_batch_size=int(p["batch_size"]),
        per_device_eval_batch_size=int(p["batch_size"]) * 2,
        gradient_accumulation_steps=int(p["grad_accum"]),
        learning_rate=float(p["lr"]),
        weight_decay=float(p["weight_decay"]),
        warmup_steps=float(p["warmup_ratio"]),  # < 1 = share of the steps
        label_smoothing_factor=float(p["label_smoothing"]),
        eval_strategy="epoch",
        save_strategy="epoch" if save else "no",
        logging_strategy="epoch",
        load_best_model_at_end=save,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2 if save else None,
        fp16=_device_fp16(p),
        report_to="none",
        seed=int(cfg.seed),
        dataloader_num_workers=0,  # Windows-safe
        disable_tqdm=not save,
    )
    collator = (
        DataCollatorForTokenClassification(tok) if task == "ner" else DataCollatorWithPadding(tok)
    )
    callbacks = []
    if save and int(p["patience"] or 0) > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(p["patience"])))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tok,
        compute_metrics=_compute_metrics(task, labels),
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume_from)
    return trainer


def last_checkpoint(final_dir: Path) -> str | None:
    """Newest checkpoint-N folder of an interrupted final training."""
    cks = [c for c in Path(final_dir).glob("checkpoint-*") if (c / "trainer_state.json").exists()]
    return str(max(cks, key=lambda c: int(c.name.split("-")[-1]))) if cks else None


def epoch_history(log_history: list[dict]) -> pd.DataFrame:
    rows: dict[float, dict] = {}
    for h in log_history:
        if "epoch" not in h:
            continue
        r = rows.setdefault(round(float(h["epoch"]), 3), {"epoch": round(float(h["epoch"]), 3)})
        for k, v in h.items():
            if isinstance(v, int | float) and k not in ("epoch", "step", "total_flos"):
                r[k.replace("eval_", "val_")] = v
    return pd.DataFrame(sorted(rows.values(), key=lambda r: r["epoch"]))


# --------------------------------------------------------------------------- inference
def load_trained(model_dir: Path):
    from transformers import (
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    import json

    conf = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    archs = " ".join(conf.get("architectures") or [])
    auto = (
        AutoModelForTokenClassification
        if "TokenClassification" in archs
        else AutoModelForSequenceClassification
    )
    model = auto.from_pretrained(str(model_dir))
    task = "ner" if "TokenClassification" in archs else "classify"
    return tok, model, task


def infer(model, tok, task: str, inputs: list, max_length: int = 256, batch: int = 32):
    """classify: inputs = texts -> probs (n, classes). ner: inputs = token lists -> tag lists."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    out: list = []
    for i in range(0, len(inputs), batch):
        chunk = inputs[i : i + batch]
        enc = tok(
            chunk,
            is_split_into_words=task == "ner",
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits
        if task == "ner":
            pred = logits.argmax(-1).cpu().numpy()
            for j, words in enumerate(chunk):
                tags = ["O"] * len(words)
                seen = set()
                for pos, w in enumerate(enc.word_ids(j)):
                    if w is not None and w not in seen:
                        tags[w] = id2label[int(pred[j, pos])]
                        seen.add(w)
                out.append(tags)
        else:
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out) if task != "ner" and out else out


def spans_text(tokens: list[str], tags: list[str], joiner: str = " ") -> list[tuple[str, str]]:
    return [
        (e, joiner.join(tokens[s : t + 1]))
        for e, s, t in sorted(entity_spans(tags), key=lambda x: x[1])
    ]


# --------------------------------------------------------------------------- MLflow model
class TextPyfunc(mlflow.pyfunc.PythonModel):
    """Registry model: DataFrame with a 'text' column -> label + confidence (or entities)."""

    def load_context(self, context):
        self.tok, self.model, self.task = load_trained(Path(context.artifacts["model"]))

    def predict(self, context, model_input, params=None):
        texts = [str(t) for t in model_input["text"]]
        if self.task == "ner":
            toks = [split_words(t) for t in texts]
            tags = infer(self.model, self.tok, "ner", toks)
            return pd.DataFrame(
                {"entities": [str(spans_text(w, t)) for w, t in zip(toks, tags, strict=True)]}
            )
        probs = infer(self.model, self.tok, "classify", texts)
        names = self.model.config.id2label
        return pd.DataFrame(
            {"label": [names[int(i)] for i in probs.argmax(1)], "confidence": probs.max(1)}
        )


def split_words(text: str) -> list[str]:
    """Words for NER on raw text: PyThaiNLP (Thai + mixed) if installed, else whitespace."""
    try:
        from pythainlp.tokenize import word_tokenize

        return [w for w in word_tokenize(text, keep_whitespace=False) if w.strip()]
    except ImportError:
        return text.split()


# --------------------------------------------------------------------------- main
def train_text(cfg: DictConfig) -> dict[str, float]:
    set_seed(cfg.seed)
    processed = Path(cfg.paths.processed)
    meta = load_meta(processed)
    task, labels = meta["task"], list(meta["class_names"])
    label2id = {v: i for i, v in enumerate(labels)}
    train_r, val_r = load_split(processed, "train"), load_split(processed, "val")
    eval_split = "test" if meta["n_rows"].get("test") else "val"
    test_r = load_split(processed, eval_split)
    if not val_r:
        val_r = test_r
    tune = cfg.get("tune") or {}
    space = (
        tune.get("search_space") if tune.get("name") not in (None, "default") else None
    ) or cfg.model.get("search_space")
    space = {k: v for k, v in (space or {}).items() if k in PARAMS}  # YOLO presets don't apply
    hpo = cfg.hpo
    p0 = _params(cfg)
    runs_dir = ensure_dir(Path(cfg.paths.models) / cfg.data.name / cfg.model.name)
    final_dir, model_dir, tok_dir = runs_dir / "final", runs_dir / "model", runs_dir / "tokenizer"
    out_dir = ensure_dir(cfg.paths.run_reports)

    resume = bool(cfg.get("resume"))
    prev = runstate.start(
        runs_dir,
        resume,
        dataset=cfg.data.name,
        model=cfg.model.name,
        task=task,
        tune=str(tune.get("name") or "default"),
    )
    resume_ckpt = last_checkpoint(final_dir) if resume and prev.get("stage") == "final" else None
    skip_fit = resume and prev.get("stage") == "evaluate" and (model_dir / "config.json").exists()
    if resume:
        log.info(
            "Resume: %s",
            "evaluating the trained model"
            if skip_fit
            else f"final training from {resume_ckpt}"
            if resume_ckpt
            else "keeping finished Optuna trials",
        )

    from transformers import AutoTokenizer

    if resume and tok_dir.exists():  # same tokenizer as before the stop
        tok = AutoTokenizer.from_pretrained(str(tok_dir))
    else:
        texts = [" ".join(r["tokens"]) if task == "ner" else r["text"] for r in train_r]
        tok = build_tokenizer(cfg, texts)
        tok.save_pretrained(str(tok_dir))
    maxlen = int(p0["max_length"])
    train_ds = encode(tok, train_r, task, label2id, maxlen)
    val_ds = encode(tok, val_r, task, label2id, maxlen)

    experiment_id = setup_mlflow(cfg)
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=f"{cfg.model.name}-{datetime.now():%Y%m%d-%H%M%S}",
        log_system_metrics=bool(cfg.mlflow.get("system_metrics", True)),
    ) as parent:
        mlflow.set_tags(
            {
                "dataset": cfg.data.name,
                "model": cfg.model.name,
                "task": task,
                "backend": "transformers",
                "stage": "train",
                "resumed": str(resume),
                "pretrained": str(cfg.model.get("pretrained") or "scratch"),
            }
        )
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config/config.yaml")
        mlflow.log_dict(meta, "config/data_meta.json")
        raw = Path(cfg.paths.raw) / cfg.data.name
        data_files = [f for f in raw.rglob("*") if f.is_file()] if raw.exists() else []
        splits = [processed / f"{s}.jsonl" for s in ("train", "val", "test")]
        lineage_info = lineage.log_context(data_files, raw, extra_artifacts=splits)

        # ---------- 1. Optuna ----------
        study, best_params, val_summary = None, {}, None
        if hpo.enabled and hpo.n_trials > 0 and space and not skip_fit:
            runstate.write_state(runs_dir, stage="hpo")
            hpo_epochs = int(cfg.model.get("hpo_epochs") or max(1, p0["epochs"] // 2))

            def objective(trial: optuna.Trial) -> float:
                params = suggest_params(trial, space)
                with mlflow.start_run(
                    experiment_id=experiment_id, run_name=f"trial-{trial.number}", nested=True
                ):
                    log_params(params)
                    p = _params(cfg, {**params, "epochs": hpo_epochs})
                    t0 = time.perf_counter()
                    tr = fit(
                        cfg,
                        task,
                        tok,
                        labels,
                        train_ds,
                        val_ds,
                        runs_dir / "hpo" / f"trial_{trial.number}",
                        p,
                        save=False,
                    )
                    res = {
                        k.replace("eval_", ""): float(v)
                        for k, v in tr.evaluate().items()
                        if isinstance(v, int | float)
                    }
                    res["fit_seconds"] = time.perf_counter() - t0
                    mlflow.log_metrics({f"val_{k}": v for k, v in res.items()})
                    trial.set_user_attr("val_summary", res)
                    shutil.rmtree(runs_dir / "hpo" / f"trial_{trial.number}", ignore_errors=True)
                    return res["f1"]

            log.info("Optuna: %d trials x %d epochs, optimising val f1", hpo.n_trials, hpo_epochs)
            study = run_study(
                OmegaConf.merge(hpo, {"direction": "maximize"}), cfg.seed, objective, resume=resume
            )
            best_params = suggest_params(optuna.trial.FixedTrial(study.best_params), space)
            val_summary = study.best_trial.user_attrs.get("val_summary")
            mlflow.log_metric("best_val_f1", study.best_value)
            log.info("Best val f1 = %.4f  params=%s", study.best_value, best_params)

        # ---------- 2. Final training ----------
        p = _params(cfg, best_params)
        t0 = time.perf_counter()
        history = pd.DataFrame()
        if skip_fit:
            from transformers import TrainerState

            ck = last_checkpoint(final_dir)
            if ck:
                history = epoch_history(
                    TrainerState.load_from_json(str(Path(ck) / "trainer_state.json")).log_history
                )
        else:
            runstate.write_state(runs_dir, stage="final")
            trainer = fit(
                cfg,
                task,
                tok,
                labels,
                train_ds,
                val_ds,
                final_dir,
                p,
                save=True,
                resume_from=resume_ckpt,
            )
            if model_dir.exists():
                shutil.rmtree(model_dir)
            trainer.save_model(str(model_dir))  # best epoch (load_best_model_at_end)
            tok.save_pretrained(str(model_dir))
            history = epoch_history(trainer.state.log_history)
            if resume_ckpt:
                mlflow.set_tag("resumed_from", resume_ckpt)
        fit_seconds = time.perf_counter() - t0
        runstate.write_state(runs_dir, stage="evaluate")
        log_params({f"model.{k}": v for k, v in p.items()})
        mlflow.log_params(
            {
                "pretrained": str(cfg.model.get("pretrained") or "scratch"),
                "tune": str(tune.get("name") or "default"),
                "hpo.n_trials": hpo.n_trials if hpo.enabled and space else 0,
            }
        )
        for _, row in history.iterrows():
            step = max(1, round(row["epoch"]))
            mlflow.log_metrics(
                {f"epoch/{k}": float(v) for k, v in row.items() if k != "epoch" and pd.notna(v)},
                step=step,
            )

        # ---------- 3. Evaluation ----------
        tok_e, model, _ = load_trained(model_dir)
        if task == "ner":
            pred_tags = infer(model, tok_e, "ner", [r["tokens"] for r in test_r], maxlen)
            result = ner_eval([r["tags"] for r in test_r], pred_tags, labels)
            preds = pd.DataFrame(
                {
                    "text": [" ".join(r["tokens"]) for r in test_r],
                    "true entities": [
                        "; ".join(f"{e}: {t}" for e, t in spans_text(r["tokens"], r["tags"]))
                        for r in test_r
                    ],
                    "predicted entities": [
                        "; ".join(f"{e}: {t}" for e, t in spans_text(r["tokens"], pt))
                        for r, pt in zip(test_r, pred_tags, strict=True)
                    ],
                }
            )
            preds.insert(
                1,
                "correct",
                [
                    entity_spans(r["tags"]) == entity_spans(pt)
                    for r, pt in zip(test_r, pred_tags, strict=True)
                ],
            )
        else:
            probs = infer(model, tok_e, "classify", [r["text"] for r in test_r], maxlen)
            y_true = np.array([label2id[r["label"]] for r in test_r])
            y_pred = probs.argmax(1)
            result = evaluate(
                y_true, y_pred, probs, labels, cfg.get("evaluation", {}).get("average", "macro")
            )
            preds = pd.DataFrame(
                {
                    "text": [r["text"][:500] for r in test_r],
                    "y_true": [labels[i] for i in y_true],
                    "y_pred": [labels[i] for i in y_pred],
                }
            )
            preds.insert(3, "correct", preds.y_true == preds.y_pred)
            preds.insert(4, "confidence", probs.max(1))
            if len(labels) <= 20:
                for i, c in enumerate(labels):
                    preds[f"proba_{c}"] = probs[:, i]
            if any("file" in r for r in test_r):
                preds.insert(0, "file", [r.get("file", "") for r in test_r])
        result.summary["fit_seconds"] = fit_seconds
        mlflow.log_metrics(
            {
                f"test_{k}": float(v)
                for k, v in result.summary.items()
                if isinstance(v, int | float) and math.isfinite(float(v))
            }
        )
        for _, row in result.per_class.iterrows():
            safe = "".join(ch if ch.isalnum() else "_" for ch in str(row["class"]))
            mlflow.log_metrics(
                {
                    f"test_{safe}_{m}": float(row[m])
                    for m in ("TP", "FP", "FN", "precision", "recall", "f1")
                }
            )

        # ---------- 4. Reports ----------
        write_json(
            {
                "test": result.summary,
                "val_best": val_summary,
                "best_params": best_params,
                "task": task,
            },
            out_dir / "metrics.json",
        )
        result.per_class.to_csv(out_dir / "per_class.csv", index=False)
        trials_df = trials_dataframe(study) if study else None
        importances = param_importances(study) if study and len(study.trials) > 1 else {}
        report_path = None
        if cfg.report.excel:
            run_info = {
                "dataset": cfg.data.name,
                "model": cfg.model.name,
                "pretrained": str(cfg.model.get("pretrained") or "tiny BERT from scratch"),
                "task": {
                    "ner": "NER – pull out entities (scored per entity)",
                    "classify": "text classification",
                }[task],
                "classes" if task == "classify" else "entities": ", ".join(
                    labels if task == "classify" else meta.get("entities", [])
                ),
                "rows train/val/test": "/".join(
                    str(meta["n_rows"].get(s, 0)) for s in ("train", "val", "test")
                ),
                "evaluated on": eval_split,
                "hpo_metric": "f1 (validation)",
                "hpo_trials": len(study.trials) if study else 0,
                "mlflow_run_id": parent.info.run_id,
                "created": datetime.now().isoformat(timespec="seconds"),
                **lineage_info,
            }
            report_path = write_report(
                out_dir / cfg.report.filename,
                run_info,
                result,
                best_params,
                trials_df,
                importances,
                preds,
                flatten_dict(OmegaConf.to_container(cfg, resolve=True)),
                val_summary,
                history=history,
            )
            log.info("Excel report: %s", report_path)
        for f in [out_dir / "metrics.json", out_dir / "per_class.csv", report_path]:
            if f:
                mlflow.log_artifact(str(f), artifact_path="reports")
        if trials_df is not None:
            trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)
            mlflow.log_artifact(str(out_dir / "optuna_trials.csv"), artifact_path="reports")

        # ---------- 5. Model ----------
        mlflow.pyfunc.log_model(
            name="model",
            python_model=TextPyfunc(),
            artifacts={"model": str(model_dir)},
            pip_requirements=["transformers", "torch", "pandas", "pythainlp"],
            metadata={"class_names": labels, "dataset": cfg.data.name, "task": task},
            registered_model_name=(
                cfg.mlflow.registered_model_name if cfg.mlflow.register_model else None
            ),
        )

    s = result.summary
    lineage.record(
        "train",
        history=lineage.history_path(cfg),
        dataset=cfg.data.name,
        model=cfg.model.name,
        task=f"text-{task}",
        tune=str(tune.get("name") or "default"),
        epochs=p["epochs"],
        hpo_trials=len(study.trials) if study else 0,
        metrics={k: v for k, v in s.items() if isinstance(v, int | float)},
        model_file=str(model_dir),
        report=str(report_path) if report_path else None,
        mlflow_run_id=parent.info.run_id,
        data_sha256=lineage_info.get("data sha256"),
        resumed=resume,
    )
    runstate.write_state(
        runs_dir, status="finished", stage="done", mlflow_run_id=parent.info.run_id
    )
    keys = (
        ["precision", "recall", "f1"]
        if task == "ner"
        else ["accuracy", "f1", "precision", "recall"]
    )
    log.info(
        "TEST(%s) %s  %s  TP=%s FP=%s FN=%s -> run %s",
        eval_split,
        task,
        "  ".join(f"{k}={s[k]:.4f}" for k in keys),
        s["TP"],
        s["FP"],
        s["FN"],
        parent.info.run_id,
    )
    return s


# --------------------------------------------------------------------------- predict
def _collect_inputs(cfg: DictConfig) -> list[tuple[str, str]]:
    """(source, text) pairs from +text=..., a table, a document / image, or a folder of them."""
    from ml_training.text_data import (
        DOC_EXTS,
        IMG_EXTS,
        TABLE_EXTS,
        TEXT_NAMES,
        extract_text,
        find_column,
        read_table,
    )

    if cfg.get("text"):
        return [("text", str(cfg.text))]
    src = Path(str(cfg.input))
    if not src.is_absolute():
        src = Path(cfg.paths.root) / src
    langs = list(cfg.data.get("ocr_langs") or ["th", "en"])
    if src.is_file() and src.suffix.lower() in TABLE_EXTS:
        df = read_table(src)
        col = find_column(df, cfg.data.get("text_column"), TEXT_NAMES, "text")
        return [(f"{src.name}:{i + 2}", str(t)) for i, t in enumerate(df[col]) if pd.notna(t)]
    files = (
        [src]
        if src.is_file()
        else sorted(f for f in src.rglob("*") if f.suffix.lower() in DOC_EXTS | IMG_EXTS)
    )
    out = []
    for f in files:
        try:
            text = extract_text(f, langs)
        except ImportError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("Skipping %s: %s", f, e)
            continue
        if text.strip():
            out.append((str(f.relative_to(src) if src.is_dir() else f.name), text))
    return out


def predict_text(cfg: DictConfig) -> Path:
    from ml_training.modeling.predict import track_prediction

    model_dir = Path(
        cfg.get("weights") or Path(cfg.paths.models) / cfg.data.name / cfg.model.name / "model"
    )
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(f"No trained model in {model_dir} – train first ({RUN} text ...)")
    tok, model, task = load_trained(model_dir)
    items = _collect_inputs(cfg)
    if not items:
        raise ValueError("Nothing to predict: no readable text found in the input")
    maxlen = int(_params(cfg)["max_length"])
    out_dir = ensure_dir(Path(cfg.paths.run_reports) / "predict")
    if task == "ner":
        words = [split_words(t) for _, t in items]
        tags = infer(model, tok, "ner", words, maxlen)
        joiner = " " if all(" " in t for _, t in items[:5]) else ""
        rows = [
            {"source": s, "class": e, "text": t}
            for (s, _), w, tg in zip(items, words, tags, strict=True)
            for e, t in spans_text(w, tg, joiner)
        ]
        df = pd.DataFrame(rows, columns=["source", "class", "text"])
        per_doc = (
            df.groupby("source").apply(
                lambda g: "; ".join(f"{a}: {b}" for a, b in zip(g["class"], g.text, strict=True)),
                include_groups=False,
            )
            if len(df)
            else pd.Series(dtype=str)
        )
        pd.DataFrame(
            {"source": [s for s, _ in items], "entities": [per_doc.get(s, "") for s, _ in items]}
        ).to_csv(out_dir / "documents.csv", index=False, encoding="utf-8-sig")
    else:
        probs = infer(model, tok, "classify", [t for _, t in items], maxlen)
        names = {int(k): v for k, v in model.config.id2label.items()}
        top = np.argsort(-probs, axis=1)[:, :3]
        df = pd.DataFrame(
            {
                "source": [s for s, _ in items],
                "prediction": [names[int(i)] for i in top[:, 0]],
                "confidence": probs.max(1),
                "top3": [
                    ", ".join(f"{names[int(i)]} {probs[r, i]:.2f}" for i in top[r])
                    for r in range(len(items))
                ],
                "text": [t[:300].replace("\n", " ") for _, t in items],
            }
        )
    out_path = Path(cfg.get("output") or out_dir / "predictions.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("%d %s -> %s", len(df), "entities" if task == "ner" else "predictions", out_path)
    info = {
        "input": str(cfg.get("input") or "text"),
        "weights": str(model_dir),
        "task": task,
        "weights_sha256": lineage.fingerprint(model_dir.glob("*.safetensors"))["sha256"],
    }
    track_prediction(cfg, info, out_path, df)
    return out_path
