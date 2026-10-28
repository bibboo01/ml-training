"""Vision backend beyond YOLO: image classification (timm) and semantic segmentation (SegFormer).

Same workflow as the other backends:
Optuna search -> final training -> evaluation (TP/FP/FN per class; IoU / Dice per class for
segmentation) -> MLflow -> Excel, with lineage, history and resume.

    uv run mlt-train data=shapes model=img_convnext
    uv run mlt-train data=shapes-seg model=seg_b0
    uv run mlt-predict data=shapes model=img_convnext +input=D:/photos

Classifiers: any timm model (model.arch: convnext_tiny, efficientnet_b0, vit_base_patch16_224,
resnet50 ... ImageNet weights downloaded once). Segmentation: SegFormer (model.hf_model =
nvidia/mit-b0 ..., or null = small SegFormer from scratch), or arch: simple_unet (small U-Net
from scratch, seg_tiny). simple_cnn (img_tiny) and simple_unet need no download.
"""

from __future__ import annotations

import json
import math
import os
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
from ml_training.metrics import EvalResult, evaluate
from ml_training.reporting.excel import write_report
from ml_training.tracking import log_params, setup_mlflow
from ml_training.utils import ensure_dir, flatten_dict, log, set_seed, write_json
from ml_training.vision_data import IMG_EXTS, load_meta, load_split

PARAMS = {
    "epochs": 20,
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "imgsz": 224,
    "patience": 5,
    "amp": True,
    "label_smoothing": 0.1,
    "warmup_epochs": 1,
    "workers": 0,
}
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def _params(cfg: DictConfig, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    given = cfg.model.get("params")
    given = OmegaConf.to_container(given, resolve=True) if given is not None else {}
    out = {k: given[k] if given.get(k) is not None else v for k, v in PARAMS.items()}
    out.update({k: v for k, v in (overrides or {}).items() if k in PARAMS})
    return out


def _hf_cache(cfg: DictConfig) -> str:
    path = str(ensure_dir(Path(cfg.paths.models) / "pretrained" / "hf"))
    os.environ.setdefault("HF_HOME", path)  # timm downloads ImageNet weights from the HF hub
    return path


# --------------------------------------------------------------------------- data
class ImageDataset:
    def __init__(
        self,
        recs: list[dict],
        task: str,
        label2id: dict[str, int],
        imgsz: int,
        train: bool,
        norm=IMAGENET,
    ) -> None:
        from torchvision.transforms import v2

        self.recs, self.task, self.label2id = recs, task, label2id
        tail = [
            v2.ToImage(),
            v2.ToDtype(__import__("torch").float32, scale=True),
            v2.Normalize(*norm),
        ]
        if task == "segment":
            aug = (
                [
                    v2.RandomResizedCrop(imgsz, scale=(0.5, 1.0), antialias=True),
                    v2.RandomHorizontalFlip(),
                ]
                if train
                else [v2.Resize((imgsz, imgsz), antialias=True)]
            )
        else:
            aug = (
                [
                    v2.RandomResizedCrop(imgsz, scale=(0.6, 1.0), antialias=True),
                    v2.RandomHorizontalFlip(),
                    v2.ColorJitter(0.2, 0.2, 0.2),
                ]
                if train
                else [v2.Resize((imgsz, imgsz), antialias=True)]
            )
        self.tf = v2.Compose(aug)
        self.tail = v2.Compose(tail)

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image
        from torchvision import tv_tensors

        r = self.recs[i]
        img = Image.open(r["image"]).convert("RGB")
        if self.task == "segment":
            mask = tv_tensors.Mask(torch.from_numpy(np.array(Image.open(r["mask"]))).long())
            img, mask = self.tf(tv_tensors.Image(np.array(img).transpose(2, 0, 1)), mask)
            return self.tail(img), mask.squeeze(0).long() if mask.dim() == 3 else mask.long()
        return self.tail(self.tf(img)), self.label2id[r["label"]]


def loader(ds: ImageDataset, batch: int, shuffle: bool, workers: int):
    import torch

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


# --------------------------------------------------------------------------- model
def simple_cnn(n_classes: int, width: int = 32):
    """4 conv blocks – learns small datasets from scratch in seconds (img_tiny); no download."""
    import torch.nn as nn

    def block(i: int, o: int):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )

    w = width
    return nn.Sequential(
        block(3, w),
        block(w, w * 2),
        block(w * 2, w * 4),
        block(w * 4, w * 4),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(w * 4, n_classes),
    )


def simple_unet(n_classes: int, width: int = 32):
    """Small U-Net (3 levels) – segmentation from scratch on small data (seg_tiny); no download."""
    import torch
    import torch.nn as nn

    def conv(i: int, o: int):
        return nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1),
            nn.BatchNorm2d(o),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            w = width
            self.e1, self.e2, self.e3 = conv(3, w), conv(w, w * 2), conv(w * 2, w * 4)
            self.mid = conv(w * 4, w * 8)
            self.u3, self.d3 = nn.ConvTranspose2d(w * 8, w * 4, 2, 2), conv(w * 8, w * 4)
            self.u2, self.d2 = nn.ConvTranspose2d(w * 4, w * 2, 2, 2), conv(w * 4, w * 2)
            self.u1, self.d1 = nn.ConvTranspose2d(w * 2, w, 2, 2), conv(w * 2, w)
            self.head, self.pool = nn.Conv2d(w, n_classes, 1), nn.MaxPool2d(2)

        def forward(self, x):
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            m = self.mid(self.pool(e3))
            d3 = self.d3(torch.cat([self.u3(m), e3], 1))
            d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
            return self.head(self.d1(torch.cat([self.u1(d2), e1], 1)))

    return UNet()


def create_classifier(arch: str, n_classes: int, pretrained: bool = False, width: int = 32):
    if arch == "simple_cnn":
        return simple_cnn(n_classes, width)
    import timm

    return timm.create_model(arch, pretrained=pretrained, num_classes=n_classes)


def build_model(cfg: DictConfig, task: str, names: list[str]):
    cache = _hf_cache(cfg)
    if task == "segment":
        arch = cfg.model.get("arch")
        if arch == "simple_unet":
            return simple_unet(len(names), int(cfg.model.get("width") or 32))
        if arch:
            raise ValueError(
                f"Model {cfg.model.name} is a classifier – use a seg_* model for segmentation data"
            )
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        id2label = dict(enumerate(names))
        label2id = {v: k for k, v in id2label.items()}
        hf = cfg.model.get("hf_model")
        if hf:
            return SegformerForSemanticSegmentation.from_pretrained(
                hf,
                num_labels=len(names),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
                cache_dir=cache,
            )
        sc = cfg.model.get("scratch") or {}
        w = int(sc.get("width", 32))
        conf = SegformerConfig(
            num_labels=len(names),
            id2label=id2label,
            label2id=label2id,
            hidden_sizes=[w, w * 2, w * 4, w * 8],
            depths=[1, 1, 1, 1],
            decoder_hidden_size=w * 4,
        )
        return SegformerForSemanticSegmentation(conf)
    if not cfg.model.get("arch"):
        raise ValueError(
            f"Model {cfg.model.name} is for segmentation – use an img_* model for classification"
        )
    return create_classifier(
        str(cfg.model.arch),
        len(names),
        bool(cfg.model.get("pretrained", True)),
        int(cfg.model.get("width") or 32),
    )


def norm_for(model, task: str):
    if task == "segment":
        return IMAGENET
    try:
        import timm

        c = timm.data.resolve_data_config({}, model=model)
        return tuple(c["mean"]), tuple(c["std"])
    except Exception:  # noqa: BLE001
        return IMAGENET


def forward(model, x, task: str):
    if task == "segment" and hasattr(model, "config"):  # SegFormer: logits at 1/4 size
        import torch.nn.functional as F

        logits = model(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return model(x)


# --------------------------------------------------------------------------- metrics
def seg_confusion(pred, gold, n: int) -> np.ndarray:
    keep = gold < n
    return np.bincount(n * gold[keep] + pred[keep], minlength=n * n).reshape(n, n)


def seg_eval(cm: np.ndarray, names: list[str]) -> EvalResult:
    """Per class pixel TP / FP / FN / TN, IoU, Dice (= F1), precision, recall + mIoU."""
    total = cm.sum()
    rows = []
    for i, name in enumerate(names):
        tp = int(cm[i, i])
        fp, fn = int(cm[:, i].sum() - tp), int(cm[i, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "class": name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "support": tp + fn,
                "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
                "precision": p,
                "recall": r,
                "f1": 2 * p * r / (p + r) if p + r else 0.0,
            }
        )
    pc = pd.DataFrame(rows)
    present = pc[pc.support > 0]
    summary = {
        "miou": float(present.iou.mean()) if len(present) else 0.0,
        "pixel_accuracy": float(np.trace(cm) / total) if total else 0.0,
        "f1": float(present.f1.mean()) if len(present) else 0.0,  # mean Dice
        "precision": float(present.precision.mean()) if len(present) else 0.0,
        "recall": float(present.recall.mean()) if len(present) else 0.0,
        "TP": int(pc.TP.sum()),
        "FP": int(pc.FP.sum()),
        "FN": int(pc.FN.sum()),
    }
    conf = pd.DataFrame(cm, index=names, columns=names)
    return EvalResult(summary=summary, per_class=pc, confusion=conf)


def run_eval(model, dl, task: str, n: int, dev):
    """classify -> (probs, gold); segment -> (confusion matrix, per-image IoU list)."""
    import torch

    model.eval()
    probs, gold, per_img = [], [], []
    cm = np.zeros((n, n), np.int64)
    with torch.no_grad():
        for x, y in dl:
            out = forward(model, x.to(dev), task).float()
            if task == "segment":
                pred = out.argmax(1).cpu().numpy()
                y = y.numpy()
                for pi, gi in zip(pred, y, strict=True):
                    c = seg_confusion(pi.ravel(), gi.ravel(), n)
                    cm += c
                    inter, union = np.diag(c), c.sum(0) + c.sum(1) - np.diag(c)
                    per_img.append(
                        float(np.mean(inter[union > 0] / union[union > 0]))
                        if (union > 0).any()
                        else 1.0
                    )
            else:
                probs.append(torch.softmax(out, 1).cpu().numpy())
                gold.append(np.asarray(y))
    if task == "segment":
        return cm, per_img
    return np.concatenate(probs), np.concatenate(gold)


def val_score(model, dl, task: str, n: int, dev) -> float:
    if task == "segment":
        cm, _ = run_eval(model, dl, task, n, dev)
        return seg_eval(cm, [str(i) for i in range(n)]).summary["miou"]
    probs, gold = run_eval(model, dl, task, n, dev)
    from sklearn.metrics import f1_score

    return float(f1_score(gold, probs.argmax(1), average="macro", zero_division=0))


# --------------------------------------------------------------------------- training loop
def fit(
    cfg,
    task: str,
    names: list[str],
    train_recs,
    val_recs,
    p: dict,
    out_dir: Path | None,
    resume: bool = False,
):
    """Train; with out_dir: last.pt each epoch (resume) + best.pt. -> (model, history, best)."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(int(cfg.seed))
    model = build_model(cfg, task, names).to(dev)
    norm = norm_for(model, task)
    l2i = {v: i for i, v in enumerate(names)}
    imgsz, bs = int(p["imgsz"]), int(p["batch_size"])
    train_dl = loader(
        ImageDataset(train_recs, task, l2i, imgsz, True, norm), bs, True, int(p["workers"])
    )
    val_dl = loader(
        ImageDataset(val_recs, task, l2i, imgsz, False, norm), bs, False, int(p["workers"])
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"])
    )
    epochs = int(p["epochs"])
    steps, warm = epochs * max(1, len(train_dl)), int(p["warmup_epochs"]) * max(1, len(train_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / max(1, warm))
            * 0.5
            * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))
        ),
    )
    amp = bool(p["amp"]) and dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=float(p["label_smoothing"]) if task == "classify" else 0.0, ignore_index=255
    )
    start, best, bad, history, best_state = 0, -1.0, 0, [], None
    last = out_dir / "last.pt" if out_dir else None
    if resume and last and last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start, best, bad, history = ck["epoch"] + 1, ck["best"], ck["bad"], ck["history"]
        if (out_dir / "best.pt").exists():
            best_state = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
        log.info("Resuming from epoch %d/%d (best val %.4f)", start + 1, epochs, best)
    for epoch in range(start, epochs):
        model.train()
        t0, total, count = time.perf_counter(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                loss = loss_fn(forward(model, x, task), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(x)
            count += len(x)
        score = val_score(model, val_dl, task, len(names), dev)
        metric = "val_miou" if task == "segment" else "val_f1"
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / max(1, count),
                metric: score,
                "lr": opt.param_groups[0]["lr"],
                "seconds": time.perf_counter() - t0,
            }
        )
        log.info(
            "epoch %d/%d  loss %.4f  %s %.4f",
            epoch + 1,
            epochs,
            total / max(1, count),
            metric,
            score,
        )
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if out_dir:
                torch.save(best_state, out_dir / "best.pt")
        else:
            bad += 1
        if last:
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best": best,
                    "bad": bad,
                    "history": history,
                },
                last,
            )
        if int(p["patience"] or 0) and bad >= int(p["patience"]):
            log.info("Early stop: no better val score for %d epochs", bad)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best


# --------------------------------------------------------------------------- save / load
def save_trained(
    model, cfg: DictConfig, task: str, names: list[str], p: dict, model_dir: Path
) -> None:
    import shutil

    import torch

    if model_dir.exists():
        shutil.rmtree(model_dir)
    ensure_dir(model_dir)
    meta = {
        "task": task,
        "class_names": names,
        "imgsz": int(p["imgsz"]),
        "norm": [list(x) for x in norm_for(model, task)],
    }
    if task == "segment" and hasattr(model, "save_pretrained"):
        model.save_pretrained(str(model_dir))
        meta["framework"] = "segformer"
    elif task == "segment":
        torch.save(model.state_dict(), model_dir / "weights.pt")
        meta |= {
            "framework": "unet",
            "arch": "simple_unet",
            "width": int(cfg.model.get("width") or 32),
        }
    else:
        torch.save(model.state_dict(), model_dir / "weights.pt")
        meta |= {
            "framework": "timm",
            "arch": str(cfg.model.arch),
            "width": int(cfg.model.get("width") or 32),
        }
    (model_dir / "vision.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_trained(model_dir: Path):
    import torch

    meta = json.loads((Path(model_dir) / "vision.json").read_text(encoding="utf-8"))
    if meta["framework"] == "segformer":
        from transformers import SegformerForSemanticSegmentation

        model = SegformerForSemanticSegmentation.from_pretrained(str(model_dir))
    elif meta["framework"] == "unet":
        model = simple_unet(len(meta["class_names"]), int(meta.get("width", 32)))
        model.load_state_dict(torch.load(Path(model_dir) / "weights.pt", map_location="cpu"))
    else:
        model = create_classifier(
            meta["arch"], len(meta["class_names"]), width=int(meta.get("width", 32))
        )
        model.load_state_dict(torch.load(Path(model_dir) / "weights.pt", map_location="cpu"))
    return model.eval(), meta


def predict_images(model, meta: dict, paths: list[Path], batch: int = 16):
    """classify -> probs (n, classes); segment -> list of class-index masks at original size."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms import v2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    s = int(meta["imgsz"])
    tf = v2.Compose(
        [
            v2.Resize((s, s), antialias=True),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(*meta["norm"]),
        ]
    )
    out: list = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i : i + batch]]
        x = torch.stack([tf(im) for im in imgs]).to(dev)
        with torch.no_grad():
            logits = forward(model, x, meta["task"]).float()
        if meta["task"] == "segment":
            for im, lg in zip(imgs, logits, strict=True):
                up = F.interpolate(
                    lg[None], size=(im.height, im.width), mode="bilinear", align_corners=False
                )
                out.append(up.argmax(1)[0].cpu().numpy().astype(np.uint8))
        else:
            out.append(torch.softmax(logits, 1).cpu().numpy())
    return out if meta["task"] == "segment" else np.concatenate(out)


PALETTE = np.array(
    [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
    ],
    np.uint8,
)


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[mask % len(PALETTE)]


def overlay(path: Path, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    from PIL import Image

    img = np.asarray(Image.open(path).convert("RGB"), np.float32)
    col = colorize(mask).astype(np.float32)
    keep = (mask > 0)[..., None]
    return np.where(keep, img * (1 - alpha) + col * alpha, img).astype(np.uint8)


class VisionPyfunc(mlflow.pyfunc.PythonModel):
    """Registry model: DataFrame with an image column -> label + confidence (or % per class)."""

    def load_context(self, context):
        self.model, self.meta = load_trained(Path(context.artifacts["model"]))

    def predict(self, context, model_input, params=None):
        paths = [Path(p) for p in model_input["image"]]
        names = self.meta["class_names"]
        res = predict_images(self.model, self.meta, paths)
        if self.meta["task"] == "segment":
            return pd.DataFrame(
                [{f"pct_{n}": float((m == i).mean()) for i, n in enumerate(names)} for m in res]
            )
        return pd.DataFrame({"label": [names[i] for i in res.argmax(1)], "confidence": res.max(1)})


# --------------------------------------------------------------------------- main
def train_vision(cfg: DictConfig) -> dict[str, float]:
    import torch

    set_seed(cfg.seed)
    processed = Path(cfg.paths.processed)
    meta = load_meta(processed)
    task, names = meta["task"], list(meta["class_names"])
    train_r, val_r = load_split(processed, "train"), load_split(processed, "val")
    eval_split = "test" if meta["n_rows"].get("test") else "val"
    test_r = load_split(processed, eval_split)
    val_r = val_r or test_r
    tune = cfg.get("tune") or {}
    space = (
        tune.get("search_space") if tune.get("name") not in (None, "default") else None
    ) or cfg.model.get("search_space")
    space = {k: v for k, v in (space or {}).items() if k in PARAMS}
    hpo = cfg.hpo
    metric = "miou" if task == "segment" else "f1"
    runs_dir = ensure_dir(Path(cfg.paths.models) / cfg.data.name / cfg.model.name)
    final_dir, model_dir = ensure_dir(runs_dir / "final"), runs_dir / "model"
    out_dir = ensure_dir(cfg.paths.run_reports)
    tune_name = str(tune.get("name") or "default")

    resume = bool(cfg.get("resume"))
    prev = runstate.start(
        runs_dir, resume, dataset=cfg.data.name, model=cfg.model.name, task=task, tune=tune_name
    )
    resume_final = resume and prev.get("stage") == "final" and (final_dir / "last.pt").exists()
    skip_fit = resume and prev.get("stage") == "evaluate" and (model_dir / "vision.json").exists()
    if not resume_final and not skip_fit:  # checkpoints of an older run must not be resumed later
        for f in ("last.pt", "best.pt"):
            (final_dir / f).unlink(missing_ok=True)
    if resume:
        log.info(
            "Resume: %s",
            "evaluating the trained model"
            if skip_fit
            else "final training from last.pt"
            if resume_final
            else "keeping finished Optuna trials",
        )

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
                "backend": "vision",
                "stage": "train",
                "resumed": str(resume),
            }
        )
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config/config.yaml")
        mlflow.log_dict(meta, "config/data_meta.json")
        files = [Path(r[k]) for r in train_r + val_r + test_r for k in ("image", "mask") if k in r]
        splits = [processed / f"{s}.jsonl" for s in ("train", "val", "test")]
        lineage_info = lineage.log_context(
            files, Path(cfg.paths.raw) / cfg.data.name, extra_artifacts=splits
        )

        # ---------- 1. Optuna ----------
        study, best_params, val_summary = None, {}, None
        if hpo.enabled and hpo.n_trials > 0 and space and not skip_fit and not resume_final:
            runstate.write_state(runs_dir, stage="hpo")
            hpo_epochs = int(cfg.model.get("hpo_epochs") or max(1, _params(cfg)["epochs"] // 3))

            def objective(trial: optuna.Trial) -> float:
                params = suggest_params(trial, space)
                with mlflow.start_run(
                    experiment_id=experiment_id, run_name=f"trial-{trial.number}", nested=True
                ):
                    log_params(params)
                    t0 = time.perf_counter()
                    _, hist, best = fit(
                        cfg,
                        task,
                        names,
                        train_r,
                        val_r,
                        _params(cfg, {**params, "epochs": hpo_epochs}),
                        None,
                    )
                    res = {metric: best, "fit_seconds": time.perf_counter() - t0}
                    mlflow.log_metrics({f"val_{k}": v for k, v in res.items()})
                    trial.set_user_attr("val_summary", res)
                    return best

            log.info(
                "Optuna: %d trials x %d epochs, optimising val %s", hpo.n_trials, hpo_epochs, metric
            )
            study = run_study(
                OmegaConf.merge(hpo, {"direction": "maximize"}), cfg.seed, objective, resume=resume
            )
            best_params = suggest_params(optuna.trial.FixedTrial(study.best_params), space)
            val_summary = study.best_trial.user_attrs.get("val_summary")
            mlflow.log_metric(f"best_val_{metric}", study.best_value)
            write_json(best_params, runs_dir / "best_params.json")
        elif (runs_dir / "best_params.json").exists() and resume:
            best_params = json.loads((runs_dir / "best_params.json").read_text(encoding="utf-8"))

        # ---------- 2. Final training ----------
        p = _params(cfg, best_params)
        t0 = time.perf_counter()
        if skip_fit:
            model, vmeta = load_trained(model_dir)
            ck = final_dir / "last.pt"
            history = pd.DataFrame(
                torch.load(ck, map_location="cpu", weights_only=False)["history"]
                if ck.exists()
                else []
            )
        else:
            runstate.write_state(runs_dir, stage="final")
            model, history, _ = fit(
                cfg, task, names, train_r, val_r, p, final_dir, resume=resume_final
            )
            save_trained(model, cfg, task, names, p, model_dir)
        fit_seconds = time.perf_counter() - t0
        runstate.write_state(runs_dir, stage="evaluate")
        log_params({f"model.{k}": v for k, v in p.items()})
        mlflow.log_params(
            {
                "arch": str(
                    cfg.model.get("arch") or cfg.model.get("hf_model") or "segformer-scratch"
                ),
                "tune": tune_name,
                "hpo.n_trials": hpo.n_trials if hpo.enabled and space else 0,
            }
        )
        for _, row in history.iterrows():
            mlflow.log_metrics(
                {f"epoch/{k}": float(v) for k, v in row.items() if k != "epoch"},
                step=int(row["epoch"]),
            )

        # ---------- 3. Evaluation ----------
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(dev)
        l2i = {v: i for i, v in enumerate(names)}
        test_dl = loader(
            ImageDataset(test_r, task, l2i, int(p["imgsz"]), False, norm_for(model, task)),
            int(p["batch_size"]),
            False,
            0,
        )
        images = []
        if task == "segment":
            cm, per_img = run_eval(model, test_dl, task, len(names), dev)
            result = seg_eval(cm, names)
            preds = pd.DataFrame({"image": [Path(r["image"]).name for r in test_r], "iou": per_img})
            preds = preds.sort_values("iou").reset_index(drop=True)
            vm = {
                "task": task,
                "imgsz": int(p["imgsz"]),
                "norm": [list(x) for x in norm_for(model, task)],
            }
            ex_dir = ensure_dir(out_dir / "examples")
            worst = [r for r in test_r if Path(r["image"]).name in set(preds.image.head(6))]
            for r, m in zip(
                worst, predict_images(model, vm, [Path(r["image"]) for r in worst]), strict=True
            ):
                from PIL import Image

                gold = np.array(Image.open(r["mask"]))
                tile = np.hstack(
                    [np.asarray(Image.open(r["image"]).convert("RGB")), colorize(gold), colorize(m)]
                )
                path = ex_dir / f"{Path(r['image']).stem}_image_true_pred.png"
                Image.fromarray(tile).save(path)
                images.append(path)
        else:
            probs, gold = run_eval(model, test_dl, task, len(names), dev)
            pred = probs.argmax(1)
            result = evaluate(
                gold, pred, probs, names, cfg.get("evaluation", {}).get("average", "macro")
            )
            preds = pd.DataFrame(
                {
                    "image": [r["image"] for r in test_r],
                    "y_true": [names[i] for i in gold],
                    "y_pred": [names[i] for i in pred],
                }
            )
            preds.insert(3, "correct", preds.y_true == preds.y_pred)
            preds.insert(4, "confidence", probs.max(1))
            if len(names) <= 20:
                for i, c in enumerate(names):
                    preds[f"proba_{c}"] = probs[:, i]
        result.summary["fit_seconds"] = fit_seconds
        mlflow.log_metrics(
            {
                f"test_{k}": float(v)
                for k, v in result.summary.items()
                if isinstance(v, int | float) and math.isfinite(float(v))
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
                "architecture": str(
                    cfg.model.get("arch") or cfg.model.get("hf_model") or "SegFormer (scratch)"
                ),
                "task": "semantic segmentation (scored per pixel: IoU / Dice)"
                if task == "segment"
                else "image classification",
                "classes": ", ".join(names),
                "images train/val/test": "/".join(
                    str(meta["n_rows"].get(s, 0)) for s in ("train", "val", "test")
                ),
                "evaluated on": eval_split,
                "hpo_metric": f"{metric} (validation)",
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
                images=images or None,
            )
            log.info("Excel report: %s", report_path)
        for f in [out_dir / "metrics.json", out_dir / "per_class.csv", report_path, *images]:
            if f:
                mlflow.log_artifact(str(f), artifact_path="reports")
        if trials_df is not None:
            trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)
            mlflow.log_artifact(str(out_dir / "optuna_trials.csv"), artifact_path="reports")

        # ---------- 5. Model ----------
        mlflow.pyfunc.log_model(
            name="model",
            python_model=VisionPyfunc(),
            artifacts={"model": str(model_dir)},
            pip_requirements=["torch", "torchvision", "timm", "transformers", "pandas", "pillow"],
            metadata={"class_names": names, "dataset": cfg.data.name, "task": task},
            registered_model_name=cfg.mlflow.registered_model_name
            if cfg.mlflow.register_model
            else None,
        )

    s = result.summary
    lineage.record(
        "train",
        history=lineage.history_path(cfg),
        dataset=cfg.data.name,
        model=cfg.model.name,
        task=f"image-{task}",
        tune=tune_name,
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
        ["miou", "pixel_accuracy", "f1"]
        if task == "segment"
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
def predict_vision(cfg: DictConfig) -> Path:
    from PIL import Image

    from ml_training.modeling.predict import track_prediction

    model_dir = Path(
        cfg.get("weights") or Path(cfg.paths.models) / cfg.data.name / cfg.model.name / "model"
    )
    if not (model_dir / "vision.json").exists():
        raise FileNotFoundError(
            f"No trained model in {model_dir} – train first (.\\run.ps1 vision ...)"
        )
    model, meta = load_trained(model_dir)
    if cfg.get("input"):
        src = Path(str(cfg.input))
        if not src.is_absolute():
            src = Path(cfg.paths.root) / src
        paths = (
            [src]
            if src.is_file()
            else sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
        )
    else:  # no +input: the dataset's test split (val when there is none)
        from ml_training.vision_data import load_split

        processed = Path(cfg.paths.processed)
        recs = load_split(processed, "test") or load_split(processed, "val")
        src, paths = processed / "test.jsonl", [Path(r["image"]) for r in recs]
    if not paths:
        raise ValueError(f"No images in {src}")
    names = meta["class_names"]
    out_dir = ensure_dir(Path(cfg.paths.run_reports) / "predict")
    res = predict_images(model, meta, paths)
    if meta["task"] == "segment":
        rows = []
        for path, m in zip(paths, res, strict=True):
            Image.fromarray(overlay(path, m)).save(out_dir / f"{path.stem}_overlay.png")
            Image.fromarray(m).save(out_dir / f"{path.stem}_mask.png")
            rows.append(
                {
                    "source": path.name,
                    "class": names[
                        int(np.bincount(m.ravel(), minlength=len(names))[1:].argmax()) + 1
                    ]
                    if len(names) > 1 and (m > 0).any()
                    else names[0],
                    **{f"pct_{n}": float((m == i).mean()) for i, n in enumerate(names)},
                }
            )
        df = pd.DataFrame(rows)
    else:
        top = np.argsort(-res, axis=1)[:, :3]
        df = pd.DataFrame(
            {
                "source": [p.name for p in paths],
                "prediction": [names[i] for i in top[:, 0]],
                "confidence": res.max(1),
                "top3": [
                    ", ".join(f"{names[i]} {res[r, i]:.2f}" for i in top[r])
                    for r in range(len(paths))
                ],
            }
        )
    out_path = Path(cfg.get("output") or out_dir / "predictions.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("%d images -> %s", len(df), out_path)
    info = {
        "input": str(src),
        "weights": str(model_dir),
        "task": meta["task"],
        "weights_sha256": lineage.fingerprint(model_dir.glob("*.*"))["sha256"],
    }
    track_prediction(cfg, info, out_path, df)
    return out_path
