"""Heatmaps for Ultralytics YOLO models (no extra packages – plain torch hooks).

EigenCAM (Muhammad & Yeasin, 2020): first principal component of the neck feature maps.
  Class-agnostic "where is the model looking" – needs no gradients, works for every task.
Grad-CAM (Selvaraju et al., 2017) per prediction: gradient of *that* box's class score
  (or the predicted class for classify) w.r.t. the same feature maps -> "why this box".

All maps are returned in original-image pixels, scaled 0..1.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping the aspect ratio and pad to size x size (like Ultralytics)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_LINEAR
    )
    return canvas, r, left, top


def center_crop(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Classify preprocessing: short side -> size, centre crop. Returns crop, scale, x0, y0."""
    h, w = img.shape[:2]
    r = size / min(h, w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0, y0 = (nw - size) // 2, (nh - size) // 2
    return resized[y0 : y0 + size, x0 : x0 + size], r, x0, y0


def _norm(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)
    m = cam.max()
    return cam / m if m > 0 else cam


class YoloExplainer:
    """Forward/backward hooks on the layers that feed the YOLO head."""

    def __init__(self, yolo, task: str, imgsz: int) -> None:
        self.task, self.imgsz = task, int(imgsz)
        self.net = yolo.model.float().eval()
        self.device = next(self.net.parameters()).device
        layers = self.net.model
        idx = [len(layers) - 2] if task == "classify" else list(layers[-1].f)
        self.layer_idx = idx
        self.acts: list[torch.Tensor] = []
        self.handles = [layers[i].register_forward_hook(self._hook) for i in idx]
        self.nc = len(self.net.names)

    def _hook(self, _module, _inp, out) -> None:
        t = out[0] if isinstance(out, list | tuple) else out
        if t.requires_grad:
            t.retain_grad()
        self.acts.append(t)

    def close(self) -> None:
        for h in self.handles:
            h.remove()

    # ------------------------------------------------------------------ forward
    def run(self, img: np.ndarray) -> dict[str, Any]:
        """Forward one BGR image; keeps what is needed for EigenCAM / Grad-CAM."""
        if self.task == "classify":
            inp, r, ox, oy = center_crop(img, self.imgsz)
        else:
            inp, r, ox, oy = letterbox(img, self.imgsz)
        x = torch.from_numpy(np.ascontiguousarray(inp[..., ::-1])).permute(2, 0, 1).float() / 255
        x = x[None].to(self.device).requires_grad_(True)
        self.acts = []
        with torch.enable_grad():
            out = self.net(x)
        pred = out[0] if isinstance(out, list | tuple) else out
        if isinstance(pred, list | tuple):  # segment: [preds, protos]
            pred = pred[0]
        self.state = {"pred": pred, "r": r, "ox": ox, "oy": oy, "shape": img.shape[:2]}
        return self.state

    def _to_original(self, cam: np.ndarray) -> np.ndarray:
        """Map a map in network-input pixels back onto the original image."""
        h, w = self.state["shape"]
        r, ox, oy = self.state["r"], self.state["ox"], self.state["oy"]
        if self.task == "classify":  # crop region of the resized image -> original
            nw, nh = round(w * r), round(h * r)
            full = np.zeros((nh, nw), np.float32)
            full[oy : oy + self.imgsz, ox : ox + self.imgsz] = cam[: nh - oy, : nw - ox]
            return cv2.resize(full, (w, h), interpolation=cv2.INTER_LINEAR)
        nw, nh = round(w * r), round(h * r)
        crop = cam[oy : oy + nh, ox : ox + nw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)

    def _combine(self, maps: list[np.ndarray]) -> np.ndarray:
        size = self.imgsz
        acc = np.zeros((size, size), np.float32)
        for m in maps:
            acc += _norm(
                cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
            )
        return _norm(self._to_original(_norm(acc)))

    # ------------------------------------------------------------------ maps
    def eigencam(self) -> np.ndarray:
        maps = []
        for a in self.acts:
            act = a[0].detach().float()
            c, h, w = act.shape
            m = act.reshape(c, -1).T  # (HW, C)
            m = m - m.mean(0, keepdim=True)
            try:
                _, _, vh = torch.linalg.svd(m, full_matrices=False)
            except RuntimeError:
                continue
            proj = (m @ vh[0]).reshape(h, w)
            if proj.sum() < 0:  # SVD sign is arbitrary: keep the "object" side positive
                proj = -proj
            maps.append(proj.cpu().numpy())
        return self._combine(maps) if maps else np.zeros(self.state["shape"], np.float32)

    def gradcam(self, score: torch.Tensor, level: int | None = None) -> np.ndarray:
        """Grad-CAM of one score. level = index of the feature map (stride) that produced the
        box – using only that scale gives a much cleaner map than mixing all three."""
        for a in self.acts:
            a.grad = None
        self.net.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)
        maps = []
        acts = [self.acts[level]] if level is not None and level < len(self.acts) else self.acts
        for a in acts:
            if a.grad is None:
                continue
            g, act = a.grad[0], a[0].detach()
            weights = g.mean(dim=(1, 2))
            maps.append(torch.relu((weights[:, None, None] * act).sum(0)).cpu().numpy())
        if not maps:
            return np.zeros(self.state["shape"], np.float32)
        cam = self._combine(maps)
        return _norm(cv2.GaussianBlur(cam, (0, 0), max(1.0, min(cam.shape) / 150)))

    def box_score(self, box_xyxy: list[float], cls: int) -> tuple[torch.Tensor | None, int | None]:
        """The raw head score that produced a final box (matched by IoU among all anchors)."""
        pred = self.state["pred"]
        r, ox, oy = self.state["r"], self.state["ox"], self.state["oy"]
        b = torch.tensor(
            [
                box_xyxy[0] * r + ox,
                box_xyxy[1] * r + oy,
                box_xyxy[2] * r + ox,
                box_xyxy[3] * r + oy,
            ],
            device=pred.device,
        )
        if pred.dim() != 3:
            return None, None
        if pred.shape[2] >= pred.shape[1]:  # (1, 4 + nc + extra, anchors), boxes xywh
            xywh = pred[0, :4].T
            scores = pred[0, 4 : 4 + self.nc]
            boxes = torch.cat([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
            cls_score = scores[cls]
        else:  # end-to-end (1, N, 6+): x1 y1 x2 y2 score cls
            boxes = pred[0, :, :4]
            cls_score = pred[0, :, 4] * (pred[0, :, 5].round() == cls)
        lt = torch.maximum(boxes[:, :2], b[:2])
        rb = torch.minimum(boxes[:, 2:], b[2:])
        inter = (rb - lt).clamp(min=0).prod(1)
        area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(1)
        iou = inter / ((b[2:] - b[:2]).prod() + area - inter + 1e-9)
        k = int(torch.argmax(iou.detach() * cls_score.detach()))
        level = None
        if pred.shape[2] >= pred.shape[1]:  # anchors are ordered level by level (stride 8, 16, 32)
            sizes = np.cumsum([a.shape[2] * a.shape[3] for a in self.acts])
            if sizes[-1] == pred.shape[2]:
                level = int(np.searchsorted(sizes, k, side="right"))
        return cls_score[k], level

    def class_score(self, cls: int) -> torch.Tensor:
        pred = self.state["pred"]
        return pred[0, cls] if pred.dim() == 2 else pred.reshape(-1)[cls]


def overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(heat, alpha, img, 1 - alpha, 0)
