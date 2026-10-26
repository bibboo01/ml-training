"""PyTorch Lightning MLP exposed through the scikit-learn estimator API.

Because it behaves like any sklearn classifier it plugs into the same pipeline, Optuna search,
metrics and MLflow logging as the classical models. Install with ``uv sync --extra dl``.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class LightningMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_dims=(64, 32),
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 8,
        val_fraction: float = 0.15,
        random_state: int = 42,
        accelerator: str = "auto",
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.accelerator = accelerator

    def fit(self, X, y):
        import lightning as L
        import torch
        from lightning.pytorch.callbacks import EarlyStopping
        from torch.utils.data import DataLoader, TensorDataset

        L.seed_everything(self.random_state, workers=True, verbose=False)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_idx = np.searchsorted(self.classes_, y).astype(np.int64)

        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(X))
        n_val = max(1, int(len(X) * self.val_fraction))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        def loader(ix, shuffle):
            ds = TensorDataset(torch.from_numpy(X[ix]), torch.from_numpy(y_idx[ix]))
            # num_workers=0 keeps it safe on Windows (no multiprocessing spawn issues)
            return DataLoader(ds, batch_size=int(self.batch_size), shuffle=shuffle, num_workers=0)

        if MLPModule is None:
            raise ImportError("lightning_mlp needs the 'dl' extra: uv sync --extra dl")
        self.module_ = MLPModule(
            X.shape[1],
            list(self.hidden_dims),
            len(self.classes_),
            self.dropout,
            self.lr,
            self.weight_decay,
        )
        trainer = L.Trainer(
            max_epochs=int(self.max_epochs),
            accelerator=self.accelerator,
            devices=1,
            callbacks=[EarlyStopping("val_loss", patience=int(self.patience))],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(self.module_, loader(tr_idx, True), loader(val_idx, False))
        self.module_.cpu().eval()
        return self

    def predict_proba(self, X):
        import torch

        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            return torch.softmax(self.module_(X), dim=1).numpy()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


try:  # torch/lightning are an optional extra – only needed when this model is selected
    import lightning as L
    import torch
    from torch import nn

    class MLPModule(L.LightningModule):
        def __init__(self, n_in, hidden, n_out, dropout, lr, weight_decay):
            super().__init__()
            self.lr, self.weight_decay = lr, weight_decay
            layers, prev = [], n_in
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, n_out))
            self.net = nn.Sequential(*layers)
            self.loss = nn.CrossEntropyLoss()

        def forward(self, x):
            return self.net(x)

        def training_step(self, batch, _):
            x, y = batch
            loss = self.loss(self(x), y)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, _):
            x, y = batch
            self.log("val_loss", self.loss(self(x), y))

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

except ImportError:  # pragma: no cover
    MLPModule = None
