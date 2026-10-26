# notebooks/

Notebooks are for **exploration only**. Reusable code goes in `src/ml_training/` and is imported:

```python
from ml_training.features import load_splits
```

Naming convention (Cookiecutter Data Science v2): `<step>.<version>-<initials>-<description>.ipynb`,
e.g. `1.0-bz-explore-breast-cancer.ipynb`.

Use the project kernel: `uv run jupyter lab` (ipykernel is in the `dev` group; add `jupyterlab`
with `uv add --group dev jupyterlab`).
