# Same tasks as run.ps1 for macOS/Linux/WSL.
.PHONY: setup pipeline train compare ui optuna test
setup:    ; uv sync --group dev
pipeline: ; uv run dvc repro && uv run dvc metrics show
train:    ; uv run mlt-train $(ARGS)
compare:  ; uv run mlt-train -m model=random_forest,logistic_regression,gradient_boosting,svm
ui:       ; uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
optuna:   ; uv run --with optuna-dashboard optuna-dashboard sqlite:///optuna.db
test:     ; uv run ruff check src tests && uv run pytest
