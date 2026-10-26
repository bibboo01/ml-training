<#
  Windows helper for ml-training.  Usage (PowerShell, from the project folder):
    .\run.ps1 setup        # install uv deps (+ git/dvc init if needed)
    .\run.ps1 pipeline     # dvc repro: prepare -> featurize -> train
    .\run.ps1 train  model=gradient_boosting hpo.n_trials=50
    .\run.ps1 compare      # train several models in one Hydra multirun
    .\run.ps1 ui           # open MLflow UI at http://127.0.0.1:5000
    .\run.ps1 optuna       # open Optuna dashboard at http://127.0.0.1:8080
    .\run.ps1 test         # ruff + pytest
  YOLO object detection:
    .\run.ps1 setup-yolo   # install Ultralytics (CPU torch)
    .\run.ps1 setup-gpu    # install Ultralytics + CUDA torch (NVIDIA GPU)
    .\run.ps1 yolo         # dvc repro pipelines/yolo/dvc.yaml
    .\run.ps1 train data=yolo data.name=helmet model=yolo11n
    .\run.ps1 predict data=yolo data.name=helmet model=yolo11n +input=D:/images
  If scripts are blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#>
param(
    [Parameter(Position = 0)][string]$Task = "help",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
# After setup-gpu, stop `uv run` from swapping the CUDA torch back to the CPU build.
if (Test-Path .gpu) { $env:UV_NO_SYNC = "1" }
$CudaIndex = "https://download.pytorch.org/whl/cu126"

function Need-Uv {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "Installing uv..." -ForegroundColor Yellow
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    }
}

switch ($Task) {
    "setup" {
        Need-Uv
        uv sync --group dev
        if (-not (Test-Path .git)) { git init -b main }
        if (-not (Test-Path .dvc)) { uv run dvc init }
        Write-Host "Ready. Next: .\run.ps1 pipeline" -ForegroundColor Green
    }
    "setup-dl" { Need-Uv; uv sync --group dev --extra dl }
    "setup-yolo" { Need-Uv; uv sync --group dev --extra yolo; Remove-Item .gpu -ErrorAction SilentlyContinue }
    "setup-gpu" {
        Need-Uv
        Remove-Item Env:UV_NO_SYNC -ErrorAction SilentlyContinue
        uv sync --group dev --extra yolo
        uv pip install --reinstall torch torchvision --index-url $CudaIndex
        New-Item .gpu -ItemType File -Force | Out-Null
        uv run --no-sync python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.version.cuda)"
    }
    "yolo"     { uv run dvc repro pipelines/yolo/dvc.yaml @Rest; uv run dvc metrics show }
    "prepare"  { uv run mlt-prepare @Rest }
    "featurize" { uv run mlt-featurize @Rest }
    "train"    { uv run mlt-train @Rest }
    "predict"  { uv run mlt-predict @Rest }
    "pipeline" { uv run dvc repro @Rest; uv run dvc metrics show }
    "compare"  { uv run mlt-train -m "model=random_forest,logistic_regression,gradient_boosting,svm" @Rest }
    "ui"       { uv run mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5000 }
    "optuna"   { uv run --with optuna-dashboard optuna-dashboard "sqlite:///optuna.db" }
    "test"     { uv run ruff check src tests; uv run pytest }
    default    { (Get-Content $PSCommandPath -TotalCount 17) -join "`n" | Write-Host }
}
