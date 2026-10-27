<#
  Windows helper for ml-training.  Easiest: double-click run.bat (or type .\run.ps1) for a menu.
  YOLO object detection (dataset = folder name in data\raw, or a full path to the folder):
    .\run.ps1 detect helmet                 # train yolo11n, full (Optuna tuning + 100 epochs)
    .\run.ps1 detect helmet m quick         # yolo11m, quick test (no tuning, 30 epochs)
    .\run.ps1 detect helmet all quick       # every size n,s,m,l,x one after another
    .\run.ps1 detect D:\datasets\helmet s   # dataset outside the project (copied to data\raw)
    .\run.ps1 detect helmet n full model.params.epochs=200   # extra Hydra overrides at the end
    .\run.ps1 results helmet                # compare all trained sizes (+ reports\helmet\compare.xlsx)
    .\run.ps1 find helmet D:\images         # detect objects with the best trained model
    .\run.ps1 find helmet D:\cam\clip.mp4 m # ... or with a chosen size
  Setup / tools:
    .\run.ps1 setup-yolo   # install (CPU)          .\run.ps1 setup-gpu   # install (NVIDIA GPU)
    .\run.ps1 ui           # MLflow UI http://127.0.0.1:5000
    .\run.ps1 optuna       # Optuna dashboard http://127.0.0.1:8080
  Tabular data:  .\run.ps1 setup | pipeline | compare | train model=svm | test
  Raw commands:  .\run.ps1 prepare|featurize|train|predict <hydra overrides>
  If scripts are blocked:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned  (or use run.bat)
#>
param(
    [Parameter(Position = 0)][string]$Task = "menu",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
# After setup-gpu, stop `uv run` from swapping the CUDA torch back to the CPU build.
if (Test-Path .gpu) { $env:UV_NO_SYNC = "1" }
$CudaIndex = "https://download.pytorch.org/whl/cu126"
$Sizes = @("yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x")
$HelpLines = 19

function Need-Uv {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "Installing uv..." -ForegroundColor Yellow
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    }
}

function Say([string]$Text, [string]$Color = "Cyan") { Write-Host $Text -ForegroundColor $Color }

# ----------------------------------------------------------------------------- YOLO helpers
# "helmet" -> data=helmet (if configs\data\helmet.yaml exists) or data=yolo data.name=helmet
# "D:\datasets\helmet" -> data=yolo data.name=helmet data.path=D:/datasets/helmet
function Get-Dataset([string]$Ds) {
    $Ds = $Ds.Trim().Trim('"').TrimEnd('\', '/')
    if (($Ds -match '[\\/:]') -and (Test-Path $Ds -PathType Container)) {
        $full = (Resolve-Path $Ds).Path
        $name = Split-Path $full -Leaf
        return @{ Name = $name; Args = @("data=yolo", "data.name=$name", "data.path='$($full -replace '\\', '/')'") }
    }
    if (Test-Path "configs\data\$Ds.yaml") { return @{ Name = $Ds; Args = @("data=$Ds") } }
    return @{ Name = $Ds; Args = @("data=yolo", "data.name=$Ds") }
}

function Get-Sizes([string]$Size) {
    if (-not $Size) { return @("yolo11n") }
    if ($Size -eq "all") { return $Sizes }
    $s = $Size.ToLower() -replace '^yolo11', ''
    if ("n", "s", "m", "l", "x" -notcontains $s) { throw "Unknown size '$Size' (use n, s, m, l, x or all)" }
    return @("yolo11$s")
}

function Get-ModeArgs([string]$Mode) {
    switch ($Mode) {
        "quick" { return @("hpo=none", "model.params.epochs=30") }
        "full" { return @() }
        default { throw "Unknown mode '$Mode' (use quick or full)" }
    }
}

function Invoke-Detect([string]$Ds, [string]$Size, [string]$Mode, [string[]]$Extra) {
    if (-not $Ds) { throw "Usage: .\run.ps1 detect <dataset> [n|s|m|l|x|all] [quick|full] [overrides]" }
    if (-not $Mode) { $Mode = "full" }
    $d = Get-Dataset $Ds
    $models = Get-Sizes $Size
    $modeArgs = Get-ModeArgs $Mode
    $dataArgs = $d.Args
    Say "Dataset: $($d.Name)   Models: $($models -join ', ')   Mode: $Mode"
    Say "[1/2] Preparing dataset..."
    uv run mlt-prepare @dataArgs
    if ($LASTEXITCODE -ne 0) { throw "Dataset check failed (see the message above)" }
    uv run mlt-featurize @dataArgs
    if ($LASTEXITCODE -ne 0) { throw "Dataset split failed (see the message above)" }
    $done = @()
    foreach ($m in $models) {
        Say "[2/2] Training $m ..."
        uv run mlt-train @dataArgs "model=$m" @modeArgs @Extra
        if ($LASTEXITCODE -ne 0) { Say "$m failed, continuing" "Yellow"; continue }
        $done += $m
        Say "  model : models\$($d.Name)\$m.pt" "Green"
        Say "  report: reports\$($d.Name)\$m\report.xlsx" "Green"
    }
    if ($done.Count -gt 0) {
        Show-Results $d.Name
        if ($done.Count -eq 1) { $script:LastReport = "reports\$($d.Name)\$($done[0])\report.xlsx" }
        else { $script:LastReport = "reports\$($d.Name)\compare.xlsx" }
    }
}

function Show-Results([string]$Name) {
    if (-not $Name) { throw "Usage: .\run.ps1 results <dataset>" }
    Say "`n=== Results: $Name ==="
    uv run python -m ml_training.compare $Name
}

function Invoke-Find([string]$Ds, [string]$InputPath, [string]$Size, [string[]]$Extra) {
    if (-not $Ds -or -not $InputPath) { throw "Usage: .\run.ps1 find <dataset> <image|folder|video|url> [n|s|m|l|x]" }
    $d = Get-Dataset $Ds
    if ($Size) { $m = (Get-Sizes $Size)[0] }
    else {
        $m = (uv run python -m ml_training.compare $d.Name --best | Select-Object -Last 1)
        if ($LASTEXITCODE -ne 0 -or -not $m) { throw "No trained model for '$($d.Name)'. Train first: .\run.ps1 detect $Ds" }
    }
    $src = $InputPath.Trim().Trim('"')
    if ($src -notmatch '://') { $src = (Resolve-Path $src).Path -replace '\\', '/' }
    $dataArgs = $d.Args
    Say "Detecting with $m on $src"
    uv run mlt-predict @dataArgs "model=$m" "+input='$src'" @Extra
    $out = "reports\$($d.Name)\$m\predict"
    if ($LASTEXITCODE -eq 0 -and (Test-Path $out)) {
        Say "Annotated files + detections.csv: $out" "Green"
        Invoke-Item $out
    }
}

# ----------------------------------------------------------------------------- menu
function Read-Choice([string]$Prompt, [string[]]$Options, [int]$Default = 1) {
    for ($i = 0; $i -lt $Options.Count; $i++) { Write-Host ("  {0}) {1}" -f ($i + 1), $Options[$i]) }
    while ($true) {
        $a = Read-Host "$Prompt [$Default]"
        if (-not $a) { return $Default - 1 }
        $n = 0
        if ([int]::TryParse($a, [ref]$n) -and $n -ge 1 -and $n -le $Options.Count) { return $n - 1 }
        Say "Please type a number 1-$($Options.Count)" "Yellow"
    }
}

function Select-Dataset([switch]$Trained) {
    if ($Trained) { $names = @(Get-ChildItem models -Directory -ErrorAction SilentlyContinue | Where-Object { Get-ChildItem $_.FullName -Filter *.pt -File } | ForEach-Object Name) }
    else { $names = @(Get-ChildItem data\raw -Directory -ErrorAction SilentlyContinue | ForEach-Object Name) }
    $names = @($names | Where-Object { $_ -ne "pretrained" })
    Say "`nDataset:"
    $opts = $names + @("Type a folder path...")
    $i = Read-Choice "Choose" $opts
    if ($i -lt $names.Count) { return $names[$i] }
    return (Read-Host "Folder path (you can drag the folder here)").Trim('"')
}

function Show-Menu {
    while ($true) {
        Say "`n========== ML Training =========="
        $i = Read-Choice "Choose" @(
            "Train YOLO (object detection)",
            "Detect objects in images / video with a trained model",
            "Compare trained YOLO models",
            "Open MLflow UI (all runs, metrics, model registry)",
            "Install / update (CPU or NVIDIA GPU)",
            "Tabular data: run the full pipeline",
            "Help (all commands)",
            "Exit")
        try {
            switch ($i) {
                0 {
                    $ds = Select-Dataset
                    Say "`nModel size (n = fastest ... x = most accurate):"
                    $sizeOpts = @("n  (nano)", "s  (small)", "m  (medium)", "l  (large)", "x  (extra large)", "all sizes, one after another")
                    $size = @("n", "s", "m", "l", "x", "all")[(Read-Choice "Choose" $sizeOpts)]
                    Say "`nTraining mode:"
                    $mode = @("quick", "full")[(Read-Choice "Choose" @("Quick test (no tuning, 30 epochs)", "Full (Optuna tuning + 100 epochs)"))]
                    $script:LastReport = $null
                    Invoke-Detect $ds $size $mode @()
                    if ($script:LastReport -and (Test-Path $script:LastReport) -and ((Read-Host "Open the Excel report? (y/n) [y]") -ne "n")) { Invoke-Item $script:LastReport }
                }
                1 {
                    $ds = Select-Dataset -Trained
                    $src = Read-Host "Image, folder or video path (drag it here) or URL"
                    Invoke-Find $ds $src "" @()
                }
                2 { $ds = Select-Dataset -Trained; Show-Results $ds; $x = "reports\$ds\compare.xlsx"; if ((Test-Path $x) -and ((Read-Host "Open compare.xlsx? (y/n) [y]") -ne "n")) { Invoke-Item $x } }
                3 { Start-Process "http://127.0.0.1:5000"; uv run mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5000 }
                4 {
                    $g = Read-Choice "Install for" @("CPU only", "NVIDIA GPU (CUDA)")
                    if ($g -eq 0) { & $PSCommandPath setup-yolo } else { & $PSCommandPath setup-gpu }
                }
                5 { uv run dvc repro; uv run dvc metrics show }
                6 { (Get-Content $PSCommandPath -TotalCount $HelpLines) -join "`n" | Write-Host }
                7 { return }
            }
        } catch { Say "Error: $($_.Exception.Message)" "Red" }
    }
}

# ----------------------------------------------------------------------------- commands
switch ($Task) {
    "menu" { Show-Menu }
    "help" { (Get-Content $PSCommandPath -TotalCount $HelpLines) -join "`n" | Write-Host }
    "detect" {
        $a = @($Rest)
        $ds = $a[0]; $size = $null; $mode = $null; $i = 1
        if ($a.Count -gt $i -and $a[$i] -notmatch '=') { $size = $a[$i]; $i++ }
        if ($a.Count -gt $i -and $a[$i] -notmatch '=') { $mode = $a[$i]; $i++ }
        $extra = @(); if ($a.Count -gt $i) { $extra = $a[$i..($a.Count - 1)] }
        Invoke-Detect $ds $size $mode $extra
    }
    "results" { Show-Results $Rest[0] }
    "find" {
        $a = @($Rest)
        $size = $null; $i = 2
        if ($a.Count -gt $i -and $a[$i] -notmatch '=') { $size = $a[$i]; $i++ }
        $extra = @(); if ($a.Count -gt $i) { $extra = $a[$i..($a.Count - 1)] }
        Invoke-Find $a[0] $a[1] $size $extra
    }
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
    "yolo-all" {
        if (@($Rest).Count -gt 0 -and $Rest[0] -notmatch '=') {   # .\run.ps1 yolo-all helmet [quick|full]
            $mode = "full"; if (@($Rest).Count -gt 1) { $mode = $Rest[1] }
            Invoke-Detect $Rest[0] "all" $mode @()
        } else {                                                  # old style: yolo-all data=yolo data.name=helmet
            foreach ($m in $Sizes) { Say "=== $m ==="; uv run mlt-train "model=$m" @Rest }
        }
    }
    "prepare"  { uv run mlt-prepare @Rest }
    "featurize" { uv run mlt-featurize @Rest }
    "train"    { uv run mlt-train @Rest }
    "predict"  { uv run mlt-predict @Rest }
    "pipeline" { uv run dvc repro @Rest; uv run dvc metrics show }
    "compare"  { uv run mlt-train -m "model=random_forest,logistic_regression,gradient_boosting,svm" @Rest }
    "ui"       { uv run mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5000 }
    "optuna"   { uv run --with optuna-dashboard optuna-dashboard "sqlite:///optuna.db" }
    "test"     { uv run ruff check src tests; uv run pytest }
    default    { Say "Unknown command '$Task'" "Yellow"; (Get-Content $PSCommandPath -TotalCount $HelpLines) -join "`n" | Write-Host }
}
