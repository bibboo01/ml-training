<#
  ml-training for Windows (Ubuntu: run.sh).  Double-click run.bat, or after `run setup` just type: run
    run                          menu  (ภาษาไทย: More tools > Language)
    run setup                    install everything (finds the NVIDIA GPU itself), then check this PC
    run train D:\data\helmet      look at the data, pick the task and model, train   (+ quick | best)
    run predict helmet D:\new      predict with the best model (image, folder, video, csv, PDF, sentence, N = forecast)
    run results helmet           scores of every model + Excel
    run status                   trained models and trainings that stopped
    run resume                   continue a training that stopped (power cut, Ctrl+C, crash)
    run check | run ui | run clean helmet | run help
  Choose everything yourself:
    run fit helmet m quick                       YOLO: n s m l x all (26m = YOLO26), quick | full, tuning preset
    run fit helmet n full data.task=segment model.params.epochs=200   (extra Hydra overrides)
    run text data\raw\reviews.csv thai           text / documents: tiny small base thai large
    run vision D:\data\leaves efficientnet       tiny resnet50 efficientnet convnext vit seg-tiny seg-b0 seg-b2
    run table D:\sales.xlsx target=amount time=date group=shop all     (lightgbm xgboost ... all)
    run explain helmet m 300                     XAI heatmaps + statistics Excel
    run history | run tune | run optuna | run test
    run train model=svm ...  /  run predict ...  /  run prepare|featurize ...   (Hydra overrides, as before)
  Setup parts: setup-gpu  setup-yolo (CPU)  setup-ocr  setup-dev
  `run` works in new terminals after `run setup`; until then use .\run.ps1 (PowerShell) or run.bat (cmd).
#>
param(
    [Parameter(Position = 0)][string]$Task = "menu",
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
$ErrorActionPreference = "Stop"
$Orig = $env:MLT_CWD; if (-not $Orig) { $Orig = (Get-Location).Path }
Set-Location $PSScriptRoot
# paths typed relative to the folder `run` was started in
if ($Rest -and $Orig.TrimEnd('\') -ne $PSScriptRoot.TrimEnd('\')) {
    $Rest = @($Rest | ForEach-Object {
        if ($_ -and $_ -notmatch '=' -and -not [IO.Path]::IsPathRooted($_) -and (Test-Path -LiteralPath (Join-Path $Orig $_))) { (Resolve-Path -LiteralPath (Join-Path $Orig $_)).Path } else { $_ }
    })
}
# Thai text in the console (Windows PowerShell 5.1 defaults to the ANSI code page)
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8; [Console]::InputEncoding = [Text.Encoding]::UTF8 } catch { }
$OutputEncoding = [Text.Encoding]::UTF8
# menu language: en (default) | th  - change in the menu: More tools > Language
$Lang = "en"; if (Test-Path .lang) { $Lang = (Get-Content .lang -TotalCount 1).Trim() }
$env:MLT_LANG = $Lang   # Python messages (inspect, errors, doctor) follow the menu language
function T([string]$Th, [string]$En) { if ($script:Lang -eq "th") { return $Th } return $En }
$env:PYTHONUTF8 = "1"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
# After setup-gpu, stop `uv run` from swapping the CUDA torch back to the CPU build.
if (Test-Path .gpu) { $env:UV_NO_SYNC = "1" }
$CudaIndex = "https://download.pytorch.org/whl/cu126"
$Sizes = @("yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x")
$HelpLines = 21

# YOLO + text (HuggingFace) + vision (timm) + LightGBM / XGBoost packages, + OCR after setup-ocr; -Gpu = CUDA build of torch
function Sync-Env([switch]$Gpu) {
    Remove-Item Env:UV_NO_SYNC -ErrorAction SilentlyContinue
    $extras = @("--extra", "yolo", "--extra", "nlp", "--extra", "vision", "--extra", "boost")
    if (Test-Path .ocr) { $extras += @("--extra", "ocr") }
    uv sync --group dev @extras
    # removing OCR (opencv-python-headless) deletes files cv2 shares with opencv-python: put them back
    uv run --no-sync python -c "import os, sys; sys.stderr = open(os.devnull, 'w'); import cv2; cv2.imshow"
    if ($LASTEXITCODE -ne 0) { uv sync --group dev @extras --reinstall-package opencv-python }
    if ($Gpu) {
        uv pip install --reinstall torch torchvision --index-url $CudaIndex
        New-Item .gpu -ItemType File -Force | Out-Null
        $env:UV_NO_SYNC = "1"
    }
    uv run --no-sync python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.version.cuda)"
}

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

# n -> yolo11n, 26m -> yolo26m, all -> yolo11n..x, all26 -> yolo26n..x
function Get-Sizes([string]$Size) {
    if (-not $Size) { return @("yolo11n") }
    $s = $Size.ToLower() -replace '^yolo', ''
    $gen = "11"
    if ($s -match '^(11|26)(.*)$') { $gen = $Matches[1]; $s = $Matches[2] }
    elseif ($s -match '^all(11|26)$') { $gen = $Matches[1]; $s = "all" }
    if ($s -eq "all") { return @("n", "s", "m", "l", "x" | ForEach-Object { "yolo$gen$_" }) }
    if ("n", "s", "m", "l", "x" -notcontains $s) { throw "Unknown size '$Size' (use n, s, m, l, x, all, or 26n ... all26)" }
    return @("yolo$gen$s")
}

function Get-ModeArgs([string]$Mode) {
    switch ($Mode) {
        "quick" { return @("hpo=none", "model.params.epochs=30") }
        "full" { return @() }
        default { throw "Unknown mode '$Mode' (use quick or full)" }
    }
}

function Invoke-Detect([string]$Ds, [string]$Size, [string]$Mode, [string[]]$Extra) {
    if (-not $Ds) { throw "Usage: run fit <dataset> [n|s|m|l|x|all] [quick|full] [overrides]" }
    if (-not $Mode) { $Mode = "full" }
    $d = Get-Dataset $Ds
    $models = Get-Sizes $Size
    $modeArgs = Get-ModeArgs $Mode
    # data.* overrides (e.g. data.task=segment) must reach prepare/featurize as well
    $dataArgs = @($d.Args) + @($Extra | Where-Object { $_ -match '^\+?data\.' })
    $Extra = @($Extra | Where-Object { $_ -notmatch '^\+?data\.' })
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
    if (-not $Name) { throw "Usage: run results <dataset>" }
    Say "`n=== Results: $Name ==="
    uv run python -m ml_training.compare $Name
}

# tiny | small | base | thai | large -> text_<size>
$TextSizes = @("tiny", "small", "base", "thai", "large")
function Get-TextModel([string]$Size) {
    if (-not $Size) { return "text_base" }
    $s = $Size.ToLower() -replace '^text_', ''
    if ($TextSizes -notcontains $s) { throw "Unknown text model '$Size' (use $($TextSizes -join ', '))" }
    return "text_$s"
}

function Get-TextDataset([string]$Ds) {
    $Ds = $Ds.Trim().Trim('"').TrimEnd('\', '/')
    if (($Ds -match '[\\/:]') -and (Test-Path $Ds)) {
        $full = (Resolve-Path $Ds).Path
        $name = [IO.Path]::GetFileNameWithoutExtension((Split-Path $full -Leaf))
        return @{ Name = $name; Args = @("data=text", "data.name=$name", "data.path='$($full -replace '\\', '/')'") }
    }
    if (Test-Path "configs\data\$Ds.yaml") { return @{ Name = $Ds; Args = @("data=$Ds") } }
    return @{ Name = $Ds; Args = @("data=text", "data.name=$Ds") }
}

function Invoke-Text([string]$Ds, [string]$Size, [string]$Mode, [string[]]$Extra) {
    if (-not $Ds) { throw "Usage: run text <dataset|file|folder> [tiny|small|base|thai|large] [quick|full] [overrides]" }
    if (-not $Mode) { $Mode = "full" }
    $d = Get-TextDataset $Ds
    $m = Get-TextModel $Size
    $dataArgs = @($d.Args) + @($Extra | Where-Object { $_ -match '^\+?data\.' })
    $Extra = @($Extra | Where-Object { $_ -notmatch '^\+?data\.' })
    $modeArgs = @()
    if ($Mode -eq "quick") { $modeArgs = @("hpo=none", "model.params.epochs=2") }
    elseif (-not ($Extra | Where-Object { $_ -match '^hpo\.n_trials=' })) { $modeArgs = @("hpo.n_trials=8") }
    Say "Text dataset: $($d.Name)   Model: $m   Mode: $Mode"
    Say "[1/2] Reading texts / documents..."
    uv run mlt-prepare @dataArgs
    if ($LASTEXITCODE -ne 0) { throw "Dataset check failed (see the message above)" }
    uv run mlt-featurize @dataArgs
    if ($LASTEXITCODE -ne 0) { throw "Dataset split failed (see the message above)" }
    Say "[2/2] Training $m ..."
    uv run mlt-train @dataArgs "model=$m" @modeArgs @Extra
    if ($LASTEXITCODE -ne 0) { throw "Training failed (see the message above)" }
    Say "  model : models\$($d.Name)\$m\model" "Green"
    Say "  report: reports\$($d.Name)\$m\report.xlsx" "Green"
    $script:LastReport = "reports\$($d.Name)\$m\report.xlsx"
}

# images beyond YOLO: short name -> model config (img_* classify, seg_* segment)
$VisionModels = [ordered]@{
    "tiny" = "img_tiny"; "resnet50" = "img_resnet50"; "efficientnet" = "img_efficientnet"; "convnext" = "img_convnext"; "vit" = "img_vit"
    "seg-tiny" = "seg_tiny"; "seg-b0" = "seg_b0"; "seg-b2" = "seg_b2"
}
function Get-VisionModel([string]$Size, [string]$Task) {
    if (-not $Size) { if ($Task -eq "segment") { return "seg_b0" } else { return "img_efficientnet" } }
    $s = $Size.ToLower()
    if ($s -match '^(img|seg)_') { if (Test-Path "configs\model\$s.yaml") { return $s } }
    elseif ($VisionModels.Contains($s)) { return $VisionModels[$s] }
    throw "Unknown image model '$Size' (use $($VisionModels.Keys -join ', '))"
}

function Get-VisionDataset([string]$Ds) {
    $Ds = $Ds.Trim().Trim('"').TrimEnd('\', '/')
    if (($Ds -match '[\\/:]') -and (Test-Path $Ds -PathType Container)) {
        $full = (Resolve-Path $Ds).Path
        $name = Split-Path $full -Leaf
        return @{ Name = $name; Args = @("data=images", "data.name=$name", "data.path='$($full -replace '\\', '/')'") }
    }
    if (Test-Path "configs\data\$Ds.yaml") { return @{ Name = $Ds; Args = @("data=$Ds") } }
    return @{ Name = $Ds; Args = @("data=images", "data.name=$Ds") }
}

function Invoke-Vision([string]$Ds, [string]$Size, [string]$Mode, [string[]]$Extra) {
    if (-not $Ds) { throw "Usage: run vision <dataset|folder> [$($VisionModels.Keys -join '|')] [quick|full] [overrides]" }
    if (-not $Mode) { $Mode = "full" }
    $d = Get-VisionDataset $Ds
    $dataArgs = @($d.Args) + @($Extra | Where-Object { $_ -match '^\+?data\.' })
    $Extra = @($Extra | Where-Object { $_ -notmatch '^\+?data\.' })
    Say "[1/2] Reading images ($($d.Name))..."
    uv run mlt-prepare @dataArgs
    if ($LASTEXITCODE -ne 0) { throw "Dataset check failed (see the message above)" }
    uv run mlt-featurize @dataArgs
    if ($LASTEXITCODE -ne 0) { throw "Dataset split failed (see the message above)" }
    $task = (Get-Content "data\processed\$($d.Name)\meta.json" -Raw | ConvertFrom-Json).task
    if (-not $Size -and $Ds -match '^shapes') { $Size = @("tiny", "seg-tiny")[[int]($task -eq "segment")] }
    $m = Get-VisionModel $Size $task
    $modeArgs = @()
    if ($Mode -eq "quick") { $modeArgs = @("hpo=none", "model.params.epochs=3") }
    elseif (-not ($Extra | Where-Object { $_ -match '^hpo\.n_trials=' })) { $modeArgs = @("hpo.n_trials=8") }
    Say "[2/2] Task: $task   Model: $m   Mode: $Mode"
    uv run mlt-train @dataArgs "model=$m" @modeArgs @Extra
    if ($LASTEXITCODE -ne 0) { throw "Training failed (see the message above)" }
    Say "  model : models\$($d.Name)\$m\model" "Green"
    Say "  report: reports\$($d.Name)\$m\report.xlsx" "Green"
    $script:LastReport = "reports\$($d.Name)\$m\report.xlsx"
}

# tables: "sales.csv" / "D:\x.xlsx" -> data=csv data.path=...; a built-in name -> data=<name>
$TableModels = @("lightgbm", "xgboost", "random_forest", "gradient_boosting", "logistic_regression", "svm")
function Get-TableDataset([string]$Ds) {
    $Ds = $Ds.Trim().Trim('"')
    if (Test-Path "configs\data\$Ds.yaml") { return @{ Name = $Ds; Args = @("data=$Ds") } }
    $file = $Ds
    if (-not (Test-Path $file -PathType Leaf)) {
        $file = @(".csv", ".xlsx", ".xls", ".parquet" | ForEach-Object { "data\raw\$Ds$_"; "data\external\$Ds$_" } | Where-Object { Test-Path $_ })[0]
    }
    if (-not $file) { throw "Table not found: $Ds (a .csv / .xlsx file, or put it in data\external)" }
    $full = (Resolve-Path $file).Path
    $name = [IO.Path]::GetFileNameWithoutExtension($full)
    return @{ Name = $name; Args = @("data=csv", "data.name=$name", "data.path='$($full -replace '\\', '/')'") }
}

function Invoke-Table([string]$Ds, [string]$Models, [string]$Mode, [string[]]$Extra) {
    if (-not $Ds) { throw "Usage: run table <file|name> target=<column> [time=<date column>] [group=<column>] [$($TableModels -join '|')|all] [quick|full]" }
    if (-not $Mode) { $Mode = "full" }
    $d = Get-TableDataset $Ds
    # short names: target=price time=date group=shop task=regress
    $short = @{ target = "data.target"; time = "data.time_column"; group = "data.group_column"; task = "data.task" }
    $Extra = @($Extra | ForEach-Object { $k, $v = $_ -split '=', 2; if ($short.ContainsKey($k)) { "$($short[$k])=$v" } else { $_ } })
    if ($Extra -match '^data\.time_column=' -and -not ($Extra -match '^data\.task=')) { $Extra += "data.task=forecast" }
    $dataArgs = @($d.Args) + @($Extra | Where-Object { $_ -match '^\+?data\.' })
    $Extra = @($Extra | Where-Object { $_ -notmatch '^\+?data\.' })
    if ($d.Args[0] -eq "data=csv" -and -not ($dataArgs -match '^data\.target=')) { throw "Which column to predict? add target=<column name>" }
    $list = @("lightgbm"); if ($Models -eq "all") { $list = $TableModels } elseif ($Models) { $list = @($Models -split ',') }
    $modeArgs = @(); if ($Mode -eq "quick") { $modeArgs = @("hpo=none") }
    Say "[1/2] Reading table ($($d.Name))..."
    uv run mlt-prepare @dataArgs
    if ($LASTEXITCODE -ne 0) { throw "Reading the table failed (see the message above)" }
    uv run mlt-featurize @dataArgs
    if ($LASTEXITCODE -ne 0) { throw "Preparing the table failed (see the message above)" }
    $task = (Get-Content "data\processed\$($d.Name)\meta.json" -Raw | ConvertFrom-Json).task
    $done = @()
    foreach ($m in $list) {
        Say "[2/2] Task: $task   Model: $m   Mode: $Mode"
        uv run mlt-train @dataArgs "model=$m" @modeArgs @Extra
        if ($LASTEXITCODE -ne 0) { Say "$m failed, continuing" "Yellow"; continue }
        $done += $m
    }
    if ($done.Count -eq 0) { throw "Training failed (see the message above)" }
    Show-Results $d.Name
    if ($done.Count -eq 1) { $script:LastReport = "reports\$($d.Name)\$($done[0])\report.xlsx" } else { $script:LastReport = "reports\$($d.Name)\compare.xlsx" }
    if ($task -eq "forecast") { Say "Forecast the future: run predict $($d.Name) 30" "Green" }
}

function Invoke-Find([string]$Ds, [string]$InputPath, [string]$Size, [string[]]$Extra) {
    if (-not $Ds -or -not $InputPath) { throw "Usage: run predict <dataset> <image|folder|video|url> [n|s|m|l|x]" }
    $d = Get-Dataset $Ds
    if ($Size -eq "tiny") { $m = @("img_tiny", "text_tiny")[[int](Test-Path "models\$($d.Name)\text_tiny")] }
    elseif ($Size -and ($Size -match '^(text_)?(tiny|small|base|thai|large)$')) { $m = Get-TextModel $Size }
    elseif ($Size -and ($Size -match '^(img|seg)_' -or $VisionModels.Contains($Size.ToLower()))) { $m = Get-VisionModel $Size "" }
    elseif ($Size -and (Test-Path "configs\model\$Size.yaml")) { $m = $Size }
    elseif ($Size) { $m = (Get-Sizes $Size)[0] }
    else {
        $m = (uv run python -m ml_training.compare $d.Name --best | Select-Object -Last 1)
        if ($LASTEXITCODE -ne 0 -or -not $m) { throw "No trained model for '$($d.Name)'. Train first: run train <data>" }
    }
    $src = $InputPath.Trim().Trim('"')
    $what = "+input='$src'"
    $tabular = Test-Path "models\$($d.Name)\$m.joblib"
    if ($tabular -and -not (Test-Path "configs\data\$($d.Name).yaml")) { $d.Args = @("data=csv", "data.name=$($d.Name)") }
    if ($tabular -and $src -match '^\d+$') { $what = "+horizon=$src" }  # forecast N periods ahead
    elseif ($src -notmatch '://') {
        if (Test-Path $src) { $src = (Resolve-Path $src).Path -replace '\\', '/'; $what = "+input='$src'" }
        elseif ($m -like "text_*") { $what = "+text='$($src -replace "'", "’")'" }  # a sentence
        else { throw "Not found: $src" }
    }
    $dataArgs = $d.Args
    Say "Predicting with $m on $src"
    uv run mlt-predict @dataArgs "model=$m" $what @Extra
    $out = "reports\$($d.Name)\$m\predict"
    if ($LASTEXITCODE -eq 0 -and (Test-Path $out)) {
        if ($m -like "text_*") {
            Say "Results (one row per text / entity): $out\predictions.csv" "Green"
            Import-Csv "$out\predictions.csv" | Select-Object -First 10 | Format-Table -AutoSize | Out-String | Write-Host
        } elseif ($tabular) {
            Say "Results: $out\predictions.csv" "Green"
            Import-Csv "$out\predictions.csv" | Select-Object -First 15 | Format-Table -AutoSize | Out-String | Write-Host
        } elseif ($m -like "img_*") {
            Say "Results (class + confidence + top 3 per image): $out\predictions.csv" "Green"
            Import-Csv "$out\predictions.csv" | Select-Object -First 10 | Format-Table -AutoSize | Out-String | Write-Host
        } elseif ($m -like "seg_*") { Say "Coloured overlays + masks + predictions.csv (% of each class per image): $out" "Green" }
        else { Say "Annotated files + detections.csv (one row per prediction): $out" "Green" }
        Invoke-Item $out
    }
}

function Invoke-Explain([string]$Ds, [string]$Size, [string]$Count, [string[]]$Extra) {
    if (-not $Ds) { throw "Usage: run explain <dataset> [n|s|m|l|x] [images]" }
    $d = Get-Dataset $Ds
    if ($Size) { $m = (Get-Sizes $Size)[0] }
    else {
        $m = (uv run python -m ml_training.compare $d.Name --best | Select-Object -Last 1)
        if ($LASTEXITCODE -ne 0 -or -not $m) { throw "No trained model for '$($d.Name)'. Train first: run fit $Ds" }
    }
    $more = @($Extra)
    if ($Count) { $more += "xai.max_images=$Count" }
    $dataArgs = $d.Args
    Say "Explaining $m on $($d.Name) (heatmaps + statistics)..."
    uv run python -m ml_training.xai.run @dataArgs "model=$m" @more
    $x = "reports\$($d.Name)\$m\xai\xai_report.xlsx"
    if ($LASTEXITCODE -eq 0 -and (Test-Path $x)) {
        Say "XAI report: $x   (heatmap images: reports\$($d.Name)\$m\xai\images)" "Green"
        $script:LastReport = $x
    }
}

# unfinished trainings as "dataset|model|stage|time" (from models\<dataset>\<model>\run_state.json)
function Get-Unfinished {
    $stages = @{ start = "starting"; hpo = "Optuna search"; final = "final training"; evaluate = "evaluation" }
    @(Get-ChildItem models\*\*\run_state.json -ErrorAction SilentlyContinue | ForEach-Object {
        $j = Get-Content $_.FullName -Raw | ConvertFrom-Json
        if ($j.status -ne "finished") {
            $st = $stages[[string]$j.stage]; if (-not $st) { $st = [string]$j.stage }
            "{0}|{1}|stopped in {2}|{3}" -f $_.Directory.Parent.Name, $_.Directory.Name, $st, $j.updated
        }
    })
}

function Invoke-Resume([string]$Ds, [string]$Size) {
    if (-not $Ds) {
        $u = Get-Unfinished
        if (-not $u) { Say "Nothing to resume - every training finished." "Green"; return }
        Say "Unfinished trainings:"
        $u | ForEach-Object { $f = $_ -split '\|'; Write-Host ("  {0,-16} {1,-9} {2}  ({3})" -f $f[0], $f[1], $f[2], $f[3]) }
        Say "Continue one with: run resume <dataset> [size]"
        return
    }
    $name = (Get-Dataset $Ds).Name
    $m = @(); if ($Size) { $m = @((Get-Sizes $Size)[0]) }
    uv run python -m ml_training.resume run $name @m
    if ($LASTEXITCODE -eq 0) {
        $done = @(Get-ChildItem "reports\$name" -Recurse -Filter report.xlsx -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
        if ($done) { $script:LastReport = $done[0].FullName; Say "Finished. Excel report: $($script:LastReport)" "Green" }
    }
}

function Invoke-Clean([string[]]$A) {
    if (-not $A -or -not $A[0]) { throw "Usage: run clean <dataset|all> [raw]" }
    $flags = @()
    if ($A -contains "raw") { $flags += "--raw" }
    uv run python -m ml_training.clean $A[0] @flags
}

# `run` in any new terminal (cmd or PowerShell): %LOCALAPPDATA%\ml-training\bin\run.cmd on the user PATH
function Install-Command {
    $bin = Join-Path $env:LOCALAPPDATA "ml-training\bin"
    $other = Get-Command run -ErrorAction SilentlyContinue | Where-Object { $_.Source -and -not $_.Source.StartsWith($bin) -and -not $_.Source.StartsWith($PSScriptRoot) }
    if ($other) { Say ((T "มีคำสั่ง 'run' อื่นอยู่แล้ว - ใช้ run.bat / .\run.ps1 แทน: " "Another 'run' command exists - keep using run.bat / .\run.ps1: ") + $other[0].Source) "Yellow"; return }
    New-Item $bin -ItemType Directory -Force | Out-Null
    Set-Content (Join-Path $bin "run.cmd") "@echo off`r`nset MLT_NOPAUSE=1`r`ncall `"$PSScriptRoot\run.bat`" %*" -Encoding Default
    $p = [Environment]::GetEnvironmentVariable("Path", "User"); if (-not $p) { $p = "" }
    if (($p -split ';') -notcontains $bin) {
        [Environment]::SetEnvironmentVariable("Path", (($p.TrimEnd(';') + ";$bin").TrimStart(';')), "User")
        Say (T "ติดตั้งคำสั่ง 'run' แล้ว - เปิดหน้าต่าง terminal ใหม่แล้วพิมพ์ run ได้จากทุกโฟลเดอร์" "Installed the 'run' command - open a new terminal and type run from any folder") "Green"
    }
}

function Invoke-Setup {
    Need-Uv
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { Say (T "พบการ์ดจอ NVIDIA - ติดตั้งแบบ GPU" "NVIDIA GPU found - installing the GPU version") "Green"; Sync-Env -Gpu }
    else { Say (T "ไม่พบการ์ดจอ NVIDIA - ติดตั้งแบบ CPU" "No NVIDIA GPU - installing the CPU version") "Yellow"; Remove-Item .gpu -ErrorAction SilentlyContinue; Sync-Env }
    Install-Command
    uv run python -m ml_training.doctor
    Say ((T "`nพร้อมแล้ว! ลองเลย:  " "`nReady! Try:  ") + "run    " + (T "หรือ" "or") + "  run train shop-sales quick") "Green"
}

function Show-Status {
    $t = @(Get-TrainedNames)
    Say ((T "โมเดลที่เทรนแล้ว: " "Trained models: ") + $(if ($t) { $t -join ', ' } else { T "ยังไม่มี" "none yet" }))
    $u = Get-Unfinished
    if ($u) {
        Say (T "งานที่ค้าง (run resume):" "Stopped trainings (run resume):") "Yellow"
        $u | ForEach-Object { $f = $_ -split '\|'; Write-Host ("  {0,-16} {1,-9} {2}  ({3})" -f $f[0], $f[1], $f[2], $f[3]) }
    } else { Say (T "ไม่มีงานค้าง" "No stopped trainings") "Green" }
}

# ----------------------------------------------------------------------------- menu
function Read-Choice([string]$Prompt, [string[]]$Options, [int]$Default = 1) {
    for ($i = 0; $i -lt $Options.Count; $i++) { Write-Host ("  {0}) {1}" -f ($i + 1), $Options[$i]) }
    while ($true) {
        $a = Read-Host "$Prompt [$Default]"
        if (-not $a) { return $Default - 1 }
        $n = 0
        if ([int]::TryParse($a, [ref]$n) -and $n -ge 1 -and $n -le $Options.Count) { return $n - 1 }
        Say (T "พิมพ์ตัวเลข 1-$($Options.Count)" "Please type a number 1-$($Options.Count)") "Yellow"
    }
}

function Ask-Yes([string]$Question, [bool]$Default = $true) {
    $d = @("n", "y")[[int]$Default]
    $a = Read-Host "$Question (y/n) [$d]"
    if (-not $a) { return $Default }
    return $a.Trim().ToLower() -in @("y", "yes", "ใช่")
}

function Open-Report {
    if ($script:LastReport -and (Test-Path $script:LastReport) -and (Ask-Yes (T "เปิดรายงาน Excel เลยไหม" "Open the Excel report?"))) { Invoke-Item $script:LastReport }
}

function Get-TrainedNames {
    @(Get-ChildItem models -Directory -ErrorAction SilentlyContinue | Where-Object { "pretrained", "runs" -notcontains $_.Name } |
        Where-Object { (Get-ChildItem $_.FullName -File | Where-Object { ".pt", ".joblib" -contains $_.Extension }) -or (Test-Path "$($_.FullName)\*\model") } |
        ForEach-Object Name)
}

function Select-Dataset([switch]$Trained) {
    if ($Trained) { $names = Get-TrainedNames }
    else { $names = @(Get-ChildItem data\raw -Directory -ErrorAction SilentlyContinue | ForEach-Object Name) }
    $names = @($names | Where-Object { "pretrained", "runs" -notcontains $_ })
    if ($Trained -and -not $names) { throw (T "ยังไม่มีโมเดลที่เทรนแล้ว - เทรนก่อนจากเมนู 1" "No trained model yet - train one first (menu 1)") }
    Say (T "`nชุดข้อมูล:" "`nDataset:")
    $opts = $names + @((T "พิมพ์ / ลากโฟลเดอร์มาวาง..." "Type a folder path..."))
    if (-not $Trained) { $opts += (T "ชุดตัวอย่าง (ดาวน์โหลดอัตโนมัติ ไว้ลองแต่ละงาน)..." "Sample dataset (auto-download, to try each task)...") }
    $i = Read-Choice (T "เลือก" "Choose") $opts
    if ($i -lt $names.Count) { return $names[$i] }
    if ($i -eq $names.Count) { return (Read-Host (T "ที่อยู่โฟลเดอร์ (ลากโฟลเดอร์มาวางได้)" "Folder path (you can drag the folder here)")).Trim('"') }
    $samples = @("coco8", "coco8-seg", "coco8-pose", "dota8", "mnist160")
    $j = Read-Choice (T "ชุดตัวอย่าง" "Sample") @("coco8       detect", "coco8-seg   segment", "coco8-pose  pose", "dota8       obb", "mnist160    classify")
    return $samples[$j]
}

function Select-TextDataset {
    $names = @(Get-ChildItem data\raw -ErrorAction SilentlyContinue | Where-Object { $_.PSIsContainer -or ".csv", ".xlsx", ".jsonl", ".json", ".tsv", ".conll" -contains $_.Extension } | ForEach-Object Name)
    Say (T "`nข้อมูลข้อความ (โฟลเดอร์หรือไฟล์ใน data\raw):" "`nText dataset (folder or file in data\raw):")
    $opts = $names + @((T "พิมพ์ / ลากไฟล์หรือโฟลเดอร์มาวาง..." "Type a file / folder path..."),
        (T "ตัวอย่าง: thai-sentiment  (จัดหมวดรีวิว บวก / ลบ / กลาง)" "Sample: thai-sentiment  (classify reviews)"),
        (T "ตัวอย่าง: thai-ner        (ดึงชื่อคน บริษัท วันที่ จำนวนเงิน)" "Sample: thai-ner        (names, companies, dates, amounts)"),
        (T "ตัวอย่าง: thai-docs       (แยกใบแจ้งหนี้ / ใบเสร็จ / สัญญา / ใบเสนอราคา)" "Sample: thai-docs       (invoice / receipt / contract / quotation)"))
    $i = Read-Choice (T "เลือก" "Choose") $opts
    if ($i -lt $names.Count) { return "data\raw\" + $names[$i] }
    if ($i -eq $names.Count) { return (Read-Host (T "ที่อยู่ไฟล์หรือโฟลเดอร์ (ลากมาวางได้)" "File or folder path (you can drag it here)")).Trim('"') }
    return @("thai-sentiment", "thai-ner", "thai-docs")[$i - $names.Count - 1]
}

function Select-TableDataset {
    $names = @(Get-ChildItem data\external, data\raw -File -ErrorAction SilentlyContinue | Where-Object { ".csv", ".xlsx", ".xls", ".parquet" -contains $_.Extension } | ForEach-Object FullName)
    Say (T "`nตาราง (CSV / Excel ใน data\external หรือ data\raw):" "`nTable (CSV / Excel in data\external or data\raw):")
    $opts = @($names | ForEach-Object { Resolve-Path $_ -Relative }) + @((T "พิมพ์ / ลากไฟล์มาวาง..." "Type a file path..."),
        (T "ตัวอย่าง: house-prices  (ทายราคาบ้าน = ทายตัวเลข)" "Sample: house-prices  (predict a price = regression)"),
        (T "ตัวอย่าง: shop-sales    (พยากรณ์ยอดขายรายวัน 3 สาขา)" "Sample: shop-sales    (forecast daily sales of 3 shops)"),
        (T "ตัวอย่าง: breast_cancer (จัดหมวด 2 กลุ่ม)" "Sample: breast_cancer (classify, 2 classes)"))
    $i = Read-Choice (T "เลือก" "Choose") $opts
    if ($i -lt $names.Count) { return $names[$i] }
    if ($i -eq $names.Count) { return (Read-Host (T "ที่อยู่ไฟล์ (ลากมาวางได้)" "File path (you can drag it here)")).Trim('"') }
    return @("house-prices", "shop-sales", "breast_cancer")[$i - $names.Count - 1]
}

function Select-ImageDataset {
    $names = @(Get-ChildItem data\raw -Directory -ErrorAction SilentlyContinue | ForEach-Object Name)
    Say (T "`nชุดภาพ (โฟลเดอร์ใน data\raw: โฟลเดอร์ย่อยตามหมวด หรือ images\ + masks\):" "`nImage dataset (folder in data\raw: one sub-folder per class, or images\ + masks\):")
    $opts = $names + @((T "พิมพ์ / ลากโฟลเดอร์มาวาง..." "Type a folder path..."),
        (T "ตัวอย่าง: shapes      (จัดหมวด วงกลม / สี่เหลี่ยม / สามเหลี่ยม)" "Sample: shapes      (classify circle / square / triangle)"),
        (T "ตัวอย่าง: shapes-seg  (แยกทุกพิกเซล วงกลม / สี่เหลี่ยม)" "Sample: shapes-seg  (pixel segmentation of circles / squares)"))
    $i = Read-Choice (T "เลือก" "Choose") $opts
    if ($i -lt $names.Count) { return $names[$i] }
    if ($i -eq $names.Count) { return (Read-Host (T "ที่อยู่โฟลเดอร์ (ลากมาวางได้)" "Folder path (you can drag it here)")).Trim('"') }
    return @("shapes", "shapes-seg")[$i - $names.Count - 1]
}

# presets = configs\tune\*.yaml (built-in ones first, then yours)
function Get-Presets {
    $rows = @(uv run python -m ml_training.tune_menu names | Where-Object { $_ -match '\|' })
    if ($rows.Count -eq 0) { $rows = @(Get-ChildItem configs\tune -Filter *.yaml | ForEach-Object { "$($_.BaseName)|" }) }
    return $rows | ForEach-Object { $n, $about = $_ -split '\|', 2; [pscustomobject]@{ Name = $n; About = $about } }
}

function Select-Preset {
    $p = @(Get-Presets)
    $i = Read-Choice (T "เลือก" "Choose") @($p | ForEach-Object { "{0,-12}{1}" -f $_.Name, $_.About })
    return $p[$i].Name
}

function Show-TuningMenu {
    while ($true) {
        Say (T "`n---------- ตั้งค่าการจูน (YOLO) ----------" "`n---------- Tuning settings ----------")
        $k = Read-Choice (T "เลือก" "Choose") @(
            (T "ดู preset ทั้งหมด (epochs, จำนวนรอบจูน, freeze, ค่าที่จูน)" "Show all presets (epochs, trials, freeze, how many settings are tuned)"),
            (T "สร้าง preset ของฉัน (ถามเฉพาะค่าที่ใช้บ่อย)" "Create my preset (asks only the common settings)"),
            (T "แก้ preset ใน Notepad (ตรวจให้เมื่อปิด)" "Edit a preset in Notepad (checked when you close it)"),
            (T "กลับ" "Back"))
        switch ($k) {
            0 { uv run python -m ml_training.tune_menu list }
            1 { uv run python -m ml_training.tune_menu create }
            2 {
                $name = Select-Preset
                Start-Process notepad "configs\tune\$name.yaml" -Wait
                uv run python -m ml_training.tune_menu check $name
                if ($LASTEXITCODE -ne 0) { Say (T "ไฟล์มีปัญหา (ดูด้านบน) - เปิดแก้อีกครั้ง" "The file has a problem (see above) - open it again and fix it.") "Red" }
                else { Say (T "Preset '$name' ใช้ได้" "Preset '$name' is OK.") "Green" }
            }
            3 { return }
        }
    }
}

# ----------------------------------------------------------------------------- automatic training
# sample name -> what the inspector would say (samples are generated / downloaded on first use)
$SampleKinds = [ordered]@{
    "shapes"         = @("vision", "classify", "ภาพ: จัดหมวดรูปทรง (วงกลม / สี่เหลี่ยม / สามเหลี่ยม)", "images: classify shapes")
    "shapes-seg"     = @("vision", "semantic", "ภาพ: แยกทุกพิกเซล (segmentation)", "images: pixel segmentation")
    "coco8"          = @("yolo", "detect", "ภาพ: ตรวจจับวัตถุ YOLO (8 ภาพ ดาวน์โหลดเล็กน้อย)", "images: YOLO object detection (small download)")
    "thai-sentiment" = @("text", "classify", "ข้อความ: จัดหมวดรีวิวภาษาไทย", "text: classify Thai reviews")
    "thai-docs"      = @("text", "classify", "เอกสาร: แยกประเภทเอกสาร 4 แบบ", "documents: 4 document types")
    "house-prices"   = @("table", "regress", "ตาราง: ทายราคาบ้าน (ทายตัวเลข)", "table: predict house prices")
    "shop-sales"     = @("table", "forecast", "ตาราง: พยากรณ์ยอดขายรายวัน", "table: forecast daily sales")
}

function Invoke-Wizard([string]$Given, [string]$Level) {
    Say (T "`n========== เทรนโมเดลใหม่แบบอัตโนมัติ ==========" "`n========== Train a new model automatically ==========")
    if ($Given) { $ans = $Given.Trim().Trim('"') } else {
    Say (T "ลากโฟลเดอร์หรือไฟล์ข้อมูลมาวางในหน้าต่างนี้แล้วกด Enter  หรือพิมพ์หมายเลขด้านล่าง" "Drag a data folder or file into this window and press Enter, or type a number below") "White"
    $have = @(Get-ChildItem data\raw, data\external -ErrorAction SilentlyContinue | Where-Object { ($_.PSIsContainer -and $_.Name -notmatch '^\.') -or ".csv", ".xlsx", ".xls", ".parquet", ".jsonl" -contains $_.Extension } | Sort-Object Name)
    $have = @($have | Where-Object { -not $SampleKinds.Contains($_.BaseName) })
    $opts = @($have | ForEach-Object { (T "ข้อมูลของฉัน: " "my data: ") + (Resolve-Path $_.FullName -Relative) })
    $opts += @($SampleKinds.Keys | ForEach-Object { (T "ตัวอย่าง: " "sample: ") + ("{0,-15}" -f $_) + (T $SampleKinds[$_][2] $SampleKinds[$_][3]) })
    for ($i = 0; $i -lt $opts.Count; $i++) { Write-Host ("  {0}) {1}" -f ($i + 1), $opts[$i]) }
    $ans = (Read-Host (T "ข้อมูล" "Data")).Trim().Trim('"')
    if (-not $ans) { return }
    $n = 0
    if ([int]::TryParse($ans, [ref]$n) -and $n -ge 1 -and $n -le $opts.Count) {
        if ($n -le $have.Count) { $ans = $have[$n - 1].FullName } else { $ans = @($SampleKinds.Keys)[$n - $have.Count - 1] }
    }
    }
    if ($SampleKinds.Contains($ans)) {
        $s = $SampleKinds[$ans]
        $info = [pscustomobject]@{ kind = $s[0]; task = $s[1]; name = $ans; path = $ans; summary = (T $s[2] $s[3]); task_th = ""; gpu = $false; vram_gb = 0; thai = $true; target = $null; time_column = $null; group_column = $null; columns = @() }
        $sample = $true
    } else {
        Say (T "กำลังตรวจข้อมูล..." "Looking at the data...")
        $json = uv run python -m ml_training.inspect_data $ans --json | Select-Object -Last 1
        try { $info = $json | ConvertFrom-Json } catch { throw (T "อ่านผลตรวจข้อมูลไม่ได้: $json" "Could not read the data check: $json") }
        if ($info.error) { Say $info.error "Red"; return }
        $sample = $false
    }
    Say ""
    Say (T "พบข้อมูล:" "Found:")
    Say "  $($info.summary)" "White"
    if ($info.task_th) { Say ((T "  งานที่เหมาะ: " "  Task: ") + $info.task_th) "White" }
    if (-not $sample) {
        if ($info.gpu) { Say ((T "  เครื่องนี้: การ์ดจอ " "  This PC: GPU ") + "$($info.gpu_name) ($($info.vram_gb) GB)") "Green" }
        elseif ($info.kind -ne "table") { Say (T "  เครื่องนี้ไม่มีการ์ดจอ NVIDIA - จะช้า แนะนำเลือก 'ทดลองเร็ว' ก่อน" "  No NVIDIA GPU - training will be slow, try 'quick' first") "Yellow" }
    }

    $extra = @()
    if ($info.kind -eq "table" -and -not $sample) {
        Say ((T "`nคอลัมน์: " "`nColumns: ") + ($info.columns -join ', '))
        $t = (Read-Host ((T "คอลัมน์ที่ต้องการให้โมเดลทาย" "Column to predict") + " [$($info.target)]")).Trim()
        if (-not $t) { $t = $info.target }
        if ($info.columns -notcontains $t) { Say (T "ไม่มีคอลัมน์ '$t' ในตาราง" "No column '$t' in the table") "Red"; return }
        $extra += "target=$t"
        if ($info.time_column -and (Ask-Yes (T "เป็นข้อมูลตามเวลา (คอลัมน์ '$($info.time_column)') ต้องการพยากรณ์อนาคตไหม" "Time series on '$($info.time_column)' - forecast the future?"))) {
            $extra += "time=$($info.time_column)"
            if ($info.group_column) { $extra += "group=$($info.group_column)"; Say (T "  แยกพยากรณ์ตาม '$($info.group_column)'" "  one series per '$($info.group_column)'") }
        } elseif ($info.task -eq "forecast") { $extra += "task=regress" }
    }

    $preset = @{ quick = 0; standard = 1; full = 1; best = 2; accurate = 2 }
    if ($Level -and $preset.ContainsKey($Level.ToLower())) { $level = $preset[$Level.ToLower()] } else {
    $Level = ""
    Say (T "`nต้องการแบบไหน:" "`nHow thorough:")
    $level = Read-Choice (T "เลือก" "Choose") @(
        (T "ทดลองเร็ว     - ไม่กี่นาที ไว้ดูว่าระบบทำงาน / ข้อมูลใช้ได้" "Quick        - a few minutes, to check the data works"),
        (T "มาตรฐาน       - จูนค่าอัตโนมัติ ได้โมเดลที่ดี (แนะนำ)" "Standard     - automatic tuning, a good model (recommended)"),
        (T "แม่นที่สุด     - โมเดลใหญ่ขึ้น / เทียบหลายโมเดล ใช้เวลานาน" "Most accurate - bigger model / compare several, takes long")) 2
    }
    $mode = @("quick", "full", "full")[$level]
    $ds = $info.path
    $script:LastReport = $null
    switch ($info.kind) {
        "yolo" {
            $size = @("n", "s", "m")[$level]; if (-not $info.gpu -and -not $sample -and $level -gt 0) { $size = "n" }
            $tune = @("default", "default", "accurate")[$level]
            Say ((T "`nจะเทรน: YOLO11" "`nWill train: YOLO11") + "$size  ($($info.task), $mode)") "White"
            if (-not $Level -and -not (Ask-Yes (T "เริ่มเลยไหม" "Start now?"))) { return }
            Invoke-Detect $ds $size $mode @("tune=$tune", "data.task=$($info.task)")
        }
        "vision" {
            if ($info.task -eq "semantic") { $size = @("seg-b0", "seg-b0", "seg-b2")[$level]; $vt = "segment" }
            else { $size = @("efficientnet", "efficientnet", "convnext")[$level]; $vt = "classify" }
            if ($sample) { $size = @("tiny", "seg-tiny")[[int]($vt -eq "segment")] }
            Say ((T "`nจะเทรน: " "`nWill train: ") + "$size  ($vt, $mode)") "White"
            if (-not $Level -and -not (Ask-Yes (T "เริ่มเลยไหม" "Start now?"))) { return }
            Invoke-Vision $ds $size $mode @("data.task=$vt")
        }
        "text" {
            $std = @("base", "thai")[[int][bool]$info.thai]
            $size = @("small", $std, "large")[$level]
            if ($level -eq 2 -and $info.vram_gb -lt 12) { $size = $std; Say (T "  การ์ดจอเล็กกว่า 12 GB - ใช้โมเดล $std แทน large" "  GPU under 12 GB - using $std instead of large") "Yellow" }
            if ($sample) { $size = "tiny" }
            Say ((T "`nจะเทรน: โมเดลข้อความ " "`nWill train: text model ") + "$size  ($($info.task), $mode)") "White"
            if (-not $Level -and -not (Ask-Yes (T "เริ่มเลยไหม" "Start now?"))) { return }
            Invoke-Text $ds $size $mode @()
        }
        "table" {
            $models = @("lightgbm", "lightgbm", "all")[$level]
            Say ((T "`nจะเทรน: " "`nWill train: ") + "$models  ($mode)") "White"
            if (-not $Level -and -not (Ask-Yes (T "เริ่มเลยไหม" "Start now?"))) { return }
            Invoke-Table $ds $models $mode $extra
        }
    }
    $nm = Split-Path ([string]$ds).TrimEnd('\', '/') -Leaf
    if (Test-Path $ds -PathType Leaf) { $nm = [IO.Path]::GetFileNameWithoutExtension($nm) }
    Say (T "`nเสร็จแล้ว! ทำนายข้อมูลใหม่: run predict $nm <ไฟล์/โฟลเดอร์> · ดูผล: run results $nm" "`nDone! Predict new data: run predict $nm <file/folder> · scores: run results $nm") "Green"
    Open-Report
}

# ----------------------------------------------------------------------------- menus
function Show-TrainMenu {
    Say (T "`n---------- เทรนแบบเลือกเอง ----------" "`n---------- Train (choose yourself) ----------")
    $k = Read-Choice (T "เลือก" "Choose") @(
        (T "YOLO: ตรวจจับวัตถุ / แยกวัตถุ / จัดหมวด / pose / กรอบเอียง" "YOLO (detect / segment / classify / pose / obb)"),
        (T "ข้อความ / เอกสาร: จัดหมวดข้อความ ดึงข้อมูลสำคัญ (NER) แยกประเภทเอกสาร" "TEXT / DOCUMENTS (classify texts, NER, document folders)"),
        (T "ภาพ: จัดหมวดด้วย ResNet / EfficientNet / ConvNeXt / ViT หรือแยกทุกพิกเซล" "IMAGES beyond YOLO (classify, pixel segmentation)"),
        (T "ตาราง (CSV / Excel): จัดหมวด ทายตัวเลข พยากรณ์อนาคต" "TABLES (classify, predict a number, forecast)"),
        (T "กลับ" "Back"))
    $script:LastReport = $null
    switch ($k) {
        0 {
            $ds = Select-Dataset
            Say (T "`nรุ่น YOLO:" "`nYOLO version:")
            $gen = @("", "26")[(Read-Choice (T "เลือก" "Choose") @((T "YOLO11 (เสถียร ค่าเริ่มต้น)" "YOLO11 (proven, default)"), (T "YOLO26 (ใหม่ล่าสุด)" "YOLO26 (newest)")))]
            Say (T "`nขนาดโมเดล (n = เร็วที่สุด ... x = แม่นที่สุด):" "`nModel size (n = fastest ... x = most accurate):")
            $sizeOpts = @("n  (nano)", "s  (small)", "m  (medium)", "l  (large)", "x  (extra large)", (T "ทุกขนาด ทีละตัว แล้วเทียบกัน" "all sizes, one after another"))
            $size = $gen + @("n", "s", "m", "l", "x", "all")[(Read-Choice (T "เลือก" "Choose") $sizeOpts)]
            Say (T "`nโหมด:" "`nTraining mode:")
            $mode = @("quick", "full")[(Read-Choice (T "เลือก" "Choose") @((T "ทดลองเร็ว (ไม่จูน 30 epochs)" "Quick test (no tuning, 30 epochs)"), (T "เต็ม (Optuna จูน + epochs ตาม preset)" "Full (Optuna tuning + epochs from the tuning preset)")))]
            Say (T "`nPreset การจูน:" "`nTuning preset:")
            $tune = Select-Preset
            Invoke-Detect $ds $size $mode @("tune=$tune")
        }
        1 {
            $ds = Select-TextDataset
            Say (T "`nโมเดล (อ่านภาษาไทยได้ทุกตัว):" "`nModel (all read Thai):")
            $k2 = Read-Choice (T "เลือก" "Choose") @(
                (T "tiny   เทรนจากศูนย์ ไม่ต้องดาวน์โหลด - ไว้ลองระบบเท่านั้น" "tiny   trained from scratch, no download - only to try the pipeline"),
                (T "small  DistilBERT หลายภาษา - เร็ว ใช้ CPU ได้" "small  multilingual DistilBERT - fast, OK on CPU"),
                (T "base   XLM-RoBERTa - แม่นและรองรับไทยดี (แนะนำ GPU)" "base   XLM-RoBERTa base - strong default (GPU recommended)"),
                (T "thai   WangchanBERTa - ดีที่สุดสำหรับข้อความไทยล้วน" "thai   WangchanBERTa - best for Thai-only text"),
                (T "large  XLM-RoBERTa large - แม่นที่สุด (GPU 12 GB+)" "large  XLM-RoBERTa large - most accurate (GPU 12 GB+)")) 3
            Say (T "`nโหมด:" "`nTraining mode:")
            $mode = @("quick", "full")[(Read-Choice (T "เลือก" "Choose") @((T "ทดลองเร็ว (ไม่จูน 2 epochs)" "Quick test (no tuning, 2 epochs)"), (T "เต็ม (Optuna 8 รอบ + เทรนจริง)" "Full (Optuna 8 trials + final training)")))]
            Invoke-Text $ds $TextSizes[$k2] $mode @()
        }
        2 {
            $ds = Select-ImageDataset
            $seg = $ds -eq "shapes-seg" -or ((Test-Path "data\raw\$ds") -and (Get-ChildItem "data\raw\$ds" -Directory | Where-Object { "masks", "mask", "SegmentationClass" -contains $_.Name }))
            if ($ds -match '^shapes') { $size = @("tiny", "seg-tiny")[[int][bool]$seg] }
            elseif ($seg) {
                Say (T "`nโมเดลแยกทุกพิกเซล (SegFormer):" "`nSegmentation model (SegFormer, pretrained):")
                $size = @("seg-b0", "seg-b2")[(Read-Choice (T "เลือก" "Choose") @((T "seg-b0  เร็ว ใช้การ์ดจอไหนก็ได้" "seg-b0  fast, any GPU"), (T "seg-b2  แม่นกว่า ต้อง GPU 8 GB+" "seg-b2  more accurate, GPU 8 GB+")))]
            } else {
                Say (T "`nโมเดลจัดหมวดภาพ (ดาวน์โหลด weight ครั้งแรกครั้งเดียว):" "`nClassification model (ImageNet-pretrained, downloaded once):")
                $size = @("efficientnet", "resnet50", "convnext", "vit")[(Read-Choice (T "เลือก" "Choose") @(
                    (T "efficientnet  เล็ก เร็ว แม่นดี (แนะนำ)" "efficientnet  EfficientNet-B0 - small and fast, good default"),
                    (T "resnet50      มาตรฐาน ทนทาน" "resnet50      ResNet-50 - classic, robust"),
                    (T "convnext      มักแม่นที่สุดในขนาดนี้" "convnext      ConvNeXt-Tiny - usually the most accurate here"),
                    (T "vit           Vision Transformer ข้อมูลเยอะ ต้อง GPU 8 GB+" "vit           ViT-B/16 - best with lots of data (GPU 8 GB+)")))]
            }
            Say (T "`nโหมด:" "`nTraining mode:")
            $mode = @("quick", "full")[(Read-Choice (T "เลือก" "Choose") @((T "ทดลองเร็ว (ไม่จูน 3 epochs)" "Quick test (no tuning, 3 epochs)"), (T "เต็ม (Optuna 8 รอบ + เทรนจริง)" "Full (Optuna 8 trials + final training)")))]
            Invoke-Vision $ds $size $mode @()
        }
        3 {
            $ds = Select-TableDataset
            $extra = @()
            if (-not (Test-Path "configs\data\$ds.yaml")) {
                $cols = @()
                if ($ds -like "*.csv") { $cols = @((Get-Content $ds -TotalCount 1 -Encoding UTF8) -split ',' | ForEach-Object { $_.Trim('"') }) }
                if ($cols) { Say ((T "`nคอลัมน์: " "`nColumns: ") + ($cols -join ', ')) }
                $extra += "target=" + (Read-Host (T "คอลัมน์ที่ต้องการทาย (target)" "Column to predict (target)"))
                $t = Read-Host (T "คอลัมน์วันที่ / เวลา ถ้าต้องการพยากรณ์อนาคต (เว้นว่าง = ไม่ใช่ข้อมูลตามเวลา)" "Date / time column for forecasting (empty = not a time series)")
                if ($t) {
                    $extra += "time=$t"
                    $g = Read-Host (T "คอลัมน์ที่แยกหลายชุด เช่น สาขา / สินค้า (เว้นว่าง = ชุดเดียว)" "Column that separates several series, e.g. shop or product (empty = one series)")
                    if ($g) { $extra += "group=$g" }
                }
            }
            Say (T "`nโมเดล:" "`nModel:")
            $k2 = Read-Choice (T "เลือก" "Choose") @(
                (T "lightgbm      เร็ว มักแม่นที่สุดกับตาราง (แนะนำ)" "lightgbm      LightGBM - fast, usually the most accurate on tables"),
                (T "xgboost       แม่นมาก" "xgboost       XGBoost - very accurate gradient boosting"),
                (T "random_forest ทนทาน ตั้งค่าน้อย" "random_forest Random forest - robust, few settings"),
                (T "all           เทรนทุกโมเดลแล้วเทียบกัน" "all           train every model and compare"))
            $models = @("lightgbm", "xgboost", "random_forest", "all")[$k2]
            Say (T "`nโหมด:" "`nTraining mode:")
            $mode = @("quick", "full")[(Read-Choice (T "เลือก" "Choose") @((T "ทดลองเร็ว (ไม่จูน)" "Quick (no tuning)"), (T "เต็ม (Optuna 30 รอบ + โมเดลสุดท้าย)" "Full (Optuna 30 trials + final model)")) 2)]
            Invoke-Table $ds $models $mode $extra
        }
        4 { return }
    }
    Open-Report
}

function Show-MoreMenu {
    while ($true) {
        Say (T "`n---------- เครื่องมือเพิ่มเติม ----------" "`n---------- More tools ----------")
        $k = Read-Choice (T "เลือก" "Choose") @(
            (T "ตรวจเครื่อง (การ์ดจอ ส่วนที่ติดตั้งแล้ว พื้นที่ดิสก์)" "Check this computer (GPU, installed parts, disk)"),
            (T "ติดตั้ง / อัปเดต (หาการ์ดจอเอง + ติดตั้งคำสั่ง run)" "Install / update (finds the GPU, installs the run command)"),
            (T "ติดตั้ง OCR (อ่านเอกสารสแกน / รูปเอกสาร)" "Install OCR (scanned documents)"),
            (T "ตั้งค่าการจูน YOLO (ดู / สร้าง / แก้ preset)" "YOLO tuning presets (view / create / edit)"),
            (T "เปิด MLflow (ประวัติทุกการเทรน กราฟ และโมเดลที่ลงทะเบียน)" "Open MLflow UI (all runs, metrics, model registry)"),
            (T "เปิด Optuna dashboard (การจูน)" "Open Optuna dashboard"),
            (T "ประวัติการใช้งาน (เตรียมข้อมูล / เทรน / ทำนาย / ล้าง)" "History (every prepare / train / predict / clean)"),
            (T "ล้างข้อมูล / ผลลัพธ์ (ถามก่อนลบ)" "Clean data / results (asks before deleting)"),
            (T "ตัวอย่าง DVC pipeline (ข้อมูลตาราง breast cancer)" "Tabular demo: run the DVC pipeline (breast cancer)"),
            (T "คำสั่งทั้งหมด (สำหรับพิมพ์เอง)" "Help (all commands)"),
            (T "Language / ภาษา: เปลี่ยนเป็น English" "Language / ภาษา: switch to Thai"),
            (T "กลับ" "Back"))
        try {
            switch ($k) {
                0 { uv run python -m ml_training.doctor }
                1 { Invoke-Setup }
                2 { & $PSCommandPath setup-ocr }
                3 { Show-TuningMenu }
                4 { Start-Process "http://127.0.0.1:5000"; uv run mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5000 }
                5 { Start-Process "http://127.0.0.1:8080"; uv run optuna-dashboard sqlite:///optuna.db --port 8080 }
                6 {
                    uv run python -m ml_training.lineage 30 --excel
                    if ((Test-Path logs\history.xlsx) -and (Ask-Yes (T "เปิด history.xlsx ไหม" "Open history.xlsx?") $false)) { Invoke-Item logs\history.xlsx }
                }
                7 {
                    $names = @(@("data\processed", "models", "reports") | ForEach-Object { Get-ChildItem $_ -Directory -ErrorAction SilentlyContinue } | ForEach-Object Name | Where-Object { "pretrained", "runs", "figures" -notcontains $_ } | Sort-Object -Unique)
                    Say (T "`nล้างอะไร?" "`nClean what?")
                    $opts = $names + @((T "ทุกชุดข้อมูล (ผลทั้งหมด + ประวัติ MLflow / Optuna)" "ALL datasets (every result + MLflow / Optuna history)"))
                    $c = Read-Choice (T "เลือก" "Choose") $opts
                    $target = "all"; if ($c -lt $names.Count) { $target = $names[$c] }
                    $raw = Ask-Yes (T "ลบข้อมูลดิบใน data\raw ด้วยไหม" "Also delete the raw dataset in data\raw?") $false
                    if ($raw) { Invoke-Clean @($target, "raw") } else { Invoke-Clean @($target) }
                }
                8 { uv run dvc repro; uv run dvc metrics show }
                9 { (Get-Content $PSCommandPath -TotalCount $HelpLines -Encoding UTF8) -join "`n" | Write-Host }
                10 {
                    $script:Lang = @("en", "th")[[int]($script:Lang -eq "en")]
                    Set-Content .lang $script:Lang
                    $env:MLT_LANG = $script:Lang
                    Say (T "เปลี่ยนเป็นภาษาไทยแล้ว" "Switched to English") "Green"
                }
                11 { return }
            }
        } catch { Say ((T "ผิดพลาด: " "Error: ") + $_.Exception.Message) "Red" }
    }
}

function Show-Menu {
    while ($true) {
        Say (T "`n================ ML Training · ระบบเทรนโมเดล AI ================" "`n================ ML Training ================")
        $trained = @(Get-TrainedNames)
        if ($trained) { Say ((T "โมเดลที่เทรนแล้ว: " "Trained: ") + ($trained -join ', ')) "DarkGray" }
        $u = Get-Unfinished
        if ($u) { Say ((T "มี {0} งานเทรนที่หยุดกลางทาง - เลือก 5 เพื่อเทรนต่อ: {1}" "{0} training(s) stopped before the end - choose 5 to continue: {1}") -f $u.Count, (($u | ForEach-Object { ($_ -split '\|')[0..1] -join ' ' }) -join ', ')) "Yellow" }
        $i = Read-Choice (T "เลือก" "Choose") @(
            (T "เทรนโมเดลใหม่ - ลากข้อมูลมาวาง ระบบเลือกงานและโมเดลให้ (แนะนำ)" "Train a new model - drag your data in, the right task and model are chosen (recommended)"),
            (T "เทรนแบบเลือกเอง (YOLO / ข้อความ / ภาพ / ตาราง)" "Train, choosing yourself (YOLO / text / images / tables)"),
            (T "ใช้โมเดลทำนายข้อมูลใหม่ (ภาพ วิดีโอ ข้อความ เอกสาร ตาราง พยากรณ์)" "Predict new data with a trained model"),
            (T "ดูผล / เทียบโมเดล (เปิด Excel)" "Results / compare models (Excel)"),
            (T "เทรนต่อจากที่ค้าง (ไฟดับ / ปิดเครื่อง / Ctrl+C)" "Resume a training that stopped (power cut / Ctrl+C / crash)"),
            (T "อธิบายโมเดล YOLO (XAI: โมเดลมองตรงไหน ทำไมตัดสินแบบนั้น)" "Explain a YOLO model (XAI heatmaps + statistics)"),
            (T "เครื่องมือเพิ่มเติม (ติดตั้ง ตรวจเครื่อง MLflow ล้างข้อมูล ภาษา ...)" "More tools (install, check PC, MLflow, clean, language ...)"),
            (T "ออก" "Exit"))
        try {
            switch ($i) {
                0 { Invoke-Wizard }
                1 { Show-TrainMenu }
                2 {
                    $ds = Select-Dataset -Trained
                    Say (T "ลากไฟล์ / โฟลเดอร์มาวาง (ภาพ วิดีโอ CSV PDF Word) หรือพิมพ์ประโยค / URL" "Drag a file / folder here (image, video, CSV, PDF, Word) or type a sentence / URL")
                    Say (T "  ข้อมูลตามเวลา: พิมพ์จำนวนงวด เช่น 30 = พยากรณ์ 30 งวดข้างหน้า" "  time series: type a number, e.g. 30 = forecast 30 periods ahead") "DarkGray"
                    $src = Read-Host (T "ข้อมูลที่จะทำนาย" "What to predict")
                    Invoke-Find $ds $src "" @()
                }
                3 {
                    $ds = Select-Dataset -Trained
                    Show-Results $ds
                    $x = "reports\$ds\compare.xlsx"
                    if ((Test-Path $x) -and (Ask-Yes (T "เปิด compare.xlsx ไหม" "Open compare.xlsx?"))) { Invoke-Item $x }
                }
                4 {
                    $u = Get-Unfinished
                    if (-not $u) { Say (T "ไม่มีงานค้าง - ทุกงานเทรนจบแล้ว" "Nothing to resume - every training finished.") "Green"; continue }
                    Say (T "`nงานที่ค้าง:" "`nUnfinished trainings:")
                    $k = Read-Choice (T "เลือก" "Choose") @($u | ForEach-Object { $f = $_ -split '\|'; "{0,-16} {1,-9} {2}  ({3})" -f $f[0], $f[1], $f[2], $f[3] })
                    $f = $u[$k] -split '\|'
                    uv run python -m ml_training.resume run $f[0] $f[1]
                    $script:LastReport = "reports\$($f[0])\$($f[1])\report.xlsx"
                    if ($LASTEXITCODE -eq 0) { Open-Report }
                }
                5 {
                    $ds = Select-Dataset -Trained
                    $n = Read-Host (T "วิเคราะห์ภาพทดสอบกี่ภาพ [100]" "How many test images to analyse? [100]")
                    $script:LastReport = $null
                    Invoke-Explain $ds "" $n @()
                    Open-Report
                }
                6 { Show-MoreMenu }
                7 { return }
            }
        } catch { Say ((T "ผิดพลาด: " "Error: ") + $_.Exception.Message) "Red" }
    }
}

# ----------------------------------------------------------------------------- commands
switch ($Task) {
    "menu" { Show-Menu }
    "help" { (Get-Content $PSCommandPath -TotalCount $HelpLines) -join "`n" | Write-Host }
    { $_ -in "fit", "detect" } {
        # words after the dataset may come in any order: size, quick|full, tuning preset, key=value
        $a = @($Rest)
        $ds = $a[0]; $size = $null; $mode = $null; $extra = @()
        for ($i = 1; $i -lt $a.Count; $i++) {
            $t = $a[$i]
            if ($t -match '=') { $extra += $t }
            elseif ("quick", "full" -contains $t) { $mode = $t }
            elseif (Test-Path "configs\tune\$t.yaml") { $extra += "tune=$t" }
            elseif (-not $size) { $size = $t }
            else { throw "Unknown argument '$t'" }
        }
        Invoke-Detect $ds $size $mode $extra
        break
    }
    "results" { if ($Rest -and $Rest[0]) { Show-Results $Rest[0] } else { Show-Results (Select-Dataset -Trained) } }
    "status" { Show-Status }
    "check" { uv run python -m ml_training.doctor }
    "train" {
        # run train <data> [quick|standard|best]; Hydra overrides (key=value) = the raw trainer
        if (-not $Rest) { Invoke-Wizard }
        elseif ($Rest[0] -match '=' -or $Rest[0] -like '-*') { uv run mlt-train @Rest }
        else { Invoke-Wizard $Rest[0] $(if (@($Rest).Count -gt 1) { $Rest[1] } else { "" }) }
        break
    }
    "predict" {
        if (-not $Rest -or $Rest[0] -match '=' -or $Rest[0] -like '-*') { uv run mlt-predict @Rest }
        else {
            $a = @($Rest); $size = $null; $i = 2
            if ($a.Count -gt $i -and $a[$i] -notmatch '=') { $size = $a[$i]; $i++ }
            $extra = @(); if ($a.Count -gt $i) { $extra = $a[$i..($a.Count - 1)] }
            Invoke-Find $a[0] $a[1] $size $extra
        }
        break
    }
    "text" {
        # text <dataset> [size] [quick|full] [preset] [key=value] in any order after the dataset
        $a = @($Rest); $size = $null; $mode = $null; $extra = @()
        for ($i = 1; $i -lt $a.Count; $i++) {
            $t = $a[$i]
            if ($t -match '=') { $extra += $t }
            elseif ("quick", "full" -contains $t) { $mode = $t }
            elseif (-not $size) { $size = $t }
            else { throw "Unknown argument '$t'" }
        }
        Invoke-Text $a[0] $size $mode $extra
        if ($script:LastReport -and (Test-Path $script:LastReport)) { Invoke-Item $script:LastReport }
        break
    }
    { $_ -in "vision", "image", "images" } {
        # vision <dataset> [model] [quick|full] [key=value] in any order after the dataset
        $a = @($Rest); $size = $null; $mode = $null; $extra = @()
        for ($i = 1; $i -lt $a.Count; $i++) {
            $t = $a[$i]
            if ($t -match '=') { $extra += $t }
            elseif ("quick", "full" -contains $t) { $mode = $t }
            elseif (-not $size) { $size = $t }
            else { throw "Unknown argument '$t'" }
        }
        Invoke-Vision $a[0] $size $mode $extra
        if ($script:LastReport -and (Test-Path $script:LastReport)) { Invoke-Item $script:LastReport }
        break
    }
    { $_ -in "auto", "wizard" } { Invoke-Wizard $(if ($Rest) { $Rest[0] } else { "" }) $(if (@($Rest).Count -gt 1) { $Rest[1] } else { "" }); break }
    "doctor" { uv run python -m ml_training.doctor }
    { $_ -in "table", "tabular" } {
        # table <file|name> [model|all] [quick|full] [target=.. time=.. group=.. key=value] in any order
        $a = @($Rest); $models = $null; $mode = $null; $extra = @()
        for ($i = 1; $i -lt $a.Count; $i++) {
            $t = $a[$i]
            if ($t -match '=') { $extra += $t }
            elseif ("quick", "full" -contains $t) { $mode = $t }
            elseif (-not $models) { $models = $t }
            else { throw "Unknown argument '$t'" }
        }
        Invoke-Table $a[0] $models $mode $extra
        if ($script:LastReport -and (Test-Path $script:LastReport)) { Invoke-Item $script:LastReport }
        break
    }
    "resume" { Invoke-Resume $Rest[0] $Rest[1]; if ($script:LastReport) { Say "Report: $($script:LastReport)" } }
    { $_ -in "explain", "xai" } {
        # explain <dataset> [size] [images] [key=value ...]
        $a = @($Rest); $size = $null; $count = $null; $extra = @()
        for ($i = 1; $i -lt $a.Count; $i++) {
            $t = $a[$i]
            if ($t -match '=') { $extra += $t }
            elseif ($t -match '^\d+$') { $count = $t }
            else { $size = $t }
        }
        Invoke-Explain $a[0] $size $count $extra
        break
    }
    "clean" { Invoke-Clean @($Rest) }
    "tune" { Show-TuningMenu }
    "history" { $n = 30; if ($Rest) { $n = $Rest[0] }; uv run python -m ml_training.lineage $n --excel }
    "find" {
        $a = @($Rest)
        $size = $null; $i = 2
        if ($a.Count -gt $i -and $a[$i] -notmatch '=') { $size = $a[$i]; $i++ }
        $extra = @(); if ($a.Count -gt $i) { $extra = $a[$i..($a.Count - 1)] }
        Invoke-Find $a[0] $a[1] $size $extra
    }
    { $_ -in "setup", "install" } { Invoke-Setup; break }
    "setup-dev" {
        Need-Uv
        uv sync --group dev
        if (-not (Test-Path .git)) { git init -b main }
        if (-not (Test-Path .dvc)) { uv run dvc init }
        Write-Host "Ready. Next: run pipeline" -ForegroundColor Green
    }
    "setup-dl" { Need-Uv; uv sync --group dev --extra dl }
    "setup-yolo" { Need-Uv; Remove-Item .gpu -ErrorAction SilentlyContinue; Sync-Env }
    "setup-gpu" { Need-Uv; Sync-Env -Gpu }
    "setup-ocr" { Need-Uv; New-Item .ocr -ItemType File -Force | Out-Null; Sync-Env -Gpu:(Test-Path .gpu) }
    "yolo"     { uv run dvc repro pipelines/yolo/dvc.yaml @Rest; uv run dvc metrics show }
    "yolo-all" {
        if (@($Rest).Count -gt 0 -and $Rest[0] -notmatch '=') {   # run yolo-all helmet [quick|full]
            $mode = "full"; if (@($Rest).Count -gt 1) { $mode = $Rest[1] }
            Invoke-Detect $Rest[0] "all" $mode @()
        } else {                                                  # old style: yolo-all data=yolo data.name=helmet
            foreach ($m in $Sizes) { Say "=== $m ==="; uv run mlt-train "model=$m" @Rest }
        }
    }
    "prepare"  { uv run mlt-prepare @Rest }
    "featurize" { uv run mlt-featurize @Rest }
    "pipeline" { uv run dvc repro @Rest; uv run dvc metrics show }
    "compare"  { uv run mlt-train -m "model=random_forest,logistic_regression,gradient_boosting,svm" @Rest }
    "ui"       { uv run mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5000 }
    "optuna"   { uv run --with optuna-dashboard optuna-dashboard "sqlite:///optuna.db" }
    "test"     { uv run ruff check src tests; uv run pytest }
    default    { Say "Unknown command '$Task'" "Yellow"; (Get-Content $PSCommandPath -TotalCount $HelpLines) -join "`n" | Write-Host }
}
