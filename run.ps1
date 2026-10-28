<#
  Windows helper for ml-training.  Easiest: double-click run.bat (or type .\run.ps1) for a menu.
  YOLO, every task: detect / segment / classify / pose / obb (read from the dataset automatically)
  dataset = folder in data\raw, a full path, or a sample: coco8 coco8-seg coco8-pose dota8 mnist160
    .\run.ps1 fit helmet                    # yolo11n, full (Optuna tuning + 100 epochs)
    .\run.ps1 fit helmet m quick            # yolo11m, quick test (no tuning, 30 epochs)
    .\run.ps1 fit helmet all quick          # every size n,s,m,l,x  (26m / all26 = YOLO26)
    .\run.ps1 fit D:\datasets\leaves s      # dataset outside the project (copied to data\raw)
    .\run.ps1 fit helmet s full accurate    # tuning preset: fast balanced accurate small_data freeze augment custom
    .\run.ps1 fit helmet n full data.task=segment model.params.epochs=200   # extra overrides
    .\run.ps1 text reviews                  # TEXT / DOCUMENTS (Thai OK): classify texts, NER, document folders
    .\run.ps1 text thai-ner tiny quick      # sizes tiny small base thai large · samples thai-sentiment thai-ner thai-docs
    .\run.ps1 find reviews "ส่งไว ของดี"     # predict a sentence, a CSV, a PDF / Word / image, or a folder of them
    .\run.ps1 vision flowers convnext       # IMAGES beyond YOLO: classify (resnet50 efficientnet convnext vit) or
    .\run.ps1 vision roads seg-b0 quick     #   pixel segmentation (seg-b0 seg-b2) · samples shapes shapes-seg (tiny / seg-tiny)
    .\run.ps1 table D:\sales.xlsx target=price  # TABLES (CSV / Excel): classify, predict a number (regression) or
    .\run.ps1 table sales.csv target=sales time=date group=shop all   #   forecast a time series · models lightgbm xgboost ... all
    .\run.ps1 find shop-sales 30            # forecast the next 30 periods · samples house-prices shop-sales breast_cancer iris
    .\run.ps1 resume helmet                 # continue a training that stopped (power cut, Ctrl+C, crash)
    .\run.ps1 results helmet                # compare all trained sizes (+ reports\helmet\compare.xlsx)
    .\run.ps1 find helmet D:\images         # predict with the best trained model
    .\run.ps1 find helmet D:\cam\clip.mp4 m # ... or with a chosen size
    .\run.ps1 explain helmet               # XAI: heatmaps + why each detection + statistics Excel
    .\run.ps1 explain helmet m 300         # ... model size m, 300 test images (default 100)
    .\run.ps1 clean helmet                  # delete results of one dataset (asks first; add raw = data too)
    .\run.ps1 clean all                     # delete every result + MLflow/Optuna history
    .\run.ps1 tune                          # make / view / edit tuning presets (common settings only)
    .\run.ps1 history                       # every prepare / train / predict / clean (+ logs\history.xlsx)
  Setup / tools:
    .\run.ps1 setup-yolo   # install (CPU)          .\run.ps1 setup-gpu   # install (NVIDIA GPU)
    .\run.ps1 setup-ocr    # + OCR for scanned PDFs / document images (Thai + English)
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
$HelpLines = 37

# YOLO + text (HuggingFace) + vision (timm) + LightGBM / XGBoost packages, + OCR after setup-ocr; -Gpu = CUDA build of torch
function Sync-Env([switch]$Gpu) {
    Remove-Item Env:UV_NO_SYNC -ErrorAction SilentlyContinue
    $extras = @("--extra", "yolo", "--extra", "nlp", "--extra", "vision", "--extra", "boost")
    if (Test-Path .ocr) { $extras += @("--extra", "ocr") }
    uv sync --group dev @extras
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
    if (-not $Ds) { throw "Usage: .\run.ps1 fit <dataset> [n|s|m|l|x|all] [quick|full] [overrides]" }
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
    if (-not $Name) { throw "Usage: .\run.ps1 results <dataset>" }
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
    if (-not $Ds) { throw "Usage: .\run.ps1 text <dataset|file|folder> [tiny|small|base|thai|large] [quick|full] [overrides]" }
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
    if (-not $Ds) { throw "Usage: .\run.ps1 vision <dataset|folder> [$($VisionModels.Keys -join '|')] [quick|full] [overrides]" }
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
    if (-not $Ds) { throw "Usage: .\run.ps1 table <file|name> target=<column> [time=<date column>] [group=<column>] [$($TableModels -join '|')|all] [quick|full]" }
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
    if ($task -eq "forecast") { Say "Forecast the future: .\run.ps1 find $($d.Name) 30" "Green" }
}

function Invoke-Find([string]$Ds, [string]$InputPath, [string]$Size, [string[]]$Extra) {
    if (-not $Ds -or -not $InputPath) { throw "Usage: .\run.ps1 find <dataset> <image|folder|video|url> [n|s|m|l|x]" }
    $d = Get-Dataset $Ds
    if ($Size -eq "tiny") { $m = @("img_tiny", "text_tiny")[[int](Test-Path "models\$($d.Name)\text_tiny")] }
    elseif ($Size -and ($Size -match '^(text_)?(tiny|small|base|thai|large)$')) { $m = Get-TextModel $Size }
    elseif ($Size -and ($Size -match '^(img|seg)_' -or $VisionModels.Contains($Size.ToLower()))) { $m = Get-VisionModel $Size "" }
    elseif ($Size -and (Test-Path "configs\model\$Size.yaml")) { $m = $Size }
    elseif ($Size) { $m = (Get-Sizes $Size)[0] }
    else {
        $m = (uv run python -m ml_training.compare $d.Name --best | Select-Object -Last 1)
        if ($LASTEXITCODE -ne 0 -or -not $m) { throw "No trained model for '$($d.Name)'. Train first: .\run.ps1 fit $Ds  (or .\run.ps1 text $Ds)" }
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
    if (-not $Ds) { throw "Usage: .\run.ps1 explain <dataset> [n|s|m|l|x] [images]" }
    $d = Get-Dataset $Ds
    if ($Size) { $m = (Get-Sizes $Size)[0] }
    else {
        $m = (uv run python -m ml_training.compare $d.Name --best | Select-Object -Last 1)
        if ($LASTEXITCODE -ne 0 -or -not $m) { throw "No trained model for '$($d.Name)'. Train first: .\run.ps1 fit $Ds" }
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
        Say "Continue one with: .\run.ps1 resume <dataset> [size]"
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
    if (-not $A -or -not $A[0]) { throw "Usage: .\run.ps1 clean <dataset|all> [raw]" }
    $flags = @()
    if ($A -contains "raw") { $flags += "--raw" }
    uv run python -m ml_training.clean $A[0] @flags
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
    if ($Trained) { $names = @(Get-ChildItem models -Directory -ErrorAction SilentlyContinue | Where-Object { (Get-ChildItem $_.FullName -File | Where-Object { ".pt", ".joblib" -contains $_.Extension }) -or (Test-Path "$($_.FullName)\*\model") } | ForEach-Object Name) }
    else { $names = @(Get-ChildItem data\raw -Directory -ErrorAction SilentlyContinue | ForEach-Object Name) }
    $names = @($names | Where-Object { "pretrained", "runs" -notcontains $_ })
    Say "`nDataset:"
    $opts = $names + @("Type a folder path...")
    if (-not $Trained) { $opts += "Sample dataset (auto-download, to try each task)..." }
    $i = Read-Choice "Choose" $opts
    if ($i -lt $names.Count) { return $names[$i] }
    if ($i -eq $names.Count) { return (Read-Host "Folder path (you can drag the folder here)").Trim('"') }
    $samples = @("coco8", "coco8-seg", "coco8-pose", "dota8", "mnist160")
    $j = Read-Choice "Sample" @("coco8       detect", "coco8-seg   segment", "coco8-pose  pose", "dota8       obb (rotated boxes)", "mnist160    classify")
    return $samples[$j]
}

function Select-TextDataset {
    $names = @(Get-ChildItem data\raw -ErrorAction SilentlyContinue | Where-Object { $_.PSIsContainer -or ".csv", ".xlsx", ".jsonl", ".json", ".tsv", ".conll" -contains $_.Extension } | ForEach-Object Name)
    Say "`nText dataset (folder or file in data\raw):"
    $opts = $names + @("Type a file / folder path...", "Sample: thai-sentiment  (classify reviews)", "Sample: thai-ner        (pull out names, companies, dates, amounts)", "Sample: thai-docs       (invoice / receipt / contract / quotation documents)")
    $i = Read-Choice "Choose" $opts
    if ($i -lt $names.Count) { return "data\raw\" + $names[$i] }
    if ($i -eq $names.Count) { return (Read-Host "File or folder path (you can drag it here)").Trim('"') }
    return @("thai-sentiment", "thai-ner", "thai-docs")[$i - $names.Count - 1]
}

function Select-TableDataset {
    $names = @(Get-ChildItem data\external, data\raw -File -ErrorAction SilentlyContinue | Where-Object { ".csv", ".xlsx", ".xls", ".parquet" -contains $_.Extension } | ForEach-Object FullName)
    Say "`nTable (CSV / Excel in data\external or data\raw):"
    $opts = @($names | ForEach-Object { Resolve-Path $_ -Relative }) + @("Type a file path...", "Sample: house-prices  (predict a price = regression)", "Sample: shop-sales    (forecast daily sales of 3 shops)", "Sample: breast_cancer (classify, 2 classes)")
    $i = Read-Choice "Choose" $opts
    if ($i -lt $names.Count) { return $names[$i] }
    if ($i -eq $names.Count) { return (Read-Host "File path (you can drag it here)").Trim('"') }
    return @("house-prices", "shop-sales", "breast_cancer")[$i - $names.Count - 1]
}

function Select-ImageDataset {
    $names = @(Get-ChildItem data\raw -Directory -ErrorAction SilentlyContinue | ForEach-Object Name)
    Say "`nImage dataset (folder in data\raw: one sub-folder per class, or images\ + masks\):"
    $opts = $names + @("Type a folder path...", "Sample: shapes      (classify circle / square / triangle)", "Sample: shapes-seg  (pixel segmentation of circles / squares)")
    $i = Read-Choice "Choose" $opts
    if ($i -lt $names.Count) { return $names[$i] }
    if ($i -eq $names.Count) { return (Read-Host "Folder path (you can drag it here)").Trim('"') }
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
    $i = Read-Choice "Choose" @($p | ForEach-Object { "{0,-12}{1}" -f $_.Name, $_.About })
    return $p[$i].Name
}

function Show-TuningMenu {
    while ($true) {
        Say "`n---------- Tuning settings ----------"
        $k = Read-Choice "Choose" @(
            "Show all presets (epochs, trials, freeze, how many settings are tuned)",
            "Create my preset (asks only the common settings)",
            "Edit a preset in Notepad (checked when you close it)",
            "Back")
        switch ($k) {
            0 { uv run python -m ml_training.tune_menu list }
            1 { uv run python -m ml_training.tune_menu create }
            2 {
                $name = Select-Preset
                Start-Process notepad "configs\tune\$name.yaml" -Wait
                uv run python -m ml_training.tune_menu check $name
                if ($LASTEXITCODE -ne 0) { Say "The file has a problem (see above) - open it again and fix it." "Red" }
                else { Say "Preset '$name' is OK." "Green" }
            }
            3 { return }
        }
    }
}

function Show-Menu {
    while ($true) {
        Say "`n========== ML Training =========="
        $u = Get-Unfinished
        if ($u) { Say ("{0} training(s) stopped before the end - choose 5 to continue: {1}" -f $u.Count, (($u | ForEach-Object { ($_ -split '\|')[0..1] -join ' ' }) -join ', ')) "Yellow" }
        $i = Read-Choice "Choose" @(
            "Train YOLO (detect / segment / classify / pose / obb)",
            "Train TEXT / DOCUMENTS (classify texts, pull out info = NER, document folders; Thai OK)",
            "Train IMAGES beyond YOLO (classify: ResNet / EfficientNet / ConvNeXt / ViT, pixel segmentation)",
            "Train TABLES (CSV / Excel: classify, predict a number, forecast a time series)",
            "Resume a training that stopped (power cut / Ctrl+C / crash)",
            "Predict with a trained model (images / video / texts / documents / tables / forecast)",
            "Compare trained YOLO models",
            "Explain a model: XAI heatmaps + why it detects + statistics Excel",
            "Tuning settings (view / create / edit presets)",
            "Open MLflow UI (all runs, metrics, model registry)",
            "Install / update (CPU or NVIDIA GPU)",
            "Clean data / results (asks before deleting)",
            "History (every prepare / train / predict / explain / clean)",
            "Tabular demo: run the DVC pipeline (breast cancer)",
            "Help (all commands)",
            "Exit")
        try {
            switch ($i) {
                0 {
                    $ds = Select-Dataset
                    Say "`nYOLO version:"
                    $gen = @("", "26")[(Read-Choice "Choose" @("YOLO11 (proven, default)", "YOLO26 (newest)"))]
                    Say "`nModel size (n = fastest ... x = most accurate):"
                    $sizeOpts = @("n  (nano)", "s  (small)", "m  (medium)", "l  (large)", "x  (extra large)", "all sizes, one after another")
                    $size = $gen + @("n", "s", "m", "l", "x", "all")[(Read-Choice "Choose" $sizeOpts)]
                    Say "`nTraining mode:"
                    $mode = @("quick", "full")[(Read-Choice "Choose" @("Quick test (no tuning, 30 epochs)", "Full (Optuna tuning + epochs from the tuning preset)"))]
                    Say "`nTuning preset (what Optuna tries + fixed training settings):"
                    $tune = Select-Preset
                    $script:LastReport = $null
                    Invoke-Detect $ds $size $mode @("tune=$tune")
                    if ($script:LastReport -and (Test-Path $script:LastReport) -and ((Read-Host "Open the Excel report? (y/n) [y]") -ne "n")) { Invoke-Item $script:LastReport }
                }
                1 {
                    $ds = Select-TextDataset
                    Say "`nModel (all read Thai):"
                    $k = Read-Choice "Choose" @(
                        "tiny   trained from scratch, no download - only to try the pipeline",
                        "small  multilingual DistilBERT - fast, OK on CPU",
                        "base   XLM-RoBERTa base - strong default (GPU recommended)",
                        "thai   WangchanBERTa - best for Thai-only text",
                        "large  XLM-RoBERTa large - most accurate (GPU 12 GB+)") 3
                    Say "`nTraining mode:"
                    $mode = @("quick", "full")[(Read-Choice "Choose" @("Quick test (no tuning, 2 epochs)", "Full (Optuna 8 trials + final training)"))]
                    $script:LastReport = $null
                    Invoke-Text $ds $TextSizes[$k] $mode @()
                    if ($script:LastReport -and (Test-Path $script:LastReport) -and ((Read-Host "Open the Excel report? (y/n) [y]") -ne "n")) { Invoke-Item $script:LastReport }
                }
                2 {
                    $ds = Select-ImageDataset
                    $seg = $ds -eq "shapes-seg" -or ((Test-Path "data\raw\$ds") -and (Get-ChildItem "data\raw\$ds" -Directory | Where-Object { "masks", "mask", "SegmentationClass" -contains $_.Name }))
                    if ($ds -match '^shapes') { $size = @("tiny", "seg-tiny")[[int][bool]$seg] }
                    elseif ($seg) {
                        Say "`nSegmentation model (SegFormer, pretrained):"
                        $size = @("seg-b0", "seg-b2")[(Read-Choice "Choose" @("seg-b0  fast, any GPU (3.7M params)", "seg-b2  more accurate, GPU 8 GB+ (25M params)"))]
                    } else {
                        Say "`nClassification model (ImageNet-pretrained, downloaded once):"
                        $size = @("efficientnet", "resnet50", "convnext", "vit")[(Read-Choice "Choose" @(
                            "efficientnet  EfficientNet-B0 - small and fast, good default",
                            "resnet50      ResNet-50 - classic, robust",
                            "convnext      ConvNeXt-Tiny - usually the most accurate here",
                            "vit           ViT-B/16 - Vision Transformer, best with lots of data (GPU 8 GB+)"))]
                    }
                    Say "`nTraining mode:"
                    $mode = @("quick", "full")[(Read-Choice "Choose" @("Quick test (no tuning, 3 epochs)", "Full (Optuna 8 trials + final training)"))]
                    $script:LastReport = $null
                    Invoke-Vision $ds $size $mode @()
                    if ($script:LastReport -and (Test-Path $script:LastReport) -and ((Read-Host "Open the Excel report? (y/n) [y]") -ne "n")) { Invoke-Item $script:LastReport }
                }
                3 {
                    $ds = Select-TableDataset
                    $extra = @()
                    if (-not (Test-Path "configs\data\$ds.yaml")) {
                        $cols = @()
                        if ($ds -like "*.csv") { $cols = @((Get-Content $ds -TotalCount 1) -split ',' | ForEach-Object { $_.Trim('"') }) }
                        if ($cols) { Say "`nColumns: $($cols -join ', ')" }
                        $extra += "target=" + (Read-Host "Column to predict (target)")
                        $t = Read-Host "Date / time column for forecasting (empty = not a time series)"
                        if ($t) {
                            $extra += "time=$t"
                            $g = Read-Host "Column that separates several series, e.g. shop or product (empty = one series)"
                            if ($g) { $extra += "group=$g" }
                        }
                    }
                    Say "`nModel:"
                    $k = Read-Choice "Choose" @(
                        "lightgbm      LightGBM - fast, usually the most accurate on tables",
                        "xgboost       XGBoost - very accurate gradient boosting",
                        "random_forest Random forest - robust, few settings",
                        "all           train every model and compare (lightgbm xgboost random_forest gradient_boosting linear svm)")
                    $models = @("lightgbm", "xgboost", "random_forest", "all")[$k]
                    Say "`nTraining mode:"
                    $mode = @("quick", "full")[(Read-Choice "Choose" @("Quick (no tuning)", "Full (Optuna 30 trials + final model)") 2)]
                    $script:LastReport = $null
                    Invoke-Table $ds $models $mode $extra
                    if ($script:LastReport -and (Test-Path $script:LastReport) -and ((Read-Host "Open the Excel report? (y/n) [y]") -ne "n")) { Invoke-Item $script:LastReport }
                }
                4 {
                    $u = Get-Unfinished
                    if (-not $u) { Say "Nothing to resume - every training finished." "Green"; continue }
                    Say "`nUnfinished trainings:"
                    $k = Read-Choice "Choose" @($u | ForEach-Object { $f = $_ -split '\|'; "{0,-16} {1,-9} {2}  ({3})" -f $f[0], $f[1], $f[2], $f[3] })
                    $f = $u[$k] -split '\|'
                    $script:LastReport = $null
                    uv run python -m ml_training.resume run $f[0] $f[1]
                    $x = "reports\$($f[0])\$($f[1])\report.xlsx"
                    if ($LASTEXITCODE -eq 0 -and (Test-Path $x) -and ((Read-Host "Open the Excel report? (y/n) [y]") -ne "n")) { Invoke-Item $x }
                }
                5 {
                    $ds = Select-Dataset -Trained
                    $src = Read-Host "Image / folder / video / URL / text / CSV path (drag it here), or a number = forecast periods"
                    Invoke-Find $ds $src "" @()
                }
                6 { $ds = Select-Dataset -Trained; Show-Results $ds; $x = "reports\$ds\compare.xlsx"; if ((Test-Path $x) -and ((Read-Host "Open compare.xlsx? (y/n) [y]") -ne "n")) { Invoke-Item $x } }
                7 {
                    $ds = Select-Dataset -Trained
                    $n = Read-Host "How many test images to analyse? [100]"
                    $script:LastReport = $null
                    Invoke-Explain $ds "" $n @()
                    if ($script:LastReport -and ((Read-Host "Open the XAI Excel report? (y/n) [y]") -ne "n")) { Invoke-Item $script:LastReport }
                }
                8 { Show-TuningMenu }
                9 { Start-Process "http://127.0.0.1:5000"; uv run mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5000 }
                10 {
                    $g = Read-Choice "Install for" @("CPU only", "NVIDIA GPU (CUDA)")
                    if ($g -eq 0) { & $PSCommandPath setup-yolo } else { & $PSCommandPath setup-gpu }
                }
                11 {
                    $names = @(@("data\processed", "models", "reports") | ForEach-Object { Get-ChildItem $_ -Directory -ErrorAction SilentlyContinue } | ForEach-Object Name | Where-Object { "pretrained", "runs", "figures" -notcontains $_ } | Sort-Object -Unique)
                    Say "`nClean what?"
                    $opts = $names + @("ALL datasets (every result + MLflow / Optuna history)")
                    $k = Read-Choice "Choose" $opts
                    $target = "all"; if ($k -lt $names.Count) { $target = $names[$k] }
                    $raw = (Read-Host "Also delete the raw dataset in data\raw? (y/n) [n]") -eq "y"
                    if ($raw) { Invoke-Clean @($target, "raw") } else { Invoke-Clean @($target) }
                }
                12 {
                    uv run python -m ml_training.lineage 30 --excel
                    if ((Test-Path logs\history.xlsx) -and ((Read-Host "Open history.xlsx? (y/n) [n]") -eq "y")) { Invoke-Item logs\history.xlsx }
                }
                13 { uv run dvc repro; uv run dvc metrics show }
                14 { (Get-Content $PSCommandPath -TotalCount $HelpLines) -join "`n" | Write-Host }
                15 { return }
            }
        } catch { Say "Error: $($_.Exception.Message)" "Red" }
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
    "results" { Show-Results $Rest[0] }
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
    "setup" {
        Need-Uv
        uv sync --group dev
        if (-not (Test-Path .git)) { git init -b main }
        if (-not (Test-Path .dvc)) { uv run dvc init }
        Write-Host "Ready. Next: .\run.ps1 pipeline" -ForegroundColor Green
    }
    "setup-dl" { Need-Uv; uv sync --group dev --extra dl }
    "setup-yolo" { Need-Uv; Remove-Item .gpu -ErrorAction SilentlyContinue; Sync-Env }
    "setup-gpu" { Need-Uv; Sync-Env -Gpu }
    "setup-ocr" { Need-Uv; New-Item .ocr -ItemType File -Force | Out-Null; Sync-Env -Gpu:(Test-Path .gpu) }
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
