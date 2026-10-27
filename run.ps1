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
    .\run.ps1 results helmet                # compare all trained sizes (+ reports\helmet\compare.xlsx)
    .\run.ps1 find helmet D:\images         # predict with the best trained model
    .\run.ps1 find helmet D:\cam\clip.mp4 m # ... or with a chosen size
    .\run.ps1 clean helmet                  # delete results of one dataset (asks first; add raw = data too)
    .\run.ps1 clean all                     # delete every result + MLflow/Optuna history
    .\run.ps1 tune                          # make / view / edit tuning presets (common settings only)
    .\run.ps1 history                       # every prepare / train / predict / clean (+ logs\history.xlsx)
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
$HelpLines = 25

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

function Invoke-Find([string]$Ds, [string]$InputPath, [string]$Size, [string[]]$Extra) {
    if (-not $Ds -or -not $InputPath) { throw "Usage: .\run.ps1 find <dataset> <image|folder|video|url> [n|s|m|l|x]" }
    $d = Get-Dataset $Ds
    if ($Size) { $m = (Get-Sizes $Size)[0] }
    else {
        $m = (uv run python -m ml_training.compare $d.Name --best | Select-Object -Last 1)
        if ($LASTEXITCODE -ne 0 -or -not $m) { throw "No trained model for '$($d.Name)'. Train first: .\run.ps1 fit $Ds" }
    }
    $src = $InputPath.Trim().Trim('"')
    if ($src -notmatch '://') { $src = (Resolve-Path $src).Path -replace '\\', '/' }
    $dataArgs = $d.Args
    Say "Predicting with $m on $src"
    uv run mlt-predict @dataArgs "model=$m" "+input='$src'" @Extra
    $out = "reports\$($d.Name)\$m\predict"
    if ($LASTEXITCODE -eq 0 -and (Test-Path $out)) {
        Say "Annotated files + detections.csv (one row per prediction): $out" "Green"
        Invoke-Item $out
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
    if ($Trained) { $names = @(Get-ChildItem models -Directory -ErrorAction SilentlyContinue | Where-Object { Get-ChildItem $_.FullName -Filter *.pt -File } | ForEach-Object Name) }
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
        $i = Read-Choice "Choose" @(
            "Train YOLO (detect / segment / classify / pose / obb)",
            "Predict images / video with a trained model",
            "Compare trained YOLO models",
            "Tuning settings (view / create / edit presets)",
            "Open MLflow UI (all runs, metrics, model registry)",
            "Install / update (CPU or NVIDIA GPU)",
            "Clean data / results (asks before deleting)",
            "History (every prepare / train / predict / clean)",
            "Tabular data: run the full pipeline",
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
                    $ds = Select-Dataset -Trained
                    $src = Read-Host "Image, folder or video path (drag it here) or URL"
                    Invoke-Find $ds $src "" @()
                }
                2 { $ds = Select-Dataset -Trained; Show-Results $ds; $x = "reports\$ds\compare.xlsx"; if ((Test-Path $x) -and ((Read-Host "Open compare.xlsx? (y/n) [y]") -ne "n")) { Invoke-Item $x } }
                3 { Show-TuningMenu }
                4 { Start-Process "http://127.0.0.1:5000"; uv run mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5000 }
                5 {
                    $g = Read-Choice "Install for" @("CPU only", "NVIDIA GPU (CUDA)")
                    if ($g -eq 0) { & $PSCommandPath setup-yolo } else { & $PSCommandPath setup-gpu }
                }
                6 {
                    $names = @(@("data\processed", "models", "reports") | ForEach-Object { Get-ChildItem $_ -Directory -ErrorAction SilentlyContinue } | ForEach-Object Name | Where-Object { "pretrained", "runs", "figures" -notcontains $_ } | Sort-Object -Unique)
                    Say "`nClean what?"
                    $opts = $names + @("ALL datasets (every result + MLflow / Optuna history)")
                    $k = Read-Choice "Choose" $opts
                    $target = "all"; if ($k -lt $names.Count) { $target = $names[$k] }
                    $raw = (Read-Host "Also delete the raw dataset in data\raw? (y/n) [n]") -eq "y"
                    if ($raw) { Invoke-Clean @($target, "raw") } else { Invoke-Clean @($target) }
                }
                7 {
                    uv run python -m ml_training.lineage 30 --excel
                    if ((Test-Path logs\history.xlsx) -and ((Read-Host "Open history.xlsx? (y/n) [n]") -eq "y")) { Invoke-Item logs\history.xlsx }
                }
                8 { uv run dvc repro; uv run dvc metrics show }
                9 { (Get-Content $PSCommandPath -TotalCount $HelpLines) -join "`n" | Write-Host }
                10 { return }
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
