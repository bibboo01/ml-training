#!/usr/bin/env bash
# ml-training for Ubuntu / Linux (Windows: run.bat / run.ps1).  After `bash run.sh setup` just type: run
#   run                          menu
#   run setup                    install everything (finds the NVIDIA GPU itself), then check the server
#   run train ~/data/helmet      look at the data, pick the task and model, train   (+ quick | best)
#   run predict helmet ~/new     predict with the best model (image, folder, video, csv, sentence, N = forecast)
#   run results helmet           scores of every model + Excel
#   run status                   trained models, stopped trainings, background jobs
#   run resume                   continue a training that stopped
#   run bg train ~/data/helmet   any command in the background - safe to log out of SSH
#   run logs | run stop          follow / stop the background job
#   run check | run ui | run clean helmet | run help
# Choose everything yourself:
#   run fit helmet m quick                     YOLO: n s m l x all (26m = YOLO26), quick | full, tuning preset
#   run text data/raw/reviews.csv thai         text models: tiny small base thai large
#   run vision ~/data/leaves efficientnet      tiny resnet50 efficientnet convnext vit seg-tiny seg-b0 seg-b2
#   run table sales.csv target=amount time=date group=shop all
#   run explain helmet | run history | run tune | run optuna | run test
#   run train model=svm ...  /  run predict ... /  run prepare|featurize ...   (Hydra overrides, as before)
# Setup parts: setup-gpu  setup-cpu  setup-system (Ubuntu packages, sudo)  setup-ocr  setup-dev
set -uo pipefail
ORIG_PWD=$PWD
cd "$(dirname "$(readlink -f "$0")")" || exit 1
SELF="$PWD/run.sh"
ME=${USER:-$(id -un)}
HELP_LINES=20

LANG_UI=en; [[ -f .lang ]] && LANG_UI=$(head -n1 .lang | tr -d '[:space:]')
export MLT_LANG=$LANG_UI   # Python messages (inspect, errors, doctor) follow the menu language
T() { if [[ $LANG_UI == th ]]; then printf '%s' "$1"; else printf '%s' "$2"; fi; }

export PYTHONUTF8=1 MLFLOW_DISABLE_AGENT_HINT=1 PYTHONIOENCODING=utf-8
PATH_BEFORE=$PATH
export PATH="$HOME/.local/bin:$PATH"
# after a CUDA reinstall (old driver), stop `uv run` from swapping torch back to the default build
[[ -f .gpu ]] && export UV_NO_SYNC=1
CUDA_INDEX="https://download.pytorch.org/whl/cu126"
CPU_INDEX="https://download.pytorch.org/whl/cpu"
LAST_REPORT=""

# shellcheck disable=SC2034  # read indirectly by say()
if [[ -t 1 ]]; then C_CYAN=$'\e[36m' C_GREEN=$'\e[32m' C_YELLOW=$'\e[33m' C_RED=$'\e[31m' C_GREY=$'\e[90m' C_WHITE=$'\e[1m' C_OFF=$'\e[0m'
else C_CYAN="" C_GREEN="" C_YELLOW="" C_RED="" C_GREY="" C_WHITE="" C_OFF=""; fi
say() { local c="C_${2:-CYAN}"; printf '%s%b%s\n' "${!c}" "$1" "$C_OFF"; }
die() { say "$1" RED; return 1; }
py() { uv run python "$@"; }

# ----------------------------------------------------------------------------- setup
need_uv() {
    command -v uv >/dev/null && return 0
    say "Installing uv..." YELLOW
    curl -LsSf https://astral.sh/uv/install.sh | sh || return 1
    export PATH="$HOME/.local/bin:$PATH"
}

cuda_ok() { uv run --no-sync python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }

# YOLO + text (HuggingFace) + vision (timm) + LightGBM / XGBoost, + OCR after setup-ocr
sync_env() {  # $1 = gpu | cpu
    unset UV_NO_SYNC
    local extras=(--extra yolo --extra nlp --extra vision --extra boost)
    [[ -f .ocr ]] && extras+=(--extra ocr)
    rm -f .gpu
    uv sync --group dev "${extras[@]}" || return 1
    # removing OCR (opencv-python-headless) deletes files cv2 shares with opencv-python: put them back
    uv run --no-sync python -c "import cv2; cv2.imshow" 2>/dev/null || uv sync --group dev "${extras[@]}" --reinstall-package opencv-python
    if [[ $1 == cpu ]]; then
        # the default Linux PyTorch carries ~3 GB of CUDA libraries; a CPU server does not need them
        uv pip install --reinstall torch torchvision --index-url "$CPU_INDEX" && touch .gpu ||
            say "$(T "ดาวน์โหลด PyTorch รุ่น CPU ไม่ได้ - ใช้รุ่นปกติต่อ (ใช้ได้ แค่ใหญ่กว่า)" "Could not download the CPU build of PyTorch - keeping the default one (works, just bigger)")" YELLOW
    elif command -v nvidia-smi >/dev/null && ! cuda_ok; then
        # the default build needs a recent driver; CUDA 12.6 runs on drivers >= 525
        say "$(T "ไดรเวอร์ NVIDIA เก่ากว่า PyTorch รุ่นปกติ - ติดตั้งรุ่น CUDA 12.6 แทน" "The NVIDIA driver is older than the default PyTorch build needs - installing the CUDA 12.6 build")" YELLOW
        uv pip install --reinstall torch torchvision --index-url "$CUDA_INDEX" && touch .gpu
    fi
    [[ -f .gpu ]] && export UV_NO_SYNC=1
    uv run --no-sync python -c "import torch; print('PyTorch', torch.__version__, '- CUDA available:', torch.cuda.is_available(), torch.version.cuda)"
}

setup_system() {
    local pkgs=(libgl1 libglib2.0-0 git curl fonts-thai-tlwg)
    say "$(T "จะติดตั้งแพ็กเกจ Ubuntu: " "Will install Ubuntu packages: ")${pkgs[*]}"
    local s=""; [[ $EUID -ne 0 ]] && s=sudo
    $s apt-get update && $s apt-get install -y "${pkgs[@]}"
}

# `run` anywhere: ~/.local/bin/run -> this script (Ubuntu puts ~/.local/bin on PATH at login)
install_command() {
    local bin="$HOME/.local/bin" have
    have=$(command -v run 2>/dev/null)
    if [[ -n $have && $(readlink -f "$have") != "$SELF" ]]; then
        say "$(T "มีคำสั่ง 'run' อื่นอยู่แล้ว ($have) - ใช้ run แทน" "Another 'run' command exists ($have) - keep using run")" YELLOW
        return 0
    fi
    mkdir -p "$bin" && chmod +x "$SELF" && ln -sfn "$SELF" "$bin/run" || return 0
    say "$(T "ติดตั้งคำสั่ง 'run' แล้ว - ใช้ได้จากทุกโฟลเดอร์" "Installed the 'run' command - works from any folder")" GREEN
    [[ ":$PATH_BEFORE:" == *":$bin:"* ]] || say "  $(T "เปิด terminal ใหม่ หรือพิมพ์:" "open a new terminal, or type:") source ~/.profile" GREY
}

setup_all() {
    need_uv || return 1
    if ! uv run --no-sync python -c "import ctypes.util as u, sys; sys.exit(0 if u.find_library('GL') else 1)" 2>/dev/null &&
        ! ldconfig -p 2>/dev/null | grep -q libGL.so.1; then
        say "$(T "ยังไม่มี libGL ที่ YOLO / OpenCV ต้องใช้" "libGL that YOLO / OpenCV need is missing")" YELLOW
        if [[ -t 0 ]] && ask_yes "$(T "ติดตั้งแพ็กเกจ Ubuntu ตอนนี้ไหม (ใช้ sudo)" "Install the Ubuntu packages now (uses sudo)?")" y; then setup_system
        else say "  $(T "ติดตั้งทีหลังได้ด้วย: run setup-system" "install them later with: run setup-system")" GREY; fi
    fi
    if command -v nvidia-smi >/dev/null; then
        say "$(T "พบการ์ดจอ NVIDIA - ติดตั้งแบบ GPU" "NVIDIA GPU found - installing the GPU version")" GREEN; sync_env gpu || return 1
    else
        say "$(T "ไม่พบการ์ดจอ NVIDIA - ติดตั้งแบบ CPU" "No NVIDIA GPU - installing the CPU version")" YELLOW; sync_env cpu || return 1
    fi
    install_command
    py -m ml_training.doctor
    say "\n$(T "พร้อมแล้ว! ลองเลย:" "Ready! Try:")  run    $(T "หรือ" "or")  run train shop-sales quick" GREEN
}

show_status() {
    local t u; t=$(trained_names | paste -sd, - | sed 's/,/, /g')
    say "$(T "โมเดลที่เทรนแล้ว: " "Trained models: ")${t:-$(T "ยังไม่มี" "none yet")}"
    u=$(get_unfinished)
    if [[ -n $u ]]; then say "$(T "งานที่ค้าง (run resume):" "Stopped trainings (run resume):")" YELLOW
        while IFS='|' read -r a b c d; do printf '  %-16s %-9s %s  (%s)\n' "$a" "$b" "$c" "$d"; done <<<"$u"; fi
    say "$(T "งานเบื้องหลัง:" "Background jobs:")"; show_jobs
}

stop_job() {
    local f pid cmd found=()
    for f in logs/bg/*.pid; do [[ -f $f ]] || continue; read -r pid cmd <"$f"; kill -0 "$pid" 2>/dev/null && found+=("$pid|$cmd"); done
    ((${#found[@]})) || { say "$(T "ไม่มีงานเบื้องหลังที่กำลังทำงาน" "No background job is running")"; return 0; }
    local opts=() r; for r in "${found[@]}"; do opts+=("PID ${r%%|*}  ${r#*|}"); done
    read_choice "$(T "หยุดงานไหน" "Stop which job")" 1 "${opts[@]}"
    pid=${found[CHOICE]%%|*}
    ask_yes "$(T "หยุด PID $pid? (เทรนต่อทีหลังได้ด้วย run resume)" "Stop PID $pid? (continue later with run resume)")" y || return 0
    kill -- -"$pid" 2>/dev/null || kill "$pid"; say "$(T "หยุดแล้ว" "Stopped")" GREEN
}

# ----------------------------------------------------------------------------- dataset helpers
# sets DS_NAME and DS_ARGS.  "helmet" -> data=helmet (configs/data/helmet.yaml) or data=$1 data.name=helmet
#                            "/srv/x/helmet" -> data=$1 data.name=helmet data.path=/srv/x/helmet
clean_path() { local p="${1#\"}"; p="${p%\"}"; p="${p#\'}"; p="${p%\'}"; p="${p/#\~/$HOME}"; [[ $p == / ]] || p="${p%/}"; printf '%s' "$p"; }
resolve_ds() {  # $1 = config for free folders (yolo|images|text), $2 = dataset, $3 = file|dir
    local kind=$1 ds; ds=$(clean_path "$2")
    if [[ $ds == */* && -e $ds ]] && { [[ $3 != dir ]] || [[ -d $ds ]]; }; then
        local full; full=$(readlink -f "$ds")
        DS_NAME=$(basename "$full"); [[ -f $full ]] && DS_NAME="${DS_NAME%.*}"
        DS_ARGS=("data=$kind" "data.name=$DS_NAME" "data.path='$full'")
    elif [[ -f configs/data/$ds.yaml ]]; then DS_NAME=$ds; DS_ARGS=("data=$ds")
    else DS_NAME=$(basename "$ds"); DS_ARGS=("data=$kind" "data.name=$DS_NAME"); fi
}

# n -> yolo11n, 26m -> yolo26m, all -> yolo11n..x, all26 -> yolo26n..x  (sets SIZES)
get_sizes() {
    local s="${1:-n}" gen=11; s="${s,,}"; s="${s#yolo}"
    if [[ $s =~ ^all(11|26)$ ]]; then gen=${BASH_REMATCH[1]}; s=all
    elif [[ $s =~ ^(11|26)(.*)$ ]]; then gen=${BASH_REMATCH[1]}; s=${BASH_REMATCH[2]}; fi
    if [[ $s == all ]]; then SIZES=("yolo${gen}n" "yolo${gen}s" "yolo${gen}m" "yolo${gen}l" "yolo${gen}x"); return 0; fi
    [[ $s =~ ^[nsmlx]$ ]] || { die "Unknown size '$1' (use n, s, m, l, x, all, or 26n ... all26)"; return 1; }
    SIZES=("yolo$gen$s")
}

# EXTRA -> DATA_EXTRA (data.* overrides, also needed by prepare / featurize) + OTHER_EXTRA
split_extra() {
    DATA_EXTRA=(); OTHER_EXTRA=()
    local e; for e in "$@"; do if [[ $e =~ ^\+?data\. ]]; then DATA_EXTRA+=("$e"); else OTHER_EXTRA+=("$e"); fi; done
}

prepare_data() {  # $1 = what is read (for the message)
    say "[1/2] $1 ($DS_NAME)..."
    uv run mlt-prepare "${DS_ARGS[@]}" "${DATA_EXTRA[@]}" || { die "Dataset check failed (see the message above)"; return 1; }
    uv run mlt-featurize "${DS_ARGS[@]}" "${DATA_EXTRA[@]}" || { die "Dataset split failed (see the message above)"; return 1; }
}
meta_task() { uv run --no-sync python -c "import json,sys; print(json.load(open(sys.argv[1]))['task'])" "data/processed/$DS_NAME/meta.json"; }
has_hpo_trials() { local e; for e in "$@"; do [[ $e == hpo.n_trials=* ]] && return 0; done; return 1; }

show_results() {
    [[ -n ${1:-} ]] || { die "Usage: run results <dataset>"; return 1; }
    say "\n=== Results: $1 ==="
    py -m ml_training.compare "$1"
}

# ----------------------------------------------------------------------------- training
invoke_detect() {  # dataset size mode extra...
    local ds=${1:-} size=${2:-} mode=${3:-full}; shift 3 2>/dev/null || shift $#
    [[ -n $ds ]] || { die "Usage: run fit <dataset> [n|s|m|l|x|all] [quick|full] [overrides]"; return 1; }
    resolve_ds yolo "$ds" dir; get_sizes "$size" || return 1; split_extra "$@"
    local mode_args=(); case $mode in quick) mode_args=(hpo=none model.params.epochs=30) ;; full) ;; *) die "Unknown mode '$mode' (use quick or full)"; return 1 ;; esac
    say "Dataset: $DS_NAME   Models: ${SIZES[*]}   Mode: $mode"
    prepare_data "Preparing dataset" || return 1
    local m done=()
    for m in "${SIZES[@]}"; do
        say "[2/2] Training $m ..."
        if ! uv run mlt-train "${DS_ARGS[@]}" "${DATA_EXTRA[@]}" "model=$m" "${mode_args[@]}" "${OTHER_EXTRA[@]}"; then say "$m failed, continuing" YELLOW; continue; fi
        done+=("$m")
        say "  model : models/$DS_NAME/$m.pt" GREEN
        say "  report: reports/$DS_NAME/$m/report.xlsx" GREEN
    done
    ((${#done[@]})) || { die "Training failed (see the message above)"; return 1; }
    show_results "$DS_NAME"
    if ((${#done[@]} == 1)); then LAST_REPORT="reports/$DS_NAME/${done[0]}/report.xlsx"; else LAST_REPORT="reports/$DS_NAME/compare.xlsx"; fi
}

TEXT_SIZES=(tiny small base thai large)
get_text_model() {
    local s="${1:-base}"; s="${s,,}"; s="${s#text_}"
    [[ " ${TEXT_SIZES[*]} " == *" $s "* ]] || { die "Unknown text model '$1' (use ${TEXT_SIZES[*]})"; return 1; }
    TEXT_MODEL="text_$s"
}

invoke_text() {
    local ds=${1:-} size=${2:-} mode=${3:-full}; shift 3 2>/dev/null || shift $#
    [[ -n $ds ]] || { die "Usage: run text <dataset|file|folder> [${TEXT_SIZES[*]}] [quick|full] [overrides]"; return 1; }
    resolve_ds text "$ds" any; get_text_model "$size" || return 1; split_extra "$@"
    local m=$TEXT_MODEL mode_args=()
    if [[ $mode == quick ]]; then mode_args=(hpo=none model.params.epochs=2); elif ! has_hpo_trials "$@"; then mode_args=(hpo.n_trials=8); fi
    say "Text dataset: $DS_NAME   Model: $m   Mode: $mode"
    prepare_data "Reading texts / documents" || return 1
    say "[2/2] Training $m ..."
    uv run mlt-train "${DS_ARGS[@]}" "${DATA_EXTRA[@]}" "model=$m" "${mode_args[@]}" "${OTHER_EXTRA[@]}" || { die "Training failed (see the message above)"; return 1; }
    say "  model : models/$DS_NAME/$m/model" GREEN
    say "  report: reports/$DS_NAME/$m/report.xlsx" GREEN
    LAST_REPORT="reports/$DS_NAME/$m/report.xlsx"
}

declare -A VISION_MODELS=([tiny]=img_tiny [resnet50]=img_resnet50 [efficientnet]=img_efficientnet [convnext]=img_convnext [vit]=img_vit [seg-tiny]=seg_tiny [seg-b0]=seg_b0 [seg-b2]=seg_b2)
VISION_KEYS="tiny resnet50 efficientnet convnext vit seg-tiny seg-b0 seg-b2"
get_vision_model() {  # size task
    local s="${1,,}"
    if [[ -z $s ]]; then if [[ $2 == segment ]]; then VISION_MODEL=seg_b0; else VISION_MODEL=img_efficientnet; fi; return 0; fi
    if [[ $s =~ ^(img|seg)_ && -f configs/model/$s.yaml ]]; then VISION_MODEL=$s; return 0; fi
    [[ -n ${VISION_MODELS[$s]:-} ]] || { die "Unknown image model '$1' (use $VISION_KEYS)"; return 1; }
    VISION_MODEL=${VISION_MODELS[$s]}
}

invoke_vision() {
    local ds=${1:-} size=${2:-} mode=${3:-full}; shift 3 2>/dev/null || shift $#
    [[ -n $ds ]] || { die "Usage: run vision <dataset|folder> [${VISION_KEYS// /|}] [quick|full] [overrides]"; return 1; }
    resolve_ds images "$ds" dir; split_extra "$@"
    prepare_data "Reading images" || return 1
    local task; task=$(meta_task)
    [[ -z $size && $ds == shapes* ]] && { size=tiny; [[ $task == segment ]] && size="seg-tiny"; }
    get_vision_model "$size" "$task" || return 1
    local m=$VISION_MODEL mode_args=()
    if [[ $mode == quick ]]; then mode_args=(hpo=none model.params.epochs=3); elif ! has_hpo_trials "$@"; then mode_args=(hpo.n_trials=8); fi
    say "[2/2] Task: $task   Model: $m   Mode: $mode"
    uv run mlt-train "${DS_ARGS[@]}" "${DATA_EXTRA[@]}" "model=$m" "${mode_args[@]}" "${OTHER_EXTRA[@]}" || { die "Training failed (see the message above)"; return 1; }
    say "  model : models/$DS_NAME/$m/model" GREEN
    say "  report: reports/$DS_NAME/$m/report.xlsx" GREEN
    LAST_REPORT="reports/$DS_NAME/$m/report.xlsx"
}

TABLE_MODELS=(lightgbm xgboost random_forest gradient_boosting logistic_regression svm)
resolve_table() {
    local ds; ds=$(clean_path "$1")
    if [[ -f configs/data/$ds.yaml ]]; then DS_NAME=$ds; DS_ARGS=("data=$ds"); return 0; fi
    local file=$ds ext d
    if [[ ! -f $file ]]; then
        file=""
        for d in data/raw data/external; do for ext in csv xlsx xls parquet; do [[ -z $file && -f $d/$ds.$ext ]] && file=$d/$ds.$ext; done; done
    fi
    [[ -n $file ]] || { die "Table not found: $ds (a .csv / .xlsx file, or put it in data/external)"; return 1; }
    local full; full=$(readlink -f "$file"); DS_NAME=$(basename "${full%.*}")
    DS_ARGS=("data=csv" "data.name=$DS_NAME" "data.path='$full'")
}

invoke_table() {  # file models mode extra...
    local ds=${1:-} models=${2:-} mode=${3:-full}; shift 3 2>/dev/null || shift $#
    [[ -n $ds ]] || { die "Usage: run table <file|name> target=<column> [time=<date column>] [group=<column>] [${TABLE_MODELS[*]}|all] [quick|full]"; return 1; }
    resolve_table "$ds" || return 1
    # short names: target=price time=date group=shop task=regress
    local e k v ext=() has_time=0 has_task=0 has_target=0
    for e in "$@"; do
        k=${e%%=*}; v=${e#*=}
        case $k in target) e="data.target=$v" ;; time) e="data.time_column=$v" ;; group) e="data.group_column=$v" ;; task) e="data.task=$v" ;; esac
        [[ $e == data.time_column=* ]] && has_time=1; [[ $e == data.task=* ]] && has_task=1; [[ $e == data.target=* ]] && has_target=1
        ext+=("$e")
    done
    ((has_time && !has_task)) && ext+=(data.task=forecast)
    split_extra "${ext[@]}"
    [[ ${DS_ARGS[0]} == data=csv && $has_target == 0 ]] && { die "Which column to predict? add target=<column name>"; return 1; }
    local list=(lightgbm); if [[ $models == all ]]; then list=("${TABLE_MODELS[@]}"); elif [[ -n $models ]]; then IFS=, read -ra list <<<"$models"; fi
    local mode_args=(); [[ $mode == quick ]] && mode_args=(hpo=none)
    prepare_data "Reading table" || return 1
    local task m done=(); task=$(meta_task)
    for m in "${list[@]}"; do
        say "[2/2] Task: $task   Model: $m   Mode: $mode"
        if ! uv run mlt-train "${DS_ARGS[@]}" "${DATA_EXTRA[@]}" "model=$m" "${mode_args[@]}" "${OTHER_EXTRA[@]}"; then say "$m failed, continuing" YELLOW; continue; fi
        done+=("$m")
    done
    ((${#done[@]})) || { die "Training failed (see the message above)"; return 1; }
    show_results "$DS_NAME"
    if ((${#done[@]} == 1)); then LAST_REPORT="reports/$DS_NAME/${done[0]}/report.xlsx"; else LAST_REPORT="reports/$DS_NAME/compare.xlsx"; fi
    [[ $task == forecast ]] && say "Forecast the future: run predict $DS_NAME 30" GREEN
    return 0
}

best_model() { uv run python -m ml_training.compare "$1" --best 2>/dev/null | tail -n1; }

show_csv() {  # file rows
    uv run --no-sync python -c "import sys, pandas as pd; pd.set_option('display.width', 200); print(pd.read_csv(sys.argv[1]).head(int(sys.argv[2])).to_string(index=False))" "$1" "$2"
}

invoke_find() {  # dataset input [size] extra...
    local ds=${1:-} src=${2:-} size=${3:-}; shift 3 2>/dev/null || shift $#
    [[ -n $ds && -n $src ]] || { die "Usage: run predict <dataset> <image|folder|video|url|csv|sentence|N> [model]"; return 1; }
    resolve_ds yolo "$ds" dir
    local m
    if [[ $size == tiny ]]; then m=img_tiny; [[ -d models/$DS_NAME/text_tiny ]] && m=text_tiny
    elif [[ $size =~ ^(text_)?(tiny|small|base|thai|large)$ ]]; then get_text_model "$size"; m=$TEXT_MODEL
    elif [[ -n $size && ( $size =~ ^(img|seg)_ || -n ${VISION_MODELS[${size,,}]:-} ) ]]; then get_vision_model "$size" "" || return 1; m=$VISION_MODEL
    elif [[ -n $size && -f configs/model/$size.yaml ]]; then m=$size
    elif [[ -n $size ]]; then get_sizes "$size" || return 1; m=${SIZES[0]}
    else
        m=$(best_model "$DS_NAME")
        [[ -n $m ]] || { die "No trained model for '$DS_NAME'. Train first: run auto <data>"; return 1; }
    fi
    src=$(clean_path "$src")
    local what="+input='$src'" tabular=0
    [[ -f models/$DS_NAME/$m.joblib ]] && tabular=1
    ((tabular)) && [[ ! -f configs/data/$DS_NAME.yaml ]] && DS_ARGS=("data=csv" "data.name=$DS_NAME")
    if ((tabular)) && [[ $src =~ ^[0-9]+$ ]]; then what="+horizon=$src"   # forecast N periods ahead
    elif [[ $src != *://* ]]; then
        if [[ -e $src ]]; then src=$(readlink -f "$src"); what="+input='$src'"
        elif [[ $m == text_* ]]; then what="+text='${src//\'/’}'"   # a sentence
        else die "Not found: $src"; return 1; fi
    fi
    say "Predicting with $m on $src"
    uv run mlt-predict "${DS_ARGS[@]}" "model=$m" "$what" "$@" || return 1
    local out="reports/$DS_NAME/$m/predict"
    [[ -d $out ]] || return 0
    if [[ $m == text_* || $m == img_* ]] || ((tabular)); then
        say "Results: $out/predictions.csv" GREEN; [[ -f $out/predictions.csv ]] && show_csv "$out/predictions.csv" 15
    elif [[ $m == seg_* ]]; then say "Coloured overlays + masks + predictions.csv (% of each class per image): $out" GREEN
    else say "Annotated files + detections.csv (one row per prediction): $out" GREEN; fi
}

invoke_explain() {  # dataset size count extra...
    local ds=${1:-} size=${2:-} count=${3:-}; shift 3 2>/dev/null || shift $#
    [[ -n $ds ]] || { die "Usage: run explain <dataset> [n|s|m|l|x] [images]"; return 1; }
    resolve_ds yolo "$ds" dir
    local m more=("$@")
    if [[ -n $size ]]; then get_sizes "$size" || return 1; m=${SIZES[0]}
    else m=$(best_model "$DS_NAME"); [[ -n $m ]] || { die "No trained model for '$DS_NAME'. Train first: run fit $ds"; return 1; }; fi
    [[ -n $count ]] && more+=("xai.max_images=$count")
    say "Explaining $m on $DS_NAME (heatmaps + statistics)..."
    py -m ml_training.xai.run "${DS_ARGS[@]}" "model=$m" "${more[@]}" || return 1
    local x="reports/$DS_NAME/$m/xai/xai_report.xlsx"
    [[ -f $x ]] && { say "XAI report: $x   (heatmap images: reports/$DS_NAME/$m/xai/images)" GREEN; LAST_REPORT=$x; }
    return 0
}

# unfinished trainings as "dataset|model|stage|time" (from models/<dataset>/<model>/run_state.json)
get_unfinished() {
    compgen -G "models/*/*/run_state.json" >/dev/null || return 0
    uv run --no-sync python - <<'EOF'
import glob, json, os
stages = {"start": "starting", "hpo": "Optuna search", "final": "final training", "evaluate": "evaluation"}
for f in sorted(glob.glob("models/*/*/run_state.json")):
    try:
        j = json.load(open(f, encoding="utf-8"))
    except Exception:
        continue
    if j.get("status") != "finished":
        model_dir = os.path.dirname(f)
        st = stages.get(str(j.get("stage")), str(j.get("stage")))
        print(f"{os.path.basename(os.path.dirname(model_dir))}|{os.path.basename(model_dir)}|stopped in {st}|{j.get('updated', '')}")
EOF
}

invoke_resume() {
    local ds=${1:-} size=${2:-}
    if [[ -z $ds ]]; then
        local u; u=$(get_unfinished)
        [[ -n $u ]] || { say "Nothing to resume - every training finished." GREEN; return 0; }
        say "Unfinished trainings:"
        while IFS='|' read -r a b c d; do printf '  %-16s %-9s %s  (%s)\n' "$a" "$b" "$c" "$d"; done <<<"$u"
        say "Continue one with: run resume <dataset> [size]"
        return 0
    fi
    resolve_ds yolo "$ds" dir
    local m=(); [[ -n $size ]] && { get_sizes "$size" || return 1; m=("${SIZES[0]}"); }
    py -m ml_training.resume run "$DS_NAME" "${m[@]}" || return 1
    LAST_REPORT=$(ls -t reports/"$DS_NAME"/*/report.xlsx 2>/dev/null | head -n1)
    [[ -n $LAST_REPORT ]] && say "Finished. Excel report: $LAST_REPORT" GREEN
    return 0
}

invoke_clean() {
    [[ -n ${1:-} ]] || { die "Usage: run clean <dataset|all> [raw]"; return 1; }
    local flags=(); [[ " $* " == *" raw "* ]] && flags=(--raw)
    py -m ml_training.clean "$1" "${flags[@]}"
}

# ----------------------------------------------------------------------------- server: web UIs, background runs
tunnel_hint() {  # port
    local host; host=$(hostname -I 2>/dev/null | awk '{print $1}')
    say "$(T "เปิดจากเครื่องตัวเอง: รันคำสั่งนี้บนเครื่องของคุณ (ไม่ใช่บน server) แล้วเปิด" "Open it from your own computer: run this there (not on the server), then open") http://localhost:$1" GREEN
    say "    ssh -N -L $1:localhost:$1 $ME@${host:-<server>}" WHITE
}
run_ui() { tunnel_hint 5000; uv run mlflow ui --backend-store-uri "sqlite:///mlflow.db" --host 127.0.0.1 --port 5000; }
run_optuna() { tunnel_hint 8080; uv run --with optuna-dashboard optuna-dashboard "sqlite:///optuna.db" --host 127.0.0.1 --port 8080; }

run_bg() {  # any run command, detached from the terminal; output -> logs/bg/<time>-<command>.log
    [[ $# -gt 0 ]] || { die "Usage: run bg <command ...>   e.g. run bg fit helmet m"; return 1; }
    mkdir -p logs/bg
    local log; log="logs/bg/$(date +%Y%m%d-%H%M%S)-$1.log"
    printf '$ run %s\n' "$*" >"$log"
    MLT_BG=1 nohup setsid bash "$SELF" "$@" </dev/null >>"$log" 2>&1 &
    echo "$! $*" >"${log%.log}.pid"
    say "$(T "เริ่มทำงานเบื้องหลังแล้ว (PID $!) - ออกจาก SSH / ปิดหน้าต่างได้ งานจะยังทำต่อ" "Started in the background (PID $!) - you can log out of SSH, it keeps running")" GREEN
    say "  $(T "ดูความคืบหน้า:" "follow it:") run logs      $(T "งานทั้งหมด:" "all jobs:") run jobs      $(T "หยุด:" "stop:") kill $!"
    BG_LOG=$log
}

show_jobs() {
    local f pid cmd any=0
    for f in logs/bg/*.pid; do
        [[ -f $f ]] || continue
        read -r pid cmd <"$f"; any=1
        if kill -0 "$pid" 2>/dev/null; then printf '  %srunning%s  PID %-7s %s   (%s)\n' "$C_GREEN" "$C_OFF" "$pid" "$cmd" "${f%.pid}.log"
        else printf '  %sended%s    PID %-7s %s   (%s: %s)\n' "$C_GREY" "$C_OFF" "$pid" "$cmd" "${f%.pid}.log" "$(tail -n1 "${f%.pid}.log" | cut -c1-60)"; fi
    done
    ((any)) || say "$(T "ยังไม่มีงานเบื้องหลัง - เริ่มด้วย run bg <คำสั่ง>" "No background jobs yet - start one with run bg <command>")"
}

follow_log() {
    local log=${1:-}; [[ -n $log ]] || log=$(ls -t logs/bg/*.log 2>/dev/null | head -n1)
    [[ -n $log ]] || { show_jobs; return 0; }
    say "$log   ($(T "กด Ctrl+C เพื่อเลิกดู - งานยังทำต่อ" "Ctrl+C stops watching - the job keeps running"))" GREY
    tail -n 40 -f "$log"
}

# ----------------------------------------------------------------------------- menu helpers
read_choice() {  # prompt default option... -> CHOICE (0-based)
    local prompt=$1 def=$2 i a; shift 2
    for ((i = 0; i < $#; i++)); do printf '  %d) %s\n' $((i + 1)) "${@:i+1:1}"; done
    if [[ ! -t 0 ]]; then CHOICE=$((def - 1)); echo "$prompt: $def"; return 0; fi   # background run: the default
    while true; do
        read -r -e -p "$prompt [$def]: " a || exit 0
        [[ -z $a ]] && { CHOICE=$((def - 1)); return 0; }
        if [[ $a =~ ^[0-9]+$ ]] && ((a >= 1 && a <= $#)); then CHOICE=$((a - 1)); return 0; fi
        say "$(T "พิมพ์ตัวเลข 1-$#" "Please type a number 1-$#")" YELLOW
    done
}
ask_yes() {  # question default(y|n)
    local d=${2:-y} a
    if [[ ! -t 0 ]]; then echo "$1: $d"; [[ $d == y ]]; return; fi
    read -r -p "$1 (y/n) [$d]: " a || exit 0
    a=${a:-$d}; [[ ${a,,} == y || ${a,,} == yes || $a == ใช่ ]]
}
ask() { local a; read -r -e -p "$1: " a || exit 0; clean_path "$a"; }
report_hint() {
    [[ -n $LAST_REPORT && -e $LAST_REPORT ]] || return 0
    say "$(T "รายงาน Excel:" "Excel report:") $PWD/$LAST_REPORT" GREEN
    say "  $(T "คัดลอกมาเปิดบนเครื่องตัวเอง:" "copy it to your computer:") scp $ME@$(hostname -I 2>/dev/null | awk '{print $1}'):$PWD/$LAST_REPORT ." GREY
}

trained_names() {
    local d
    for d in models/*/; do
        d=${d%/}; d=${d#models/}
        [[ $d == pretrained || $d == runs || $d == '*' ]] && continue
        compgen -G "models/$d/*.pt" >/dev/null || compgen -G "models/$d/*.joblib" >/dev/null || compgen -G "models/$d/*/model" >/dev/null || continue
        echo "$d"
    done
}

select_trained() {
    local names; mapfile -t names < <(trained_names)
    ((${#names[@]})) || { die "$(T "ยังไม่มีโมเดลที่เทรนแล้ว - เทรนก่อนจากเมนู 1" "No trained model yet - train one first (menu 1)")"; return 1; }
    say "$(T "\nชุดข้อมูล:" "\nDataset:")"
    read_choice "$(T "เลือก" "Choose")" 1 "${names[@]}"
    PICK=${names[$CHOICE]}
}

# lists data/raw (+ data/external) entries and samples; sets PICK
select_from() {  # title  filter(dirs|files|both)  sample...  ("name|description")
    local title=$1 filter=$2 f names=() opts=() s; shift 2
    for f in data/raw/* data/external/*; do
        [[ -e $f ]] || continue
        case $filter in dirs) [[ -d $f ]] || continue ;; files) [[ -f $f && $f =~ \.(csv|xlsx|xls|parquet)$ ]] || continue ;;
            both) [[ -d $f || $f =~ \.(csv|xlsx|xls|jsonl|json|tsv|conll)$ ]] || continue ;; esac
        [[ $f == data/external/* && -d $f ]] && continue
        names+=("$f"); opts+=("$f")
    done
    opts+=("$(T "พิมพ์ที่อยู่ไฟล์ / โฟลเดอร์..." "Type a file / folder path...")")
    for s in "$@"; do opts+=("$(T "ตัวอย่าง: " "Sample: ")${s#*|}"); done
    say "\n$title"
    read_choice "$(T "เลือก" "Choose")" 1 "${opts[@]}"
    if ((CHOICE < ${#names[@]})); then PICK=${names[$CHOICE]}
    elif ((CHOICE == ${#names[@]})); then PICK=$(ask "$(T "ที่อยู่" "Path")")
    else s=${*:CHOICE-${#names[@]}:1}; PICK=${s%%|*}; fi
}

presets() {  # "name|about" lines
    local rows; rows=$(py -m ml_training.tune_menu names 2>/dev/null | grep '|')
    [[ -n $rows ]] || rows=$(for f in configs/tune/*.yaml; do f=${f##*/}; echo "${f%.yaml}|"; done)
    printf '%s\n' "$rows"
}
select_preset() {
    local rows opts=() r; mapfile -t rows < <(presets)
    for r in "${rows[@]}"; do opts+=("$(printf '%-12s%s' "${r%%|*}" "${r#*|}")"); done
    read_choice "$(T "เลือก" "Choose")" 1 "${opts[@]}"
    PICK=${rows[$CHOICE]%%|*}
}

tuning_menu() {
    while true; do
        say "$(T "\n---------- ตั้งค่าการจูน (YOLO) ----------" "\n---------- Tuning settings ----------")"
        read_choice "$(T "เลือก" "Choose")" 1 \
            "$(T "ดู preset ทั้งหมด" "Show all presets (epochs, trials, freeze, how many settings are tuned)")" \
            "$(T "สร้าง preset ของฉัน (ถามเฉพาะค่าที่ใช้บ่อย)" "Create my preset (asks only the common settings)")" \
            "$(T "แก้ preset ด้วย editor (ตรวจให้เมื่อปิด)" "Edit a preset in ${EDITOR:-nano} (checked when you close it)")" \
            "$(T "กลับ" "Back")"
        case $CHOICE in
            0) py -m ml_training.tune_menu list ;;
            1) py -m ml_training.tune_menu create ;;
            2) select_preset; "${EDITOR:-nano}" "configs/tune/$PICK.yaml"
               if py -m ml_training.tune_menu check "$PICK"; then say "$(T "Preset '$PICK' ใช้ได้" "Preset '$PICK' is OK.")" GREEN
               else say "$(T "ไฟล์มีปัญหา (ดูด้านบน) - แก้อีกครั้ง" "The file has a problem (see above) - edit it again.")" RED; fi ;;
            *) return 0 ;;
        esac
    done
}

# runs a run command now, or in the background when the user wants (long trainings over SSH)
launch() {
    if [[ -z ${MLT_BG:-} ]] && ask_yes "$(T "ทำงานเบื้องหลังไหม (ปิด SSH ได้ งานไม่หยุด)" "Run in the background (safe to close SSH)?")" n; then
        run_bg "$@"
        ask_yes "$(T "ดูความคืบหน้าตอนนี้เลยไหม" "Watch the progress now?")" y && follow_log "$BG_LOG"
        return 0
    fi
    dispatch "$@"
}

# ----------------------------------------------------------------------------- automatic training
SAMPLES=("shapes|vision|classify|ภาพ: จัดหมวดรูปทรง|images: classify shapes"
    "shapes-seg|vision|semantic|ภาพ: แยกทุกพิกเซล (segmentation)|images: pixel segmentation"
    "coco8|yolo|detect|ภาพ: ตรวจจับวัตถุ YOLO (ดาวน์โหลดเล็กน้อย)|images: YOLO object detection (small download)"
    "thai-sentiment|text|classify|ข้อความ: จัดหมวดรีวิวภาษาไทย|text: classify Thai reviews"
    "thai-docs|text|classify|เอกสาร: แยกประเภทเอกสาร 4 แบบ|documents: 4 document types"
    "house-prices|table|regress|ตาราง: ทายราคาบ้าน (ทายตัวเลข)|table: predict house prices"
    "shop-sales|table|forecast|ตาราง: พยากรณ์ยอดขายรายวัน|table: forecast daily sales")

# inspect_data --json -> shell variables I_kind, I_task, I_summary ... (one python call)
inspect_vars() {
    py -m ml_training.inspect_data "$1" --json | tail -n1 | uv run --no-sync python -c '
import json, shlex, sys
try:
    d = json.loads(sys.stdin.read())
except Exception as e:
    d = {"error": f"could not read the data check: {e}"}
for k in ("error", "kind", "task", "name", "path", "summary", "task_th", "gpu", "gpu_name", "vram_gb", "thai", "target", "time_column", "group_column"):
    v = d.get(k)
    v = "" if v in (None, False) else ("1" if v is True else str(v))
    print(f"I_{k}={shlex.quote(v)}")
print("I_columns=(" + " ".join(shlex.quote(str(c)) for c in d.get("columns") or []) + ")")'
}

wizard() {
    local given=${1:-} preset_level=${2:-} ans sample=0 s
    say "$(T "\n========== เทรนโมเดลใหม่แบบอัตโนมัติ ==========" "\n========== Train a new model automatically ==========")"
    if [[ -n $given ]]; then ans=$(clean_path "$given")
    else
        say "$(T "พิมพ์ที่อยู่โฟลเดอร์ / ไฟล์ข้อมูล (Tab เติมชื่อให้) หรือเลือกหมายเลข" "Type the path of a data folder / file (Tab completes it) or pick a number")" WHITE
        local have=() opts=() f n
        for f in data/raw/* data/external/*; do
            [[ -e $f ]] || continue
            [[ -d $f || $f =~ \.(csv|xlsx|xls|parquet|jsonl)$ ]] || continue
            n=$(basename "${f%.*}"); [[ -d $f ]] && n=$(basename "$f")
            for s in "${SAMPLES[@]}"; do [[ ${s%%|*} == "$n" ]] && continue 2; done
            have+=("$f"); opts+=("$(T "ข้อมูลของฉัน: " "my data: ")$f")
        done
        for s in "${SAMPLES[@]}"; do IFS='|' read -r n _ _ th en <<<"$s"; opts+=("$(T "ตัวอย่าง: " "sample: ")$(printf '%-15s' "$n")$(T "$th" "$en")"); done
        for ((n = 0; n < ${#opts[@]}; n++)); do printf '  %d) %s\n' $((n + 1)) "${opts[n]}"; done
        ans=$(ask "$(T "ข้อมูล" "Data")")
        [[ -n $ans ]] || return 0
        if [[ $ans =~ ^[0-9]+$ ]] && ((ans >= 1 && ans <= ${#opts[@]})); then
            if ((ans <= ${#have[@]})); then ans=$(readlink -f "${have[ans - 1]}"); else s=${SAMPLES[ans - ${#have[@]} - 1]}; ans=${s%%|*}; fi
        fi
    fi
    local I_error="" I_kind="" I_task="" I_name="" I_path="" I_summary="" I_task_th="" I_gpu="" I_gpu_name="" I_vram_gb="" I_thai="" I_target="" I_time_column="" I_group_column="" I_columns=()
    for s in "${SAMPLES[@]}"; do
        if [[ ${s%%|*} == "$ans" ]]; then
            local th en
            # shellcheck disable=SC2034
            IFS='|' read -r I_name I_kind I_task th en <<<"$s"
            I_path=$ans; I_summary=$(T "$th" "$en"); I_thai=1; I_vram_gb=0; sample=1
        fi
    done
    if ((!sample)); then
        say "$(T "กำลังตรวจข้อมูล..." "Looking at the data...")"
        eval "$(inspect_vars "$ans")"
        [[ -z $I_error ]] || { say "$I_error" RED; return 1; }
    fi
    say "\n$(T "พบข้อมูล:" "Found:")"
    say "  $I_summary" WHITE
    [[ -n $I_task_th ]] && say "$(T "  งานที่เหมาะ: " "  Task: ")$I_task_th" WHITE
    if ((!sample)); then
        if [[ -n $I_gpu ]]; then say "$(T "  เครื่องนี้: การ์ดจอ " "  This server: GPU ")$I_gpu_name ($I_vram_gb GB)" GREEN
        elif [[ $I_kind != table ]]; then say "$(T "  ไม่พบการ์ดจอ NVIDIA - จะช้า แนะนำเลือก 'Quick' ก่อน (หรือรัน run doctor)" "  No NVIDIA GPU found - training will be slow, try 'Quick' first (or run run doctor)")" YELLOW; fi
    fi

    local extra=() t
    if [[ $I_kind == table ]] && ((!sample)); then
        say "$(T "\nคอลัมน์: " "\nColumns: ")$(IFS=,; echo "${I_columns[*]}" | sed 's/,/, /g')"
        t=""; [[ -t 0 ]] && { read -r -e -p "$(T "คอลัมน์ที่ต้องการให้โมเดลทาย" "Column to predict") [$I_target]: " t || exit 0; }
        t=${t:-$I_target}
        [[ " ${I_columns[*]} " == *" $t "* ]] || { say "$(T "ไม่มีคอลัมน์ '$t' ในตาราง" "No column '$t' in the table")" RED; return 1; }
        extra+=("target=$t")
        if [[ -n $I_time_column ]] && ask_yes "$(T "เป็นข้อมูลตามเวลา (คอลัมน์ '$I_time_column') ต้องการพยากรณ์อนาคตไหม" "Time series on '$I_time_column' - forecast the future?")" y; then
            extra+=("time=$I_time_column")
            [[ -n $I_group_column ]] && { extra+=("group=$I_group_column"); say "$(T "  แยกพยากรณ์ตาม '$I_group_column'" "  one series per '$I_group_column'")"; }
        elif [[ $I_task == forecast ]]; then extra+=(task=regress); fi
    fi

    case $preset_level in quick) CHOICE=0 ;; standard | full) CHOICE=1 ;; best | accurate) CHOICE=2 ;; *) preset_level="" ;; esac
    [[ -z $preset_level ]] && say "$(T "\nต้องการแบบไหน:" "\nHow thorough:")" && read_choice "$(T "เลือก" "Choose")" 2 \
        "$(T "Quick         - ไม่กี่นาที ไว้ดูว่าข้อมูลใช้ได้" "Quick         - a few minutes, to check the data works")" \
        "$(T "Standard      - จูนค่าอัตโนมัติ ได้โมเดลที่ดี (แนะนำ)" "Standard      - automatic tuning, a good model (recommended)")" \
        "$(T "Most accurate - โมเดลใหญ่ขึ้น / เทียบหลายโมเดล ใช้เวลานาน" "Most accurate - bigger model / compare several, takes long")"
    local level=$CHOICE mode=full size cmd=(); ((level == 0)) && mode=quick
    case $I_kind in
        yolo)
            local sizes=(n s m) tunes=(default default accurate); size=${sizes[level]}
            [[ -z $I_gpu ]] && ((!sample && level > 0)) && size=n
            say "$(T "\nจะเทรน: " "\nWill train: ")YOLO11$size  ($I_task, $mode)" WHITE
            cmd=(fit "$I_path" "$size" "$mode" "tune=${tunes[level]}" "data.task=$I_task") ;;
        vision)
            local vt=classify sizes=(efficientnet efficientnet convnext)
            [[ $I_task == semantic ]] && { vt=segment; sizes=(seg-b0 seg-b0 seg-b2); }
            size=${sizes[level]}; ((sample)) && { size=tiny; [[ $vt == segment ]] && size="seg-tiny"; }
            say "$(T "\nจะเทรน: " "\nWill train: ")$size  ($vt, $mode)" WHITE
            cmd=(vision "$I_path" "$size" "$mode" "data.task=$vt") ;;
        text)
            local std=base; [[ -n $I_thai ]] && std=thai
            local sizes=(small "$std" large); size=${sizes[level]}
            if ((level == 2)) && [[ ${I_vram_gb%.*} -lt 12 ]]; then size=$std; say "$(T "  การ์ดจอเล็กกว่า 12 GB - ใช้ $std แทน large" "  GPU under 12 GB - using $std instead of large")" YELLOW; fi
            ((sample)) && size=tiny
            say "$(T "\nจะเทรน: โมเดลข้อความ " "\nWill train: text model ")$size  ($I_task, $mode)" WHITE
            cmd=(text "$I_path" "$size" "$mode") ;;
        table)
            local ms=(lightgbm lightgbm all)
            say "$(T "\nจะเทรน: " "\nWill train: ")${ms[level]}  ($mode)" WHITE
            cmd=(table "$I_path" "${ms[level]}" "$mode" "${extra[@]}") ;;
        *) die "$(T "ไม่รู้จักชนิดข้อมูล" "Unknown kind of data"): $I_kind"; return 1 ;;
    esac
    [[ -n $preset_level ]] || ask_yes "$(T "เริ่มเลยไหม" "Start now?")" y || return 0
    say "  = run ${cmd[*]}" GREY
    launch "${cmd[@]}" || return 1
    [[ -n ${BG_LOG:-} ]] && return 0
    say "$(T "\nเสร็จแล้ว! ทำนายข้อมูลใหม่: run predict $DS_NAME <ไฟล์/โฟลเดอร์> · ดูผล: run results $DS_NAME" "\nDone! Predict new data: run predict $DS_NAME <file/folder> · scores: run results $DS_NAME")" GREEN
    report_hint
}

# ----------------------------------------------------------------------------- menus
train_menu() {
    say "$(T "\n---------- เทรนแบบเลือกเอง ----------" "\n---------- Train (choose yourself) ----------")"
    read_choice "$(T "เลือก" "Choose")" 1 \
        "$(T "YOLO: ตรวจจับ / แยกวัตถุ / จัดหมวด / pose / กรอบเอียง" "YOLO (detect / segment / classify / pose / obb)")" \
        "$(T "ข้อความ / เอกสาร: จัดหมวด ดึงข้อมูลสำคัญ (NER) แยกประเภทเอกสาร" "TEXT / DOCUMENTS (classify texts, NER, document folders)")" \
        "$(T "ภาพ: จัดหมวดด้วย ResNet / EfficientNet / ConvNeXt / ViT หรือแยกทุกพิกเซล" "IMAGES beyond YOLO (classify, pixel segmentation)")" \
        "$(T "ตาราง (CSV / Excel): จัดหมวด ทายตัวเลข พยากรณ์อนาคต" "TABLES (classify, predict a number, forecast)")" \
        "$(T "กลับ" "Back")"
    local k=$CHOICE ds size mode extra=() quick full
    quick=$(T "Quick (ไม่จูน)" "Quick test (no tuning)"); full=$(T "Full (Optuna จูน + เทรนจริง)" "Full (Optuna tuning + final training)")
    LAST_REPORT=""
    case $k in
        0)
            select_from "$(T "ชุดข้อมูล YOLO:" "YOLO dataset:")" dirs "coco8|coco8       detect" "coco8-seg|coco8-seg   segment" "coco8-pose|coco8-pose  pose" "dota8|dota8       obb" "mnist160|mnist160    classify"
            ds=$PICK
            say "$(T "\nรุ่น YOLO:" "\nYOLO version:")"; read_choice "$(T "เลือก" "Choose")" 1 "YOLO11 $(T "(เสถียร ค่าเริ่มต้น)" "(proven, default)")" "YOLO26 $(T "(ใหม่ล่าสุด)" "(newest)")"
            local gen=""; ((CHOICE == 1)) && gen=26
            say "$(T "\nขนาดโมเดล (n = เร็วที่สุด ... x = แม่นที่สุด):" "\nModel size (n = fastest ... x = most accurate):")"
            local sz=(n s m l x all); read_choice "$(T "เลือก" "Choose")" 1 "n  (nano)" "s  (small)" "m  (medium)" "l  (large)" "x  (extra large)" "$(T "ทุกขนาด แล้วเทียบกัน" "all sizes, one after another")"
            size=$gen${sz[CHOICE]}
            say "$(T "\nโหมด:" "\nTraining mode:")"; read_choice "$(T "เลือก" "Choose")" 1 "$quick, 30 epochs" "$full"
            mode=quick; ((CHOICE == 1)) && mode=full
            say "$(T "\nPreset การจูน:" "\nTuning preset:")"; select_preset
            launch fit "$ds" "$size" "$mode" "tune=$PICK" ;;
        1)
            select_from "$(T "ข้อมูลข้อความ (โฟลเดอร์หรือไฟล์):" "Text dataset (folder or file):")" both "thai-sentiment|thai-sentiment  $(T "(จัดหมวดรีวิว)" "(classify reviews)")" "thai-ner|thai-ner        $(T "(ดึงชื่อคน บริษัท วันที่ จำนวนเงิน)" "(names, companies, dates, amounts)")" "thai-docs|thai-docs       $(T "(แยกประเภทเอกสาร)" "(invoice / receipt / contract / quotation)")"
            ds=$PICK
            say "$(T "\nโมเดล (อ่านภาษาไทยได้ทุกตัว):" "\nModel (all read Thai):")"
            read_choice "$(T "เลือก" "Choose")" 3 \
                "tiny   $(T "เทรนจากศูนย์ ไว้ลองระบบเท่านั้น" "trained from scratch, no download - only to try the pipeline")" \
                "small  $(T "DistilBERT หลายภาษา - เร็ว" "multilingual DistilBERT - fast, OK on CPU")" \
                "base   $(T "XLM-RoBERTa - แม่นและรองรับไทยดี" "XLM-RoBERTa base - strong default (GPU recommended)")" \
                "thai   $(T "WangchanBERTa - ดีที่สุดสำหรับไทยล้วน" "WangchanBERTa - best for Thai-only text")" \
                "large  $(T "XLM-RoBERTa large - แม่นที่สุด (GPU 12 GB+)" "XLM-RoBERTa large - most accurate (GPU 12 GB+)")"
            size=${TEXT_SIZES[CHOICE]}
            say "$(T "\nโหมด:" "\nTraining mode:")"; read_choice "$(T "เลือก" "Choose")" 1 "$quick, 2 epochs" "$full"
            mode=quick; ((CHOICE == 1)) && mode=full
            launch text "$ds" "$size" "$mode" ;;
        2)
            select_from "$(T "ชุดภาพ (โฟลเดอร์ย่อยตามหมวด หรือ images/ + masks/):" "Image dataset (one sub-folder per class, or images/ + masks/):")" dirs "shapes|shapes      $(T "(จัดหมวดรูปทรง)" "(classify circle / square / triangle)")" "shapes-seg|shapes-seg  $(T "(แยกทุกพิกเซล)" "(pixel segmentation)")"
            ds=$PICK
            local seg=0; [[ $ds == shapes-seg ]] && seg=1
            [[ -d $ds/masks || -d $ds/mask || -d $ds/SegmentationClass ]] && seg=1
            if [[ $ds == shapes* ]]; then size=tiny; ((seg)) && size="seg-tiny"
            elif ((seg)); then
                say "$(T "\nโมเดลแยกทุกพิกเซล (SegFormer):" "\nSegmentation model (SegFormer, pretrained):")"
                local o=(seg-b0 seg-b2); read_choice "$(T "เลือก" "Choose")" 1 "seg-b0  $(T "เร็ว" "fast, any GPU")" "seg-b2  $(T "แม่นกว่า GPU 8 GB+" "more accurate, GPU 8 GB+")"; size=${o[CHOICE]}
            else
                say "$(T "\nโมเดลจัดหมวดภาพ:" "\nClassification model (ImageNet-pretrained, downloaded once):")"
                local o=(efficientnet resnet50 convnext vit)
                read_choice "$(T "เลือก" "Choose")" 1 "efficientnet  $(T "เล็ก เร็ว แม่นดี (แนะนำ)" "small and fast, good default")" "resnet50      $(T "มาตรฐาน ทนทาน" "classic, robust")" "convnext      $(T "มักแม่นที่สุด" "usually the most accurate here")" "vit           $(T "ข้อมูลเยอะ GPU 8 GB+" "best with lots of data (GPU 8 GB+)")"
                size=${o[CHOICE]}
            fi
            say "$(T "\nโหมด:" "\nTraining mode:")"; read_choice "$(T "เลือก" "Choose")" 1 "$quick, 3 epochs" "$full"
            mode=quick; ((CHOICE == 1)) && mode=full
            launch vision "$ds" "$size" "$mode" ;;
        3)
            select_from "$(T "ตาราง (CSV / Excel):" "Table (CSV / Excel):")" files "house-prices|house-prices  $(T "(ทายราคาบ้าน)" "(predict a price = regression)")" "shop-sales|shop-sales    $(T "(พยากรณ์ยอดขาย)" "(forecast daily sales of 3 shops)")" "breast_cancer|breast_cancer $(T "(จัดหมวด 2 กลุ่ม)" "(classify, 2 classes)")"
            ds=$PICK
            if [[ ! -f configs/data/$ds.yaml ]]; then
                [[ $ds == *.csv ]] && say "$(T "\nคอลัมน์: " "\nColumns: ")$(head -n1 "$ds" | tr -d '"\r' | sed 's/,/, /g')"
                local t g; t=$(ask "$(T "คอลัมน์ที่ต้องการทาย (target)" "Column to predict (target)")"); extra+=("target=$t")
                t=$(ask "$(T "คอลัมน์วันที่ ถ้าต้องการพยากรณ์อนาคต (เว้นว่าง = ไม่ใช่)" "Date / time column for forecasting (empty = not a time series)")")
                if [[ -n $t ]]; then
                    extra+=("time=$t")
                    g=$(ask "$(T "คอลัมน์ที่แยกหลายชุด เช่น สาขา (เว้นว่าง = ชุดเดียว)" "Column that separates several series, e.g. shop (empty = one series)")")
                    [[ -n $g ]] && extra+=("group=$g")
                fi
            fi
            say "$(T "\nโมเดล:" "\nModel:")"
            local o=(lightgbm xgboost random_forest all)
            read_choice "$(T "เลือก" "Choose")" 1 "lightgbm      $(T "เร็ว มักแม่นที่สุด (แนะนำ)" "fast, usually the most accurate on tables")" "xgboost       $(T "แม่นมาก" "very accurate gradient boosting")" "random_forest $(T "ทนทาน" "robust, few settings")" "all           $(T "ทุกโมเดลแล้วเทียบกัน" "train every model and compare")"
            local models=${o[CHOICE]}
            say "$(T "\nโหมด:" "\nTraining mode:")"; read_choice "$(T "เลือก" "Choose")" 2 "$quick" "$(T "Full (Optuna 30 รอบ)" "Full (Optuna 30 trials + final model)")"
            mode=quick; ((CHOICE == 1)) && mode=full
            launch table "$ds" "$models" "$mode" "${extra[@]}" ;;
        *) return 0 ;;
    esac
    [[ -z ${BG_LOG:-} ]] && report_hint
    return 0
}

more_menu() {
    while true; do
        say "$(T "\n---------- เครื่องมือเพิ่มเติม ----------" "\n---------- More tools ----------")"
        read_choice "$(T "เลือก" "Choose")" 1 \
            "$(T "ตรวจเครื่อง (การ์ดจอ ไดรเวอร์ ส่วนที่ติดตั้ง ดิสก์)" "Check this server (GPU, driver, installed parts, disk)")" \
            "$(T "ติดตั้ง / อัปเดต (หาการ์ดจอเอง)" "Install / update (finds the GPU itself)")" \
            "$(T "ติดตั้งแพ็กเกจ Ubuntu ที่ต้องใช้ (sudo)" "Install the Ubuntu packages it needs (sudo)")" \
            "$(T "ติดตั้ง OCR (อ่านเอกสารสแกน)" "Install OCR (scanned documents)")" \
            "$(T "งานเบื้องหลัง (ดูสถานะ / ดู log)" "Background jobs (status / follow the log)")" \
            "$(T "ตั้งค่าการจูน YOLO" "YOLO tuning presets (view / create / edit)")" \
            "$(T "เปิด MLflow (ผ่าน SSH tunnel)" "Open MLflow UI (through an SSH tunnel)")" \
            "$(T "เปิด Optuna dashboard (ผ่าน SSH tunnel)" "Open Optuna dashboard (through an SSH tunnel)")" \
            "$(T "ประวัติการใช้งาน" "History (every prepare / train / predict / clean)")" \
            "$(T "ล้างข้อมูล / ผลลัพธ์ (ถามก่อนลบ)" "Clean data / results (asks before deleting)")" \
            "$(T "คำสั่งทั้งหมด" "Help (all commands)")" \
            "$(T "Language / ภาษา: เปลี่ยนเป็น English" "Language / ภาษา: switch to Thai")" \
            "$(T "กลับ" "Back")"
        case $CHOICE in
            0) py -m ml_training.doctor ;;
            1) setup_all ;;
            2) setup_system ;;
            3) touch .ocr; need_uv && sync_env "$( [[ -f .gpu && $(uv run --no-sync python -c 'import torch;print(torch.version.cuda or "")' 2>/dev/null) == "" ]] && echo cpu || echo gpu)" ;;
            4) show_jobs; compgen -G "logs/bg/*.log" >/dev/null && ask_yes "$(T "ดู log ล่าสุดไหม" "Follow the latest log?")" y && follow_log ;;
            5) tuning_menu ;;
            6) run_ui ;;
            7) run_optuna ;;
            8) py -m ml_training.lineage 30 --excel ;;
            9) local names=() d o
               for d in data/processed/*/ models/*/ reports/*/; do d=${d%/}; d=${d##*/}; [[ $d == '*' || $d == pretrained || $d == runs || $d == figures ]] && continue; names+=("$d"); done
               mapfile -t names < <(printf '%s\n' "${names[@]}" | sort -u | sed '/^$/d')
               say "$(T "\nล้างอะไร?" "\nClean what?")"
               read_choice "$(T "เลือก" "Choose")" 1 "${names[@]}" "$(T "ทุกชุดข้อมูล (ผลทั้งหมด + ประวัติ MLflow / Optuna)" "ALL datasets (every result + MLflow / Optuna history)")"
               o=all; ((CHOICE < ${#names[@]})) && o=${names[CHOICE]}
               if ask_yes "$(T "ลบข้อมูลดิบใน data/raw ด้วยไหม" "Also delete the raw dataset in data/raw?")" n; then invoke_clean "$o" raw; else invoke_clean "$o"; fi ;;
            10) head -n "$HELP_LINES" "$SELF" | tail -n +2 | sed 's/^# \{0,1\}//' ;;
            11) if [[ $LANG_UI == en ]]; then LANG_UI=th; else LANG_UI=en; fi
                echo "$LANG_UI" >.lang; export MLT_LANG=$LANG_UI
                say "$(T "เปลี่ยนเป็นภาษาไทยแล้ว" "Switched to English")" GREEN ;;
            *) return 0 ;;
        esac
    done
}

main_menu() {
    while true; do
        BG_LOG=""
        say "$(T "\n================ ML Training · ระบบเทรนโมเดล AI ================" "\n================ ML Training ================")"
        local trained u n
        trained=$(trained_names | paste -sd, - | sed 's/,/, /g')
        [[ -n $trained ]] && say "$(T "โมเดลที่เทรนแล้ว: " "Trained: ")$trained" GREY
        u=$(get_unfinished)
        if [[ -n $u ]]; then
            n=$(wc -l <<<"$u")
            say "$(T "มี $n งานเทรนที่หยุดกลางทาง - เลือก 5 เพื่อเทรนต่อ: " "$n training(s) stopped before the end - choose 5 to continue: ")$(cut -d'|' -f1,2 <<<"$u" | tr '|' ' ' | paste -sd, -)" YELLOW
        fi
        n=$(for f in logs/bg/*.pid; do [[ -f $f ]] && read -r p _ <"$f" && kill -0 "$p" 2>/dev/null && echo; done | wc -l)
        ((n > 0)) && say "$(T "มี $n งานกำลังทำงานเบื้องหลัง - ดูได้ที่ 7 > 5 หรือ run logs" "$n background job(s) running - see 7 > 5 or run logs")" GREEN
        read_choice "$(T "เลือก" "Choose")" 1 \
            "$(T "เทรนโมเดลใหม่ - ใส่ที่อยู่ข้อมูล ระบบเลือกงานและโมเดลให้ (แนะนำ)" "Train a new model - give the data path, the right task and model are chosen (recommended)")" \
            "$(T "เทรนแบบเลือกเอง (YOLO / ข้อความ / ภาพ / ตาราง)" "Train, choosing yourself (YOLO / text / images / tables)")" \
            "$(T "ใช้โมเดลทำนายข้อมูลใหม่" "Predict new data with a trained model")" \
            "$(T "ดูผล / เทียบโมเดล" "Results / compare models")" \
            "$(T "เทรนต่อจากที่ค้าง (ไฟดับ / Ctrl+C / SSH หลุด)" "Resume a training that stopped (Ctrl+C / crash / SSH dropped)")" \
            "$(T "อธิบายโมเดล YOLO (XAI)" "Explain a YOLO model (XAI heatmaps + statistics)")" \
            "$(T "เครื่องมือเพิ่มเติม (ติดตั้ง ตรวจเครื่อง งานเบื้องหลัง MLflow ภาษา ...)" "More tools (install, check server, background jobs, MLflow, language ...)")" \
            "$(T "ออก" "Exit")"
        case $CHOICE in
            0) wizard ;;
            1) train_menu ;;
            2) select_trained || continue
               say "$(T "ที่อยู่ไฟล์ / โฟลเดอร์ (ภาพ วิดีโอ CSV PDF Word) หรือพิมพ์ประโยค / URL" "Path of a file / folder (image, video, CSV, PDF, Word) or type a sentence / URL")"
               say "$(T "  ข้อมูลตามเวลา: พิมพ์จำนวนงวด เช่น 30 = พยากรณ์ 30 งวดข้างหน้า" "  time series: type a number, e.g. 30 = forecast 30 periods ahead")" GREY
               local src; read -r -e -p "$(T "ข้อมูลที่จะทำนาย" "What to predict"): " src || exit 0
               invoke_find "$PICK" "$src" "" ;;
            3) select_trained || continue; show_results "$PICK"; LAST_REPORT="reports/$PICK/compare.xlsx"; report_hint ;;
            4) u=$(get_unfinished)
               [[ -n $u ]] || { say "$(T "ไม่มีงานค้าง - ทุกงานเทรนจบแล้ว" "Nothing to resume - every training finished.")" GREEN; continue; }
               local rows opts=() r a b c d; mapfile -t rows <<<"$u"
               for r in "${rows[@]}"; do IFS='|' read -r a b c d <<<"$r"; opts+=("$(printf '%-16s %-9s %s  (%s)' "$a" "$b" "$c" "$d")"); done
               say "$(T "\nงานที่ค้าง:" "\nUnfinished trainings:")"; read_choice "$(T "เลือก" "Choose")" 1 "${opts[@]}"
               IFS='|' read -r a b _ <<<"${rows[CHOICE]}"
               launch resume-model "$a" "$b" ;;
            5) select_trained || continue
               local cnt; read -r -p "$(T "วิเคราะห์ภาพทดสอบกี่ภาพ [100]" "How many test images to analyse? [100]"): " cnt || exit 0
               LAST_REPORT=""; invoke_explain "$PICK" "" "$cnt" && report_hint ;;
            6) more_menu ;;
            *) return 0 ;;
        esac
    done
}

# ----------------------------------------------------------------------------- commands
# words after the dataset may come in any order: size / model, quick|full, tuning preset, key=value
parse_args() {  # allow_preset  args... -> P_SIZE P_MODE P_EXTRA
    local preset=$1 t; shift
    P_SIZE=""; P_MODE=""; P_EXTRA=()
    for t in "$@"; do
        if [[ $t == *=* ]]; then P_EXTRA+=("$t")
        elif [[ $t == quick || $t == full ]]; then P_MODE=$t
        elif ((preset)) && [[ -f configs/tune/$t.yaml ]]; then P_EXTRA+=("tune=$t")
        elif [[ -z $P_SIZE ]]; then P_SIZE=$t
        else die "Unknown argument '$t'"; return 1; fi
    done
}

dispatch() {
    local task=${1:-menu}; shift 2>/dev/null
    case $task in
        menu) main_menu ;;
        help | -h | --help) head -n "$HELP_LINES" "$SELF" | tail -n +2 | sed 's/^# \{0,1\}//' ;;
        fit | detect) local ds=${1:-}; shift 2>/dev/null; parse_args 1 "$@" && invoke_detect "$ds" "$P_SIZE" "${P_MODE:-full}" "${P_EXTRA[@]}" ;;
        text) local ds=${1:-}; shift 2>/dev/null; parse_args 0 "$@" && invoke_text "$ds" "$P_SIZE" "${P_MODE:-full}" "${P_EXTRA[@]}" ;;
        vision | image | images) local ds=${1:-}; shift 2>/dev/null; parse_args 0 "$@" && invoke_vision "$ds" "$P_SIZE" "${P_MODE:-full}" "${P_EXTRA[@]}" ;;
        table | tabular) local ds=${1:-}; shift 2>/dev/null; parse_args 0 "$@" && invoke_table "$ds" "$P_SIZE" "${P_MODE:-full}" "${P_EXTRA[@]}" ;;
        auto | wizard) wizard "${1:-}" "${2:-}" ;;
        train)  # run train <data> [quick|standard|best]; Hydra overrides (key=value) = the raw trainer
            if [[ $# -eq 0 ]]; then wizard
            elif [[ $1 == *=* || $1 == -* ]]; then uv run mlt-train "$@"
            else wizard "$1" "${2:-}"; fi ;;
        results) if [[ -n ${1:-} ]]; then show_results "$1"; else select_trained && show_results "$PICK"; fi ;;
        status) show_status ;;
        stop) stop_job ;;
        check) py -m ml_training.doctor ;;
        predict) if [[ $# -eq 0 || $1 == *=* || $1 == -* ]]; then uv run mlt-predict "$@"; else dispatch find "$@"; fi ;;
        find)
            local ds=${1:-} src=${2:-} size=""; shift 2 2>/dev/null || shift $#
            [[ $# -gt 0 && ${1:-} != *=* ]] && { size=$1; shift; }
            invoke_find "$ds" "$src" "$size" "$@" ;;
        explain | xai)
            local ds=${1:-} size="" count="" more=() t; shift 2>/dev/null
            for t in "$@"; do if [[ $t == *=* ]]; then more+=("$t"); elif [[ $t =~ ^[0-9]+$ ]]; then count=$t; else size=$t; fi; done
            invoke_explain "$ds" "$size" "$count" "${more[@]}" ;;
        resume) invoke_resume "${1:-}" "${2:-}" ;;
        resume-model) py -m ml_training.resume run "$1" "$2" && { LAST_REPORT="reports/$1/$2/report.xlsx"; report_hint; } ;;
        clean) invoke_clean "$@" ;;
        tune) tuning_menu ;;
        history) py -m ml_training.lineage "${1:-30}" --excel ;;
        doctor) py -m ml_training.doctor ;;
        bg) run_bg "$@" ;;
        jobs) show_jobs ;;
        logs | log) follow_log "${1:-}" ;;
        setup | install) setup_all ;;
        setup-dev) need_uv && uv sync --group dev && { [[ -d .git ]] || git init -b main; } && { [[ -d .dvc ]] || uv run dvc init; } && say "Ready. Next: run pipeline" GREEN ;;
        setup-gpu) need_uv && sync_env gpu ;;
        setup-cpu | setup-yolo) need_uv && sync_env cpu ;;
        setup-ocr) touch .ocr; need_uv && sync_env "$( [[ -f .gpu ]] && uv run --no-sync python -c 'import torch,sys;sys.exit(0 if torch.version.cuda else 1)' 2>/dev/null && echo gpu || { [[ -f .gpu ]] && echo cpu || echo gpu; })" ;;
        setup-system) setup_system ;;
        yolo) uv run dvc repro pipelines/yolo/dvc.yaml "$@" && uv run dvc metrics show ;;
        yolo-all) invoke_detect "${1:-}" all "${2:-full}" ;;
        prepare) uv run mlt-prepare "$@" ;;
        featurize) uv run mlt-featurize "$@" ;;
        pipeline) uv run dvc repro "$@" && uv run dvc metrics show ;;
        compare) uv run mlt-train -m "model=random_forest,logistic_regression,gradient_boosting,svm" "$@" ;;
        ui) run_ui ;;
        optuna) run_optuna ;;
        test) uv run ruff check src tests && uv run pytest ;;
        *) say "Unknown command '$task'" YELLOW; head -n "$HELP_LINES" "$SELF" | tail -n +2 | sed 's/^# \{0,1\}//'; return 1 ;;
    esac
}

# paths typed relative to the folder `run` was started in (run train helmet  inside ~/data)
ARGS=()
for a in "$@"; do
    if [[ $ORIG_PWD != "$PWD" && -n $a && $a != *=* && $a != /* && -e $ORIG_PWD/$a ]]; then a=$(readlink -f "$ORIG_PWD/$a"); fi
    ARGS+=("$a")
done
dispatch "${ARGS[@]}"
