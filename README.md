# ml-training · ระบบเทรนโมเดล AI อัตโนมัติ (MLOps Level 3)

ระบบเทรนโมเดลที่ใช้ได้กับ **dataset ใดก็ได้** (ตาราง CSV / Excel / Parquet, OpenML, sklearn) และ **โมเดลใดก็ได้**
ที่มี API แบบ scikit-learn (รวมถึง PyTorch Lightning) พร้อม

- **ปรับ hyperparameter อัตโนมัติ** ด้วย Optuna (กำหนด search space ใน YAML ไม่ต้องเขียนโค้ด)
- **ติดตามการทดลอง** ด้วย MLflow (params, metrics, artifacts, Model Registry)
- **คำนวณ TP / FP / FN / TN** ทั้งแบบ binary และ multiclass (one-vs-rest ต่อคลาส) + precision, recall,
  specificity, F1, ROC-AUC, MCC ฯลฯ
- **ส่งออกรายงาน Excel** พร้อมกราฟ native ของ Excel (แก้ไขกราฟต่อใน Excel ได้)
- **เวอร์ชันข้อมูล + pipeline** ด้วย DVC (`data/raw` → `data/processed` → model)
- **ตั้งค่าทั้งหมดผ่าน Hydra** (`configs/*.yaml`) และ override จาก command line ได้

ใช้งานบน Windows ได้เต็มรูปแบบ (มี `run.ps1`, ใช้ SQLite แทน server, path แบบ Windows-safe)

---

## 1. ติดตั้ง (Windows)

ต้องมี [Git for Windows](https://git-scm.com/download/win) ก่อน จากนั้นเปิด **PowerShell** ในโฟลเดอร์โปรเจกต์ เช่น `D:\ml-trianing`

```powershell
# ครั้งแรกถ้า PowerShell บล็อกสคริปต์
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

.\run.ps1 setup          # ติดตั้ง uv (ถ้ายังไม่มี) + Python 3.12 + dependencies ทั้งหมด
.\run.ps1 setup-dl       # (ไม่บังคับ) ติดตั้ง PyTorch + Lightning สำหรับโมเดล deep learning
```

`uv` จะสร้าง virtualenv ใน `.venv\` และล็อกเวอร์ชันทุกแพ็กเกจใน `uv.lock` ให้ได้ผลซ้ำได้ทุกเครื่อง

> ต้องการ GPU (CUDA)? ติดตั้ง torch แบบ CUDA ตาม https://pytorch.org/get-started/locally/ แล้วตั้ง
> `model.params.accelerator=gpu`

## 2. ลองรันตัวอย่าง (breast cancer, 569 แถว, binary)

```powershell
.\run.ps1 pipeline       # = dvc repro : prepare -> featurize -> train (Optuna 30 trials)
.\run.ps1 ui             # เปิด MLflow UI ที่ http://127.0.0.1:5000
```

ผลลัพธ์ที่ได้

| ไฟล์ | เนื้อหา |
|---|---|
| `reports/breast_cancer/random_forest/report.xlsx` | รายงาน Excel พร้อมกราฟ |
| `reports/breast_cancer/random_forest/metrics.json` | metrics ทั้งหมด (DVC อ่านไฟล์นี้) |
| `reports/breast_cancer/random_forest/figures/*.png` | confusion matrix, ROC curve |
| `models/breast_cancer/random_forest.joblib` | pipeline (preprocess + model) พร้อมใช้ |
| `mlflow.db`, `mlruns/` | ประวัติการทดลอง + Model Registry |
| `optuna.db` | ประวัติ Optuna study (ดูด้วย `.\run.ps1 optuna`) |

### รายงาน Excel มีชีตอะไรบ้าง

| ชีต | เนื้อหา | กราฟ |
|---|---|---|
| Summary | ข้อมูล run, metrics test vs validation, best hyperparameters | Bar chart metrics |
| Confusion | Confusion matrix (heatmap) + กล่อง TP / FP / FN / TN | Stacked column TP/FP/FN/TN ต่อคลาส |
| PerClass | TP, FP, FN, TN, precision, recall, specificity, F1, FPR, FNR, NPV ต่อคลาส | Column chart |
| ROC | จุด FPR/TPR/threshold | ROC curve + เส้น chance |
| HPO_Trials | ทุก trial ของ Optuna + best-so-far | Optimisation history |
| ParamImportance | ความสำคัญของแต่ละ hyperparameter | Bar chart |
| Predictions | y_true, y_pred, ความน่าจะเป็น, ถูก/ผิด ของทุกแถวใน test set | (มี filter) |
| Config | ค่า config ทั้งหมดของ run นั้น | |

## 3. ใช้กับข้อมูลของคุณเอง

วางไฟล์ไว้ที่ `data\external\` แล้วระบุคอลัมน์เป้าหมาย (มีไฟล์ตัวอย่าง `data/external/churn_demo.csv`
ที่มีทั้งคอลัมน์ตัวเลข ข้อความ และค่าว่าง ลองรันคำสั่งด้านล่างโดยเปลี่ยน `churn` เป็น `churn_demo` ได้เลย)

```powershell
$A = "data=csv data.name=churn data.path=data/external/churn.csv data.target=Churn data.drop_columns=[customer_id]"
uv run mlt-prepare   $A.Split()
uv run mlt-featurize $A.Split()
uv run mlt-train     $A.Split() model=gradient_boosting hpo.n_trials=50
```

- คอลัมน์ตัวเลข → เติมค่าว่างด้วย median + scale อัตโนมัติ
- คอลัมน์ข้อความ/หมวดหมู่ → เติมค่าว่าง + one-hot อัตโนมัติ
- target เป็นข้อความหรือตัวเลขก็ได้ (2 คลาสหรือมากกว่า)
- ถ้าจะใช้บ่อย ให้ copy `configs/data/csv.yaml` เป็น `configs/data/churn.yaml` แล้วเรียก `data=churn`

ดึงจาก OpenML: `uv run mlt-prepare data=openml data.loader=credit-g data.name=credit_g`

ทำให้ DVC ติดตามข้อมูลของคุณ: แก้ `vars` ใน `dvc.yaml` (dataset / model / n_trials) แล้ว `.\run.ps1 pipeline`
และ `uv run dvc add data/external/churn.csv` เพื่อเก็บเวอร์ชันไฟล์ต้นฉบับ

## 4. เปลี่ยนโมเดล / ปรับจูน

```powershell
uv run mlt-train model=logistic_regression           # โมเดลที่มีให้: random_forest, logistic_regression,
uv run mlt-train model=svm hpo.n_trials=100          #   gradient_boosting, svm, lightning_mlp
uv run mlt-train hpo=none                            # ไม่จูน ใช้ model.params ตรงๆ
uv run mlt-train hpo.metric=recall                   # จูนเพื่อลด FN (เช่นงานแพทย์)
uv run mlt-train evaluation.average=binary           # รายงาน precision/recall ของคลาส positive
uv run mlt-train evaluation.threshold=0.3            # เปลี่ยน threshold ของ binary
.\run.ps1 compare                                    # เทรนหลายโมเดลต่อกัน แล้วเทียบใน MLflow UI
```

**เพิ่มโมเดลใหม่ไม่ต้องเขียนโค้ด** – สร้าง `configs/model/xgboost.yaml`

```yaml
name: xgboost
backend: sklearn
class_path: xgboost.XGBClassifier      # ติดตั้งก่อน: uv add xgboost
params: {n_estimators: 300, random_state: ${seed}}
search_space:
  max_depth: {type: int, low: 3, high: 10}
  learning_rate: {type: float, low: 0.01, high: 0.3, log: true}
  subsample: {type: float, low: 0.5, high: 1.0}
```

แล้วรัน `uv run mlt-train model=xgboost` (ประเภทใน search space: `int`, `float`, `categorical`; ใส่ `log: true` / `step` ได้)

## 5. ใช้โมเดลทำนาย

```powershell
uv run mlt-predict +input=data/external/new_rows.csv                          # ใช้ไฟล์ .joblib ล่าสุด
uv run mlt-predict +input=new.csv +model_uri=models:/breast_cancer-random_forest/latest   # จาก Model Registry
```

## 6. โครงสร้างโปรเจกต์ (Cookiecutter Data Science v2)

```
ml-training/
├── configs/                 Hydra configs
│   ├── config.yaml          ค่าหลัก (paths, mlflow, evaluation, report)
│   ├── data/                breast_cancer, iris, csv, openml
│   ├── model/               random_forest, logistic_regression, gradient_boosting, svm, lightning_mlp
│   └── hpo/                 optuna, none
├── data/
│   ├── external/            ไฟล์จากภายนอก (ของคุณ)
│   ├── raw/                 ข้อมูลดิบ ห้ามแก้มือ (DVC)
│   ├── interim/             ข้อมูลระหว่างทาง
│   └── processed/           train/val/test + meta.json (DVC)
├── models/                  โมเดลที่เทรนแล้ว (DVC)
├── notebooks/               สำรวจข้อมูลเท่านั้น – โค้ดจริงอยู่ใน src/
├── reports/                 รายงาน Excel, metrics, รูป
├── src/ml_training/
│   ├── dataset.py           stage 1: โหลดข้อมูล -> data/raw
│   ├── features.py          stage 2: clean + split -> data/processed
│   ├── metrics.py           TP/FP/FN/TN + metrics ทั้งหมด
│   ├── hpo.py               Optuna จาก search_space ใน YAML
│   ├── modeling/
│   │   ├── factory.py       model abstraction: Pipeline(preprocess -> model ใดก็ได้)
│   │   ├── lightning_models.py   PyTorch Lightning MLP แบบ sklearn API
│   │   ├── train.py         stage 3: tune -> train -> evaluate -> MLflow -> Excel
│   │   └── predict.py       ทำนาย batch
│   └── reporting/           excel.py (XlsxWriter native charts), plots.py (PNG)
├── tests/                   pytest (unit + end-to-end)
├── dvc.yaml / dvc.lock      DVC pipeline
├── pyproject.toml / uv.lock dependencies (uv)
├── run.ps1                  คำสั่งลัดสำหรับ Windows (Makefile สำหรับ Linux/macOS)
└── docs/ROADMAP.md          แผนไป Level 4 และ 5
```

## 7. หลักการที่ทำให้ได้ Level 3

- **Reproducible**: `uv.lock` ล็อก dependency, `seed` ใน config, DVC ล็อก hash ของข้อมูลและโมเดลใน `dvc.lock`
- **Config-driven**: ไม่มีค่า hard-code ทุกอย่างอยู่ใน YAML และ override ได้, config ของทุก run ถูกเก็บใน MLflow
- **ไม่มี data leakage**: preprocessing ถูก fit เฉพาะบน train และบันทึกไปกับโมเดล; Optuna ใช้ validation set,
  test set ใช้ครั้งเดียวตอนท้าย
- **Tracked**: ทุก trial เป็น nested run ใน MLflow, โมเดลสุดท้ายถูก register ใน Model Registry อัตโนมัติ
- **แยก notebooks ออกจาก src/** และทดสอบได้ด้วย `.\run.ps1 test`

## 8. ทำงานร่วมกับ Git / DVC remote

```powershell
git add . ; git commit -m "first experiment"
uv run dvc remote add -d storage D:\dvc-storage     # หรือ s3://, gdrive://, ssh://
uv run dvc push                                     # สำรองข้อมูล + โมเดล
uv run dvc metrics diff                             # เทียบ metrics กับ commit ก่อน
```

ดูแผนต่อไปที่ [docs/ROADMAP.md](docs/ROADMAP.md)
