# Roadmap: Level 3 → Level 5

ตอนนี้: **Level 3** (Development) – โครงสร้างมาตรฐาน, config-driven, data/model versioning, experiment tracking,
HPO อัตโนมัติ, metrics + รายงานอัตโนมัติ

| มิติ | Level 3 (ตอนนี้) ✅ | Level 4 | Level 5 |
|---|---|---|---|
| Development | CCDS v2, Hydra, uv, DVC, MLflow, Optuna | Ruff + Pytest + pre-commit, CI (GitHub Actions) รัน test + `dvc repro` บน PR, Great Expectations ตรวจคุณภาพข้อมูล | CI เทรนซ้ำอัตโนมัติเมื่อข้อมูลเปลี่ยน, auto-promote โมเดลใน Registry เมื่อดีกว่า champion |
| Deployment | `.joblib` + MLflow Model Registry, `mlt-predict` | Docker image + FastAPI `/predict` โหลดโมเดลจาก Registry (`models:/name@champion`) | Blue/green หรือ canary, rollback อัตโนมัติ, Kubernetes / cloud endpoint |
| Operation | รันมือผ่าน `run.ps1` / `dvc repro` | Orchestration ด้วย Prefect หรือ ZenML (ตั้งเวลา, retry, แจ้งเตือน) | Evidently ตรวจ data/prediction drift → trigger retrain อัตโนมัติ, dashboard monitoring |

## จุดที่เตรียมไว้แล้วในโค้ด

- **Ruff / Pytest**: config อยู่ใน `pyproject.toml`, มี `tests/` (unit + end-to-end) – `uv run pytest`
- **pre-commit**: เพิ่มไฟล์ `.pre-commit-config.yaml` (ruff, ruff-format, nbstripout) แล้ว `uv add --group dev pre-commit`
- **Great Expectations**: ใส่การตรวจใน `features.py` หลังโหลด raw (schema, ค่าว่าง, ช่วงค่า) ก่อน split
- **Prefect / ZenML**: แต่ละ stage มีฟังก์ชัน `run(cfg)` ที่เรียกได้โดยตรง (`dataset.run`, `features.run`, `train.train`)
  จึงห่อเป็น `@task` / `@step` ได้ทันที
- **FastAPI / Docker**: โหลดโมเดลด้วย `mlflow.sklearn.load_model("models:/<name>/latest")` – pipeline รวม
  preprocessing แล้ว รับ DataFrame ดิบได้เลย; class names อยู่ใน model metadata
- **Evidently**: เก็บ `data/processed/<name>/train.csv` เป็น reference dataset เทียบกับข้อมูลจริงใน production
- **MLflow server**: เปลี่ยน `mlflow.tracking_uri` ใน `configs/config.yaml` เป็น `http://<server>:5000` เพื่อใช้ร่วมกันเป็นทีม

## ลำดับที่แนะนำ

1. pre-commit + GitHub Actions (test + `dvc repro` บน PR) → Development L4
2. Great Expectations ใน stage featurize
3. FastAPI + Dockerfile → Deployment L4
4. Prefect flow ตั้งเวลา retrain → Operation L4
5. Evidently drift report + trigger → Operation L5
