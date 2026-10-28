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

## เริ่มใช้งานแบบง่ายที่สุด (YOLO ทุกงาน)

รองรับทุกงานของ YOLO และ**ระบบดูจากไฟล์เองว่าเป็นงานอะไร** ไม่ต้องตั้งค่า:

| งาน | ใช้ทำอะไร | dataset หน้าตาแบบไหน | ลองด้วยตัวอย่าง |
|---|---|---|---|
| detect | ตีกรอบวัตถุ | `labels/*.txt` แถวละ 5 ค่า | `coco8` |
| segment | ระบายขอบวัตถุ (polygon) | `labels/*.txt` แถวเป็นจุด polygon | `coco8-seg` |
| pose | จุด keypoint (คน, มือ, สัตว์) | `data.yaml` มี `kpt_shape` | `coco8-pose` |
| obb | กรอบหมุนได้ (ภาพมุมสูง, เอกสาร) | `labels/*.txt` แถวละ 9 ค่า | `dota8` |
| classify | จำแนกทั้งรูป | โฟลเดอร์ละ class: `train/<class>/*.jpg` หรือ `<class>/*.jpg` | `mnist160` |

1. วางโฟลเดอร์ dataset (export แบบ YOLO จาก Roboflow / CVAT / Label Studio) ไว้ที่ `data\raw\<ชื่อ>\` เช่น `data\raw\helmet\`
2. **ดับเบิลคลิก `run.bat`** แล้วเลือกจากเมนู (ไม่ต้องแก้ ExecutionPolicy):

```
========== ML Training ==========
  1) Train YOLO (detect / segment / classify / pose / obb)  <- dataset > YOLO11/26 > ขนาด > quick/full
  2) Predict images / video with a trained model            <- ลาก/วางโฟลเดอร์รูปหรือวิดีโอได้เลย
  3) Compare trained YOLO models                            <- ตารางเทียบทุกขนาด + compare.xlsx
  4) Open MLflow UI
  5) Install / update (CPU or NVIDIA GPU)                   <- ทำข้อนี้ก่อนครั้งแรก
  ...
```

หรือพิมพ์คำสั่งสั้นใน PowerShell (ระบบเตรียม/แบ่งข้อมูลให้เองก่อนเทรน):

```powershell
.\run.ps1 fit helmet                   # yolo11n แบบเต็ม (Optuna + 100 epochs)
.\run.ps1 fit helmet m quick           # yolo11m แบบลองเร็ว (ไม่จูน, 30 epochs)
.\run.ps1 fit helmet all quick         # ครบทุกขนาด n,s,m,l,x
.\run.ps1 fit helmet 26s               # YOLO26 ขนาด s   (all26 = YOLO26 ครบทุกขนาด)
.\run.ps1 fit D:\datasets\leaves s     # dataset อยู่นอกโปรเจกต์ (คัดลอกเข้า data\raw ให้)
.\run.ps1 fit coco8-seg n quick        # ลองงาน segment ด้วยชุดตัวอย่าง (ดาวน์โหลดให้เอง)
.\run.ps1 text reviews                 # ข้อความ / เอกสาร (ภาษาไทยได้) ดูหัวข้อ "ข้อความ และเอกสาร"
.\run.ps1 vision flowers               # จัดหมวดภาพ ViT / ConvNeXt / EfficientNet หรือ segmentation ดูหัวข้อ "ภาพแบบอื่น"
.\run.ps1 table sales.csv target=sales  # ตาราง: จัดหมวด / ทายตัวเลข / พยากรณ์ ดูหัวข้อ "ข้อมูลตาราง"
.\run.ps1 resume helmet                # เทรนต่อจากที่ค้าง (ไฟดับ / Ctrl+C / เครื่องค้าง) · resume เฉยๆ = ดูรายการที่ค้าง
.\run.ps1 results helmet               # เทียบผลทุกขนาด
.\run.ps1 find helmet D:\images        # ทำนายด้วยโมเดลที่ดีที่สุด (หรือ ... D:\clip.mp4 m)
.\run.ps1 fit helmet s full accurate    # เลือกแบบการจูน (ดูตารางด้านล่าง)
.\run.ps1 clean helmet                 # ลบผลเทรนของ helmet (ถามยืนยันก่อน · เติม raw = ลบรูปด้วย)
.\run.ps1 clean all                    # ลบผลทั้งหมด + ประวัติ MLflow / Optuna
.\run.ps1 history                      # ประวัติทุกคำสั่งที่เคยรัน (+ logs\history.xlsx)
.\run.ps1 help                         # ดูคำสั่งทั้งหมด
```

### เทรนค้างกลางทาง → เทรนต่อได้

ทุกการเทรน YOLO จดสถานะไว้ใน `models\<dataset>\<model>\run_state.json` (คำสั่งที่ใช้ + กำลังทำขั้นไหน)
ถ้าไฟดับ กด Ctrl+C หรือเครื่องค้าง ให้รัน `.\run.ps1 resume helmet` (หรือเมนูข้อ 2 · เมนูจะเตือนสีเหลืองเองถ้ามีงานค้าง)
ระบบใช้การตั้งค่าเดิมทั้งหมด (preset, ขนาด, epochs ...) แล้วต่อจากจุดที่หยุด:

| หยุดตอน | ตอน resume |
|---|---|
| Optuna กำลังค้นหา | trial ที่เสร็จแล้วเก็บไว้ · trial ที่โดนตัดถูกทำเครื่องหมาย FAIL แล้วทำเฉพาะที่ยังขาด |
| เทรนรอบสุดท้าย | ต่อจาก epoch ล่าสุดใน `last.pt` (optimizer, learning rate, EMA เหมือนไม่เคยหยุด) |
| ประเมินผล / ทำรายงาน | ไม่เทรนซ้ำ ใช้ `best.pt` ที่ได้แล้ว ประเมินและทำ Excel ใหม่ |

ทุกครั้งที่ resume เป็น MLflow run ใหม่ (tag `resumed=True`) กราฟทุก epoch รวมช่วงก่อนหยุดด้วย
ถ้าเครื่องมีหลายโมเดลค้าง ให้ระบุขนาด เช่น `.\run.ps1 resume helmet s`

### เลือกแบบการจูน (tuning preset)

เลือกได้ในเมนู หรือพิมพ์ชื่อต่อท้ายคำสั่ง `fit` (ลำดับคำไม่สำคัญ) ไฟล์อยู่ใน `configs\tune\`

| preset | เหมาะกับ | สิ่งที่ทำ |
|---|---|---|
| `default` | ทั่วไป | ใช้ค่าในไฟล์โมเดล (`configs\model\yolo11*.yaml`) |
| `fast` | อยากเห็นผลเร็ว | 50 epochs, Optuna 10 รอบ จูนแค่ lr / momentum / weight decay |
| `balanced` | ใช้งานจริงทั่วไป | จูน optimizer + loss gain + augmentation หลัก, 30 รอบ |
| `accurate` | ต้องการแม่นสุด (เครื่อง GPU) | 300 epochs, cosine LR, จูนกว้างสุดรวมชนิด optimizer (SGD/AdamW), 50 รอบ |
| `small_data` | รูปไม่กี่ร้อยรูป | freeze backbone 10 ชั้น, augmentation แรง, patience 60 |
| `freeze` | วัตถุคล้าย COCO, อยากเร็ว | freeze backbone แล้วจูนแค่ learning rate |
| `augment` | โมเดลจำข้อมูลเกิน (overfit) | จูนเฉพาะ augmentation |
| `custom` | ปรับเองทุกอย่าง | แก้ `configs\tune\custom.yaml` ได้เลย มีทุกค่าที่ใช้บ่อยพร้อมคำอธิบาย |

- **ทำ preset ของตัวเองแบบง่าย**: เมนู *Tuning settings* (หรือ `.\run.ps1 tune`) → *Create my preset*
  เลือก preset ตั้งต้น แล้วตอบแค่ค่าที่ใช้บ่อย (กด Enter = ใช้ค่าเดิม): epochs, ขนาดภาพ, batch, patience, freeze,
  optimizer, device, จำนวนรอบ Optuna, epochs ต่อรอบ → ตั้งชื่อ เช่น `odb_gpu` → ขึ้นเป็นตัวเลือกตอนเทรนทันที
- ปรับละเอียด: เมนูเดียวกัน → *Edit a preset in Notepad* ปิด Notepad แล้วระบบตรวจไฟล์ให้ว่าใช้ได้
- ค่าที่พิมพ์ต่อท้ายชนะ preset เสมอ เช่น `.\run.ps1 fit helmet accurate hpo.n_trials=80 model.params.imgsz=960`
- โหมด `quick` ยังใช้ค่าคงที่ของ preset (เช่น freeze) แต่ข้ามการจูนและเทรน 30 epochs
- preset ที่ใช้ถูกบันทึกใน Excel (Summary) และ MLflow (param `tune`) เพื่อเทียบย้อนหลังได้

### ระบบเก็บประวัติอะไรบ้าง

ทุกการเทรน (ทั้ง YOLO และข้อมูลตาราง) เป็น 1 run ใน MLflow (`.\run.ps1 ui`) ที่เก็บ:

| หมวด | สิ่งที่เก็บ | ดูที่ไหนใน MLflow |
|---|---|---|
| ตั้งค่า | config Hydra ทั้งหมด, preset, hyperparameter ทุกตัว, ค่าที่ดีที่สุดจาก Optuna | Parameters, `config/` |
| ผลลัพธ์ | metrics ของ test, TP/FP/FN(/TN) ต่อ class, ทุก Optuna trial (nested run) | Metrics, `reports/` |
| ระหว่างเทรน | mAP / loss / learning rate **ทุก epoch** เป็นกราฟ | Metrics `epoch/...` → Chart |
| เครื่อง | CPU / RAM / ดิสก์ / GPU ระหว่างเทรน (ทุก ~10 วินาที) | System metrics |
| ข้อมูล | ลายนิ้วมือ sha256 ของรูป+label ที่ใช้จริง, รายชื่อไฟล์ train/val/test | tag `data.sha256`, `lineage/` |
| โค้ด | git commit, ไฟล์ที่แก้ยังไม่ commit (diff), คำสั่งที่รัน | tags `git.*`, `command`, `lineage/` |
| สภาพแวดล้อม | Python, เวอร์ชันไลบรารี (torch, ultralytics ...), ชื่อเครื่อง, GPU, `uv.lock` | `lineage/environment.json` |
| ไฟล์ | รายงาน Excel, กราฟ, โฟลเดอร์ผล Ultralytics ทั้งหมด, `best.pt` (ใน Model Registry) + `last.pt` | `reports/`, `ultralytics/`, `weights/` |
| การใช้งาน | ทุกครั้งที่ predict / find: input, weights (+ hash), จำนวนผลต่อ class, CSV | run ที่ tag `stage=predict` |
| XAI | ทุกครั้งที่ explain: ตัวเลขสรุป, Excel XAI, ภาพ heatmap ใน Gallery | run ที่ tag `stage=xai` |

นอกจากนี้ทุกคำสั่ง (prepare / featurize / train / predict / explain / clean) ถูกจดเป็น 1 บรรทัดใน `logs\history.jsonl`
ดูด้วย `.\run.ps1 history` (หรือเมนู *History*) ได้ตาราง + `logs\history.xlsx` · คำสั่ง `clean` ไม่ลบไฟล์นี้
และการลบเองก็ถูกจดไว้ด้วย · ปิดบางอย่างได้ใน `configs/config.yaml` หมวด `mlflow:`
(`system_metrics`, `log_weights`, `log_predictions`)

ต่อท้ายด้วยค่า Hydra ได้เสมอ เช่น `.\run.ps1 fit helmet n full model.params.epochs=200 model.params.batch=-1`
ถ้าระบบเดางานผิด บังคับได้ด้วย `data.task=segment` (หรือ detect / pose / obb / classify) · คำสั่งเดิม `detect` ยังใช้ได้
รายละเอียดเชิงลึกอยู่ในหัวข้อ 5

### XAI: โมเดลมองตรงไหน และทำไมถึงตัดสินแบบนั้น

```powershell
.\run.ps1 explain helmet            # โมเดลที่ดีที่สุดของ helmet, ภาพ test 100 ภาพ
.\run.ps1 explain helmet m 300      # ขนาด m, 300 ภาพ  (หรือเมนูข้อ 4 "Explain a model")
.\run.ps1 explain helmet xai.conf=0.4 xai.split=val   # ปรับค่าอื่นได้ (ดู configs/config.yaml หมวด xai:)
```

ทำได้ทุกงาน (detect / segment / pose / obb / classify) ผลอยู่ที่ `reports\<dataset>\<model>\xai\`

| วิธี | ตอบคำถาม |
|---|---|
| **EigenCAM** | ภาพรวม โมเดล "มอง" ตรงไหนของภาพ (ไม่ขึ้นกับ class) |
| **Grad-CAM รายกรอบ** | กรอบนี้ถูกตัดสินว่าเป็น class นี้ เพราะเห็นอะไร (จุดร้อน = หลักฐาน) |
| **attention lift / pointing game** | ความร้อนอยู่ในกรอบจริงไหม: >1 = มองที่วัตถุ · <1 = ใช้บริบทรอบๆ ตัดสิน |
| **จับคู่กับ label จริง** | ทุกกรอบเป็น TP หรือ FP แบบไหน (background / localization / class / duplicate) และทุกวัตถุที่พลาดเป็น FN แบบไหน (missed / wrong class / poor box) |

ชีตใน `xai_report.xlsx`:
**Summary** (ตัวเลขสำคัญพร้อม 95% CI และคำอธิบายภาษาไทย) ·
**Stats** (ทดสอบสถิติพร้อมคำตอบ เช่น ตอนทายถูกโมเดลมองที่วัตถุมากกว่าตอนทายผิดไหม, ขนาดวัตถุมีผลไหม ใช้ Mann-Whitney U, Spearman, Chi-square + Cramér's V, point-biserial, Kruskal-Wallis พร้อม p-value และ effect size) ·
**PerClass** (P/R/F1 + Wilson 95% CI, ชนิดความผิดพลาด, recall ตามขนาด, conf ที่ดีที่สุดต่อ class) ·
**Detections / Objects / Images** (ทุกกรอบ / ทุกวัตถุ / ทุกภาพ คลิกลิงก์เปิดภาพ heatmap ได้) ·
**Errors**, **Size**, **Calibration** (ECE / Brier: conf เชื่อได้แค่ไหน), **Threshold** (P/R/F1 ทุกค่า conf) ·
**Gallery** (ภาพที่ผิดมากที่สุดและดีที่สุด: กรอบ | EigenCAM | Grad-CAM) · **Glossary** (ความหมายทุกคำ)

classify จะได้ Confusion matrix, คู่ class ที่สับสนบ่อย, ภาพที่ผิดทั้งที่มั่นใจ และ Grad-CAM ของทั้ง class ที่ทายและ class ที่ถูก

### ข้อความ และเอกสาร (ภาษาไทยได้) – `.\run.ps1 text`

ติดตั้งครั้งแรก (หรือหลังอัปเดตนี้) ให้รัน `.\run.ps1 setup-gpu` (หรือ `setup-yolo` ถ้าไม่มี GPU) อีกครั้ง
เพื่อลง transformers / PyThaiNLP · ถ้าต้องอ่านเอกสารสแกนหรือรูปภาพ เพิ่ม `.\run.ps1 setup-ocr`

```powershell
.\run.ps1 text thai-sentiment tiny quick     # ลองระบบด้วยชุดตัวอย่าง (ไม่ต้องดาวน์โหลด)
.\run.ps1 text reviews                       # data\raw\reviews (ไฟล์หรือโฟลเดอร์) · โมเดล base · จูน Optuna 8 รอบ
.\run.ps1 text D:\data\invoices thai         # โฟลเดอร์เอกสารนอกโปรเจกต์ · โมเดลภาษาไทย
.\run.ps1 find reviews "ส่งไว ของดีมาก"         # ทำนายประโยคเดียว
.\run.ps1 find invoices D:\scans              # ทำนายทั้งโฟลเดอร์ (PDF / Word / รูป / txt) หรือไฟล์ CSV
```

ระบบดูไฟล์แล้วเลือกงานเอง (บังคับได้ด้วย `data.task=classify` หรือ `data.task=ner`):

| ข้อมูลที่วางใน `data\raw\<ชื่อ>\` | งาน | ตัวอย่างใช้งาน |
|---|---|---|
| CSV / Excel / JSONL ที่มีคอลัมน์ข้อความ + คอลัมน์ label (หาชื่อคอลัมน์ให้เอง) | **จัดหมวดข้อความ** | sentiment, intent แชตบอต, จัดหมวดเรื่องร้องเรียน |
| JSONL `{"tokens": [...], "tags": ["B-PER", "I-PER", "O", ...]}` หรือไฟล์ CoNLL | **NER – ดึงข้อมูลสำคัญ** | ชื่อ, บริษัท, วันที่, จำนวนเงิน จากใบแจ้งหนี้ / สัญญา |
| โฟลเดอร์ย่อยตาม class ข้างในเป็น .txt .pdf .docx หรือรูป (.jpg .png .tif) | **จัดหมวดเอกสาร** | แยกใบแจ้งหนี้ / ใบเสร็จ / สัญญา / ใบเสนอราคา |

มีคอลัมน์ `split` (train / val / test) หรือไฟล์ `train.csv` `val.csv` `test.csv` ระบบจะใช้ตามนั้น ไม่มีก็แบ่งให้ 70/15/15 แบบสมดุลต่อ class
PDF ใช้ข้อความในไฟล์ · PDF สแกนและรูปภาพใช้ OCR (EasyOCR ไทย+อังกฤษ) · ข้อความที่ดึงได้เก็บไว้ที่ `data\interim\<ชื่อ>\texts\` ไม่ต้อง OCR ซ้ำ

| ขนาด | โมเดล (HuggingFace) | เหมาะกับ |
|---|---|---|
| `tiny` | BERT เล็กเทรนจากศูนย์ (ไม่ต้องดาวน์โหลด) | ลองระบบเท่านั้น |
| `small` | distilbert-base-multilingual-cased | เร็ว ใช้ CPU ได้ |
| `base` (ค่าเริ่มต้น) | xlm-roberta-base | แม่นและรองรับไทยดี (แนะนำ GPU) |
| `thai` | WangchanBERTa | ข้อความภาษาไทยล้วน มักดีที่สุด |
| `large` | xlm-roberta-large | แม่นที่สุด ต้อง GPU 12 GB+ |

ใช้โมเดลอื่นจาก HuggingFace ได้ทุกตัว เช่น `.\run.ps1 text reviews base model.pretrained=microsoft/mdeberta-v3-base`
ได้ทุกอย่างเหมือน YOLO: Optuna จูน (lr, batch, weight decay, warmup), Excel (Summary, Confusion, PerClass = TP/FP/FN ต่อ class
หรือต่อชนิดข้อมูลสำหรับ NER, TrainingHistory, Predictions), MLflow + Model Registry, history, `resume` และ `results` เทียบโมเดล
ตัวอย่างในตัว: `thai-sentiment` (รีวิว 3 อารมณ์), `thai-ner` (ชื่อ บริษัท สถานที่ วันที่ เงิน), `thai-docs` (เอกสาร 4 ประเภท)

### ภาพแบบอื่นนอกจาก YOLO – `.\run.ps1 vision`

จัดหมวดภาพด้วยโมเดลมาตรฐาน (ResNet / EfficientNet / ConvNeXt / ViT จาก timm) และ **semantic segmentation**
(บอก class ทุกพิกเซล เช่น ถนน รอยร้าว แผล พื้นที่เพาะปลูก) ด้วย SegFormer · ติดตั้งครั้งแรกหลังอัปเดตนี้ให้รัน `.\run.ps1 setup-gpu` อีกครั้ง

```powershell
.\run.ps1 vision shapes                     # ลองระบบ: จัดหมวดรูปทรง (ไม่ต้องดาวน์โหลด)
.\run.ps1 vision shapes-seg                 # ลองระบบ: segmentation ทุกพิกเซล (ไม่ต้องดาวน์โหลด)
.\run.ps1 vision flowers                    # data\raw\flowers · efficientnet · จูน Optuna 8 รอบ
.\run.ps1 vision D:\data\leaves convnext quick
.\run.ps1 vision roads seg-b2               # โฟลเดอร์ images\ + masks\
.\run.ps1 find flowers D:\new-photos         # ทำนายทั้งโฟลเดอร์ -> predictions.csv (+ ภาพระบายสีสำหรับ segmentation)
```

| ข้อมูลใน `data\raw\<ชื่อ>\` | งาน |
|---|---|
| โฟลเดอร์ย่อยตาม class (`cat\*.jpg`, `dog\*.jpg`) จะอยู่ใน `train\ val\ test\` ด้วยก็ได้ | **จัดหมวดภาพ** |
| `images\` + `masks\` ชื่อไฟล์ตรงกัน (mask เป็น PNG ค่าพิกเซล = เลข class, 0 = พื้นหลัง) + `classes.txt` | **segmentation** |
| Roboflow export แบบ "Semantic Segmentation Masks" (`x.jpg` + `x_mask.png` + `_classes.csv`) | **segmentation** |

| ชื่อ | โมเดล | เหมาะกับ |
|---|---|---|
| `tiny` / `seg-tiny` | CNN / U-Net เล็กเทรนจากศูนย์ (ไม่ต้องดาวน์โหลด) | ลองระบบเท่านั้น |
| `efficientnet` (ค่าเริ่มต้น) | EfficientNet-B0 | เล็ก เร็ว แม่นดี |
| `resnet50` | ResNet-50 | มาตรฐาน ทนทาน |
| `convnext` | ConvNeXt-Tiny | มักแม่นที่สุดในขนาดนี้ |
| `vit` | ViT-B/16 | ข้อมูลเยอะ ต้อง GPU 8 GB+ |
| `seg-b0` (ค่าเริ่มต้นของ segmentation) | SegFormer-B0 | เร็ว ใช้ GPU ไหนก็ได้ |
| `seg-b2` | SegFormer-B2 | แม่นกว่า ต้อง GPU 8 GB+ |

ใช้โมเดลอื่นของ timm ได้ทุกตัว เช่น `.\run.ps1 vision flowers efficientnet model.arch=efficientnet_b3`
ได้ครบเหมือนงานอื่น: Optuna, Excel (classify: Confusion, PerClass, Predictions · segmentation: IoU / Dice ต่อ class, mIoU,
pixel accuracy, ภาพตัวอย่าง ภาพจริง | คำตอบ | ที่โมเดลทาย), MLflow Registry, history, `resume`, `results`

### ข้อมูลตาราง: จัดหมวด ทายตัวเลข พยากรณ์อนาคต – `.\run.ps1 table`

ไฟล์ CSV / Excel / Parquet ไฟล์เดียว บอกแค่คอลัมน์ที่ต้องการทาย ระบบเลือกงานเอง:
label → **จัดหมวด** · ตัวเลขหลายค่า (ราคา ยอดขาย อุณหภูมิ) → **regression** · ใส่คอลัมน์วันที่ → **พยากรณ์อนาคต (time series)**

```powershell
.\run.ps1 table house-prices quick                         # ลองระบบ: ทายราคาบ้าน (ไม่ต้องดาวน์โหลด)
.\run.ps1 table shop-sales all                             # ลองระบบ: พยากรณ์ยอดขายรายวัน 3 สาขา เทียบทุกโมเดล
.\run.ps1 find shop-sales 30                               # ยอดขาย 30 วันข้างหน้า -> predictions.csv
.\run.ps1 table D:\data\churn.xlsx target=Churn             # จัดหมวด (ลูกค้าจะเลิกใช้ไหม)
.\run.ps1 table sales.csv target=sales time=date group=shop # พยากรณ์ (sales.csv อยู่ใน data\external)
.\run.ps1 find churn D:\new_customers.csv                  # ทำนายไฟล์ใหม่
```

| โมเดล | ใช้กับ |
|---|---|
| `lightgbm` (ค่าเริ่มต้น) | เร็ว มักแม่นที่สุดกับข้อมูลตาราง |
| `xgboost` | แม่นมาก (`model.params.device=cuda` ใช้ GPU) |
| `random_forest` · `gradient_boosting` | ทนทาน ตั้งค่าน้อย |
| `logistic_regression` (ตัวเลข = Ridge) · `svm` (ตัวเลข = SVR) | เส้นตรง / ข้อมูลน้อย |
| `all` | เทรนทุกตัวแล้วเทียบใน `compare.xlsx` |

พยากรณ์อนาคต: ระบบสร้างข้อมูลย้อนหลังให้เอง (ค่าเมื่อ 1, 2, 3, 7, 14 งวดก่อน ค่าเฉลี่ยเคลื่อนที่ 7 / 28 งวด วันในสัปดาห์ เดือน ฯลฯ)
แบ่ง train / val / test ตามเวลา (test = ช่วงล่าสุด ไม่สุ่ม ไม่มีข้อมูลอนาคตรั่ว) · หลายสาขา / สินค้าในไฟล์เดียวใช้ `group=` ·
ปรับได้ด้วย `data.lags=[1,7,30]` `data.rolling=[7,30]` `data.horizon=60`

Excel ของงานตัวเลขมี Summary (R², MAE, RMSE, MAPE, MASE = เทียบกับการเดาว่า "เท่าเดิม" พร้อมคำอธิบายทุกค่า), ActualVsPredicted,
Residuals, Forecast (กราฟจริง vs โมเดลตามเวลา), FeatureImportance (คอลัมน์ไหนสำคัญ), Predictions, HPO_Trials

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

## 5. YOLO งานรูปภาพ (detect / segment / classify / pose / obb)

ใช้ระบบเดียวกัน (Hydra + Optuna + MLflow + DVC + Excel) กับงานรูปภาพทุกแบบของ Ultralytics YOLO
งานถูกอ่านจาก dataset (`data.task: auto`) แล้วระบบสลับ weights ให้เอง เช่น `model=yolo11n` กับ dataset segment
จะเทรน `yolo11n-seg.pt`, กับ dataset classify จะเทรน `yolo11n-cls.pt` (ขนาดภาพ 224) · มีทั้ง `yolo11n..x` และ `yolo26n..x`

### 5.1 ติดตั้ง (ครั้งเดียว)

```powershell
.\run.ps1 setup-gpu      # มีการ์ดจอ NVIDIA (แนะนำ – เร็วกว่า CPU หลายสิบเท่า)
.\run.ps1 setup-yolo     # ไม่มี GPU (ใช้ CPU)
```

### 5.2 วาง dataset

คัดลอกโฟลเดอร์ dataset ที่ label แล้ว (export แบบ **YOLO** จาก Roboflow / CVAT / Label Studio) ไปไว้ที่
`data\raw\<ชื่อ dataset>\` เช่น `data\raw\helmet\`

```
data/raw/helmet/
├── data.yaml            ← names + train/val/test (path แบบ ../train/images ของ Roboflow ใช้ได้)
├── train/images/*.jpg   train/labels/*.txt
├── valid/images/*.jpg   valid/labels/*.txt
└── test/images/*.jpg    test/labels/*.txt     (ถ้ามี)
```

หรือแบบง่ายสุด `images/` + `labels/` + `classes.txt` ระบบจะแบ่ง train/val/test ให้เอง (val 20%, test 10%)

- **segment / obb**: โครงสร้างเหมือนกันทุกอย่าง ต่างแค่เนื้อในไฟล์ label (polygon / 4 มุม)
- **pose**: ต้องมี `data.yaml` ที่มี `kpt_shape` (และ `flip_idx` ถ้ามี) – Roboflow export ให้อยู่แล้ว
- **classify**: ไม่มี labels ใช้โฟลเดอร์แทน `train/<class>/*.jpg` + `valid/<class>/...` หรือแค่ `<class>/*.jpg`
  (ระบบแบ่ง val/test ให้แบบรักษาสัดส่วนทุก class และใช้ hard link จึงไม่กินพื้นที่เพิ่ม)

### 5.3 เทรน

```powershell
uv run mlt-prepare   data=yolo data.name=helmet          # ตรวจไฟล์
uv run mlt-featurize data=yolo data.name=helmet          # แบ่ง split + นับกรอบต่อ class
uv run mlt-train     data=yolo data.name=helmet model=yolo11n
```

- Optuna จะลอง hyperparameter (lr0, momentum, augmentation, loss gain ฯลฯ) แบบสั้น `model.hpo_epochs` (10 epochs)
  จำนวน `hpo.n_trials` ครั้ง แล้วเทรนตัวจริงด้วยค่าที่ดีที่สุด `model.params.epochs` (100 epochs)
- ปรับได้ เช่น `hpo.n_trials=20 model.hpo_epochs=15 model.params.epochs=200 model.params.imgsz=960 model.params.batch=-1`
- ข้ามการจูน: `hpo=none`  · เลือกขนาด: `model=yolo11n|yolo11s|yolo11m|yolo11l|yolo11x` (n เร็วสุด, x แม่นสุด) · เทรนครบทุกขนาด: `.\run.ps1 fit helmet all` · ใช้ weights ของคุณ: `model.weights=models/pretrained/my.pt`
- ลองระบบก่อนด้วยตัวอย่าง: `coco8` (detect), `coco8-seg`, `coco8-pose`, `dota8` (obb), `mnist160` (classify)
  เช่น `.\run.ps1 fit dota8 n quick`
- หรือผ่าน DVC: แก้ `vars` ใน `pipelines/yolo/dvc.yaml` แล้ว `.\run.ps1 yolo`
  (เก็บเวอร์ชันรูปด้วย `uv run dvc add data/raw/helmet`)

ผลลัพธ์: `reports/<dataset>/<model>/report.xlsx`, `models/<dataset>/<model>.pt` (best weights),
โมเดลถูก register ใน MLflow ชื่อ `<dataset>-<model>`, ไฟล์ดิบของ Ultralytics อยู่ที่ `models/<dataset>/<model>/final/`

### 5.4 Metrics

| ค่า | ความหมาย |
|---|---|
| mAP50 / mAP50-95 | ค่ามาตรฐาน (บน test ถ้ามี ไม่งั้นใช้ val) – segment = mAP ของ mask, pose = mAP ของ keypoint, obb = กรอบหมุน (mAP ของกรอบปกติอยู่ใน `box_map50*`) |
| top1 / top5 | classify: สัดส่วนรูปที่ทายถูกอันดับ 1 / อยู่ใน 5 อันดับแรก |
| TP | กรอบที่ทายถูก class และซ้อนกับกรอบจริง IoU ≥ 0.45 ที่ confidence ≥ 0.25 |
| FP | กรอบที่ทายผิด class หรือทายในที่ที่ไม่มีวัตถุ (background) |
| FN | วัตถุจริงที่หาไม่เจอหรือทายเป็น class อื่น |
| TN | **ไม่มีใน detect/segment/pose/obb** (พื้นหลังมีได้ไม่จำกัด จึงนับไม่ได้) · **classify มี TN** (นับแบบ one-vs-rest ต่อ class) |

เปลี่ยนเกณฑ์ confidence ได้ด้วย `model.evaluation.conf=0.4` รายงาน Excel มีชีตเพิ่ม: **Dataset** (จำนวนกรอบต่อ class/split),
**TrainingHistory** (mAP และ loss ทุก epoch), **Plots** (confusion matrix, PR/F1 curve, results.png)

### 5.5 ทำนาย

```powershell
uv run mlt-predict data=yolo data.name=helmet model=yolo11n +input=D:/images        # โฟลเดอร์รูป
uv run mlt-predict data=yolo data.name=helmet model=yolo11n +input=D:/cam/clip.mp4 +conf=0.4
```

ได้รูป/วิดีโอที่วาดผลแล้ว + `detections.csv` ใน `reports/<dataset>/<model>/predict/` หนึ่งแถวต่อหนึ่งผลทำนาย:
detect = class, conf, x1, y1, x2, y2 · segment = + `polygon` · pose = + `keypoints` (x, y, conf) ·
obb = cx, cy, w, h, angle · classify = class, conf, `top5`

## 6. ใช้โมเดลทำนาย (ข้อมูลตาราง)

```powershell
uv run mlt-predict +input=data/external/new_rows.csv                          # ใช้ไฟล์ .joblib ล่าสุด
uv run mlt-predict +input=new.csv +model_uri=models:/breast_cancer-random_forest/latest   # จาก Model Registry
```

## 7. โครงสร้างโปรเจกต์ (Cookiecutter Data Science v2)

```
ml-training/
├── configs/                 Hydra configs
│   ├── config.yaml          ค่าหลัก (paths, mlflow, evaluation, report)
│   ├── data/                breast_cancer, iris, csv, openml, coco8, yolo
│   ├── model/               random_forest, logistic_regression, gradient_boosting, svm, lightning_mlp,
│   │                        yolo11n/s/m/l/x
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
│   ├── detection_data.py    dataset YOLO: ตรวจ, แบ่ง split, นับกรอบ
│   ├── tracking.py          MLflow helpers
│   ├── modeling/
│   │   ├── factory.py       model abstraction: Pipeline(preprocess -> model ใดก็ได้)
│   │   ├── lightning_models.py   PyTorch Lightning MLP แบบ sklearn API
│   │   ├── train.py         stage 3: tune -> train -> evaluate -> MLflow -> Excel
│   │   ├── yolo.py          backend YOLO (Ultralytics) + TP/FP/FN ต่อ class
│   │   └── predict.py       ทำนาย batch
│   └── reporting/           excel.py (XlsxWriter native charts), plots.py (PNG)
├── tests/                   pytest (unit + end-to-end)
├── dvc.yaml / dvc.lock      DVC pipeline (ข้อมูลตาราง)
├── pipelines/yolo/dvc.yaml  DVC pipeline (YOLO)
├── pyproject.toml / uv.lock dependencies (uv)
├── run.bat / run.ps1        เมนู + คำสั่งลัดสำหรับ Windows (Makefile สำหรับ Linux/macOS)
└── docs/ROADMAP.md          แผนไป Level 4 และ 5
```

## 8. หลักการที่ทำให้ได้ Level 3

- **Reproducible**: `uv.lock` ล็อก dependency, `seed` ใน config, DVC ล็อก hash ของข้อมูลและโมเดลใน `dvc.lock`
- **Config-driven**: ไม่มีค่า hard-code ทุกอย่างอยู่ใน YAML และ override ได้, config ของทุก run ถูกเก็บใน MLflow
- **ไม่มี data leakage**: preprocessing ถูก fit เฉพาะบน train และบันทึกไปกับโมเดล; Optuna ใช้ validation set,
  test set ใช้ครั้งเดียวตอนท้าย
- **Tracked**: ทุก trial เป็น nested run ใน MLflow, โมเดลสุดท้ายถูก register ใน Model Registry อัตโนมัติ
- **แยก notebooks ออกจาก src/** และทดสอบได้ด้วย `.\run.ps1 test`

## 9. ทำงานร่วมกับ Git / DVC remote

```powershell
git add . ; git commit -m "first experiment"
uv run dvc remote add -d storage D:\dvc-storage     # หรือ s3://, gdrive://, ssh://
uv run dvc push                                     # สำรองข้อมูล + โมเดล
uv run dvc metrics diff                             # เทียบ metrics กับ commit ก่อน
```

ดูแผนต่อไปที่ [docs/ROADMAP.md](docs/ROADMAP.md)
