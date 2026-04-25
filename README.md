# IoT Intrusion Detection System

A machine-learning pipeline for detecting cyberattacks in IoT network traffic, built on the **CICIoT2023** dataset.

Two detection pipelines are provided: a fine-grained **34-class classifier** (the main deployed system) and a coarse-grained **2/8-class classifier** used for comparison. The browser UI and Docker deployment serve the 34-class pipeline only.

---

## Pipelines

### Pipeline 1 — 34-Class Detection (`34 classes/`) — Main Deployed System
Fine-grained classification of network flows into **34 specific attack types or Benign**.  
Includes a **Detection Mode** for running predictions on CSV files and a **Monitoring Mode** for drift detection and real-time system health assessment.

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Random Forest | 73.12% | 0.622 |
| Logistic Regression | 68.84% | 0.549 |

### Pipeline 2 — 2/8-Class Detection (`2_8_CLASSES/`) — Comparison Only
Coarse-grained two-stage classification. Not deployed — used for performance comparison only.

1. **Binary stage** — Attack vs. Benign
2. **8-Class stage** — Attack family: DDoS, DoS, Mirai, Recon, BruteForce, Spoofing, Web, Benign

| Model | Binary Accuracy | 8-Class Accuracy |
|---|---|---|
| Random Forest | 99.14% | 82.35% |
| Logistic Regression | 97.72% | 67.76% |

---

## Quick Start (Docker — Recommended)

The fastest way to run the project. No Python installation required.

**Step 1 — Pull the image:**
```powershell
docker pull reuhdev/iot-ids:latest
```

**Step 2 — Run and open automatically (PowerShell):**
```powershell
docker run -d -p 8000:8000 reuhdev/iot-ids:latest; Start-Sleep 3; Start-Process "chrome" "http://127.0.0.1:8000/static/index.html"
```
This starts the container in the background and opens the UI in Chrome automatically.

**Or open manually after starting:**
```powershell
docker run -d -p 8000:8000 reuhdev/iot-ids:latest
```
Then open: http://127.0.0.1:8000/static/index.html

**Step 3 — Stop the container when done:**
```powershell
docker stop $(docker ps -q --filter ancestor=reuhdev/iot-ids:latest)
```

> Docker Hub: https://hub.docker.com/r/reuhdev/iot-ids  
> The Docker image uses Logistic Regression only. Random Forest is excluded due to file size (336 MB). A clear message is shown if RF is selected.

---

## How to Test the Model

### Step 1 — Get a CSV file

Demo CSV files are included in `34 classes/demo/`. Use any of the following:

| File | Description |
|---|---|
| `demo_stable.csv` | ~87% benign traffic with mild recon activity |
| `demo_warning.csv` | ~67% attack traffic (2 attack types) |
| `demo_critical.csv` | ~90% attack traffic (9 diverse attack types) |
| `Backdoor_Malware.pcap.csv` | Single-class attack sample — Backdoor Malware |
| `SqlInjection.pcap.csv` | Single-class attack sample — SQL Injection |

Each file contains network flow records with the 39 features the model expects. You can also use any CSV exported from the CICIoT2023 dataset.

---

### Step 2 — Open the UI

After starting the server (via Docker or locally), open:

```
http://127.0.0.1:8000/static/index.html
```

You will see two tabs at the top: **Detection Mode** and **Monitoring Mode**.

---

### Step 3 — Detection Mode (Run Predictions)

Detection Mode classifies each row in your CSV and shows the predicted attack type with a confidence score.

1. Click the **Detection Mode** tab
2. Select a model — **Logistic Regression** (available in Docker) or **Random Forest** (local only)
3. Drag and drop a CSV file onto the upload area, or click to browse
4. Click **Run Prediction**
5. Results appear below:
   - Each row is assigned a predicted class (e.g., `BENIGN`, `DDOS-SYN_FLOOD`, `SQLINJECTION`)
   - A confidence score (0–100%) is shown for each prediction
   - A class distribution chart shows the breakdown of all predicted classes

**Good files to test Detection Mode:** `Backdoor_Malware.pcap.csv`, `SqlInjection.pcap.csv`

---

### Step 4 — Monitoring Mode (System Health)

Monitoring Mode analyzes a batch of traffic and reports whether the system is operating normally or under attack.

1. Click the **Monitoring Mode** tab
2. Drag and drop a CSV file onto the upload area
3. The system will show one of three statuses:

| Status | Meaning |
|---|---|
| 🟢 Stable | Traffic is mostly benign — system is healthy |
| 🟡 Warning | Moderate attack presence detected |
| 🔴 Critical | High volume or diverse attack traffic detected |

4. A detailed breakdown shows attack type distribution and drift indicators

**Good files to test Monitoring Mode:** `demo_stable.csv`, `demo_warning.csv`, `demo_critical.csv`

---

## Running Locally (Without Docker)

### Requirements

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch (`torch`) is only needed for the 34-class Logistic Regression training script — not for running the API.

### 34-Class Pipeline (Main System)

```bash
cd "34 classes/deployment"
uvicorn api_34:app --reload
```

Open: http://127.0.0.1:8000/static/index.html

> When running locally, both **Random Forest** and **Logistic Regression** are available if the model files exist in `34 classes/models/trained/`.

### 2/8-Class Pipeline (Comparison)

```bash
cd 2_8_CLASSES/deployment
uvicorn api_2_8:app --reload
```

Open: http://127.0.0.1:8000

---

## Models

### Random Forest
- 200 estimators, max depth 20, `class_weight='balanced'`
- Best accuracy; large file size (~336 MB for 34-class, ~25 GB for 2/8-class multiclass)
- Not included in Docker or GitHub — retrain locally using scripts in `training/`

### Logistic Regression
- `solver='lbfgs'`, `max_iter=1000`, `class_weight='balanced'`
- Lightweight and fast; included directly in the repository
- 34-class version trained with a PyTorch wrapper for GPU acceleration

---

## Project Structure

```
Project/
├── 34 classes/
│   ├── data/
│   │   ├── metadata_34.json          # feature list, class names, split sizes
│   │   └── preprocess-34.py          # preprocessing script
│   ├── demo/
│   │   ├── demo_stable.csv           # ~87% benign — expected: Stable
│   │   ├── demo_warning.csv          # ~67% attack — expected: Warning
│   │   ├── demo_critical.csv         # ~90% attack — expected: Critical
│   │   ├── Backdoor_Malware.pcap.csv # single-class attack sample
│   │   └── SqlInjection.pcap.csv     # single-class attack sample
│   ├── deployment/
│   │   ├── api_34.py                 # FastAPI app (serves UI + predictions)
│   │   └── static/
│   │       └── index.html            # browser UI (Detection + Monitoring modes)
│   ├── models/
│   │   ├── preprocessing/
│   │   │   ├── scaler_34.pkl
│   │   │   └── label_encoder_34.pkl
│   │   └── trained/
│   │       ├── logistic_34.pkl       # included (16 KB)
│   │       └── rf_34.pkl             # NOT included — 336 MB, retrain locally
│   ├── results/
│   │   ├── logistic_34/              # confusion matrix, metrics, per-class report
│   │   └── rf_34/
│   └── training/
│       ├── train-34-logistic.py
│       └── train-34-rf.py
│
├── 2_8_CLASSES/
│   ├── data/
│   │   ├── metadata_2_8.json
│   │   └── preprocess_2_8.py
│   ├── deployment/
│   │   ├── api_2_8.py
│   │   └── static/
│   │       └── index.html
│   ├── models/
│   │   ├── preprocessing/
│   │   │   ├── scaler_2_8.pkl
│   │   │   └── label_encoder_2_8.pkl
│   │   └── trained/
│   │       ├── logreg_binary_2_8.pkl      # included (4 KB)
│   │       ├── logreg_multiclass_2_8.pkl  # included (4 KB)
│   │       ├── binary_rf_2_8.pkl          # NOT included — 724 MB
│   │       └── multiclass_rf_2_8.pkl      # NOT included — 25 GB
│   ├── results/
│   │   ├── logistic_2_8/
│   │   └── rf_2_8/
│   └── training/
│       ├── train_logistic_2_8.py
│       └── train_rf_2_8.py
│
├── Dockerfile
├── .dockerignore
├── main_api.py                        # Docker entry point
├── requirements.txt
└── requirements-deploy.txt
```

---

## Running the Full Pipeline Locally

Follow these steps in order to reproduce the full project from scratch.

### Prerequisites

```bash
pip install -r requirements.txt
```

Place the raw CICIoT2023 CSV files (`Merged01.csv` — `Merged06.csv`) in:
- `34 classes/data/raw/`
- `2_8_CLASSES/data/raw/`

---

### Pipeline 1 — 34-Class (Fine-Grained Detection)

**Step 1 — Preprocess the data**
```bash
cd "34 classes"
python data/preprocess-34.py
```
Outputs scaled parquet splits to `data/processed_data_34/` and saves `scaler_34.pkl`, `label_encoder_34.pkl` to `models/preprocessing/`.

**Step 2 — Train Logistic Regression**
```bash
python training/train-34-logistic.py
```
Saves `models/trained/logistic_34.pkl`.

**Step 3 — Train Random Forest** *(optional — 336 MB model, takes time)*
```bash
python training/train-34-rf.py
```
Saves `models/trained/rf_34.pkl`.

**Step 4 — Generate result images**
```bash
python results/generate_results_34.py
```
Saves confusion matrices, metrics tables, and per-class reports to `results/logistic_34/` and `results/rf_34/`.

**Step 5 — Run the API**
```bash
cd deployment
uvicorn api_34:app --reload
```
Open: http://127.0.0.1:8000/static/index.html

---

### Pipeline 2 — 2/8-Class (Coarse-Grained Comparison)

**Step 1 — Preprocess the data**
```bash
cd 2_8_CLASSES
python data/preprocess_2_8.py
```
Outputs scaled parquet splits to `data/processed_data_2_8/` and saves `scaler_2_8.pkl`, `label_encoder_2_8.pkl` to `models/preprocessing/`.

**Step 2 — Train Logistic Regression**
```bash
python training/train_logistic_2_8.py
```
Saves `models/trained/logreg_binary_2_8.pkl` and `models/trained/logreg_multiclass_2_8.pkl`.

**Step 3 — Train Random Forest** *(optional — very large models, requires significant RAM)*
```bash
python training/train_rf_2_8.py
```
Saves `models/trained/binary_rf_2_8.pkl` and `models/trained/multiclass_rf_2_8.pkl`.

**Step 4 — Generate result images**
```bash
python results/generate_results_2_8.py
```
Saves confusion matrices, metrics tables, and per-class reports to `results/logistic_2_8/` and `results/rf_2_8/`.

**Step 5 — Run the API** *(comparison/evaluation only)*
```bash
cd deployment
uvicorn api_2_8:app --reload
```
Open: http://127.0.0.1:8000

---

## Dataset

**CICIoT2023** — Canadian Institute for Cybersecurity  
34 attack categories across IoT network traffic scenarios.
