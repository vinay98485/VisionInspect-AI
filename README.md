# VisionInspect AI

An industrial manufacturing defect detection and quality inspection system. It combines deep-learning computer vision, anomaly detection, semantic segmentation, and automated decision-making to perform real-time quality control on the shop floor.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [System Architecture](#3-system-architecture)
4. [AI Inspection Pipeline](#4-ai-inspection-pipeline)
5. [ML Models and Their Roles](#5-ml-models-and-their-roles)
6. [Severity Scoring and Quality Decisions](#6-severity-scoring-and-quality-decisions)
7. [Backend API Overview](#7-backend-api-overview)
8. [Database Overview](#8-database-overview)
9. [Frontend Architecture](#9-frontend-architecture)
10. [Docker Architecture](#10-docker-architecture)
11. [Required Model Artifacts](#11-required-model-artifacts)
12. [Required Calibration Files](#12-required-calibration-files)
13. [Environment Variables](#13-environment-variables)
14. [Docker Quickstart](#14-docker-quickstart)
15. [Application URLs and Ports](#15-application-urls-and-ports)
16. [Authentication Flow](#16-authentication-flow)
17. [Basic Inspection Workflow](#17-basic-inspection-workflow)
18. [Stopping the Stack](#18-stopping-the-stack)
19. [Troubleshooting](#19-troubleshooting)
20. [End-to-End Validation Results](#20-end-to-end-validation-results)
21. [Known Limitations](#21-known-limitations)
22. [Repository Structure](#22-repository-structure)
23. [Running Tests (Non-Docker)](#23-running-tests-non-docker)

---

## 1. Project Overview

VisionInspect AI is a web-based industrial QC platform designed for factory floor deployment. Quality Engineers upload product images through a React frontend. The FastAPI backend coordinates a multi-stage AI pipeline that detects anomalies, segments defect regions, scores severity, and issues an automated Accept/Reject decision. All results are persisted in PostgreSQL and entered into a Supervisor Review Queue for human audit.

The entire application stack (frontend, backend, database) ships as a single Docker Compose environment.

---

## 2. Key Features

- **Automated AI Inspection**: Full end-to-end defect detection, segmentation, classification, and scoring.
- **Severity Engine**: Multi-factor 0–100 severity score (Size 30%, Location 25%, Defect Type 25%, AI Confidence 20%).
- **Quality Decision Engine**: Automated Accept/Reject based on severity thresholds.
- **Supervisor Review Queue**: Every inspection enters a review queue. Supervisors can audit and override the AI decision.
- **Manufacturing Analytics**: Real-time dashboards for Accept/Reject trends, defect rates, and yield.
- **Role-Based Access Control**: Quality Engineer (upload, inspect, view) and Factory Supervisor (review queue, override, reports).
- **Visual Overlay**: Defect contour and bounding-box overlay image generated for each inspection.
- **Dockerized Stack**: One-command deployment via Docker Compose.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Docker Compose                   │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │
│  │   Frontend   │  │   Backend    │  │   DB     │  │
│  │  React+Nginx │  │   FastAPI    │  │ Postgres │  │
│  │  port 80     │  │  port 8000   │  │ port5432 │  │
│  └──────┬───────┘  └──────┬───────┘  └────┬─────┘  │
│         │                 │               │        │
│         └──── HTTP ───────┘               │        │
│                           └── SQLAlchemy ─┘        │
│                                                     │
│  Backend container also mounts:                     │
│    ./ai/models/*.pt  (read-only model weights)      │
│    ./ai/models/*.csv (calibration files)            │
│    uploads_volume    (uploaded images)              │
└─────────────────────────────────────────────────────┘
```

---

## 4. AI Inspection Pipeline

The `InspectionPipeline` orchestrates seven sequential stages:

| Stage | Component | Description |
|---|---|---|
| 1 | Category Classifier (ResNet18) | Identifies product category (e.g., bottle, pill) |
| 2 | Anomaly Detector (Deep Feature Memory Bank) | Compares feature embeddings to normal references; produces an anomaly score |
| 3 | U-Net Segmenter | Generates a pixel-level mask of the defective region |
| 4 | Defect Classifier (Hierarchical ResNet18) | Identifies the specific defect subtype |
| 5 | YOLO11n Object Detector | Secondary bounding-box localization of discrete defect objects |
| 6 | Severity Engine | Computes weighted multi-factor severity score and level |
| 7 | Decision Engine | Issues Accept or Reject quality decision with recommendation |

---

## 5. ML Models and Their Roles

### Required (MUST exist on host for startup)
| File | Role |
|---|---|
| `ai/models/normal_features_layer3.pt` | Normal-class feature memory bank for anomaly detection (728 MB) |
| `ai/models/defect_segmenter_unet.pt` | U-Net segmentation model (119 MB) |

### Optional (comment out mounts in `docker-compose.yml` if not present)
| File | Role |
|---|---|
| `ai/models/category_classifier_resnet18.pt` | ResNet18 category classifier |
| `ai/models/defect_classifier_hierarchical.pt` | Hierarchical defect-type classifier |
| `ai/weights/yolo/baseline/weights/best.pt` | YOLO11n object detection weights |

> **Warning**: If you uncomment an optional volume mount in `docker-compose.yml` but the file is not present on the host, Docker creates an **empty directory** in place of the file. This causes `torch.load()` to crash at startup. Only uncomment a mount when the file physically exists.

---

## 6. Severity Scoring and Quality Decisions

### Severity Score Formula

```
severity_score = (size_score × 0.30) +
                 (location_score × 0.25) +
                 (defect_type_score × 0.25) +
                 (confidence_score × 0.20)
```

All sub-scores are 0–100. The final score is clamped to [0, 100].

### Severity Levels and Quality Decisions

| Severity Score | Level    | Quality Decision |
|----------------|----------|-----------------|
| 0 – 39         | Low      | Accept           |
| 40 – 59        | Medium   | Reject           |
| 60 – 79        | High     | Reject           |
| 80 – 100       | Critical | Reject           |

### Recommendation Logic

Recommendations are derived from the `(defect_type, quality_decision, severity_level)` combination and provide actionable guidance (e.g., "SCRAP", "REWORK", "PASS").

---

## 7. Backend API Overview

Base URL: `http://localhost:8000`

### Authentication
| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Register a new user (JSON body: `name`, `email`, `password`, `role_id`, optionally `supervisor_registration_code`) |
| POST | `/auth/login` | Log in; returns `access_token` (JSON body: `email`, `password`) |
| GET | `/auth/me` | Get current user profile (requires Bearer token) |

### Inspections
| Method | Path | Description |
|---|---|---|
| POST | `/images/upload` | Upload image file (multipart `file` field; requires Role 1 token) |
| POST | `/images/{image_id}/inspect` | Trigger AI inspection on an uploaded image (requires Role 1 token) |
| GET | `/images/` | List all images (Role 1 or 2) |
| GET | `/images/{image_id}` | Get full inspection record (Role 1 or 2) |
| GET | `/images/{image_id}/file` | Download original image file |
| GET | `/images/{image_id}/inspection-overlay` | Download defect overlay PNG |
| POST | `/images/{image_id}/review` | Submit supervisor review decision (Role 2) |
| GET | `/images/supervisor/review-queue` | Get all images for review (Role 2) |

### Analytics
| Method | Path | Description |
|---|---|---|
| GET | `/analytics/...` | Various dashboard aggregation endpoints |

### Roles
| role_id | Role Name |
|---|---|
| 1 | Quality Engineer |
| 2 | Factory Supervisor |

---

## 8. Database Overview

PostgreSQL 14. Three tables:

| Table | Description |
|---|---|
| `roles` | `id`, `role` — seeded automatically on startup (`1 = Quality Engineer`, `2 = Factory Supervisor`) |
| `users` | `id`, `name`, `email`, `password_hash`, `role_id`, `is_active` |
| `images` | Full inspection record per uploaded image including all AI scores, severity, quality decision, and supervisor review fields |

The database is initialized (tables created, roles seeded) automatically on backend startup via `python -m app.database.init_db`.

---

## 9. Frontend Architecture

- **Framework**: React 19 + Vite
- **Styling**: Tailwind CSS
- **API**: `frontend/src/services/api.js` — reads `VITE_API_BASE` environment variable (injected at Docker build time via `--build-arg`); falls back to `http://127.0.0.1:8000`.
- **Routing**: React Router (handled by Nginx `try_files` fallback for SPA).
- **Auth**: JWT stored client-side; all API requests include `Authorization: Bearer <token>`.
- **Production server**: Nginx (not `npm run dev`). The Vite dev server is used only for local development.

---

## 10. Docker Architecture

Three services in `docker-compose.yml`:

| Service | Image | Port | Notes |
|---|---|---|---|
| `db` | `postgres:14-alpine` | 5432 | Named volume `postgres_data` |
| `backend` | `visioninspect_ai-backend` | 8000 | Built from `backend/Dockerfile`. Mounts model weights (read-only) and `uploads_volume`. |
| `frontend` | `visioninspect_ai-frontend` | 80 | Multi-stage build: Node 20 builder → Nginx Alpine |

**Backend Dockerfile highlights**:
- Base: `python:3.11-slim`
- System deps: `libgl1`, `libglib2.0-0` (required by OpenCV)
- Dependencies: installed from `backend/requirements.txt` using the official PyTorch CPU index (`https://download.pytorch.org/whl/cpu`) to avoid downloading unused CUDA libraries (saves ~1.6 GB)
- BuildKit pip cache (`--mount=type=cache,target=/root/.cache/pip`) enables faster rebuilds
- PyTorch version: **2.14.0+cpu** (CPU-only; `torch.cuda.is_available() = False`)
- Final image size: **~1.24 GB**

---

## 11. Required Model Artifacts

The following model files **must exist on the host** before starting the Docker stack. They are never committed to the repository (GitHub 100 MB file limit) and must be obtained separately.

| Host Path | Size | Description |
|---|---|---|
| `./ai/models/normal_features_layer3.pt` | 728 MB | Normal feature memory bank |
| `./ai/models/defect_segmenter_unet.pt` | 119 MB | U-Net defect segmenter |

These are mounted read-only into the backend container:
```yaml
- ./ai/models/normal_features_layer3.pt:/app/ai/models/normal_features_layer3.pt:ro
- ./ai/models/defect_segmenter_unet.pt:/app/ai/models/defect_segmenter_unet.pt:ro
```

---

## 12. Required Calibration Files

These small configuration files are committed to the repository and automatically included in the Docker image via `COPY ai/ ./ai/`:

| File | Description |
|---|---|
| `ai/models/anomaly_thresholds.csv` | Per-category anomaly detection thresholds |
| `ai/models/size_score_boundaries.csv` | Defect area → size score mapping |
| `ai/models/segmentation_threshold.txt` | U-Net binary mask threshold |
| `ai/models/segmentation_postprocessing.json` | Morphological post-processing config |

---

## 13. Environment Variables

Copy `.env.example` to `.env` and fill in values before starting the stack.

| Variable | Default (in compose) | Description |
|---|---|---|
| `POSTGRES_USER` | `visioninspect` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `visioninspect` | PostgreSQL password |
| `POSTGRES_DB` | `visioninspect_db` | PostgreSQL database name |
| `DATABASE_URL` | `postgresql+psycopg2://visioninspect:visioninspect@db:5432/visioninspect_db` | Full SQLAlchemy connection URL. **Must use `postgresql+psycopg2://` prefix** (not bare `postgresql://`) to correctly select the installed `psycopg2-binary` driver. |
| `SUPERVISOR_REGISTRATION_CODE` | `default_secret_code` | Required code for registering Supervisor accounts |
| `FRONTEND_ORIGIN` | `http://localhost` | Allowed CORS origin for the backend |
| `VITE_API_BASE` | `http://localhost:8000` | Backend API URL baked into the frontend at build time |

> **Security**: Never commit your actual `.env` file. It is listed in `.gitignore`.

---

## 14. Docker Quickstart

### Prerequisites
- Docker Desktop ≥ 4.x (includes Docker Compose V2)
- At least 10 GB free disk space (base images + model weights)
- Model artifacts present at `./ai/models/` (see [Section 11](#11-required-model-artifacts))

### Steps

**1. Clone the repository:**
```bash
git clone <repo-url>
cd VisionInspect_AI
```

**2. Configure environment:**
```bash
cp .env.example .env
# Edit .env and set a strong POSTGRES_PASSWORD and SUPERVISOR_REGISTRATION_CODE
```

**3. Verify required model files exist:**
```bash
ls -lh ai/models/normal_features_layer3.pt ai/models/defect_segmenter_unet.pt
```

**4. Build and start the stack:**
```bash
docker compose up -d
```
The first build will take **3–5 minutes** as it downloads the Python base image and installs all PyPI packages including PyTorch (~200 MB CPU wheel).

**5. Monitor backend startup** (model loading takes 30–60 seconds):
```bash
docker compose logs -f backend
```
Look for: `INFO: Application startup complete.`

**6. Verify all services are running:**
```bash
docker compose ps
```
Expected:
```
NAME                           STATUS
visioninspect_ai-backend-1     Up (healthy)
visioninspect_ai-db-1          Up (healthy)
visioninspect_ai-frontend-1    Up
```

**7. Verify health:**
```bash
curl http://localhost:8000/
# → {"message":"VisionInspect AI Backend Running"}

curl -I http://localhost
# → HTTP/1.1 200 OK (nginx)
```

---

## 15. Application URLs and Ports

| Service | URL |
|---|---|
| Frontend (React UI) | http://localhost |
| Backend API | http://localhost:8000 |
| API Interactive Docs | http://localhost:8000/docs |
| PostgreSQL | localhost:5432 |

---

## 16. Authentication Flow

1. **Register** a Quality Engineer account:
```bash
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"name":"Test Engineer","email":"engineer@example.com","password":"SecurePass123!","role_id":1}'
```

2. **Log in** to obtain a JWT:
```bash
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"engineer@example.com","password":"SecurePass123!"}'
# Returns: {"access_token": "<JWT>", ...}
```

3. Use the token for subsequent requests:
```bash
curl http://localhost:8000/auth/me \
  -H "Authorization: Bearer <JWT>"
```

**Password requirements**: ≥ 8 characters, at least one uppercase, one lowercase, one digit, one special character.

**Supervisor registration** additionally requires the `supervisor_registration_code` field to match the value set in `.env`.

---

## 17. Basic Inspection Workflow

```
1. Register / Login  →  get JWT

2. Upload image
   POST /images/upload
   -F "file=@/path/to/image.png"
   → returns image_id

3. Trigger inspection
   POST /images/{image_id}/inspect
   → returns full inspection result JSON

4. Retrieve stored record
   GET /images/{image_id}

5. Download overlay
   GET /images/{image_id}/inspection-overlay
```

**Minimal end-to-end example (curl):**
```bash
TOKEN="<JWT from login>"

# 1. Upload
RESP=$(curl -s -X POST http://localhost:8000/images/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@mvtec_anomaly_detection/bottle/test/good/000.png")
IMAGE_ID=$(echo $RESP | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

# 2. Inspect
curl -s -X POST http://localhost:8000/images/$IMAGE_ID/inspect \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

---

## 18. Stopping the Stack

```bash
# Stop containers (preserves volumes / database data)
docker compose down

# Stop and remove all volumes including PostgreSQL data
docker compose down -v
```

> **Warning**: `docker compose down -v` permanently deletes the PostgreSQL database volume. All user accounts and inspection records will be lost.

---

## 19. Troubleshooting

### Backend crashes immediately with `ModuleNotFoundError: No module named 'psycopg'`
**Cause**: `DATABASE_URL` uses `postgresql://` prefix instead of `postgresql+psycopg2://`.  
**Fix**: Ensure `.env` contains `DATABASE_URL=postgresql+psycopg2://...`.

### Backend crashes with `ModuleNotFoundError: No module named 'ultralytics'`
**Cause**: Stale Docker image built before `ultralytics` was added to `requirements.txt`.  
**Fix**: `docker compose build --no-cache backend`

### Backend crashes with `IsADirectoryError` on a model file
**Cause**: An optional volume mount in `docker-compose.yml` is uncommented, but the `.pt` file does not exist on the host. Docker creates an empty directory instead.  
**Fix**: Comment out or remove the volume mount for the missing optional model file.

### Build context transfer is very slow (multiple GB)
**Cause**: A Python virtual environment (`backend/myvenv/`) was not excluded from the Docker context.  
**Fix**: The `.dockerignore` already excludes `**/myvenv` and `**/venv`. Ensure your local virtual environment is inside one of these directories.

### Backend image is unexpectedly large (>6 GB)
**Cause**: Installing PyTorch from the default PyPI registry pulls in `nvidia-*` CUDA packages even on CPU-only builds.  
**Fix**: The `backend/Dockerfile` explicitly uses `--extra-index-url https://download.pytorch.org/whl/cpu` to install the CPU wheel. Do not remove this flag. Verified image size: **~1.24 GB**.

### `docker compose up` fails because Docker daemon is not running
**Fix**: Start Docker Desktop, wait for it to reach the "running" state, then retry.

---

## 20. End-to-End Validation Results

The complete Docker Compose stack was validated on **Apple M2 (arm64 / aarch64)** using MVTec Anomaly Detection dataset images.

**Validated PyTorch environment:**
- `torch.__version__`: `2.14.0+cpu`
- `torch.version.cuda`: `None`
- `torch.cuda.is_available()`: `False`
- NVIDIA packages installed: None

---

### Test Case 1 — Normal Specimen

| Field | Value |
|---|---|
| Image | `mvtec_anomaly_detection/bottle/test/good/000.png` |
| Category | `bottle` |
| Action | Upload → Inspect |
| Inspection Decision | **NORMAL** |
| Severity Level | **Low** |
| Quality Decision | **Accept** |
| Recommendation | **PASS** |
| Detected Objects (YOLO) | 0 |
| DB Persistence | ✅ Confirmed |

---

### Test Case 2 — Defective Specimen

| Field | Value |
|---|---|
| Image | `mvtec_anomaly_detection/bottle/test/broken_large/000.png` |
| Category | `bottle` |
| Defect Type | `broken_large` |
| Action | Upload → Inspect |
| Inspection Decision | **DEFECTIVE** |
| Severity Level | **Critical** |
| Quality Decision | **Reject** |
| Recommendation | **SCRAP** |
| Detected Objects (YOLO) | 5 |
| DB Persistence | ✅ Confirmed |

---

### Validated Services

| Service | Status |
|---|---|
| PostgreSQL 14 | ✅ Healthy |
| FastAPI Backend | ✅ Healthy (`/` returns 200) |
| React/Nginx Frontend | ✅ Running (HTTP 200 on port 80) |
| Backend→DB connection | ✅ Verified |
| Model loading (`normal_features_layer3.pt`) | ✅ Verified |
| Model loading (`defect_segmenter_unet.pt`) | ✅ Verified |
| Calibration file loading | ✅ Verified |
| Defect overlay generation | ✅ Verified |
| Backend restart / persistence | ✅ Verified |

---

## 21. Known Limitations

- **YOLO object detection**: YOLO11n struggles with subtle pixel-level anomalies (color shifts, thin cracks). U-Net segmentation is the primary ground truth for the severity engine; YOLO provides secondary bounding-box overlays.
- **MVTec dataset dependency**: Training scripts and calibration pipelines are written for the MVTec Anomaly Detection dataset. The runtime container does **not** require the MVTec dataset.
- **CPU inference only**: The Docker deployment uses CPU inference. Model loading (especially `normal_features_layer3.pt` at 728 MB) takes 30–60 seconds on first startup.
- **No cloud deployment**: This project has been validated for local Docker deployment only. Cloud or Kubernetes deployment is outside the current scope.

---

## 22. Repository Structure

```
VisionInspect_AI/
├── ai/
│   ├── dataset/           # MVTec data loaders and YOLO dataset prep scripts
│   ├── evaluation/        # Validation scripts, regression suites, benchmarks
│   ├── models/            # PyTorch model architectures, pipelines, calibration files
│   └── weights/           # YOLO baseline weights (gitignored)
├── backend/
│   ├── app/
│   │   ├── database/      # SQLAlchemy connection, init, migration scripts
│   │   ├── models/        # SQLAlchemy ORM models
│   │   ├── routers/       # FastAPI route handlers (auth, images, analytics)
│   │   ├── schemas/       # Pydantic request/response schemas
│   │   ├── security/      # JWT, password hashing, RBAC authorization
│   │   └── services/      # Business logic (image service, auth service, recommendations)
│   ├── tests/             # Backend unit and integration tests
│   ├── Dockerfile         # Backend Docker image definition
│   └── requirements.txt   # Python dependencies
├── docs/                  # Project documentation and UML diagrams
├── frontend/
│   ├── src/
│   │   ├── components/    # React UI components
│   │   ├── pages/         # Route-level page components
│   │   └── services/      # API client (api.js)
│   ├── Dockerfile         # Frontend multi-stage Docker build
│   └── nginx.conf         # Nginx config for React Router SPA fallback
├── .dockerignore          # Docker build context exclusions
├── .env.example           # Environment variable template
├── .gitignore             # Git exclusions
└── docker-compose.yml     # Full stack Docker Compose definition
```

---

## 23. Running Tests (Non-Docker)

For local development without Docker:

```bash
# Backend setup
cd backend
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt

# Initialize DB (requires a local PostgreSQL instance)
export DATABASE_URL="postgresql+psycopg2://<user>:<pass>@localhost:5432/visioninspect_db"
python -m app.database.init_db

# Start backend dev server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Run backend tests
PYTHONPATH=backend:. python -m unittest discover -s backend/tests

# Run AI regression suite
PYTHONPATH=. python ai/evaluation/test_regression_inspection.py
```

```bash
# Frontend setup
cd frontend
npm install
npm run dev   # Dev server at http://localhost:5173
```
