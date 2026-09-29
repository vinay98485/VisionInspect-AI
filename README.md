# VisionInspect AI

## 1. Project Overview & Architecture

VisionInspect AI is an industrial manufacturing defect detection and quality inspection system. It combines deep learning computer vision, advanced anomaly detection, semantic segmentation, and automated decision-making to perform real-time quality control on the shop floor. 

The core architecture operates via a multi-stage AI pipeline:
1. **Category Classification**: ResNet18 identifies the product type (e.g., bottle, pill, cable).
2. **Anomaly Detection**: Deep feature embeddings isolate defective items from normal items using memory banks.
3. **Semantic Segmentation**: U-Net generates sub-pixel masks of the defect regions.
4. **Defect Classification**: Identifies the specific defect subtype (e.g., scratch, crack, contamination).
5. **Object Detection (Secondary)**: YOLO11n provides secondary bounding-box localization for discrete objects.
6. **Severity Scoring Engine**: A multi-factor engine calculating a 0-100 severity score based on Size (30%), Location (25%), Defect Type (25%), and AI Confidence (20%).
7. **Automated Quality Decision**: Determines Accept/Reject based on the severity level.

## 2. Key Features

- **Automated Quality Decision Engine**:
  - Low (0-39) → Accept
  - Medium (40-59) → Reject
  - High (60-79) → Reject
  - Critical (80-100) → Reject
- **Supervisor Review Workflow**: Every inspection automatically enters a Supervisor Review Queue. Supervisors can manually audit the AI's `quality_decision` and formally "Approve" or "Reject" the batch, providing full operational traceability.
- **Manufacturing Analytics**: Real-time dashboards plotting Accept vs. Reject trends, defect rates, and yield percentages.
- **Role-Based Access Control**:
  - **Quality Engineer**: Upload images, execute inspections, view analytics.
  - **Factory Supervisor**: Access review queue, override AI decisions, generate CSV production reports.

## 3. Technology Stack

- **Backend**: Python, FastAPI, SQLAlchemy, PostgreSQL
- **AI/ML**: PyTorch, Ultralytics YOLO, ResNet18, U-Net, OpenCV, Scikit-Learn
- **Frontend**: React 19, Vite, Tailwind CSS

## 4. Repository Structure

```
VisionInspect_AI/
├── ai/
│   ├── evaluation/            # Validation scripts, regression suites, benchmarks
│   ├── weights/               # Ultralytics YOLO baseline weights
│   └── models/                # PyTorch architectures, scorers, pipelines, configs
├── backend/
│   ├── app/                   # FastAPI application, database schemas, APIs
│   └── tests/                 # Backend unit and integration tests
├── docs/                      # Extensive project documentation and UML diagrams
└── frontend/                  # React + Vite application
```

## 5. Quickstart & Installation

### 1. Database Setup
```bash
createdb visioninspect_db
export DATABASE_URL="postgresql://<username>:<password>@localhost:5432/visioninspect_db"
```

### 2. Backend Setup
```bash
cd backend
python -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 6. Testing & Deployment (Milestone 4 Status)
The project is currently in the **Milestone 4: Final Testing & Documentation** phase. 
Extensive automated unit testing, API validation, and UI audits have been successfully completed. 
The system is ready for containerized deployment (e.g., Docker).

**Known Limitations:**
- YOLO11n object detection struggles with subtle pixel-level anomalies (e.g., color shifts, thin cracks). U-Net segmentation serves as the primary ground truth for the severity engine, while YOLO acts as an optional overlay.

To run verification suites:
```bash
# Run backend API tests
PYTHONPATH=backend:. ./backend/myvenv/bin/python -m unittest discover -s backend/tests

# Run AI regression suite
PYTHONPATH=. ./backend/myvenv/bin/python ai/evaluation/test_regression_inspection.py
```
