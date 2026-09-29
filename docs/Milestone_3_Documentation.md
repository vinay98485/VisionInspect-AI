# VisionInspect AI — Milestone 3 Technical Documentation
## Defect Classification, Severity Scoring & Manufacturing Analytics

---

**Project Title:** VisionInspect AI — Manufacturing Defect Detection & Quality Inspection System  
**Author:** Vinay Kumar Mandalapu  
**Milestone:** 3 — Defect Classification & Manufacturing Analytics (Weeks 5 & 6)  
**Date:** September 2026  
**Version:** 3.0  
**Repository Baseline:** Production-Stabilized, Validated on MVTec Anomaly Detection Benchmark  

---

## Table of Contents

1. [Title & Executive Summary](#1-title--executive-summary)
2. [Milestone 3 Scope](#2-milestone-3-scope)
3. [Official Milestone 3 Requirements](#3-official-milestone-3-requirements)
4. [Implementation Summary](#4-implementation-summary)
5. [Defect Categorization Workflows](#5-defect-categorization-workflows)
6. [Severity Scoring & Quality Risk Assessments](#6-severity-scoring--quality-risk-assessments)
7. [Quality Assessment Modules](#7-quality-assessment-modules)
8. [Supervisor Review Workflow](#8-supervisor-review-workflow)
9. [Production Quality Reports](#9-production-quality-reports)
10. [Defect Analytics Dashboard](#10-defect-analytics-dashboard)
11. [Trend Monitoring Workflows](#11-trend-monitoring-workflows)
12. [Quality Recommendations (Downstream Guidance)](#12-quality-recommendations-downstream-guidance)
13. [Operational Insights](#13-operational-insights)
14. [End-to-End Manufacturing Quality Workflow](#14-end-to-end-manufacturing-quality-workflow)
15. [API & Backend Implementation](#15-api--backend-implementation)
16. [Frontend Implementation](#16-frontend-implementation)
17. [Database Design & Persistence](#17-database-design--persistence)
18. [Verification & Testing](#18-verification--testing)
19. [Requirement-to-Implementation Traceability Matrix](#19-requirement-to-implementation-traceability-matrix)
20. [Milestone 3 Completion Summary](#20-milestone-3-completion-summary)
21. [Milestone 4 Handover](#21-milestone-4-handover)

---

## 1. Title & Executive Summary

VisionInspect AI is an industrial-grade computer vision platform engineered to automate quality inspection across discrete manufacturing assembly lines. Building directly upon the foundational infrastructure of Milestone 1 (authentication, RBAC, file storage) and the core AI inspection engine of Milestone 2 (category classification, patch anomaly detection, U-Net localization, and YOLO11n object detection), **Milestone 3** delivers the operational intelligence layer of the system: **Defect Classification, Severity Scoring & Manufacturing Analytics**.

In manufacturing production lines, detecting a defect is only the initial step; quality managers require actionable intelligence:
- What exact defect subtype occurred (e.g., crack, contamination, bent wire, hole)?
- What is the quantitative risk to structural integrity and product function?
- Should the part be accepted, reworked, cleaned, or scrapped?
- What are the line-level yield trends over 7-day, 30-day, and all-time operating horizons?
- How can factory supervisors review AI decisions and export official audit documentation?

Milestone 3 implements complete end-to-end solutions for these requirements: category-conditioned hierarchical defect classification, an authenticated 4-factor Severity Scoring engine, automated Accept/Reject quality decisioning, a dedicated Factory Supervisor audit workflow, real-time analytics dashboards powered by Recharts, multi-horizon trend monitoring, exportable CSV Production Quality Reports, and downstream advisory rework guidance.

---

## 2. Milestone 3 Scope

The development scope for Milestone 3 corresponds to Weeks 5 and 6 of the official project curriculum:

- **Defect Categorization Workflows**: Linking autonomous category recognition with hierarchical defect classification heads to resolve specific defect subtypes across all 15 industrial categories.
- **Severity Scoring & Risk Assessment**: Implementing the official 4-factor weighted scoring formula combining defect physical size (30%), spatial location (25%), defect hazard type (25%), and detection confidence (20%).
- **Automated Quality Assessment**: Determining automated pass/fail quality decisions based on calibrated severity thresholds and one-class anomaly verification.
- **Factory Supervisor Governance**: Providing a protected review queue where authorized supervisors audit inspection records, sign off with approval or rejection, and record mandatory inspection notes.
- **Production Quality Reports**: Engineering an automated report generation service that compiles executive summaries, pass yields, and granular inspection records into downloadable CSV reports.
- **Defect Analytics & Trend Monitoring**: Delivering interactive dashboard visualizations tracking total inspections, defect rates, average severity, category distributions, defect Pareto breakdowns, and time-series trends over 7-day, 30-day, and all-time horizons.
- **Downstream Quality Guidance**: Providing deterministic, advisory rework recommendations (`PASS`, `CLEAN`, `REWORK`, `SCRAP`) to optimize plant floor disposition.

---

## 3. Official Milestone 3 Requirements

The official project specification (*"AI_Manufacturing Defect Detection & Quality Inspection System"*) defines the formal requirements for Milestone 3 (Weeks 5 & 6) under the **Defect Classification & Manufacturing Analytics** phase:

### Official High-Level Requirements
1. **Implement defect categorization workflows**: Classify detected flaws into granular defect categories conditioned on product type.
2. **Generate severity scoring reports and quality risk assessments**: Compute multi-parameter severity ratings and categorize defect risk into standardized severity bands.
3. **Build quality assessment modules**: Automate pass/fail quality decisions and enforce quality inspection standards.
4. **Generate production quality reports**: Provide exportable quality documentation summarizing production health, defect rates, and inspection histories.
5. **Build defect analytics dashboards**: Present real-time operational telemetry, defect distributions, and inspection statistics.
6. **Develop trend monitoring workflows**: Track quality metrics and defect frequency over multi-day operational horizons.

### Subsystem Module Requirements (Modules 5, 6, and 7)
- **Module 5: Defect Classification Module**: Defect type classification, severity scoring, defect categorization, confidence analysis.
- **Module 6: Quality Control Module**: Pass/fail decision generation, inspection reporting, quality recommendations, production monitoring.
- **Module 7: Manufacturing Analytics Dashboard Module**: Defect trend analysis, production quality reports, inspection statistics, operational insights.

---

## 4. Implementation Summary

The Milestone 3 architecture bridges AI inference outputs with factory operations:

```
                           Raw Inspection Results
                 (Category, Defect Subtype, Mask, Bounding Boxes)
                                     │
                                     ▼
                    Defect Categorization & Scoring Engine
             ┌───────────────────────┴───────────────────────┐
             ▼                                               ▼
     Physical Measurement                           Risk Evaluation
     • Defect Area % (U-Net)                        • Defect Type Score (Lookup)
     • Size Score (Percentile Map)                  • Centroid Distance (Location)
     • YOLO Object Count & Boxes                    • Anomaly Margin Confidence
             │                                               │
             └───────────────────────┬───────────────────────┘
                                     ▼
                        Multi-Factor Severity Formula
         Severity = 0.30×Size + 0.25×Location + 0.25×Type + 0.20×Confidence
                                     │
                                     ▼
                         Automated Quality Decision
                   • Severity >= 60.0 & Defective ──► REJECT
                   • Severity < 60.0 or Normal   ──► ACCEPT
                                     │
                                     ▼
                    Advisory Quality Recommendation
                      • PASS / CLEAN / REWORK / SCRAP
                                     │
                                     ▼
                    PostgreSQL Persistence (28 Columns)
                                     │
       ┌─────────────────────────────┼─────────────────────────────┐
       ▼                             ▼                             ▼
Factory Supervisor Queue       Analytics Aggregation        Report Export Service
• Review & Audit Trail         • Total Volume & Yield       • Executive Summary
• Mandatory Review Notes       • Category & Defect Pareto   • Granular Audit Table
• Sign-off (Approve/Reject)    • 7d / 30d / All Trends      • Downloadable CSV
```

---

## 5. Defect Categorization Workflows

### 5.1 Hierarchical Defect Subtype Classification
In manufacturing inspection, knowing that an item is defective is insufficient; line operators must know whether a cable suffers from `cut_inner_insulation`, `bent_wire`, or `missing_cable`.

Implemented in [`ai/models/defect_classifier.py`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/ai/models/defect_classifier.py):
- **Backbone**: Deep ResNet18 feature extractor.
- **Hierarchical Head Architecture**: 15 distinct linear output layers, each specialized for the defect taxonomy of a specific MVTec category.
- **Category Conditioning**: When `CategoryClassifier` detects category $C$, only classification head $H_C$ is invoked.
- **Subtype Output**: Returns the specific defect label string alongside softmax classification confidence.

### 5.2 Decision Fusion Reconciliation
The raw classifier prediction is reconciled against the patch-based Anomaly Detector via [`ai/models/decision_fusion.py`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/ai/models/decision_fusion.py):
- If the classifier predicts a defect but the anomaly score is below threshold $T_c$, the sample is assigned to **Quadrant D** and resolved to `"normal"`, preventing false alarms.
- If the anomaly score exceeds $T_c$ but the classifier predicted nominal, the sample is assigned to **Quadrant B** and categorized as `"unclassified_anomaly"`, ensuring novel anomalies are captured.
- When both corroborate a defect, the specimen is assigned to **Quadrant C** with the verified defect subtype.

---

## 6. Severity Scoring & Quality Risk Assessments

### 6.1 Official 4-Factor Weighted Formula
VisionInspect AI implements the official severity formulation defined in the project specification:

$$\text{Severity Score} = (0.30 \times \text{Size}) + (0.25 \times \text{Location}) + (0.25 \times \text{Defect Type}) + (0.20 \times \text{Confidence})$$

The mathematical formulation is implemented in [`ai/models/severity_scorer.py`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/ai/models/severity_scorer.py).

### 6.2 Component Scoring Methodologies

#### 1. Defect Size Score (30% Weight)
- **Module**: [`ai/models/size_scorer.py`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/ai/models/size_scorer.py)
- **Input**: Defect area percentage from U-Net segmentation:
  $$\text{Area } (\%) = \left(\frac{\text{Defective Pixels}}{224 \times 224}\right) \times 100\%$$
- **Calibration**: Industrial materials exhibit radically different nominal flaw dimensions (e.g., a $1.5\%$ area crack on a `pill` is massive, while $1.5\%$ on a `carpet` is small). Size scores are calibrated using category-specific empirical percentiles ($p_{10}$ to $p_{95}$) recorded in `ai/models/size_score_boundaries.csv`.
- **Mapping**: Piecewise linear interpolation mapping physical area into a standardized $[0, 100]$ score.

#### 2. Defect Location Score (25% Weight)
- **Module**: [`ai/models/location_scorer.py`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/ai/models/location_scorer.py)
- **Centrality Factor (70%)**: Flaws near the center of a component typically compromise functional load-bearing paths or aesthetic focal areas:
  $$\text{Centroid Factor} = \left(1.0 - \frac{d(\text{Centroid}, \text{Center})}{d_{\text{max}}}\right) \times 100$$
- **Area Factor (30%)**: Normalized defect coverage contributing to spatial impact.
- **Combined Location Score**: Weighted combination yielding $[0, 100]$.

#### 3. Defect Type Score (25% Weight)
- **Module**: [`ai/models/defect_type_scorer.py`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/ai/models/defect_type_scorer.py)
- **Hazard Lookup Table**: Flaws are assigned static risk scores based on structural hazard:
  - Critical structural fractures (`broken_large`, `crack`, `hole`): **90–95 points**
  - Moderate functional flaws (`bent_wire`, `cut_inner_insulation`, `thread`): **75–85 points**
  - Minor cosmetic flaws (`scratch`, `contamination`, `color`): **40–60 points**
  - `unclassified_anomaly`: **70.0 points**
  - Confirmed `normal` / `good`: **0.0 points**

#### 4. Detection Confidence (20% Weight)
- **Module**: [`ai/models/confidence.py`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/ai/models/confidence.py)
- **Anomaly Margin Interpolation**: Computes model confidence based on the distance between the test anomaly score and category operating boundaries:
  - If score $\le \text{normal\_boundary}$: $0\%$ defect confidence.
  - If score $\ge \text{defect\_boundary}$: $100\%$ defect confidence.
  - Intermediate scores interpolate linearly.

### 6.3 Standardized Severity Levels
Severity scores map to four standardized industrial risk tiers:

| Severity Level | Score Range | Industrial Meaning | Action Required |
| :--- | :---: | :--- | :--- |
| **Critical** | **80.00 – 100.00** | Major structural defect or complete part fracture | Immediate rejection; quarantine lot |
| **High** | **60.00 – 79.99** | Significant quality deviation exceeding tolerance | Product rejection; route for repair/rework |
| **Medium** | **40.00 – 59.99** | Moderate quality concern near tolerance boundary | Inspection review required by supervisor |
| **Low** | **0.00 – 39.99** | Minor cosmetic defect within allowable limits | Product generally acceptable |

### 6.4 Note on Specification Arithmetic Example
In the official specification document (*"AI_Manufacturing Defect Detection & Quality Inspection System"*), an illustrative example is provided on pages 5–6:
- Size Score: 85
- Location Score: 90
- Defect Type Score: 95
- Confidence Score: 92
- Stated Calculated Severity Score in text: 88

Applying the official 4-factor formula yields:
$$\text{Severity} = (85 \times 0.30) + (90 \times 0.25) + (95 \times 0.25) + (92 \times 0.20) = 25.50 + 22.50 + 23.75 + 18.40 = 90.15$$
Both $90.15$ and the illustrative $88$ fall into the **Critical** band ($80\text{--}100$). VisionInspect AI implements the strict, exact mathematical formulation directly in [`ai/models/severity_scorer.py`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/ai/models/severity_scorer.py), guaranteeing reproducible floating-point calculations across all inspections.

---

## 7. Quality Assessment Modules

### 7.1 Automated Pass/Fail Decisioning
The automated quality decision is rendered by `SeverityScorer.get_quality_decision()`:
- **`Reject`**: Assigned when a specimen is classified as `DEFECTIVE` and its computed `Severity Score` $\ge 60.00$ (High or Critical severity tier).
- **`Accept`**: Assigned when a specimen is confirmed `NORMAL` (Decision Fusion Quadrants A or D) or when minor cosmetic flaws produce a `Severity Score` $< 40.00$ (Low severity tier within manufacturing tolerance).

### 7.2 Strict Invariant for Normal Specimens
When Decision Fusion resolves a sample as `NORMAL`:
- `quality_decision` = `"Accept"`
- `severity_score` = `0.00`
- `severity_level` = `"Low"`
- `predicted_area_percent` = `0.00%`
- `size_score` = `0.00`, `location_score` = `0.00`, `defect_type_score` = `0.00`, `confidence_score` = `0.00`
- `detected_objects_count` = `0`, `bounding_boxes` = `[]`

This invariant eliminates false scrap on verified nominal production runs.

---

## 8. Supervisor Review Workflow

While AI inspection provides automated decisions, ISO 9001 and industrial quality standards require human-in-the-loop oversight for rejected lots and edge cases.

### 8.1 Role Separation (RBAC)
- **Quality Engineer (Role ID 1)**: Submits images, triggers AI inspection, views inspection results and telemetry.
- **Factory Supervisor (Role ID 2)**: Authorized to access the supervisory review queue, audit AI decisions, record binding approvals or rejections, and export compliance reports. Quality Engineers attempting supervisory actions receive `403 Forbidden`.

### 8.2 Review Workflow Lifecycle
1. **Inspection Completion**: Newly inspected images default to `supervisor_decision = NULL` (`pending` review state).
2. **Review Queue Ingestion**: Supervisors retrieve pending records via `GET /images/supervisor/review-queue`.
3. **Audit Inspection**: The supervisor inspects the high-resolution original image alongside the dual U-Net/YOLO overlay and metric breakdown.
4. **Binding Sign-Off**: The supervisor submits `POST /images/{id}/review` with:
   - `decision`: `"approved"` (overriding or confirming Accept) or `"rejected"` (confirming Reject).
   - `notes`: Required audit rationale (e.g., *"Verified critical fracture on bottle wall. Production lot quarantined."*).
5. **Audit Trail Persistence**: The system records `supervisor_decision`, `supervisor_notes`, `reviewed_by` (supervisor user ID), and `reviewed_at` (UTC timestamp) in PostgreSQL.

---

## 9. Production Quality Reports

Implemented in [`backend/app/services/report_service.py`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/backend/app/services/report_service.py), the report generation service compiles real-time production quality data into standardized, downloadable CSV reports via `GET /analytics/reports/export?format=csv`.

### 9.1 Report Structure

#### Section 1: Executive Summary Block
```csv
PRODUCTION QUALITY REPORT SUMMARY
Metric,Value
Report Generation Timestamp,2026-09-18 10:20:00 UTC
Total Inspections,248
Accepted Count,182
Rejected Count,66
Pass/Yield Percentage,73.39%
Defect Rate,26.61%
Average Severity Score,31.42
Average Confidence,88.45%
Pending Supervisor Reviews,14
Approved Supervisor Reviews,175
Rejected Supervisor Reviews,59
```

#### Section 2: Granular Inspection Audit Table
Comprises 18 detailed columns per inspection record:
`Inspection ID`, `Timestamp`, `Original Filename`, `Category`, `Defect Type`, `Classification Confidence`, `Anomaly Score`, `Predicted Area Percentage`, `Size Score`, `Location Score`, `Defect Type Score`, `Severity Score`, `Severity Level`, `AI Quality Decision`, `Supervisor Decision`, `Supervisor Notes`, `Reviewer ID`, `Review Timestamp`.

This format allows immediate ingestion into enterprise ERP/MES platforms (e.g., SAP, Rockwell FactoryTalk) and spreadsheet analysis.

---

## 10. Defect Analytics Dashboard

The manufacturing analytics backend is implemented in [`backend/app/routers/analytics.py`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/backend/app/routers/analytics.py) and surfaced through [`frontend/src/pages/SupervisorDashboard.jsx`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/frontend/src/pages/SupervisorDashboard.jsx).

### 10.1 Key Metrics Aggregated (`GET /analytics/summary`)
- **Total Inspections**: Count of all executed inspections.
- **Accepted / Rejected**: Total counts and operational defect yield rate.
- **Pending / Reviewed**: Audit governance compliance metrics.
- **Severity Tier Breakdown**: Distribution across Critical, High, Medium, and Low tiers.
- **Averages**: Line-wide average anomaly detection confidence and average severity score.
- **Category Pareto**: Inspection counts grouped by product category.
- **Defect Subtype Pareto**: Flaw frequency grouped by defect classification.
- **Supervisor Decisions**: Count of supervisor approvals vs. rejections.

### 10.2 Interactive Frontend Visualizations
Using `recharts`, the dashboard renders:
1. **KPI Metric Cards**: Real-time totals with status badges and percentages.
2. **Severity Distribution Pie Chart**: Visual breakdown of Critical, High, Medium, and Low severity proportions.
3. **Defect Type Horizontal Bar Chart**: Pareto ranking of the most frequent manufacturing defect types across lines.
4. **Category Breakdown Bar Chart**: Inspection volume and flaw rates categorized by product line.

---

## 11. Trend Monitoring Workflows

Implemented via `GET /analytics/trends?days=7|30|all`, trend monitoring aggregates daily production statistics over variable horizons.

### 11.1 Dynamic Query Parameters
- `days=7`: 7-day trailing operational window.
- `days=30`: 30-day trailing monthly quality review window (default).
- `days=all`: Complete historical production archive from initial platform deployment.

### 11.2 Daily Metric Aggregations
For each calendar day in the horizon, the service computes:
- `date`: ISO date string (`YYYY-MM-DD`).
- `total_inspections`: Total parts processed.
- `accepted`: Parts accepted by AI.
- `rejected`: Parts rejected by AI.
- `defect_rate`: Percentage of inspected parts rejected:
  $$\text{Defect Rate } (\%) = \left(\frac{\text{Rejected}}{\text{Total}}\right) \times 100\%$$
- `avg_severity`: Mean severity score of inspected parts.

### 11.3 Interactive Trend Visualization
The Factory Supervisor console renders a dual-axis interactive line/bar chart displaying daily defect rates alongside inspection volume. When quality deviations or tool wear cause sudden defect spikes, supervisors can isolate the exact shift and date of occurrence.

---

## 12. Quality Recommendations (Downstream Guidance)

Implemented in [`backend/app/services/recommendation_service.py`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/backend/app/services/recommendation_service.py), the recommendation engine translates complex AI telemetry into immediate, deterministic shop-floor disposition guidance:

| Recommendation Action | Action Label | Trigger Conditions | Operational Plant Guidance |
| :--- | :--- | :--- | :--- |
| **`PASS`** | Pass to Production | AI Decision = `Accept` or defect type is `normal`/`good` | No corrective action required. Part meets quality standards; release to production packaging. |
| **`CLEAN`** | Clean & Re-inspect | Flaw types containing `contamination`, `stain`, `glue`, `color` | Surface residue detected without irreversible structural damage. Clean using standard wash protocol and re-inspect. |
| **`REWORK`** | Route for Rework | Flaw types containing `bent_wire`, `cut_outer_insulation`, `scratch`, `rough`, `deformation` | Non-critical surface or geometric deviation capable of corrective repair. Route to rework station. |
| **`SCRAP`** | Quarantine for Scrap | Severity tier = `Critical` or flaws containing `crack`, `broken_large`, `hole`, `cut_inner_insulation` | Irreversible structural defect exceeding allowable repair limits. Quarantine component for scrap disposition. |

### Explicit Advisory Nature
Quality recommendations are downstream, deterministic operational aids. They provide immediate guidance to assembly operators but **do not override** the AI's core quality decision (`Accept`/`Reject`) or the Factory Supervisor's manual sign-off authority.

---

## 13. Operational Insights

By unifying classification, localization, severity scoring, and analytics, VisionInspect AI generates actionable operational insights for manufacturing engineers:
1. **Tool Wear & Machine Health**: Gradual upward trends in average severity scores on rigid materials (`screw`, `metal_nut`, `grid`) indicate tool wear or mold degradation before catastrophic tool breakage occurs.
2. **Material Batch Variance**: Spikes in surface contamination or stains (`carpet`, `leather`, `wood`) correlate with raw material batch inconsistencies from upstream suppliers.
3. **Yield Optimization**: Categorizing flaws into `CLEAN` and `REWORK` prevents unnecessary scrap, directly reducing manufacturing scrap costs.

---

## 14. End-to-End Manufacturing Quality Workflow

The complete end-to-end quality inspection workflow operates across four sequential phases:

```
Phase 1: Ingestion & Verification (Quality Engineer)
  1. Quality Engineer logs in to VisionInspect AI web application.
  2. Submits product photograph via drag-and-drop or batch upload.
  3. System validates MIME type, image dimensions, file size (<=5MB), and computes SHA-256 hash.

Phase 2: Automated AI Inspection
  4. Inspection pipeline automatically classifies product category (e.g. "bottle").
  5. Extracts Layer 3 patch embeddings and computes distance against normal memory bank.
  6. Predicts defect subtype and classification confidence using category-specific head.
  7. Decision Fusion arbiter evaluates operating quadrant (A, B, C, or D).
  8. If Defective: executes U-Net segmentation (area %), severity calculation, and YOLO11n object detector (bounding boxes, object count).
  9. Synthesizes dual visual overlay (red contours + green bounding boxes).
  10. Generates advisory recommendation (PASS, CLEAN, REWORK, SCRAP).
  11. Persists 28 columns to PostgreSQL database.

Phase 3: Supervisory Governance (Factory Supervisor)
  12. Factory Supervisor views pending inspections in the review queue.
  13. Reviews original image, dual overlay, and telemetry cards.
  14. Submits official approval or rejection with mandatory audit notes.
  15. Audit trail records supervisor ID and timestamp.

Phase 4: Plant-Wide Analytics & Quality Reporting
  16. Analytics engine aggregates defect rates, severity distributions, and category breakdowns.
  17. Supervisor monitors 7-day and 30-day quality trends to detect defect anomalies.
  18. Exports official Production Quality Report in CSV format for executive review.
```

---

## 15. API & Backend Implementation

Milestone 3 adds dedicated endpoints across authentication, inspection, supervisor governance, and analytics:

| Method | Endpoint Path | Access Level | Description |
| :--- | :--- | :---: | :--- |
| `POST` | `/images/{id}/inspect` | Quality Engineer (1) | Executes AI inspection pipeline, computes severity, triggers YOLO if defective |
| `GET` | `/images/{id}/inspection-overlay` | Authenticated (1, 2) | Streams dynamically rendered composite PNG with red U-Net contours and green YOLO boxes |
| `GET` | `/images/supervisor/review-queue` | Supervisor (2) | Lists completed inspections awaiting supervisory audit |
| `POST` | `/images/{id}/review` | Supervisor (2) | Records supervisory approval/rejection with mandatory notes and timestamp |
| `GET` | `/analytics/summary` | Authenticated (1, 2) | Aggregates plant-wide totals, defect rates, severity tiers, and category breakdowns |
| `GET` | `/analytics/trends` | Supervisor (2) | Computes daily time-series inspection metrics over `7`, `30`, or `all` days |
| `GET` | `/analytics/reports/export` | Supervisor (2) | Generates and streams downloadable CSV Production Quality Report |

---

## 16. Frontend Implementation

### 16.1 Inspection Details Studio ([`frontend/src/pages/ImageDetails.jsx`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/frontend/src/pages/ImageDetails.jsx))
- **Dual Visual Viewer**: Side-by-side comparative inspection showing original raw photograph alongside server-rendered composite overlay.
- **Detected Defect Objects**: Displays discrete object count with an active `YOLO11n` badge.
- **Severity Rating Card**: Displays numerical score ($0.00\text{--}100.00$) and color-coded badge (`Critical`, `High`, `Medium`, `Low`).
- **Advisory Recommendation Card**: Displays action badge (`PASS`, `CLEAN`, `REWORK`, `SCRAP`), technical rationale, and operational guidance.
- **Supervisor Sign-Off Panel**: Allows authorized supervisors to record approvals/rejections directly on the specimen page.

### 16.2 Factory Supervisor Console ([`frontend/src/pages/SupervisorDashboard.jsx`](file:///Users/vinaykumarmandalapu/Desktop/VisionInspect_AI/frontend/src/pages/SupervisorDashboard.jsx))
- **Executive Metric Strip**: Real-time KPI cards for total inspections, pass yield, defect rate, average severity, and pending reviews.
- **Interactive Recharts Visualizations**:
  - Severity distribution pie chart.
  - Defect subtype Pareto bar chart.
  - Category breakdown bar chart.
  - Daily inspection defect rate and volume trend chart.
- **Horizon Filter**: Switch dynamically between 7-day, 30-day, and all-time trend views.
- **Export Quality Report Button**: Triggers direct browser download of the structured CSV quality report.
- **Review Queue Modal**: Review interface with thumbnail preview, full telemetry, decision selector, and notes field.

---

## 17. Database Design & Persistence

The PostgreSQL `images` relational table has been expanded to 28 columns supporting complete AI and manufacturing traceability:

| Column Name | SQL Type | Description |
| :--- | :--- | :--- |
| `id` | `Integer (PK)` | Auto-incrementing primary key |
| `filename` | `String(255)` | Stored filename on disk |
| `original_filename` | `String(255)` | Original client upload filename |
| `file_size` | `Integer` | Image file size in bytes |
| `mime_type` | `String(50)` | Validated MIME type |
| `sha256_hash` | `String(64)` | Content hash for integrity & deduplication |
| `uploaded_at` | `DateTime` | Upload timestamp (UTC) |
| `uploaded_by` | `Integer (FK)` | Reference to user who uploaded image |
| `inspection_status` | `String(50)` | Lifecycle state (`pending`, `completed`, `reviewed`) |
| `category` | `String(100)` | Predicted or validated manufacturing category |
| `defect_type` | `String(100)` | Resolved defect subtype string |
| `classification_confidence` | `Float` | Category classification confidence percentage |
| `anomaly_score` | `Float` | Patch-based ResNet18 Layer 3 anomaly distance |
| `confidence_score` | `Float` | Anomaly margin confidence percentage |
| `predicted_area_percent` | `Float` | Physical defect area percentage from U-Net mask |
| `size_score` | `Float` | Percentile-mapped size severity score (0–100) |
| `location_score` | `Float` | Centroid-based spatial location severity score (0–100) |
| `defect_type_score` | `Float` | Hazard weight of defect subtype (0–95) |
| `severity_score` | `Float` | Official 4-factor combined severity score (0–100) |
| `severity_level` | `String(50)` | Standardized risk band (`Critical`, `High`, `Medium`, `Low`) |
| `quality_decision` | `String(50)` | Automated AI quality decision (`Accept`, `Reject`) |
| `recommendation_action` | `String(50)` | Advisory recommendation action (`PASS`, `CLEAN`, `REWORK`, `SCRAP`) |
| `recommendation_label` | `String(100)` | Human-readable recommendation label |
| `supervisor_decision` | `String(50)` | Supervisory sign-off decision (`approved`, `rejected`) |
| `supervisor_notes` | `Text` | Mandatory supervisory review audit notes |
| `reviewed_by` | `Integer (FK)` | Reference to supervisor user who signed off |
| `reviewed_at` | `DateTime` | Supervisory sign-off timestamp (UTC) |
| `detected_objects_count` | `Integer` | Number of discrete defect bounding boxes detected by YOLO11n |
| `bounding_boxes` | `JSON` | Serialized JSON array of bounding box coordinates & confidences |

---

## 18. Verification & Testing

Milestone 3 implementation was verified through an exhaustive test battery:

### 18.1 End-to-End API & Database Lifecycle Test
Executed in `scratch/test_e2e_yolo_api.py` against live PostgreSQL:
1. **User Authentication**: Registered and authenticated Quality Engineer (`qe_yolo_...`).
2. **Defective Specimen Lifecycle**:
   - Uploaded `bottle/test/broken_large/000.png` (ID: 248).
   - Executed inspection: `quality_decision = Reject`, `severity_score = 81.68` (Critical), `detected_objects_count = 5`.
   - Verified PostgreSQL database persistence: all 28 fields verified including JSON bounding boxes.
   - Retrieved composite overlay PNG (697,006 bytes).
3. **Normal Specimen Lifecycle**:
   - Uploaded `bottle/test/good/000.png` (ID: 249).
   - Executed inspection: `quality_decision = Accept`, `severity_score = 0.00` (Low), `detected_objects_count = 0`, `bounding_boxes = []`.
   - Verified PostgreSQL database persistence: verified zero-defect normal invariant.
4. **Serializer Exposure**: Confirmed `GET /images/{id}` returns all fields cleanly.

### 18.2 Comprehensive Test Suite (`backend/tests/test_milestone2_comprehensive.py`)
- Verified all 31 test methods covering authentication, RBAC, uploads, normal invariants, defective scoring formulas, overlay generation, database persistence, supervisor reviews, and analytics summary.

### 18.3 Client Build Verification
- Executed `npm run build` in `frontend/`: 606 modules transformed, cleanly compiled in 112ms with 0 errors.

---

## 19. Requirement-to-Implementation Traceability Matrix

| Official Requirement | Current Implementation | Implementing Code / File | Verification Evidence | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Defect Categorization Workflows** | Category Classifier (15 classes) + Hierarchical Defect Classifier (15 heads) | `ai/models/category_classifier.py`, `ai/models/defect_classifier.py` | 100% category test accuracy; regression tests passing | **DONE** |
| **Severity Scoring & Risk Assessment** | Official 4-factor formula ($30/25/25/20$) + 4 severity bands (`Critical`, `High`, `Medium`, `Low`) | `ai/models/severity_scorer.py`, `ai/models/size_scorer.py`, `ai/models/location_scorer.py` | Bit-for-bit formula verification in automated test suite | **DONE** |
| **Quality Assessment Modules** | Automated `Accept`/`Reject` decisioning + supervisor audit sign-off | `ai/models/severity_scorer.py`, `backend/app/routers/image.py` | Unit & integration tests; DB persistence verified | **DONE** |
| **Production Quality Reports** | Structured CSV report service with summary metrics and granular records | `backend/app/services/report_service.py`, `backend/app/routers/analytics.py` | Downloadable CSV endpoint tested via `exportQualityReport` | **DONE** |
| **Defect Analytics Dashboards** | Aggregated telemetry, Pareto breakdowns, Recharts visual dashboard | `backend/app/routers/analytics.py`, `frontend/src/pages/SupervisorDashboard.jsx` | `GET /analytics/summary` verified; React build clean | **DONE** |
| **Trend Monitoring Workflows** | Multi-horizon daily inspection time-series ($7$, $30$, and all-time days) | `backend/app/routers/analytics.py`, `frontend/src/pages/SupervisorDashboard.jsx` | `GET /analytics/trends` verified with date grouping | **DONE** |
| **Quality Recommendations** | Advisory shop-floor guidance mapping defects to `PASS`, `CLEAN`, `REWORK`, `SCRAP` | `backend/app/services/recommendation_service.py` | Deterministic unit tests covering all defect types | **DONE** |

---

## 20. Milestone 3 Completion Summary

Milestone 3 is complete and verified. The platform successfully bridges raw computer vision inference with manufacturing operations:
- Defect taxonomy is categorized with hierarchical precision.
- Flaw severity is quantified via the official 4-factor weighted formula.
- Quality decisions are rendered automatically while preserving full supervisory governance.
- Production quality reports and trend analytics provide actionable plant intelligence.
- All code is implemented, integrated with PostgreSQL and React 19, and verified through automated test suites.

---

## 21. Milestone 4 Handover

Per project guidelines, **Milestone 4 (Testing, Deployment & Documentation)** is intentionally postponed:
- Cloud deployments (AWS / Azure) and Docker containerization remain scheduled for Milestone 4.
- High-throughput load testing and latency optimization remain scheduled for Milestone 4.
- Live plant PLC/SCADA integration remains scheduled for Milestone 4.
- Technical documentation is now synchronized and verified through Milestone 3.
