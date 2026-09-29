import { useState, useEffect } from "react";
import { Link, useNavigate, useOutletContext } from "react-router-dom";
import {
  getCurrentUser,
  uploadImage,
  inspectImage,
  AuthError,
  ForbiddenError,
  ApiError,
} from "../services/api";
import {
  UploadIcon,
  CameraIcon,
  AlertIcon,
  CheckIcon,
  ExternalLinkIcon,
} from "../components/Icons";

const CATEGORIES = [
  "bottle",
  "cable",
  "capsule",
  "carpet",
  "grid",
  "hazelnut",
  "leather",
  "metal_nut",
  "pill",
  "screw",
  "tile",
  "toothbrush",
  "transistor",
  "wood",
  "zipper",
];

const DEFECT_TYPES = [
  "crack",
  "hole",
  "cut",
  "scratch",
  "contamination",
  "broken",
  "bent",
  "deformation",
  "color",
  "stain",
  "rough",
  "missing",
  "misplaced",
];

function Upload() {
  const navigate = useNavigate();
  const { setHeaderConfig } = useOutletContext() || {};

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  const [category, setCategory] = useState("");
  const [defectType, setDefectType] = useState("");
  const [autoClassify, setAutoClassify] = useState(true);
  const [inspectionStep, setInspectionStep] = useState(0);

  const [uploading, setUploading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);

  const [successData, setSuccessData] = useState(null);
  const [error, setError] = useState(null);
  const [errorType, setErrorType] = useState(null);

  useEffect(() => {
    setHeaderConfig?.({
      title: "New AI Inspection",
      subtitle: "Execute multi-stage deep learning pipeline on specimen frames",
      actions: (
        <Link to="/inspections" className="btn-header-secondary">
          <span>Inspection History →</span>
        </Link>
      ),
    });
  }, [setHeaderConfig]);

  useEffect(() => {
    const checkRole = async () => {
      try {
        const userData = await getCurrentUser();
        if (userData.role_id === 2) {
          setError(
            "Factory Supervisors do not have upload privileges. Image uploads and inspection execution are reserved for Quality Engineers."
          );
          setErrorType("forbidden");
        }
      } catch (err) {
        if (err instanceof AuthError) {
          setError(err.message);
          setErrorType("auth");
        }
      }
    };

    checkRole();
  }, []);

  const handleFileSelect = (selectedFile) => {
    setError(null);
    setSuccessData(null);

    if (!selectedFile) return;

    if (!["image/jpeg", "image/png"].includes(selectedFile.type)) {
      setError("Invalid file format. Only JPEG and PNG images are supported.");
      setErrorType("api");
      return;
    }

    if (selectedFile.size > 5 * 1024 * 1024) {
      setError("File size exceeds maximum limit of 5 MB.");
      setErrorType("api");
      return;
    }

    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async (event) => {
    event.preventDefault();

    setError(null);
    setErrorType(null);
    setSuccessData(null);

    if (!autoClassify) {
      if (!category) {
        setError("Please select the product category.");
        setErrorType("api");
        return;
      }

      if (!defectType) {
        setError("Please select the defect type.");
        setErrorType("api");
        return;
      }
    }

    if (!file) {
      setError("Please select an inspection image to upload.");
      setErrorType("api");
      return;
    }

    setUploading(true);
    setInspectionStep(1);

    let stepTimer = null;

    try {
      // Step 1: Upload image binary
      const uploadedImage = await uploadImage(file);
      setInspectionStep(2);

      stepTimer = setInterval(() => {
        setInspectionStep((prev) => (prev < 4 ? prev + 1 : prev));
      }, 600);

      // Step 2: Trigger AI inference pipeline
      const inspectionResult = await inspectImage(
        uploadedImage.id,
        autoClassify ? (category || null) : category,
        autoClassify ? (defectType || null) : defectType
      );

      if (stepTimer) clearInterval(stepTimer);
      setInspectionStep(5);

      setSuccessData({
        ...uploadedImage,
        inspection: inspectionResult,
      });

      setFile(null);
      setPreviewUrl(null);
      if (!autoClassify) {
        setCategory("");
        setDefectType("");
      }
    } catch (err) {
      if (stepTimer) clearInterval(stepTimer);
      if (err instanceof AuthError) {
        setError(err.message);
        setErrorType("auth");
      } else if (err instanceof ForbiddenError) {
        setError(err.message);
        setErrorType("forbidden");
      } else if (err instanceof ApiError) {
        setError(err.message);
        setErrorType("api");
      } else {
        setError("Unable to complete the image inspection. Please check the backend service.");
        setErrorType("api");
      }
    } finally {
      setUploading(false);
      setInspectionStep(0);
    }
  };

  if (errorType === "forbidden") {
    return (
      <div className="notice-card error-notice">
        <div className="notice-icon"><AlertIcon size={24} /></div>
        <div className="notice-content">
          <h3 className="notice-title">Upload Restricted</h3>
          <p className="notice-desc">{error}</p>
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={() => navigate("/supervisor/dashboard")}
            style={{ marginTop: "1rem" }}
          >
            Go to Supervisor Console
          </button>
        </div>
      </div>
    );
  }

  if (errorType === "auth") {
    return (
      <div className="notice-card error-notice">
        <div className="notice-icon"><AlertIcon size={24} /></div>
        <div className="notice-content">
          <h3 className="notice-title">Session Expired</h3>
          <p className="notice-desc">{error}</p>
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={() => navigate("/login")}
            style={{ marginTop: "1rem" }}
          >
            Log in again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="upload-page-wrapper">
      {/* Error alert */}
      {error && (
        <div className="alert-banner alert-error" style={{ marginBottom: "1.5rem" }}>
          <AlertIcon size={16} />
          <span>{error}</span>
          <button type="button" className="alert-dismiss-btn" onClick={() => setError(null)}>
            ×
          </button>
        </div>
      )}

      {/* Success Banner */}
      {successData && (
        <div className="inspection-result-banner">
          <div className="result-banner-header">
            <div className="result-banner-title-group">
              <span className="success-icon-badge"><CheckIcon size={18} /></span>
              <div>
                <h3 className="result-banner-title">Specimen Inspected Successfully</h3>
                <span className="result-banner-sub">{successData.original_filename} • Record #{successData.id}</span>
              </div>
            </div>

            <span
              className={`status-pill ${
                successData.inspection?.quality_decision === "Reject" ? "status-danger" : "status-healthy"
              }`}
              style={{ fontSize: "0.85rem", padding: "0.35rem 0.8rem" }}
            >
              Decision: {successData.inspection?.quality_decision || "Accept"}
            </span>
          </div>

          {/* Quick Metrics Grid */}
          <div className="result-metrics-row">
            <div className="result-metric-box">
              <span className="result-metric-label">Predicted Category</span>
              <span className="result-metric-val capitalize">
                {successData.inspection?.predicted_category || successData.inspection?.category || "—"}
              </span>
            </div>

            <div className="result-metric-box">
              <span className="result-metric-label">Defect Classification</span>
              <span className="result-metric-val capitalize">
                {successData.inspection?.resolved_defect_status ||
                  successData.inspection?.predicted_defect_type ||
                  successData.inspection?.defect_type ||
                  "Normal"}
              </span>
            </div>

            <div className="result-metric-box">
              <span className="result-metric-label">Confidence</span>
              <span className="result-metric-val">
                {successData.inspection?.classification_confidence != null
                  ? `${Number(successData.inspection.classification_confidence).toFixed(1)}%`
                  : "—"}
              </span>
            </div>

            <div className="result-metric-box">
              <span className="result-metric-label">Severity Level</span>
              <span className="result-metric-val">
                {successData.inspection?.severity_level || "Low"}
              </span>
            </div>
          </div>

          <div className="result-banner-actions">
            <Link to={`/images/${successData.id}`} className="btn btn-primary btn-sm">
              <span>View Full Inspection #{successData.id}</span>
              <ExternalLinkIcon size={12} />
            </Link>

            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => setSuccessData(null)}
            >
              Inspect Another Sample
            </button>
          </div>
        </div>
      )}

      {/* Main Inspection Workflow Card */}
      <div className="clean-workflow-card">
        <form onSubmit={handleUpload}>
          {/* Workflow Stage Visualizer */}
          <div className="workflow-steps-indicator" aria-label="Inspection Workflow Steps">
            <div className={`step-node ${uploading && inspectionStep >= 1 ? "active" : successData ? "completed" : "active"}`}>
              <span className="step-num">{successData ? "✓" : "1"}</span>
              <span className="step-name">Upload</span>
            </div>
            <div className="step-line" />
            <div className={`step-node ${uploading && inspectionStep >= 2 ? "active" : successData ? "completed" : ""}`}>
              <span className="step-num">{successData ? "✓" : "2"}</span>
              <span className="step-name">Classify</span>
            </div>
            <div className="step-line" />
            <div className={`step-node ${uploading && inspectionStep >= 3 ? "active" : successData ? "completed" : ""}`}>
              <span className="step-num">{successData ? "✓" : "3"}</span>
              <span className="step-name">Detect</span>
            </div>
            <div className="step-line" />
            <div className={`step-node ${uploading && inspectionStep >= 4 ? "active" : successData ? "completed" : ""}`}>
              <span className="step-num">{successData ? "✓" : "4"}</span>
              <span className="step-name">Segment</span>
            </div>
            <div className="step-line" />
            <div className={`step-node ${successData ? "completed" : ""}`}>
              <span className="step-num">{successData ? "✓" : "5"}</span>
              <span className="step-name">Verdict</span>
            </div>
          </div>

          {/* Mode Switcher */}
          <div className="classification-mode-box">
            <div className="mode-info">
              <span className="mode-title">
                {autoClassify ? "Automatic AI Classification Active" : "Manual Specification Mode"}
              </span>
              <p className="mode-desc">
                {autoClassify
                  ? "ResNet18 automated multi-class inference identifies product category (15 classes) and defect type."
                  : "Manual override active. Specify product category and defect type below."}
              </p>
            </div>

            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => setAutoClassify(!autoClassify)}
              disabled={uploading}
            >
              {autoClassify ? "Switch to Manual" : "Switch to Auto AI"}
            </button>
          </div>

          {/* Manual dropdowns if active */}
          {!autoClassify && (
            <div className="manual-spec-grid">
              <div className="form-group">
                <label className="form-label" htmlFor="category">Product Category *</label>
                <select
                  id="category"
                  className="form-input"
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  disabled={uploading}
                >
                  <option value="">Select product category</option>
                  {CATEGORIES.map((item) => (
                    <option key={item} value={item}>
                      {item.replace("_", " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="defectType">Defect Type *</label>
                <select
                  id="defectType"
                  className="form-input"
                  value={defectType}
                  onChange={(e) => setDefectType(e.target.value)}
                  disabled={uploading}
                >
                  <option value="">Select defect type</option>
                  {DEFECT_TYPES.map((item) => (
                    <option key={item} value={item}>
                      {item.replace("_", " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}

          {/* Drag & Drop Zone */}
          <div
            className={`modern-dropzone ${isDragOver ? "dragover" : ""}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => document.getElementById("specimen-file-input")?.click()}
          >
            <input
              id="specimen-file-input"
              type="file"
              accept="image/jpeg,image/png"
              onChange={(e) => handleFileSelect(e.target.files[0])}
              style={{ display: "none" }}
              disabled={uploading}
            />

            <div className="dropzone-icon-circle">
              <CameraIcon size={24} />
            </div>

            <div className="dropzone-text-group">
              <span className="dropzone-main-label">
                {file ? file.name : "Drop specimen sensor frame here, or browse"}
              </span>
              <span className="dropzone-hint-label">
                Standard optical inspection frames (JPEG, PNG up to 5 MB)
              </span>
            </div>
          </div>

          {/* File Selected Preview */}
          {file && (
            <div className="selected-file-strip">
              {previewUrl && <img src={previewUrl} alt="Preview" className="preview-mini-thumb" />}
              <div className="file-info-col">
                <span className="file-name-text">{file.name}</span>
                <span className="file-meta-text">
                  {(file.size / 1024).toFixed(1)} KB • {file.type}
                </span>
              </div>
              <button
                type="button"
                className="btn-remove-file"
                onClick={(e) => {
                  e.stopPropagation();
                  setFile(null);
                  setPreviewUrl(null);
                }}
                disabled={uploading}
              >
                Remove
              </button>
            </div>
          )}

          {/* Active Processing Step Checklist */}
          {uploading && (
            <div className="pipeline-live-checklist">
              <div className="pipeline-checklist-header">
                <div className="app-loading-spinner" style={{ width: "16px", height: "16px" }} />
                <span>AI Manufacturing Inspection Running...</span>
              </div>

              <div className="pipeline-steps-list">
                <div className={`pipeline-step-item ${inspectionStep >= 1 ? "step-active" : ""}`}>
                  <span>{inspectionStep > 1 ? "✓" : inspectionStep === 1 ? "●" : "○"}</span>
                  <span>1. Ingesting raw sensor frame to plant repository</span>
                </div>
                <div className={`pipeline-step-item ${inspectionStep >= 2 ? "step-active" : ""}`}>
                  <span>{inspectionStep > 2 ? "✓" : inspectionStep === 2 ? "●" : "○"}</span>
                  <span>2. ResNet18 Product Category Classifier inference (15 classes)</span>
                </div>
                <div className={`pipeline-step-item ${inspectionStep >= 3 ? "step-active" : ""}`}>
                  <span>{inspectionStep > 3 ? "✓" : inspectionStep === 3 ? "●" : "○"}</span>
                  <span>3. Defect classification & ResNet18 Layer3 anomaly scoring</span>
                </div>
                <div className={`pipeline-step-item ${inspectionStep >= 4 ? "step-active" : ""}`}>
                  <span>{inspectionStep > 4 ? "✓" : inspectionStep === 4 ? "●" : "○"}</span>
                  <span>4. U-Net segmentation & YOLO11n gated object detection</span>
                </div>
                <div className={`pipeline-step-item ${inspectionStep >= 5 ? "step-active" : ""}`}>
                  <span>{inspectionStep >= 5 ? "✓" : "○"}</span>
                  <span>5. Automated decision fusion & quality verdict synthesis</span>
                </div>
              </div>
            </div>
          )}

          {/* Submit Action */}
          <div className="form-submit-row">
            <button
              type="submit"
              className="btn btn-primary btn-lg"
              disabled={!file || uploading || (!autoClassify && (!category || !defectType))}
            >
              <UploadIcon size={18} />
              <span>{uploading ? "Analyzing Specimen..." : "Upload & Run AI Inspection"}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default Upload;
