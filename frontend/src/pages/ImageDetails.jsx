import { useEffect, useState } from "react";
import { useNavigate, useParams, Link, useOutletContext } from "react-router-dom";
import {
  getImage,
  getImageBlobUrl,
  getInspectionOverlayBlobUrl,
  reviewImage,
  AuthError,
  ForbiddenError,
  NotFoundError,
  ApiError,
} from "../services/api";
import {
  AlertIcon,
  CheckIcon,
  CameraIcon,
  LayersIcon,
} from "../components/Icons";

function ImageDetails() {
  const { imageId } = useParams();
  const navigate = useNavigate();
  const { user, setHeaderConfig } = useOutletContext() || {};

  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [inspectionOverlayUrl, setInspectionOverlayUrl] = useState(null);
  const [activeImageView, setActiveImageView] = useState("overlay"); // 'overlay' | 'raw'

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [errorType, setErrorType] = useState(null);

  const [supervisorNotes, setSupervisorNotes] = useState("");
  const [reviewing, setReviewing] = useState(false);
  const [reviewFeedback, setReviewFeedback] = useState(null);

  // Progressive disclosure for technical engineering scores
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  const isSupervisor = user?.role_id === 2;

  useEffect(() => {
    setHeaderConfig?.({
      title: image ? `Inspection Record #${image.id}` : `Inspection #${imageId}`,
      subtitle: image?.original_filename || "Specimen visual defect analysis",
      actions: (
        <Link
          to={isSupervisor ? "/supervisor/reviews" : "/inspections"}
          className="btn-header-secondary"
        >
          <span>← Back to {isSupervisor ? "Queue" : "Inspections"}</span>
        </Link>
      ),
    });
  }, [setHeaderConfig, image, imageId, isSupervisor]);

  useEffect(() => {
    let active = true;
    let loadedImageUrl = null;
    let loadedOverlayUrl = null;

    const fetchRecord = async () => {
      setLoading(true);
      setError(null);
      setErrorType(null);

      try {
        const data = await getImage(imageId);
        if (!active) return;

        setImage(data);
        setSupervisorNotes(data.supervisor_notes || "");

        // Load raw frame
        loadedImageUrl = await getImageBlobUrl(imageId);
        if (!active) {
          URL.revokeObjectURL(loadedImageUrl);
          return;
        }
        setImageUrl(loadedImageUrl);

        // Load AI overlay if available
        if (data.inspection_overlay_available) {
          try {
            loadedOverlayUrl = await getInspectionOverlayBlobUrl(imageId);
            if (active) {
              setInspectionOverlayUrl(loadedOverlayUrl);
              setActiveImageView("overlay");
            } else {
              URL.revokeObjectURL(loadedOverlayUrl);
            }
          } catch {
            if (active) {
              setInspectionOverlayUrl(null);
              setActiveImageView("raw");
            }
          }
        } else {
          setActiveImageView("raw");
        }
      } catch (err) {
        if (!active) return;
        if (err instanceof AuthError) {
          navigate("/login");
        } else if (err instanceof ForbiddenError) {
          setError(err.message);
          setErrorType("forbidden");
        } else if (err instanceof NotFoundError) {
          setError(`Inspection record #${imageId} was not found in the plant database.`);
          setErrorType("notfound");
        } else if (err instanceof ApiError) {
          setError(err.message);
          setErrorType("server");
        } else {
          setError("Failed to fetch inspection details from server.");
          setErrorType("server");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    fetchRecord();

    return () => {
      active = false;
      if (loadedImageUrl) URL.revokeObjectURL(loadedImageUrl);
      if (loadedOverlayUrl) URL.revokeObjectURL(loadedOverlayUrl);
    };
  }, [imageId, navigate]);

  const handleSupervisorReview = async (decision) => {
    const trimmedNotes = supervisorNotes.trim();

    if (decision === "rejected" && !trimmedNotes) {
      setReviewFeedback({
        type: "error",
        text: "Rejection requires supervisor audit notes explaining the defect condition.",
      });
      return;
    }

    setReviewing(true);
    setReviewFeedback(null);

    try {
      const updated = await reviewImage(imageId, decision, trimmedNotes);
      setImage(updated);
      setReviewFeedback({
        type: "success",
        text: `Inspection #${imageId} officially marked as ${decision.toUpperCase()}.`,
      });
    } catch (err) {
      if (err instanceof AuthError) {
        navigate("/login");
      } else {
        setReviewFeedback({
          type: "error",
          text: err.message || "Failed to submit review.",
        });
      }
    } finally {
      setReviewing(false);
    }
  };

  if (loading) {
    return (
      <div className="section-loading-container">
        <div className="app-loading-spinner" />
        <p className="section-loading-text">Loading inspection telemetry #{imageId}...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="notice-card error-notice">
        <div className="notice-icon"><AlertIcon size={24} /></div>
        <div className="notice-content">
          <h3 className="notice-title">{errorType === "notfound" ? "Record Not Found" : "Inspection Error"}</h3>
          <p className="notice-desc">{error}</p>
          <Link
            to={isSupervisor ? "/supervisor/reviews" : "/inspections"}
            className="btn btn-secondary btn-sm"
            style={{ marginTop: "1rem" }}
          >
            Return to Inspection List
          </Link>
        </div>
      </div>
    );
  }

  const hasInspection = image?.severity_score !== null && image?.severity_score !== undefined;
  const isApproved = image?.supervisor_decision === "approved";
  const isRejected = image?.supervisor_decision === "rejected";

  return (
    <div className="specimen-detail-wrapper">
      {/* Review Feedback Alert */}
      {reviewFeedback && (
        <div className={`alert-banner ${reviewFeedback.type === "success" ? "alert-success" : "alert-error"}`} style={{ marginBottom: "1.25rem" }}>
          {reviewFeedback.type === "success" ? <CheckIcon size={16} /> : <AlertIcon size={16} />}
          <span>{reviewFeedback.text}</span>
          <button type="button" className="alert-dismiss-btn" onClick={() => setReviewFeedback(null)}>
            ×
          </button>
        </div>
      )}

      {/* ==============================================================
          HERO SECTION: FOCUSED IMAGE DISPLAY + KEY VERDICT CARD
          ============================================================== */}
      <section className="specimen-hero-grid">
        {/* Left: Focused Image Viewer */}
        <div className="hero-viewer-card">
          <div className="viewer-controls-bar">
            <div className="viewer-title-group">
              <span className="viewer-title">{image.original_filename}</span>
              <span className="viewer-sub">Record ID #{image.id}</span>
            </div>

            {/* View Switcher: Overlay vs Raw */}
            {inspectionOverlayUrl && (
              <div className="segment-control subtle-segment" role="tablist">
                <button
                  type="button"
                  className={`segment-btn ${activeImageView === "overlay" ? "active" : ""}`}
                  onClick={() => setActiveImageView("overlay")}
                >
                  <LayersIcon size={14} />
                  <span>AI Overlay</span>
                </button>
                <button
                  type="button"
                  className={`segment-btn ${activeImageView === "raw" ? "active" : ""}`}
                  onClick={() => setActiveImageView("raw")}
                >
                  <CameraIcon size={14} />
                  <span>Raw Frame</span>
                </button>
              </div>
            )}
          </div>

          <div className="viewer-canvas-frame">
            {activeImageView === "overlay" && inspectionOverlayUrl ? (
              <img
                src={inspectionOverlayUrl}
                alt="AI Defect Segmentation Overlay"
                className="viewer-main-img"
              />
            ) : imageUrl ? (
              <img
                src={imageUrl}
                alt="Raw optical sensor frame"
                className="viewer-main-img"
              />
            ) : (
              <div className="viewer-placeholder">
                <CameraIcon size={32} />
                <span>Buffering visual frame...</span>
              </div>
            )}
          </div>

          <div className="viewer-footer-caption">
            {activeImageView === "overlay" && inspectionOverlayUrl ? (
              <span>
                <strong>Red contours:</strong> U-Net defect boundary • <strong>Green bounding boxes:</strong> YOLO11n object detections ({image.detected_objects_count ?? 0} found)
              </span>
            ) : (
              <span>Raw optical frame as acquired from visual sensor cell.</span>
            )}
          </div>
        </div>

        {/* Right: Key Verdict Card */}
        <div className="hero-verdict-card">
          <div className="verdict-card-top">
            <span className="verdict-label-small">Manufacturing Quality Verdict</span>
            <div className="verdict-badge-large">
              <span
                className={`status-pill ${
                  image.quality_decision === "Reject" ? "status-danger" : "status-healthy"
                }`}
                style={{ fontSize: "1.1rem", padding: "0.5rem 1rem", fontWeight: 700 }}
              >
                {image.quality_decision ? image.quality_decision.toUpperCase() : "PENDING"}
              </span>
            </div>
          </div>

          <div className="verdict-stats-list">
            <div className="verdict-stat-row">
              <span className="verdict-stat-title">Defect Severity</span>
              <span className="verdict-stat-value">
                <strong className={image.severity_level === "Critical" ? "danger-color" : ""}>
                  {image.severity_level || "Normal"}
                </strong>{" "}
                <span className="verdict-stat-sub">
                  ({hasInspection ? Number(image.severity_score).toFixed(2) : "—"})
                </span>
              </span>
            </div>

            <div className="verdict-stat-row">
              <span className="verdict-stat-title">AI Confidence</span>
              <span className="verdict-stat-value">
                {image.confidence_score
                  ? `${Number(image.confidence_score).toFixed(1)}%`
                  : image.classification_confidence
                  ? `${Number(image.classification_confidence).toFixed(1)}%`
                  : "—"}
              </span>
            </div>

            <div className="verdict-stat-row">
              <span className="verdict-stat-title">Defect Classification</span>
              <span className="verdict-stat-value capitalize">
                {image.resolved_defect_status || image.predicted_defect_type || image.defect_type || "Normal"}
              </span>
            </div>

            <div className="verdict-stat-row">
              <span className="verdict-stat-title">Product Category</span>
              <span className="verdict-stat-value capitalize">
                {image.predicted_category || image.category || "—"}
              </span>
            </div>

            {image.recommendation && image.recommendation.action !== "PENDING" && (
              <div className="verdict-recommendation-box">
                <span className="verdict-stat-title">Recommended Action</span>
                <span className="recommendation-action-tag">
                  {image.recommendation.action_label || image.recommendation.action}
                </span>
                <p className="recommendation-summary-text">
                  {image.recommendation.rationale}
                </p>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* ==============================================================
          DETAILED SECTIONS BELOW: DEEP TELEMETRY & SIGN-OFF
          ============================================================== */}
      <section className="specimen-secondary-grid">
        {/* 1. Defect Localization & Metrics */}
        <div className="clean-section-card">
          <h3 className="section-heading" style={{ fontSize: "1.05rem" }}>
            Defect Localization & Anomaly Metrics
          </h3>
          <p className="section-subheading">Computer vision spatial measurements & multi-modal detection consensus</p>

          <div className="spec-table-container" style={{ marginTop: "1rem" }}>
            <table className="meta-spec-table">
              <tbody>
                <tr>
                  <th>Detected Defect Objects</th>
                  <td>
                    <strong>{image.detected_objects_count ?? 0}</strong>
                    {image.detected_objects_count > 0 && (
                      <span className="status-pill status-neutral" style={{ marginLeft: "0.5rem" }}>
                        YOLO11n (conf 0.25)
                      </span>
                    )}
                  </td>
                </tr>
                <tr>
                  <th>Predicted Defect Area</th>
                  <td>
                    {image.predicted_area_percent != null
                      ? `${Number(image.predicted_area_percent).toFixed(2)}% of frame`
                      : "0.00%"}
                  </td>
                </tr>
                <tr>
                  <th>ResNet18 Anomaly Score</th>
                  <td>
                    {image.anomaly_score != null ? Number(image.anomaly_score).toFixed(4) : "—"}
                  </td>
                </tr>
                <tr>
                  <th>Classification Confidence</th>
                  <td>
                    {image.classification_confidence != null
                      ? `${Number(image.classification_confidence).toFixed(1)}%`
                      : "—"}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* 2. Factory Supervisor Sign-off */}
        <div className="clean-section-card">
          <div className="box-header-row">
            <div>
              <h3 className="section-heading" style={{ fontSize: "1.05rem" }}>
                Factory Supervisor Sign-off
              </h3>
              <p className="section-subheading">Formal audit review and production sign-off trail</p>
            </div>
            <span
              className={`status-pill ${
                isApproved ? "status-healthy" : isRejected ? "status-danger" : "status-warning"
              }`}
            >
              {isApproved ? "Approved" : isRejected ? "Rejected" : "Awaiting Review"}
            </span>
          </div>

          <div style={{ marginTop: "1rem" }}>
            {image.reviewed_by ? (
              <div className="signed-audit-trail">
                <div className="audit-row">
                  <span className="audit-lbl">Reviewed By</span>
                  <span className="audit-val">{image.reviewed_by.name} ({image.reviewed_by.email})</span>
                </div>
                <div className="audit-row">
                  <span className="audit-lbl">Sign-off Timestamp</span>
                  <span className="audit-val">{new Date(image.reviewed_at).toLocaleString()}</span>
                </div>
                <div className="audit-notes-box">
                  <span className="audit-lbl" style={{ display: "block", marginBottom: "0.3rem" }}>
                    Supervisor Audit Notes:
                  </span>
                  <p className="audit-notes-text">
                    {image.supervisor_notes || "No additional remarks recorded for this inspection."}
                  </p>
                </div>
              </div>
            ) : isSupervisor ? (
              <div className="supervisor-action-box">
                <div className="form-group">
                  <label className="form-label" htmlFor="audit-notes">
                    Supervisor Audit Observations {image.quality_decision === "Reject" ? "*" : ""}
                  </label>
                  <textarea
                    id="audit-notes"
                    className="form-textarea"
                    rows="3"
                    placeholder="Enter audit notes (mandatory if overriding or rejecting)..."
                    value={supervisorNotes}
                    onChange={(e) => setSupervisorNotes(e.target.value)}
                    disabled={reviewing}
                  />
                </div>

                <div className="action-button-row">
                  <button
                    type="button"
                    className="btn btn-success"
                    onClick={() => handleSupervisorReview("approved")}
                    disabled={reviewing}
                  >
                    <CheckIcon size={14} />
                    <span>{reviewing ? "Processing..." : "Approve Batch"}</span>
                  </button>
                  <button
                    type="button"
                    className="btn btn-danger"
                    onClick={() => handleSupervisorReview("rejected")}
                    disabled={reviewing}
                  >
                    <span>{reviewing ? "Processing..." : "Reject Batch"}</span>
                  </button>
                </div>
              </div>
            ) : (
              <div className="pending-review-notice">
                <span className="notice-text">
                  This inspection specimen is currently in the Factory Supervisor queue awaiting formal sign-off.
                </span>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* ==============================================================
          PROGRESSIVE DISCLOSURE: TECHNICAL METRICS & DATA PROVENANCE
          ============================================================== */}
      <section className="progressive-details-section">
        <button
          type="button"
          className="btn-disclosure-toggle"
          onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
        >
          <span>{showTechnicalDetails ? "▼ Hide Technical Pipeline Metrics" : "▶ Show Technical Pipeline Metrics & Data Provenance"}</span>
        </button>

        {showTechnicalDetails && (
          <div className="disclosed-technical-content">
            <div className="technical-grid">
              <div className="tech-box">
                <span className="tech-lbl">Size Severity Metric</span>
                <span className="tech-val">{Number(image.size_score ?? 0).toFixed(3)}</span>
              </div>
              <div className="tech-box">
                <span className="tech-lbl">Location Criticality Metric</span>
                <span className="tech-val">{Number(image.location_score ?? 0).toFixed(3)}</span>
              </div>
              <div className="tech-box">
                <span className="tech-lbl">Defect Type Severity Metric</span>
                <span className="tech-val">{Number(image.defect_type_score ?? 0).toFixed(3)}</span>
              </div>
              <div className="tech-box">
                <span className="tech-lbl">Storage Path</span>
                <span className="tech-val mono" style={{ fontSize: "0.75rem" }}>{image.storage_path || "—"}</span>
              </div>
              <div className="tech-box">
                <span className="tech-lbl">Uploader Name & ID</span>
                <span className="tech-val">{image.uploaded_by?.name || "Operator"} (UID #{image.uploaded_by?.id || "1"})</span>
              </div>
              <div className="tech-box">
                <span className="tech-lbl">Acquisition Timestamp</span>
                <span className="tech-val">{new Date(image.uploaded_at).toISOString()}</span>
              </div>
            </div>

            {image.recommendation?.guidance && (
              <div className="tech-guidance-box">
                <strong>Engineering Guidance:</strong> {image.recommendation.guidance}
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}

export default ImageDetails;
