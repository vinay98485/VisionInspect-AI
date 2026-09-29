import { useEffect, useState } from "react";
import { Link, useNavigate, useOutletContext } from "react-router-dom";
import {
  getImages,
  getAnalyticsSummary,
  getImageBlobUrl,
  AuthError,
  ForbiddenError,
  ApiError,
} from "../../services/api";
import {
  UploadIcon,
  RefreshIcon,
  AlertIcon,
  CheckIcon,
  CameraIcon,
  ExternalLinkIcon,
} from "../../components/Icons";

function QEDashboard() {
  const navigate = useNavigate();
  const { user, setHeaderConfig } = useOutletContext() || {};

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [errorType, setErrorType] = useState(null);

  const [images, setImages] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [thumbnails, setThumbnails] = useState({});
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  useEffect(() => {
    setHeaderConfig?.({
      title: "Quality Engineer Console",
      subtitle: "Inspection queue telemetry & immediate quality actions",
      actions: (
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button
            type="button"
            className="btn-header-secondary"
            onClick={() => setRefreshTrigger((prev) => prev + 1)}
          >
            <RefreshIcon size={14} />
            <span>Refresh</span>
          </button>
          <Link to="/upload" className="btn btn-primary btn-sm">
            <UploadIcon size={15} />
            <span>+ New Inspection</span>
          </Link>
        </div>
      ),
    });
  }, [setHeaderConfig]);

  useEffect(() => {
    let active = true;

    const fetchQEDashboard = async () => {
      setLoading(true);
      setError(null);
      setErrorType(null);

      try {
        if (user && user.role_id === 2) {
          navigate("/supervisor/dashboard", { replace: true });
          return;
        }

        const [imageData, analyticsData] = await Promise.all([
          getImages(),
          getAnalyticsSummary(),
        ]);

        if (!active) return;

        setImages(imageData);
        setAnalytics(analyticsData);

        // Fetch thumbnails for latest 6 images
        imageData.slice(0, 6).forEach(async (img) => {
          try {
            const url = await getImageBlobUrl(img.id);
            if (active) {
              setThumbnails((prev) => ({
                ...prev,
                [img.id]: url,
              }));
            }
          } catch {
            // Keep fallback icon
          }
        });
      } catch (err) {
        if (!active) return;
        if (err instanceof AuthError) {
          setError(err.message);
          setErrorType("auth");
        } else if (err instanceof ForbiddenError) {
          setError(err.message);
          setErrorType("forbidden");
        } else if (err instanceof ApiError) {
          setError(err.message);
          setErrorType("server");
        } else {
          setError("Failed to load Quality Engineer dashboard telemetry.");
          setErrorType("server");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    fetchQEDashboard();

    return () => {
      active = false;
    };
  }, [user, navigate, refreshTrigger]);

  if (loading) {
    return (
      <div className="section-loading-container">
        <div className="app-loading-spinner" />
        <p className="section-loading-text">Loading operator telemetry...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="notice-card error-notice">
        <div className="notice-icon"><AlertIcon size={24} /></div>
        <div className="notice-content">
          <h3 className="notice-title">{errorType === "auth" ? "Session Expired" : "Notice"}</h3>
          <p className="notice-desc">{error}</p>
          {errorType === "auth" && (
            <button className="btn btn-primary btn-sm" onClick={() => navigate("/login")}>
              Log in again
            </button>
          )}
        </div>
      </div>
    );
  }

  const totalInspections = analytics?.total_inspections ?? images.length;
  const pendingReview = analytics?.pending_review ?? images.filter((img) => !img.supervisor_decision || img.supervisor_decision === "pending").length;
  const reviewedCount = analytics?.reviewed ?? images.filter((img) => img.inspection_status === "reviewed" || img.supervisor_decision).length;
  const criticalCount = analytics?.severity?.critical ?? 0;

  // Latest inspections (preview only, up to 5)
  const recentInspections = images.slice(0, 5);

  // Critical issues or rejected records needing re-test
  const attentionItems = images
    .filter((img) => img.quality_decision === "Reject" || img.severity_level === "Critical")
    .slice(0, 4);

  return (
    <div className="overview-page-wrapper">
      {/* 1. PRIMARY KPIS */}
      <section className="metrics-strip" aria-label="Quality Engineer KPIs">
        <div className="metric-cell">
          <span className="metric-caption">Total Inspections</span>
          <div className="metric-value-row">
            <span className="metric-numeral">{totalInspections}</span>
            <span className="metric-badge-neutral">Logged</span>
          </div>
        </div>

        <div className="metric-cell">
          <span className="metric-caption">Awaiting Supervisor</span>
          <div className="metric-value-row">
            <span className="metric-numeral warning-color">{pendingReview}</span>
            <span className="status-pill status-warning">In Review</span>
          </div>
        </div>

        <div className="metric-cell">
          <span className="metric-caption">Completed Reviews</span>
          <div className="metric-value-row">
            <span className="metric-numeral success-color">{reviewedCount}</span>
            <span className="status-pill status-healthy">Closed</span>
          </div>
        </div>

        <div className="metric-cell">
          <span className="metric-caption">Critical Flags</span>
          <div className="metric-value-row">
            <span className="metric-numeral danger-color">{criticalCount}</span>
            {criticalCount > 0 ? (
              <span className="status-pill status-danger">Alert</span>
            ) : (
              <span className="status-pill status-healthy">Nominal</span>
            )}
          </div>
        </div>
      </section>

      {/* 2. QUICK LAUNCH BANNER */}
      <section className="quick-action-strip">
        <div className="quick-action-content">
          <div>
            <h3 className="quick-action-title">Capture & Analyze New Inspection Sample</h3>
            <p className="quick-action-subtitle">
              Run automated ResNet18 anomaly classification, U-Net pixel segmentation, and YOLO11n defect detection.
            </p>
          </div>
          <Link to="/upload" className="btn btn-primary">
            <UploadIcon size={16} />
            <span>Launch AI Inspection</span>
          </Link>
        </div>
      </section>

      {/* 3. ATTENTION & RECENT ACTIVITY */}
      <section className="overview-attention-section">
        {/* Left: Critical / Defective Records */}
        <div className="attention-column">
          <div className="section-title-bar">
            <div>
              <h3 className="section-heading">Defects Needing Attention</h3>
              <p className="section-subheading">Parts flagged with severe defect scores or rejection verdict</p>
            </div>
          </div>

          {attentionItems.length === 0 ? (
            <div className="empty-subtle-box">
              <CheckIcon size={20} className="empty-icon success-color" />
              <p className="empty-text">No active rejected defect records logged.</p>
            </div>
          ) : (
            <div className="attention-cards-list">
              {attentionItems.map((img) => (
                <div key={img.id} className="attention-item-row">
                  <div className="item-meta">
                    <span className="item-title">{img.original_filename}</span>
                    <span className="item-details">
                      #{img.id} • {img.predicted_category || img.category || "Part"} •{" "}
                      <span className="danger-color">{img.predicted_defect_type || img.defect_type || "Defect"}</span>
                    </span>
                  </div>
                  <div className="item-action-area">
                    <span className="status-pill status-danger">
                      {img.severity_level || "Defect"}
                    </span>
                    <Link to={`/images/${img.id}`} className="btn-table-action">
                      Inspect
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right: Recent Visual Inspections (Preview) */}
        <div className="attention-column">
          <div className="section-title-bar">
            <div>
              <h3 className="section-heading">Recent Visual Inspections</h3>
              <p className="section-subheading">Latest 5 samples processed by the inspection cell</p>
            </div>
            <Link to="/inspections" className="text-link-action">
              View All ({images.length}) →
            </Link>
          </div>

          {recentInspections.length === 0 ? (
            <div className="empty-subtle-box">
              <p className="empty-text">No inspection records yet. Start by uploading an inspection sample.</p>
              <Link to="/upload" className="btn btn-primary btn-sm" style={{ marginTop: "0.5rem" }}>
                Upload First Sample
              </Link>
            </div>
          ) : (
            <div className="attention-cards-list">
              {recentInspections.map((img) => (
                <div key={img.id} className="attention-item-row">
                  <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                    {thumbnails[img.id] ? (
                      <img src={thumbnails[img.id]} alt="" className="table-sample-thumb" style={{ width: "36px", height: "36px" }} />
                    ) : (
                      <div className="table-sample-placeholder" style={{ width: "36px", height: "36px" }}>
                        <CameraIcon size={14} />
                      </div>
                    )}
                    <div className="item-meta">
                      <Link to={`/images/${img.id}`} className="item-title">
                        {img.original_filename}
                      </Link>
                      <span className="item-details">
                        {new Date(img.uploaded_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} • ID #{img.id}
                      </span>
                    </div>
                  </div>

                  <div className="item-action-area">
                    <span
                      className={`status-pill ${
                        img.quality_decision === "Reject" ? "status-danger" : "status-healthy"
                      }`}
                    >
                      {img.quality_decision || (img.severity_score !== null ? "Evaluated" : "Pending")}
                    </span>
                    <Link to={`/images/${img.id}`} className="btn-table-action" title="View details">
                      <ExternalLinkIcon size={12} />
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export default QEDashboard;
