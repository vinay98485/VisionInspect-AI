import { useEffect, useState } from "react";
import { Link, useNavigate, useOutletContext } from "react-router-dom";
import {
  getSupervisorReviewQueue,
  getAnalyticsSummary,
  AuthError,
  ForbiddenError,
  ApiError,
} from "../../services/api";
import {
  RefreshIcon,
  AlertIcon,
  CheckIcon,
  ExternalLinkIcon,
} from "../../components/Icons";

function SupervisorDashboard() {
  const navigate = useNavigate();
  const { user, setHeaderConfig } = useOutletContext() || {};

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [errorType, setErrorType] = useState(null);

  const [queue, setQueue] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  useEffect(() => {
    setHeaderConfig?.({
      title: "Quality Overview",
      subtitle: "Plant-wide visual inspection telemetry & line performance",
      actions: (
        <button
          type="button"
          className="btn-header-action"
          onClick={() => setRefreshTrigger((prev) => prev + 1)}
          title="Refresh quality metrics"
        >
          <RefreshIcon size={14} />
          <span>Refresh</span>
        </button>
      ),
    });
  }, [setHeaderConfig]);

  useEffect(() => {
    let active = true;

    const fetchData = async () => {
      setLoading(true);
      setError(null);
      setErrorType(null);

      try {
        if (user && user.role_id !== 2) {
          setError("Supervisor access required. Redirecting to Quality Engineer console...");
          setErrorType("forbidden");
          setTimeout(() => navigate("/dashboard", { replace: true }), 1500);
          return;
        }

        const [queueData, analyticsData] = await Promise.all([
          getSupervisorReviewQueue(),
          getAnalyticsSummary(),
        ]);

        if (!active) return;

        setQueue(queueData);
        setAnalytics(analyticsData);
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
          setError("Unable to connect to the manufacturing inspection service.");
          setErrorType("server");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    fetchData();

    return () => {
      active = false;
    };
  }, [user, navigate, refreshTrigger]);

  if (loading) {
    return (
      <div className="section-loading-container">
        <div className="app-loading-spinner" />
        <p className="section-loading-text">Loading quality telemetry...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="notice-card error-notice">
        <div className="notice-icon"><AlertIcon size={24} /></div>
        <div className="notice-content">
          <h3 className="notice-title">{errorType === "auth" ? "Session Expired" : "Access Restricted"}</h3>
          <p className="notice-desc">{error}</p>
          {errorType === "auth" && (
            <button className="btn btn-primary btn-sm" onClick={() => navigate("/login")}>
              Sign In Again
            </button>
          )}
        </div>
      </div>
    );
  }

  // Derive prioritized metrics
  const totalInspections = analytics?.total_inspections ?? queue.length;
  const pendingReview = queue.filter(
    (img) => !img.supervisor_decision || img.supervisor_decision === "pending"
  ).length;
  const rejectedCount = analytics?.rejected ?? queue.filter((img) => img.supervisor_decision === "rejected").length;
  const criticalCount = analytics?.severity?.critical ?? 0;

  // Acceptance rate calculation
  const acceptedCount = analytics?.accepted ?? 0;
  const totalEvaluated = acceptedCount + rejectedCount;
  const acceptanceRate = totalEvaluated > 0 ? ((acceptedCount / totalEvaluated) * 100).toFixed(1) : "100.0";

  // Critical / high attention items from queue
  const attentionItems = queue
    .filter(
      (img) =>
        (!img.supervisor_decision || img.supervisor_decision === "pending") &&
        (img.severity_level === "Critical" || img.severity_level === "High" || (img.severity_score && Number(img.severity_score) > 0.6))
    )
    .slice(0, 4);

  // Normal pending review items
  const pendingItems = queue
    .filter((img) => !img.supervisor_decision || img.supervisor_decision === "pending")
    .slice(0, 5);

  return (
    <div className="overview-page-wrapper">
      {/* 1. PRIMARY METRICS (4 PRIORITIZED KPIS) */}
      <section className="metrics-strip" aria-label="Primary Manufacturing KPIs">
        <div className="metric-cell">
          <span className="metric-caption">Total Inspections</span>
          <div className="metric-value-row">
            <span className="metric-numeral">{totalInspections}</span>
            <span className="metric-badge-neutral">Plant-wide</span>
          </div>
        </div>

        <div className="metric-cell attention-accent">
          <span className="metric-caption">Pending Review</span>
          <div className="metric-value-row">
            <span className="metric-numeral warning-color">{pendingReview}</span>
            {pendingReview > 0 && (
              <Link to="/supervisor/reviews" className="metric-action-pill">
                Review Now →
              </Link>
            )}
          </div>
        </div>

        <div className="metric-cell">
          <span className="metric-caption">Defect Rejections</span>
          <div className="metric-value-row">
            <span className="metric-numeral danger-color">{rejectedCount}</span>
            <span className="metric-sub-label">Disapproved</span>
          </div>
        </div>

        <div className="metric-cell">
          <span className="metric-caption">Critical Issues</span>
          <div className="metric-value-row">
            <span className="metric-numeral danger-color">{criticalCount}</span>
            {criticalCount > 0 ? (
              <span className="status-pill status-danger">Immediate Action</span>
            ) : (
              <span className="status-pill status-healthy">Zero Critical</span>
            )}
          </div>
        </div>
      </section>

      {/* 2. SECONDARY OVERVIEW: QUALITY HEALTH SUMMARY */}
      <section className="overview-split-section">
        <div className="overview-summary-box">
          <div className="box-header-row">
            <div>
              <h2 className="section-heading">Quality Health & Line Performance</h2>
              <p className="section-subheading">Aggregated baseline across all automated inspection cells</p>
            </div>
            <Link to="/supervisor/analytics" className="text-link-action">
              Detailed Analytics →
            </Link>
          </div>

          <div className="health-kpi-grid">
            <div className="health-stat">
              <span className="health-stat-label">First-Pass Yield</span>
              <div className="health-stat-bar-container">
                <div className="health-stat-bar" style={{ width: `${acceptanceRate}%` }} />
              </div>
              <div className="health-stat-footer">
                <span className="health-stat-num">{acceptanceRate}%</span>
                <span className="health-stat-note">{acceptedCount} Accepted / {totalEvaluated} Total</span>
              </div>
            </div>

            <div className="health-stat">
              <span className="health-stat-label">Average AI Confidence</span>
              <span className="health-stat-value">
                {Number(analytics?.average?.confidence ?? 0).toFixed(1)}%
              </span>
              <span className="health-stat-note">Multi-model consensus</span>
            </div>

            <div className="health-stat">
              <span className="health-stat-label">Mean Severity Score</span>
              <span className="health-stat-value">
                {Number(analytics?.average?.severity ?? 0).toFixed(2)}
              </span>
              <span className="health-stat-note">0.0 (clean) to 1.0 (fatal)</span>
            </div>
          </div>
        </div>
      </section>

      {/* 3. ATTENTION SECTION */}
      <section className="overview-attention-section">
        <div className="attention-column">
          <div className="section-title-bar">
            <div>
              <h3 className="section-heading">High Priority Inspections</h3>
              <p className="section-subheading">Flagged critical defects requiring immediate supervisor review</p>
            </div>
            {attentionItems.length > 0 && (
              <span className="status-pill status-danger">{attentionItems.length} Urgent</span>
            )}
          </div>

          {attentionItems.length === 0 ? (
            <div className="empty-subtle-box">
              <CheckIcon size={20} className="empty-icon success-color" />
              <p className="empty-text">No critical defect flags currently awaiting review.</p>
            </div>
          ) : (
            <div className="attention-cards-list">
              {attentionItems.map((img) => (
                <div key={img.id} className="attention-item-row">
                  <div className="item-meta">
                    <span className="item-title">{img.original_filename}</span>
                    <span className="item-details">
                      Inspection #{img.id} • {img.predicted_category || img.category || "Part"} •{" "}
                      <span className="danger-color">{img.predicted_defect_type || img.defect_type || "Defect"}</span>
                    </span>
                  </div>
                  <div className="item-action-area">
                    <span className="status-pill status-danger">
                      Severity {Number(img.severity_score).toFixed(2)}
                    </span>
                    <Link to={`/images/${img.id}`} className="btn-table-action" title="Inspect record">
                      Review <ExternalLinkIcon size={12} />
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="attention-column">
          <div className="section-title-bar">
            <div>
              <h3 className="section-heading">Review Queue Preview</h3>
              <p className="section-subheading">{pendingReview} total items pending supervisor disposition</p>
            </div>
            <Link to="/supervisor/reviews" className="text-link-action">
              Open Queue ({pendingReview}) →
            </Link>
          </div>

          {pendingItems.length === 0 ? (
            <div className="empty-subtle-box">
              <CheckIcon size={20} className="empty-icon success-color" />
              <p className="empty-text">Queue is clear. All inspections have been signed off.</p>
            </div>
          ) : (
            <div className="attention-cards-list">
              {pendingItems.map((img) => (
                <div key={img.id} className="attention-item-row">
                  <div className="item-meta">
                    <span className="item-title">{img.original_filename}</span>
                    <span className="item-details">
                      Uploaded by {img.uploaded_by?.name || "Operator"} •{" "}
                      {new Date(img.uploaded_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </span>
                  </div>
                  <div className="item-action-area">
                    <span className="status-pill status-warning">Pending Review</span>
                    <Link to="/supervisor/reviews" className="btn-table-action">
                      Audit
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

export default SupervisorDashboard;
