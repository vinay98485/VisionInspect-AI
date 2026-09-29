import { useEffect, useState } from "react";
import { Link, useNavigate, useOutletContext } from "react-router-dom";
import {
  getSupervisorReviewQueue,
  getImageBlobUrl,
  reviewImage,
  AuthError,
  ForbiddenError,
  ApiError,
} from "../../services/api";
import {
  RefreshIcon,
  AlertIcon,
  CheckIcon,
  CameraIcon,
} from "../../components/Icons";

function ReviewQueue() {
  const navigate = useNavigate();
  const { user, setHeaderConfig } = useOutletContext() || {};

  const [images, setImages] = useState([]);
  const [thumbnails, setThumbnails] = useState({});
  const [notesByImage, setNotesByImage] = useState({});
  const [reviewingId, setReviewingId] = useState(null);

  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("all");
  const [error, setError] = useState(null);
  const [errorType, setErrorType] = useState(null);
  const [actionMessage, setActionMessage] = useState(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  useEffect(() => {
    setHeaderConfig?.({
      title: "Review Queue",
      subtitle: "Operational sign-off workspace for inspection batches",
      actions: (
        <button
          type="button"
          className="btn-header-action"
          onClick={() => {
            setActionMessage(null);
            setRefreshTrigger((prev) => prev + 1);
          }}
          title="Reload queue records"
        >
          <RefreshIcon size={14} />
          <span>Refresh Queue</span>
        </button>
      ),
    });
  }, [setHeaderConfig]);

  useEffect(() => {
    let active = true;

    const fetchQueue = async () => {
      setLoading(true);
      setError(null);
      setErrorType(null);

      try {
        if (user && user.role_id !== 2) {
          setError("Access restricted to Factory Supervisors.");
          setErrorType("forbidden");
          return;
        }

        const imageData = await getSupervisorReviewQueue();
        if (!active) return;

        setImages(imageData);

        // Prepopulate existing notes
        const initialNotes = {};
        imageData.forEach((img) => {
          if (img.supervisor_notes) {
            initialNotes[img.id] = img.supervisor_notes;
          }
        });
        setNotesByImage(initialNotes);

        // Lazy load thumbnails for first 20 records
        imageData.slice(0, 20).forEach(async (img) => {
          try {
            const url = await getImageBlobUrl(img.id);
            if (active) {
              setThumbnails((prev) => ({
                ...prev,
                [img.id]: url,
              }));
            }
          } catch {
            // Keep fallback placeholder
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
          setError("Failed to fetch supervisor review queue.");
          setErrorType("server");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    fetchQueue();

    return () => {
      active = false;
    };
  }, [user, refreshTrigger]);

  const handleNotesChange = (imageId, value) => {
    setNotesByImage((prev) => ({
      ...prev,
      [imageId]: value,
    }));
  };

  const handleReview = async (imageId, decision) => {
    const notes = (notesByImage[imageId] || "").trim();

    if (decision === "rejected" && !notes) {
      setActionMessage({
        type: "error",
        text: `Inspection #${imageId} rejection requires supervisor audit notes. Please add explanatory remarks before rejecting.`,
      });
      return;
    }

    setReviewingId(imageId);
    setActionMessage(null);

    try {
      const updatedImage = await reviewImage(imageId, decision, notes);

      setImages((prev) =>
        prev.map((img) => (img.id === imageId ? updatedImage : img))
      );

      setActionMessage({
        type: "success",
        text: `Inspection #${imageId} (${updatedImage.original_filename}) successfully marked as ${decision.toUpperCase()}.`,
      });
    } catch (err) {
      if (err instanceof AuthError) {
        navigate("/login");
      } else {
        setActionMessage({
          type: "error",
          text: err.message || "Failed to submit review decision.",
        });
      }
    } finally {
      setReviewingId(null);
    }
  };

  // Counts
  const totalCount = images.length;
  const pendingCount = images.filter(
    (img) => !img.supervisor_decision || img.supervisor_decision === "pending"
  ).length;
  const approvedCount = images.filter((img) => img.supervisor_decision === "approved").length;
  const rejectedCount = images.filter((img) => img.supervisor_decision === "rejected").length;

  const filteredImages = images.filter((img) => {
    if (filter === "pending") {
      return !img.supervisor_decision || img.supervisor_decision === "pending";
    }
    if (filter === "approved") {
      return img.supervisor_decision === "approved";
    }
    if (filter === "rejected") {
      return img.supervisor_decision === "rejected";
    }
    return true;
  });

  if (loading) {
    return (
      <div className="section-loading-container">
        <div className="app-loading-spinner" />
        <p className="section-loading-text">Loading inspection review queue...</p>
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
        </div>
      </div>
    );
  }

  return (
    <div className="operational-page-wrapper">
      {/* Action Banner */}
      {actionMessage && (
        <div className={`alert-banner ${actionMessage.type === "success" ? "alert-success" : "alert-error"}`}>
          {actionMessage.type === "success" ? <CheckIcon size={16} /> : <AlertIcon size={16} />}
          <span>{actionMessage.text}</span>
          <button
            type="button"
            className="alert-dismiss-btn"
            onClick={() => setActionMessage(null)}
          >
            ×
          </button>
        </div>
      )}

      {/* Filter Toolbar & Queue Counts */}
      <div className="queue-controls-bar">
        <div className="segment-control" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={filter === "all"}
            className={`segment-btn ${filter === "all" ? "active" : ""}`}
            onClick={() => setFilter("all")}
          >
            All <span className="tab-badge">{totalCount}</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={filter === "pending"}
            className={`segment-btn ${filter === "pending" ? "active" : ""}`}
            onClick={() => setFilter("pending")}
          >
            Pending <span className="tab-badge warning-bg">{pendingCount}</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={filter === "approved"}
            className={`segment-btn ${filter === "approved" ? "active" : ""}`}
            onClick={() => setFilter("approved")}
          >
            Approved <span className="tab-badge success-bg">{approvedCount}</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={filter === "rejected"}
            className={`segment-btn ${filter === "rejected" ? "active" : ""}`}
            onClick={() => setFilter("rejected")}
          >
            Rejected <span className="tab-badge danger-bg">{rejectedCount}</span>
          </button>
        </div>

        <div className="queue-summary-text">
          Showing <strong>{filteredImages.length}</strong> of <strong>{totalCount}</strong> inspection records
        </div>
      </div>

      {/* Interactive Review Data Table */}
      {filteredImages.length === 0 ? (
        <div className="empty-state-panel">
          <CheckIcon size={32} className="empty-state-icon" />
          <h3 className="empty-state-title">No Records Found</h3>
          <p className="empty-state-desc">
            {filter === "all"
              ? "No inspection records currently exist in the queue."
              : `There are no inspections matching the "${filter}" filter criteria.`}
          </p>
        </div>
      ) : (
        <div className="table-card-wrapper">
          <table className="modern-data-table" aria-label="Inspection Review Queue">
            <thead>
              <tr>
                <th style={{ width: "56px" }}>Frame</th>
                <th>Sample / ID</th>
                <th>Uploader</th>
                <th>Timestamp</th>
                <th>AI Verdict</th>
                <th>Status</th>
                <th style={{ minWidth: "240px" }}>Supervisor Audit Notes</th>
                <th style={{ textAlign: "right", minWidth: "160px" }}>Disposition</th>
              </tr>
            </thead>
            <tbody>
              {filteredImages.map((image) => {
                const isApproved = image.supervisor_decision === "approved";
                const isRejected = image.supervisor_decision === "rejected";
                const isProcessing = reviewingId === image.id;
                const hasAIInspection = image.severity_score !== null && image.severity_score !== undefined;

                return (
                  <tr key={image.id} className={isProcessing ? "row-processing" : ""}>
                    {/* Visual Frame */}
                    <td>
                      {thumbnails[image.id] ? (
                        <img
                          src={thumbnails[image.id]}
                          alt={image.original_filename}
                          className="table-sample-thumb"
                        />
                      ) : (
                        <div className="table-sample-placeholder" title="Sensor capture">
                          <CameraIcon size={16} />
                        </div>
                      )}
                    </td>

                    {/* Filename & ID */}
                    <td>
                      <Link to={`/images/${image.id}`} className="table-title-link">
                        {image.original_filename}
                      </Link>
                      <span className="table-sub-detail">ID #{image.id}</span>
                    </td>

                    {/* Uploader */}
                    <td>
                      <span className="table-text-medium">{image.uploaded_by?.name || "Operator"}</span>
                    </td>

                    {/* Timestamp */}
                    <td>
                      <span className="table-mono-text">
                        {new Date(image.uploaded_at).toLocaleString([], {
                          month: "short",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </span>
                    </td>

                    {/* AI Verdict */}
                    <td>
                      {hasAIInspection ? (
                        <div className="table-verdict-stack">
                          <span
                            className={`status-pill ${
                              image.quality_decision === "Reject"
                                ? "status-danger"
                                : image.severity_level === "Critical" || image.severity_level === "High"
                                ? "status-warning"
                                : "status-healthy"
                            }`}
                          >
                            {image.quality_decision === "Reject"
                              ? "AI Defective"
                              : image.severity_level || "Pass"}
                          </span>
                          <span className="table-sub-detail">
                            Score: {Number(image.severity_score).toFixed(2)}
                          </span>
                        </div>
                      ) : (
                        <span className="status-pill status-neutral">Pending AI</span>
                      )}
                    </td>

                    {/* Status */}
                    <td>
                      {isApproved ? (
                        <span className="status-pill status-healthy">
                          <span className="dot-indicator" /> Approved
                        </span>
                      ) : isRejected ? (
                        <span className="status-pill status-danger">
                          <span className="dot-indicator" /> Rejected
                        </span>
                      ) : (
                        <span className="status-pill status-warning">
                          <span className="dot-indicator" /> Pending
                        </span>
                      )}
                    </td>

                    {/* Audit Notes */}
                    <td>
                      <textarea
                        className="inline-notes-input"
                        rows="2"
                        placeholder={
                          isApproved || isRejected
                            ? "Review completed"
                            : "Enter supervisor audit notes (required for rejection)..."
                        }
                        value={notesByImage[image.id] ?? image.supervisor_notes ?? ""}
                        onChange={(e) => handleNotesChange(image.id, e.target.value)}
                        disabled={isProcessing || isApproved || isRejected}
                      />
                      {image.reviewed_by && (
                        <span className="notes-meta-caption">
                          Signed by {image.reviewed_by.name} at{" "}
                          {new Date(image.reviewed_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </span>
                      )}
                    </td>

                    {/* Action buttons */}
                    <td style={{ textAlign: "right" }}>
                      {!isApproved && !isRejected ? (
                        <div className="table-action-button-group">
                          <button
                            type="button"
                            className="btn-action-approve"
                            onClick={() => handleReview(image.id, "approved")}
                            disabled={isProcessing}
                            title="Approve inspection batch"
                          >
                            {isProcessing ? "..." : "Approve"}
                          </button>
                          <button
                            type="button"
                            className="btn-action-reject"
                            onClick={() => handleReview(image.id, "rejected")}
                            disabled={isProcessing}
                            title="Reject batch (requires notes)"
                          >
                            {isProcessing ? "..." : "Reject"}
                          </button>
                        </div>
                      ) : (
                        <span className="signed-off-label">
                          <CheckIcon size={14} /> Signed
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default ReviewQueue;
