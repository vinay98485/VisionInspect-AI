import { useEffect, useState } from "react";
import { Link, useNavigate, useOutletContext } from "react-router-dom";
import {
  getImages,
  getImageBlobUrl,
  AuthError,
  ForbiddenError,
  ApiError,
} from "../../services/api";
import {
  RefreshIcon,
  SearchIcon,
  UploadIcon,
  AlertIcon,
  CameraIcon,
  ExternalLinkIcon,
} from "../../components/Icons";

function Inspections() {
  const navigate = useNavigate();
  const { setHeaderConfig } = useOutletContext() || {};

  const [images, setImages] = useState([]);
  const [thumbnails, setThumbnails] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  // Search & filter states
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  useEffect(() => {
    setHeaderConfig?.({
      title: "Inspection History",
      subtitle: "Complete repository of captured visual inspection specimens",
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

    const fetchHistory = async () => {
      setLoading(true);
      setError(null);

      try {
        const data = await getImages();
        if (!active) return;

        setImages(data);

        // Fetch thumbnails for first 25 records
        data.slice(0, 25).forEach(async (img) => {
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
          navigate("/login");
        } else if (err instanceof ForbiddenError || err instanceof ApiError) {
          setError(err.message);
        } else {
          setError("Failed to fetch inspection history.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    fetchHistory();

    return () => {
      active = false;
    };
  }, [navigate, refreshTrigger]);

  // Filtering & search logic
  const filteredImages = images.filter((img) => {
    // 1. Search Query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      const filenameMatch = img.original_filename?.toLowerCase().includes(q);
      const idMatch = String(img.id).includes(q);
      const categoryMatch = img.category?.toLowerCase().includes(q) || img.predicted_category?.toLowerCase().includes(q);
      if (!filenameMatch && !idMatch && !categoryMatch) return false;
    }

    // 2. Status Filter
    if (statusFilter === "completed") {
      return img.inspection_status === "completed" || (img.severity_score !== null && img.severity_score !== undefined);
    }
    if (statusFilter === "pending") {
      return !img.supervisor_decision || img.supervisor_decision === "pending";
    }
    if (statusFilter === "approved") {
      return img.supervisor_decision === "approved";
    }
    if (statusFilter === "rejected") {
      return img.supervisor_decision === "rejected";
    }

    return true;
  });

  if (loading) {
    return (
      <div className="section-loading-container">
        <div className="app-loading-spinner" />
        <p className="section-loading-text">Loading inspection records...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="notice-card error-notice">
        <div className="notice-icon"><AlertIcon size={24} /></div>
        <div className="notice-content">
          <h3 className="notice-title">Error</h3>
          <p className="notice-desc">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="operational-page-wrapper">
      {/* Search & Filter Toolbar */}
      <div className="table-filter-toolbar">
        <div className="search-input-wrapper">
          <SearchIcon size={16} className="search-icon" />
          <input
            type="text"
            className="search-input"
            placeholder="Search by filename, record ID, or product category..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
          {searchQuery && (
            <button
              type="button"
              className="clear-search-btn"
              onClick={() => setSearchQuery("")}
            >
              ×
            </button>
          )}
        </div>

        <div className="segment-control" role="tablist">
          <button
            type="button"
            className={`segment-btn ${statusFilter === "all" ? "active" : ""}`}
            onClick={() => setStatusFilter("all")}
          >
            All ({images.length})
          </button>
          <button
            type="button"
            className={`segment-btn ${statusFilter === "pending" ? "active" : ""}`}
            onClick={() => setStatusFilter("pending")}
          >
            Pending
          </button>
          <button
            type="button"
            className={`segment-btn ${statusFilter === "approved" ? "active" : ""}`}
            onClick={() => setStatusFilter("approved")}
          >
            Approved
          </button>
          <button
            type="button"
            className={`segment-btn ${statusFilter === "rejected" ? "active" : ""}`}
            onClick={() => setStatusFilter("rejected")}
          >
            Rejected
          </button>
        </div>
      </div>

      {/* 7-Column Historical Inspection Table */}
      {filteredImages.length === 0 ? (
        <div className="empty-state-panel">
          <p className="empty-state-desc">
            {searchQuery || statusFilter !== "all"
              ? "No inspection records match your current search/filter criteria."
              : "No visual inspection specimens have been uploaded yet."}
          </p>
          {!searchQuery && statusFilter === "all" && (
            <Link to="/upload" className="btn btn-primary btn-sm" style={{ marginTop: "0.5rem" }}>
              Upload First Specimen
            </Link>
          )}
        </div>
      ) : (
        <div className="table-card-wrapper">
          <table className="modern-data-table" aria-label="Inspection History">
            <thead>
              <tr>
                <th style={{ width: "56px" }}>Preview</th>
                <th>Filename & ID</th>
                <th>Uploaded By</th>
                <th>Timestamp</th>
                <th>AI Status</th>
                <th>Supervisor Decision</th>
                <th style={{ textAlign: "right", minWidth: "130px" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredImages.map((image) => {
                const hasAIInspection = image.severity_score !== null && image.severity_score !== undefined;
                const isApproved = image.supervisor_decision === "approved";
                const isRejected = image.supervisor_decision === "rejected";

                return (
                  <tr key={image.id}>
                    {/* Thumbnail */}
                    <td>
                      {thumbnails[image.id] ? (
                        <img
                          src={thumbnails[image.id]}
                          alt={image.original_filename}
                          className="table-sample-thumb"
                        />
                      ) : (
                        <div className="table-sample-placeholder">
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

                    {/* Uploaded By */}
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

                    {/* AI Status */}
                    <td>
                      {hasAIInspection ? (
                        <span className="status-pill status-healthy">
                          <span className="dot-indicator" /> Completed
                        </span>
                      ) : (
                        <span className="status-pill status-warning">
                          <span className="dot-indicator" /> Pending
                        </span>
                      )}
                    </td>

                    {/* Supervisor Decision */}
                    <td>
                      {isApproved ? (
                        <span className="status-pill status-healthy">Approved</span>
                      ) : isRejected ? (
                        <span className="status-pill status-danger">Rejected</span>
                      ) : (
                        <span className="status-pill status-neutral">Awaiting Review</span>
                      )}
                    </td>

                    {/* Actions */}
                    <td style={{ textAlign: "right" }}>
                      <Link to={`/images/${image.id}`} className="btn-table-action">
                        Inspect Record <ExternalLinkIcon size={12} />
                      </Link>
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

export default Inspections;
