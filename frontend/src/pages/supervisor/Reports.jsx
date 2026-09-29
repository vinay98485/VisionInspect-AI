import { useEffect, useState } from "react";
import { useOutletContext } from "react-router-dom";
import { exportQualityReport, AuthError, ForbiddenError, ApiError } from "../../services/api";
import { ExportIcon, CheckIcon, AlertIcon, ReportsIcon } from "../../components/Icons";

function Reports() {
  const { setHeaderConfig } = useOutletContext() || {};

  const [exporting, setExporting] = useState(false);
  const [feedback, setFeedback] = useState(null);

  useEffect(() => {
    setHeaderConfig?.({
      title: "Manufacturing Quality Reports",
      subtitle: "Official plant audit logs and regulatory quality exports",
      actions: null,
    });
  }, [setHeaderConfig]);

  const handleExport = async () => {
    setExporting(true);
    setFeedback(null);

    try {
      const result = await exportQualityReport("csv");
      setFeedback({
        type: "success",
        text: `Production quality audit report successfully compiled and downloaded as ${result.filename}.`,
      });
    } catch (err) {
      let message = "Failed to compile quality report.";
      if (err instanceof AuthError || err instanceof ForbiddenError || err instanceof ApiError) {
        message = err.message;
      }
      setFeedback({
        type: "error",
        text: message,
      });
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="reports-page-wrapper">
      {feedback && (
        <div className={`alert-banner ${feedback.type === "success" ? "alert-success" : "alert-error"}`} style={{ marginBottom: "1.5rem" }}>
          {feedback.type === "success" ? <CheckIcon size={16} /> : <AlertIcon size={16} />}
          <span>{feedback.text}</span>
          <button type="button" className="alert-dismiss-btn" onClick={() => setFeedback(null)}>
            ×
          </button>
        </div>
      )}

      <div className="report-generator-card">
        <div className="report-card-icon-box">
          <ReportsIcon size={28} />
        </div>

        <div className="report-card-info">
          <h3 className="report-card-title">Production Quality & Inspection Audit Report</h3>
          <p className="report-card-desc">
            Compiles a complete plant-wide ledger of every visual inspection record. Includes raw image timestamps,
            AI classification categories, anomaly detection scores, bounding box metrics, defect area percentages,
            composite severity scores, and official factory supervisor review signatures.
          </p>

          <div className="report-metadata-badges">
            <span className="metadata-pill">Format: Comma-Separated Values (.csv)</span>
            <span className="metadata-pill">Encoding: UTF-8</span>
            <span className="metadata-pill">Target: Quality Assurance & ISO Audit</span>
          </div>

          <div className="report-action-row">
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleExport}
              disabled={exporting}
            >
              <ExportIcon size={16} />
              <span>{exporting ? "Compiling Report..." : "Generate & Download CSV"}</span>
            </button>
          </div>
        </div>
      </div>

      <div className="report-details-section">
        <h4 className="section-heading" style={{ fontSize: "1rem", marginBottom: "0.75rem" }}>
          Data Dictionary Included in Export
        </h4>
        <div className="data-dictionary-grid">
          <div className="dict-item">
            <span className="dict-field">Record Identifier & Filename</span>
            <span className="dict-desc">Original inspection image filename and unique plant ID</span>
          </div>
          <div className="dict-item">
            <span className="dict-field">Acquisition Telemetry</span>
            <span className="dict-desc">Uploading engineer name, email, and ISO timestamp</span>
          </div>
          <div className="dict-item">
            <span className="dict-field">AI ML Inference</span>
            <span className="dict-desc">ResNet18 category, defect class, confidence, and anomaly scores</span>
          </div>
          <div className="dict-item">
            <span className="dict-field">Defect Localization</span>
            <span className="dict-desc">U-Net area percentage and YOLO11n defect count</span>
          </div>
          <div className="dict-item">
            <span className="dict-field">Severity & Decision</span>
            <span className="dict-desc">Size, location, type weighted scores and quality verdict</span>
          </div>
          <div className="dict-item">
            <span className="dict-field">Supervisor Review</span>
            <span className="dict-desc">Sign-off decision, audit notes, reviewer identity, and timestamp</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Reports;
