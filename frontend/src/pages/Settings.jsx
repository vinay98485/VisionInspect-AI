import { useEffect } from "react";
import { useNavigate, useOutletContext } from "react-router-dom";
import { logout } from "../services/api";
import { UserIcon, LogOutIcon, CheckIcon } from "../components/Icons";

function Settings() {
  const navigate = useNavigate();
  const { user, setHeaderConfig } = useOutletContext() || {};

  useEffect(() => {
    setHeaderConfig?.({
      title: "Console Settings",
      subtitle: "Operator profile, role credentials, and AI architecture telemetry",
      actions: null,
    });
  }, [setHeaderConfig]);

  const handleSignOut = () => {
    logout();
    navigate("/login");
  };

  const isSupervisor = user?.role_id === 2;

  return (
    <div className="settings-page-wrapper">
      {/* 1. User Profile Card */}
      <div className="settings-section-card">
        <div className="settings-card-header">
          <div className="settings-icon-bubble">
            <UserIcon size={20} />
          </div>
          <div>
            <h3 className="settings-card-title">Operator Profile</h3>
            <p className="settings-card-subtitle">Active credentials for this terminal session</p>
          </div>
        </div>

        <div className="settings-grid">
          <div className="settings-field">
            <span className="settings-field-label">Full Name</span>
            <span className="settings-field-value">{user?.name || "Operator"}</span>
          </div>

          <div className="settings-field">
            <span className="settings-field-label">Work Email</span>
            <span className="settings-field-value">{user?.email || "—"}</span>
          </div>

          <div className="settings-field">
            <span className="settings-field-label">Console Role</span>
            <span className="settings-field-value">
              <span className={`status-pill ${isSupervisor ? "status-warning" : "status-healthy"}`}>
                {isSupervisor ? "Factory Supervisor" : "Quality Engineer"}
              </span>
            </span>
          </div>

          <div className="settings-field">
            <span className="settings-field-label">User Identifier</span>
            <span className="settings-field-value mono">UID #{user?.id || user?.user_id || "1"}</span>
          </div>
        </div>

        <div className="settings-action-footer">
          <button type="button" className="btn btn-secondary btn-sm" onClick={handleSignOut}>
            <LogOutIcon size={14} />
            <span>Sign Out of Terminal</span>
          </button>
        </div>
      </div>

      {/* 2. AI Architecture Telemetry */}
      <div className="settings-section-card">
        <div className="settings-card-header">
          <div>
            <h3 className="settings-card-title">AI Inspection System Architecture</h3>
            <p className="settings-card-subtitle">Verified multi-stage manufacturing intelligence stack</p>
          </div>
        </div>

        <div className="arch-specs-list">
          <div className="arch-spec-row">
            <div className="arch-spec-info">
              <span className="arch-spec-name">Product Category Classification</span>
              <span className="arch-spec-desc">ResNet18 Deep Neural Network (15 Industrial MVTec classes)</span>
            </div>
            <span className="status-pill status-healthy"><CheckIcon size={12} /> Active</span>
          </div>

          <div className="arch-spec-row">
            <div className="arch-spec-info">
              <span className="arch-spec-name">Anomaly Feature Extraction</span>
              <span className="arch-spec-desc">ResNet18 Layer3 Mid-Level Feature Map Anomaly Scoring</span>
            </div>
            <span className="status-pill status-healthy"><CheckIcon size={12} /> Active</span>
          </div>

          <div className="arch-spec-row">
            <div className="arch-spec-info">
              <span className="arch-spec-name">Defect Object Detection</span>
              <span className="arch-spec-desc">YOLO11n Gated Object Detector (conf=0.25, imgsz=640)</span>
            </div>
            <span className="status-pill status-healthy"><CheckIcon size={12} /> Active</span>
          </div>

          <div className="arch-spec-row">
            <div className="arch-spec-info">
              <span className="arch-spec-name">Pixel-Level Defect Segmentation</span>
              <span className="arch-spec-desc">U-Net Architecture (512x512 resolution contour extraction)</span>
            </div>
            <span className="status-pill status-healthy"><CheckIcon size={12} /> Active</span>
          </div>

          <div className="arch-spec-row">
            <div className="arch-spec-info">
              <span className="arch-spec-name">Decision Fusion & Severity Scoring</span>
              <span className="arch-spec-desc">Automated Rule Engine (Size + Location + Type weighted composite)</span>
            </div>
            <span className="status-pill status-healthy"><CheckIcon size={12} /> Active</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Settings;
