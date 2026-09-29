import { NavLink, useNavigate } from "react-router-dom";
import {
  DashboardIcon,
  QueueIcon,
  InspectionsIcon,
  UploadIcon,
  AnalyticsIcon,
  TrendsIcon,
  ReportsIcon,
  SettingsIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CloseIcon,
  LogOutIcon,
} from "./Icons";
import { logout } from "../services/api";

function Sidebar({ user, collapsed, onToggleCollapse, mobileOpen, onCloseMobile }) {
  const navigate = useNavigate();

  const isSupervisor = user?.role_id === 2;

  const handleSignOut = () => {
    logout();
    navigate("/login");
  };

  const supervisorNavItems = [
    {
      label: "Dashboard",
      to: "/supervisor/dashboard",
      icon: DashboardIcon,
    },
    {
      label: "Review Queue",
      to: "/supervisor/reviews",
      icon: QueueIcon,
    },
    {
      section: "Analytics",
    },
    {
      label: "Overview",
      to: "/supervisor/analytics",
      icon: AnalyticsIcon,
    },
    {
      label: "Trends",
      to: "/supervisor/analytics/trends",
      icon: TrendsIcon,
    },
    {
      section: "Management",
    },
    {
      label: "Reports",
      to: "/supervisor/reports",
      icon: ReportsIcon,
    },
    {
      label: "Settings",
      to: "/supervisor/settings",
      icon: SettingsIcon,
    },
  ];

  const qeNavItems = [
    {
      label: "Dashboard",
      to: "/dashboard",
      icon: DashboardIcon,
    },
    {
      label: "Inspections",
      to: "/inspections",
      icon: InspectionsIcon,
    },
    {
      label: "New Inspection",
      to: "/upload",
      icon: UploadIcon,
      accent: true,
    },
    {
      section: "Insights",
    },
    {
      label: "Analytics",
      to: "/analytics",
      icon: AnalyticsIcon,
    },
    {
      label: "Settings",
      to: "/settings",
      icon: SettingsIcon,
    },
  ];

  const navItems = isSupervisor ? supervisorNavItems : qeNavItems;

  return (
    <>
      {/* Mobile Backdrop */}
      {mobileOpen && (
        <div
          className="sidebar-backdrop"
          onClick={onCloseMobile}
          aria-hidden="true"
        />
      )}

      <aside
        className={`app-sidebar ${collapsed ? "collapsed" : ""} ${mobileOpen ? "mobile-open" : ""}`}
        aria-label="Primary Navigation"
      >
        {/* Brand Header */}
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <div className="brand-logo-mark">VI</div>
            {!collapsed && (
              <div className="brand-text-block">
                <span className="brand-title">VisionInspect</span>
                <span className="brand-badge-ai">AI</span>
              </div>
            )}
          </div>

          {/* Desktop Collapse Toggle */}
          <button
            type="button"
            className="sidebar-toggle-btn desktop-only"
            onClick={onToggleCollapse}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <ChevronRightIcon size={16} /> : <ChevronLeftIcon size={16} />}
          </button>

          {/* Mobile Close Button */}
          <button
            type="button"
            className="sidebar-toggle-btn mobile-only"
            onClick={onCloseMobile}
            aria-label="Close navigation"
          >
            <CloseIcon size={18} />
          </button>
        </div>

        {/* Role Indicator */}
        {!collapsed && (
          <div className="sidebar-role-indicator">
            <span className="role-dot" />
            <span className="role-label">
              {isSupervisor ? "Supervisor Console" : "Quality Engineer"}
            </span>
          </div>
        )}

        {/* Navigation List */}
        <nav className="sidebar-nav">
          {navItems.map((item, idx) => {
            if (item.section) {
              if (collapsed) return null;
              return (
                <div key={`sec-${idx}`} className="nav-section-title">
                  {item.section}
                </div>
              );
            }

            const Icon = item.icon;

            return (
              <NavLink
                key={item.to}
                to={item.to}
                onClick={onCloseMobile}
                className={({ isActive: navActive }) =>
                  `sidebar-nav-item ${navActive ? "active" : ""} ${item.accent ? "accent" : ""}`
                }
                title={collapsed ? item.label : undefined}
              >
                <span className="nav-icon-wrapper">
                  <Icon size={18} />
                </span>
                {!collapsed && <span className="nav-label">{item.label}</span>}
              </NavLink>
            );
          })}
        </nav>

        {/* User Profile / Footer */}
        <div className="sidebar-footer">
          <div className="sidebar-user-card" title={user?.name || user?.email}>
            <div className="user-avatar-circle">
              {(user?.name || user?.email || "U").slice(0, 1).toUpperCase()}
            </div>
            {!collapsed && (
              <div className="user-info-text">
                <span className="user-display-name">{user?.name || "Operator"}</span>
                <span className="user-email-text">{user?.email || ""}</span>
              </div>
            )}
            <button
              type="button"
              className="user-signout-btn"
              onClick={handleSignOut}
              title="Sign out of console"
              aria-label="Sign out"
            >
              <LogOutIcon size={16} />
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}

export default Sidebar;
