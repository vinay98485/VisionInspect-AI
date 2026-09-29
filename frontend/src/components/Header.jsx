import { MenuIcon } from "./Icons";

function Header({ onOpenMobile, title, subtitle, actions }) {
  // Format current date in a clean industrial format
  const currentDate = new Date().toLocaleDateString(undefined, {
    weekday: "short",
    month: "short",
    day: "numeric",
  });

  return (
    <header className="app-top-header">
      <div className="header-left">
        {/* Mobile menu trigger */}
        <button
          type="button"
          className="header-mobile-trigger"
          onClick={onOpenMobile}
          aria-label="Open navigation menu"
        >
          <MenuIcon size={20} />
        </button>

        <div className="header-title-block">
          {title && <h1 className="header-title">{title}</h1>}
          {subtitle && <p className="header-subtitle">{subtitle}</p>}
        </div>
      </div>

      <div className="header-right">
        {actions && <div className="header-actions">{actions}</div>}

        <div className="header-context-pill">
          <span className="context-time">{currentDate}</span>
          <span className="context-divider">•</span>
          <span className="context-status">
            <span className="status-ping" />
            Active
          </span>
        </div>
      </div>
    </header>
  );
}

export default Header;
