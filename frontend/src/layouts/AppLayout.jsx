import { useEffect, useState, useCallback } from "react";
import { Outlet, useNavigate } from "react-router-dom";
import Sidebar from "../components/Sidebar";
import Header from "../components/Header";
import { getCurrentUser, AuthError } from "../services/api";

function AppLayout() {
  const navigate = useNavigate();

  const [user, setUser] = useState(() => {
    try {
      const raw = localStorage.getItem("current_user");
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  });

  const [loading, setLoading] = useState(!user);
  const [collapsed, setCollapsed] = useState(() => {
    return localStorage.getItem("vi_sidebar_collapsed") === "true";
  });
  const [mobileOpen, setMobileOpen] = useState(false);

  // Dynamic header context that child routes can inject if desired
  const [headerConfig, setHeaderConfig] = useState({
    title: "",
    subtitle: "",
    actions: null,
  });

  const refreshUser = useCallback(async () => {
    try {
      const userData = await getCurrentUser();
      setUser(userData);
      return userData;
    } catch (err) {
      if (err instanceof AuthError) {
        navigate("/login", { replace: true });
      }
      return null;
    }
  }, [navigate]);

  useEffect(() => {
    let active = true;

    const initAuth = async () => {
      try {
        const userData = await getCurrentUser();
        if (active) {
          setUser(userData);
        }
      } catch (err) {
        if (active && err instanceof AuthError) {
          navigate("/login", { replace: true });
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    initAuth();

    return () => {
      active = false;
    };
  }, [navigate]);

  const handleToggleCollapse = () => {
    setCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem("vi_sidebar_collapsed", String(next));
      return next;
    });
  };

  if (loading) {
    return (
      <div className="app-loading-screen">
        <div className="app-loading-spinner" />
        <p className="app-loading-text">Connecting to VisionInspect Console...</p>
      </div>
    );
  }

  return (
    <div className={`app-shell-layout ${collapsed ? "sidebar-collapsed" : ""}`}>
      <Sidebar
        user={user}
        collapsed={collapsed}
        onToggleCollapse={handleToggleCollapse}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      <div className="app-main-viewport">
        <Header
          user={user}
          onOpenMobile={() => setMobileOpen(true)}
          title={headerConfig.title}
          subtitle={headerConfig.subtitle}
          actions={headerConfig.actions}
        />

        <main className="app-content-body">
          <Outlet context={{ user, refreshUser, setHeaderConfig }} />
        </main>
      </div>
    </div>
  );
}

export default AppLayout;
