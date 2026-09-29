import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

// Layout
import AppLayout from "./layouts/AppLayout";

// Auth Pages
import Login from "./pages/Login";
import Register from "./pages/Register";

// Quality Engineer Pages
import QEDashboard from "./pages/qe/QEDashboard";
import Inspections from "./pages/qe/Inspections";
import QEAnalytics from "./pages/qe/QEAnalytics";

// Supervisor Pages
import SupervisorDashboard from "./pages/supervisor/SupervisorDashboard";
import ReviewQueue from "./pages/supervisor/ReviewQueue";
import AnalyticsOverview from "./pages/supervisor/AnalyticsOverview";
import AnalyticsTrends from "./pages/supervisor/AnalyticsTrends";
import Reports from "./pages/supervisor/Reports";

// Shared Pages
import Upload from "./pages/Upload";
import ImageDetails from "./pages/ImageDetails";
import Settings from "./pages/Settings";

function getIsSupervisor() {
  try {
    const raw = localStorage.getItem("current_user");
    const user = raw ? JSON.parse(raw) : null;
    return user?.role_id === 2;
  } catch {
    return false;
  }
}

// Smart root redirect based on stored user session
function RootRedirect() {
  const token = localStorage.getItem("access_token");
  if (!token) {
    return <Navigate to="/login" replace />;
  }

  if (getIsSupervisor()) {
    return <Navigate to="/supervisor/dashboard" replace />;
  }

  return <Navigate to="/dashboard" replace />;
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public Authentication Routes */}
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* Persistent Shell Layout with Sidebar & Header */}
        <Route element={<AppLayout />}>
          {/* Quality Engineer Primary Routes */}
          <Route path="/dashboard" element={<QEDashboard />} />
          <Route path="/inspections" element={<Inspections />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/analytics" element={<QEAnalytics />} />
          <Route path="/settings" element={<Settings />} />

          {/* Factory Supervisor Primary Routes */}
          <Route path="/supervisor/dashboard" element={<SupervisorDashboard />} />
          <Route path="/supervisor/reviews" element={<ReviewQueue />} />
          <Route path="/supervisor/analytics" element={<AnalyticsOverview />} />
          <Route path="/supervisor/analytics/trends" element={<AnalyticsTrends />} />
          <Route path="/supervisor/reports" element={<Reports />} />
          <Route path="/supervisor/settings" element={<Settings />} />

          {/* Shared Specimen Detail Route */}
          <Route path="/images/:imageId" element={<ImageDetails />} />
        </Route>

        {/* Global Fallbacks */}
        <Route path="/" element={<RootRedirect />} />
        <Route path="*" element={<RootRedirect />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
