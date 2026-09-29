import { useEffect, useState } from "react";
import { Link, useOutletContext } from "react-router-dom";
import {
  BarChart,
  Bar,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getAnalyticsSummary, AuthError, ApiError } from "../../services/api";
import { RefreshIcon, AlertIcon, UploadIcon } from "../../components/Icons";

const COLORS = {
  cyan: "#38bdf8",
  green: "#34d399",
  red: "#f87171",
  amber: "#fbbf24",
  purple: "#a78bfa",
  text: "#94a3b8",
  grid: "rgba(255, 255, 255, 0.06)",
  tooltipBackground: "#161b26",
  tooltipBorder: "rgba(255, 255, 255, 0.12)",
};

const severityColors = [COLORS.red, COLORS.amber, COLORS.purple, COLORS.green];

function QEAnalytics() {
  const { setHeaderConfig } = useOutletContext() || {};

  const [loading, setLoading] = useState(true);
  const [analytics, setAnalytics] = useState(null);
  const [error, setError] = useState(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  useEffect(() => {
    setHeaderConfig?.({
      title: "Quality Insights",
      subtitle: "Inspection cell performance, defect frequencies, and confidence distributions",
      actions: (
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button
            type="button"
            className="btn-header-action"
            onClick={() => setRefreshTrigger((prev) => prev + 1)}
          >
            <RefreshIcon size={14} />
            <span>Refresh</span>
          </button>
          <Link to="/upload" className="btn btn-primary btn-sm">
            <UploadIcon size={14} />
            <span>+ Inspect</span>
          </Link>
        </div>
      ),
    });
  }, [setHeaderConfig]);

  useEffect(() => {
    let active = true;

    const fetchSummary = async () => {
      setLoading(true);
      setError(null);

      try {
        const data = await getAnalyticsSummary();
        if (active) {
          setAnalytics(data);
        }
      } catch (err) {
        if (!active) return;
        if (err instanceof AuthError) {
          setError(err.message);
        } else if (err instanceof ApiError) {
          setError(err.message);
        } else {
          setError("Failed to load quality insights.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    fetchSummary();

    return () => {
      active = false;
    };
  }, [refreshTrigger]);

  if (loading) {
    return (
      <div className="section-loading-container">
        <div className="app-loading-spinner" />
        <p className="section-loading-text">Loading inspection insights...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="notice-card error-notice">
        <div className="notice-icon"><AlertIcon size={24} /></div>
        <div className="notice-content">
          <h3 className="notice-title">Unavailable</h3>
          <p className="notice-desc">{error}</p>
        </div>
      </div>
    );
  }

  const severityData = [
    { name: "Critical", value: analytics?.severity?.critical ?? 0 },
    { name: "High", value: analytics?.severity?.high ?? 0 },
    { name: "Medium", value: analytics?.severity?.medium ?? 0 },
    { name: "Low", value: analytics?.severity?.low ?? 0 },
  ];

  const defectTypeData = (analytics?.defect_types ?? []).map((item) => ({
    name: item.defect_type,
    count: item.count,
  }));

  const accepted = analytics?.accepted ?? 0;
  const rejected = analytics?.rejected ?? 0;
  const total = accepted + rejected;
  const passRate = total > 0 ? ((accepted / total) * 100).toFixed(1) : "100.0";

  return (
    <div className="analytics-page-wrapper">
      {/* 1. Quality Telemetry Strip */}
      <section className="metrics-strip">
        <div className="metric-cell">
          <span className="metric-caption">First Pass Yield</span>
          <div className="metric-value-row">
            <span className="metric-numeral success-color">{passRate}%</span>
            <span className="metric-badge-neutral">{accepted} / {total}</span>
          </div>
        </div>

        <div className="metric-cell">
          <span className="metric-caption">Avg AI Confidence</span>
          <div className="metric-value-row">
            <span className="metric-numeral">{Number(analytics?.average?.confidence ?? 0).toFixed(1)}%</span>
            <span className="metric-sub-label">Consensus</span>
          </div>
        </div>

        <div className="metric-cell">
          <span className="metric-caption">Average Defect Severity</span>
          <div className="metric-value-row">
            <span className="metric-numeral">{Number(analytics?.average?.severity ?? 0).toFixed(2)}</span>
            <span className="metric-sub-label">0.0 to 1.0</span>
          </div>
        </div>

        <div className="metric-cell">
          <span className="metric-caption">Total Inspections</span>
          <div className="metric-value-row">
            <span className="metric-numeral">{analytics?.total_inspections ?? 0}</span>
            <span className="metric-sub-label">Logged</span>
          </div>
        </div>
      </section>

      {/* 2. Charts Grid */}
      <div className="chart-hero-grid">
        <div className="clean-chart-panel primary-chart-panel">
          <div className="chart-panel-header">
            <div>
              <h3 className="chart-panel-title">Severity Distribution</h3>
              <p className="chart-panel-caption">Part count across critical, high, medium, and low defect thresholds</p>
            </div>
          </div>
          <div className="chart-panel-body" style={{ height: "280px" }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={severityData} margin={{ top: 15, right: 15, left: -10, bottom: 5 }}>
                <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" />
                <XAxis dataKey="name" tick={{ fill: COLORS.text, fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fill: COLORS.text, fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: COLORS.tooltipBackground,
                    border: `1px solid ${COLORS.tooltipBorder}`,
                    borderRadius: "8px",
                    color: "#ffffff",
                  }}
                />
                <Bar dataKey="value" name="Inspections" radius={[4, 4, 0, 0]}>
                  {severityData.map((entry, index) => (
                    <Cell key={`sev-${index}`} fill={severityColors[index]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="clean-chart-panel secondary-chart-panel">
          <div className="chart-panel-header">
            <div>
              <h3 className="chart-panel-title">Common Defect Types</h3>
              <p className="chart-panel-caption">Top recurring physical defect classifications</p>
            </div>
          </div>
          <div className="chart-panel-body" style={{ height: "280px" }}>
            {defectTypeData.length === 0 ? (
              <div className="empty-chart-fallback">No defect taxonomy data available.</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={defectTypeData} layout="vertical" margin={{ top: 10, right: 20, left: 20, bottom: 5 }}>
                  <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" />
                  <XAxis type="number" allowDecimals={false} tick={{ fill: COLORS.text, fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" width={90} tick={{ fill: COLORS.text, fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: COLORS.tooltipBackground,
                      border: `1px solid ${COLORS.tooltipBorder}`,
                      borderRadius: "8px",
                      color: "#ffffff",
                    }}
                  />
                  <Bar dataKey="count" name="Count" fill={COLORS.cyan} radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default QEAnalytics;
