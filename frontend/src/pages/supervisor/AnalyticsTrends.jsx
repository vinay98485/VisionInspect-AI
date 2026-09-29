import { useEffect, useState } from "react";
import { Link, useOutletContext } from "react-router-dom";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getAnalyticsTrends, AuthError, ApiError } from "../../services/api";
import { AlertIcon, AnalyticsIcon } from "../../components/Icons";

const COLORS = {
  cyan: "#38bdf8",
  green: "#34d399",
  red: "#f87171",
  amber: "#fbbf24",
  text: "#94a3b8",
  grid: "rgba(255, 255, 255, 0.06)",
  tooltipBackground: "#161b26",
  tooltipBorder: "rgba(255, 255, 255, 0.12)",
};

function AnalyticsTrends() {
  const { setHeaderConfig } = useOutletContext() || {};

  const [trendRange, setTrendRange] = useState("30");
  const [trends, setTrends] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setHeaderConfig?.({
      title: "Production Trends",
      subtitle: "Temporal analysis of inspection volumes, defect rates, and line quality",
      actions: (
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <Link to="/supervisor/analytics" className="btn-header-secondary">
            <AnalyticsIcon size={14} />
            <span>Overview</span>
          </Link>
        </div>
      ),
    });
  }, [setHeaderConfig]);

  useEffect(() => {
    let active = true;

    const fetchTrends = async () => {
      setLoading(true);
      setError(null);

      try {
        const data = await getAnalyticsTrends(trendRange);
        if (active) {
          setTrends(data);
        }
      } catch (err) {
        if (!active) return;
        if (err instanceof AuthError) {
          setError(err.message);
        } else if (err instanceof ApiError) {
          setError(err.message);
        } else {
          setError("Failed to fetch historical trends.");
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    };

    fetchTrends();

    return () => {
      active = false;
    };
  }, [trendRange]);

  return (
    <div className="trends-page-wrapper">
      {/* Time Window Filter Toolbar */}
      <div className="trends-toolbar">
        <div className="trends-toolbar-title-group">
          <span className="trends-toolbar-label">Observation Window:</span>
        </div>

        <div className="segment-control subtle-segment" role="tablist">
          <button
            type="button"
            className={`segment-btn ${trendRange === "7" ? "active" : ""}`}
            onClick={() => setTrendRange("7")}
            disabled={loading}
          >
            Past 7 Days
          </button>
          <button
            type="button"
            className={`segment-btn ${trendRange === "30" ? "active" : ""}`}
            onClick={() => setTrendRange("30")}
            disabled={loading}
          >
            Past 30 Days
          </button>
          <button
            type="button"
            className={`segment-btn ${trendRange === "all" ? "active" : ""}`}
            onClick={() => setTrendRange("all")}
            disabled={loading}
          >
            All Time
          </button>
        </div>
      </div>

      {error && (
        <div className="alert-banner alert-error" style={{ marginBottom: "1.5rem" }}>
          <AlertIcon size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* 4 Line Charts Grid */}
      <div className="trends-charts-grid">
        {/* 1. Inspection Volume */}
        <div className="clean-chart-panel">
          <div className="chart-panel-header">
            <div>
              <h3 className="chart-panel-title">Inspection Throughput</h3>
              <p className="chart-panel-caption">Daily total inspection volume processed</p>
            </div>
          </div>
          <div className="chart-panel-body" style={{ height: "260px" }}>
            {loading ? (
              <div className="chart-loading-state">
                <div className="app-loading-spinner" />
              </div>
            ) : trends.length === 0 ? (
              <div className="empty-chart-fallback">No throughput records in this timeframe.</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trends} margin={{ top: 15, right: 20, left: -15, bottom: 5 }}>
                  <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: COLORS.text, fontSize: 11 }}
                    tickFormatter={(d) => (d && d.length >= 10 ? d.slice(5) : d)}
                  />
                  <YAxis tick={{ fill: COLORS.text, fontSize: 11 }} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: COLORS.tooltipBackground,
                      borderColor: COLORS.tooltipBorder,
                      color: "#fff",
                      borderRadius: "6px",
                    }}
                    formatter={(val) => [val, "Total Inspections"]}
                    labelFormatter={(label) => `Date: ${label}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="total_inspections"
                    stroke={COLORS.cyan}
                    strokeWidth={2}
                    dot={{ r: 2.5, fill: COLORS.cyan }}
                    activeDot={{ r: 4.5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* 2. Defect Rate */}
        <div className="clean-chart-panel">
          <div className="chart-panel-header">
            <div>
              <h3 className="chart-panel-title">Defect Rate (%)</h3>
              <p className="chart-panel-caption">Percentage of inspections classified as defective</p>
            </div>
          </div>
          <div className="chart-panel-body" style={{ height: "260px" }}>
            {loading ? (
              <div className="chart-loading-state">
                <div className="app-loading-spinner" />
              </div>
            ) : trends.length === 0 ? (
              <div className="empty-chart-fallback">No defect rate records in this timeframe.</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trends} margin={{ top: 15, right: 20, left: -15, bottom: 5 }}>
                  <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: COLORS.text, fontSize: 11 }}
                    tickFormatter={(d) => (d && d.length >= 10 ? d.slice(5) : d)}
                  />
                  <YAxis tick={{ fill: COLORS.text, fontSize: 11 }} unit="%" domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: COLORS.tooltipBackground,
                      borderColor: COLORS.tooltipBorder,
                      color: "#fff",
                      borderRadius: "6px",
                    }}
                    formatter={(val) => [`${val}%`, "Defect Rate"]}
                    labelFormatter={(label) => `Date: ${label}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="defect_rate"
                    stroke={COLORS.red}
                    strokeWidth={2}
                    dot={{ r: 2.5, fill: COLORS.red }}
                    activeDot={{ r: 4.5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* 3. Average Severity */}
        <div className="clean-chart-panel">
          <div className="chart-panel-header">
            <div>
              <h3 className="chart-panel-title">Mean Severity Trend</h3>
              <p className="chart-panel-caption">Aggregate defect severity score progression</p>
            </div>
          </div>
          <div className="chart-panel-body" style={{ height: "260px" }}>
            {loading ? (
              <div className="chart-loading-state">
                <div className="app-loading-spinner" />
              </div>
            ) : trends.length === 0 ? (
              <div className="empty-chart-fallback">No severity telemetry in this timeframe.</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trends} margin={{ top: 15, right: 20, left: -15, bottom: 5 }}>
                  <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: COLORS.text, fontSize: 11 }}
                    tickFormatter={(d) => (d && d.length >= 10 ? d.slice(5) : d)}
                  />
                  <YAxis tick={{ fill: COLORS.text, fontSize: 11 }} domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: COLORS.tooltipBackground,
                      borderColor: COLORS.tooltipBorder,
                      color: "#fff",
                      borderRadius: "6px",
                    }}
                    formatter={(val) => [Number(val).toFixed(2), "Avg Severity"]}
                    labelFormatter={(label) => `Date: ${label}`}
                  />
                  <Line
                    type="monotone"
                    dataKey="avg_severity"
                    stroke={COLORS.amber}
                    strokeWidth={2}
                    dot={{ r: 2.5, fill: COLORS.amber }}
                    activeDot={{ r: 4.5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* 4. Accept vs Reject */}
        <div className="clean-chart-panel">
          <div className="chart-panel-header">
            <div>
              <h3 className="chart-panel-title">Acceptance vs Rejection</h3>
              <p className="chart-panel-caption">Dual-stream comparison of approved vs defect batches</p>
            </div>
          </div>
          <div className="chart-panel-body" style={{ height: "260px" }}>
            {loading ? (
              <div className="chart-loading-state">
                <div className="app-loading-spinner" />
              </div>
            ) : trends.length === 0 ? (
              <div className="empty-chart-fallback">No disposition metrics in this timeframe.</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={trends} margin={{ top: 15, right: 20, left: -15, bottom: 5 }}>
                  <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tick={{ fill: COLORS.text, fontSize: 11 }}
                    tickFormatter={(d) => (d && d.length >= 10 ? d.slice(5) : d)}
                  />
                  <YAxis tick={{ fill: COLORS.text, fontSize: 11 }} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: COLORS.tooltipBackground,
                      borderColor: COLORS.tooltipBorder,
                      color: "#fff",
                      borderRadius: "6px",
                    }}
                    labelFormatter={(label) => `Date: ${label}`}
                  />
                  <Legend wrapperStyle={{ color: COLORS.text, fontSize: "11px", paddingTop: "4px" }} />
                  <Line
                    type="monotone"
                    dataKey="accepted"
                    name="Accepted"
                    stroke={COLORS.green}
                    strokeWidth={2}
                    dot={{ r: 2.5, fill: COLORS.green }}
                  />
                  <Line
                    type="monotone"
                    dataKey="rejected"
                    name="Rejected"
                    stroke={COLORS.red}
                    strokeWidth={2}
                    dot={{ r: 2.5, fill: COLORS.red }}
                  />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AnalyticsTrends;
