import { useEffect, useState } from "react";
import { Link, useOutletContext } from "react-router-dom";
import {
  BarChart,
  Bar,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  getAnalyticsSummary,
  AuthError,
  ForbiddenError,
  ApiError,
} from "../../services/api";
import { RefreshIcon, AlertIcon, TrendsIcon } from "../../components/Icons";

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
const decisionColors = [COLORS.green, COLORS.red];

function AnalyticsOverview() {
  const { setHeaderConfig } = useOutletContext() || {};

  const [loading, setLoading] = useState(true);
  const [analytics, setAnalytics] = useState(null);
  const [error, setError] = useState(null);
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  useEffect(() => {
    setHeaderConfig?.({
      title: "Quality Analytics Overview",
      subtitle: "Categorical distributions and defect taxonomy across production",
      actions: (
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <Link to="/supervisor/analytics/trends" className="btn-header-secondary">
            <TrendsIcon size={14} />
            <span>View Trends</span>
          </Link>
          <button
            type="button"
            className="btn-header-action"
            onClick={() => setRefreshTrigger((prev) => prev + 1)}
          >
            <RefreshIcon size={14} />
            <span>Refresh</span>
          </button>
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
        } else if (err instanceof ForbiddenError) {
          setError(err.message);
        } else if (err instanceof ApiError) {
          setError(err.message);
        } else {
          setError("Failed to load analytics distributions.");
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
        <p className="section-loading-text">Synthesizing plant analytics distributions...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="notice-card error-notice">
        <div className="notice-icon"><AlertIcon size={24} /></div>
        <div className="notice-content">
          <h3 className="notice-title">Analytics Unavailable</h3>
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

  const decisionData = [
    { name: "Accepted", value: analytics?.accepted ?? 0 },
    { name: "Rejected", value: analytics?.rejected ?? 0 },
  ];

  const defectTypeData = (analytics?.defect_types ?? []).map((item) => ({
    name: item.defect_type,
    count: item.count,
  }));

  const categoryData = (analytics?.categories ?? []).map((item) => ({
    name: item.category,
    count: item.count,
  }));

  return (
    <div className="analytics-page-wrapper">
      {/* Visual Hierarchy: Top Row (Primary Metrics: Severity + Decision Ratio) */}
      <div className="chart-hero-grid">
        {/* Severity Distribution */}
        <div className="clean-chart-panel primary-chart-panel">
          <div className="chart-panel-header">
            <div>
              <h3 className="chart-panel-title">Defect Severity Breakdown</h3>
              <p className="chart-panel-caption">Count of inspections categorized by critical, high, medium, and low impact</p>
            </div>
          </div>
          <div className="chart-panel-body" style={{ height: "300px" }}>
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

        {/* Quality Decisions Donut */}
        <div className="clean-chart-panel secondary-chart-panel">
          <div className="chart-panel-header">
            <div>
              <h3 className="chart-panel-title">Disposition Ratio</h3>
              <p className="chart-panel-caption">Automated Accept vs Reject split</p>
            </div>
          </div>
          <div className="chart-panel-body" style={{ height: "300px" }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={decisionData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="45%"
                  outerRadius={90}
                  innerRadius={50}
                  paddingAngle={4}
                  label={({ name, value }) => `${name}: ${value}`}
                >
                  {decisionData.map((entry, index) => (
                    <Cell key={`dec-${index}`} fill={decisionColors[index]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: COLORS.tooltipBackground,
                    border: `1px solid ${COLORS.tooltipBorder}`,
                    borderRadius: "8px",
                    color: "#ffffff",
                  }}
                />
                <Legend wrapperStyle={{ color: COLORS.text, fontSize: "12px" }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Visual Hierarchy: Bottom Row (Taxonomy & Categories) */}
      <div className="chart-secondary-grid">
        {/* Defect Types */}
        <div className="clean-chart-panel">
          <div className="chart-panel-header">
            <div>
              <h3 className="chart-panel-title">Defect Type Taxonomy</h3>
              <p className="chart-panel-caption">Frequencies across specific defect morphologies</p>
            </div>
          </div>
          <div className="chart-panel-body" style={{ height: "320px" }}>
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

        {/* Product Categories */}
        <div className="clean-chart-panel">
          <div className="chart-panel-header">
            <div>
              <h3 className="chart-panel-title">Product Category Volume</h3>
              <p className="chart-panel-caption">Inspections aggregated across manufactured product lines</p>
            </div>
          </div>
          <div className="chart-panel-body" style={{ height: "320px" }}>
            {categoryData.length === 0 ? (
              <div className="empty-chart-fallback">No category data recorded.</div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={categoryData} margin={{ top: 10, right: 15, left: -10, bottom: 45 }}>
                  <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" />
                  <XAxis
                    dataKey="name"
                    angle={-35}
                    textAnchor="end"
                    interval={0}
                    tick={{ fill: COLORS.text, fontSize: 10 }}
                  />
                  <YAxis allowDecimals={false} tick={{ fill: COLORS.text, fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: COLORS.tooltipBackground,
                      border: `1px solid ${COLORS.tooltipBorder}`,
                      borderRadius: "8px",
                      color: "#ffffff",
                    }}
                  />
                  <Bar dataKey="count" name="Inspections" fill={COLORS.purple} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AnalyticsOverview;
