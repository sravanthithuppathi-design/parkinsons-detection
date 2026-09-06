import React, { useState, useEffect, useCallback } from "react";
import {
  BarChart2, TrendingUp, Activity, Brain,
  RefreshCw, ShieldAlert, ShieldCheck, Database, Zap,
  FlaskConical, Loader2,
} from "lucide-react";
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { getLocalHistory, getStats, seedDemoData } from "../api/client.js";
import toast from "react-hot-toast";

const TIP = {
  contentStyle: {
    background:   "rgba(15,23,42,0.97)",
    border:       "1px solid rgba(255,255,255,0.1)",
    borderRadius: "14px",
    color:        "#f8fafc",
    fontSize:     12,
    padding:      "10px 14px",
    boxShadow:    "0 8px 32px rgba(0,0,0,0.5)",
  },
  cursor: { fill: "rgba(255,255,255,0.03)" },
};
const TICK = { fill: "rgba(255,255,255,0.3)", fontSize: 10 };

export default function DashboardPage() {
  const [stats,   setStats]   = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [seeding, setSeeding] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    // Always load history from localStorage — instant, no network needed
    const localData = getLocalHistory();
    setHistory(localData);

    // Try to load backend stats for any server-side metrics (best-effort)
    try {
      const s = await getStats();
      setStats(s);
    } catch {
      // Silently ignore — stats are optional, charts are computed from localStorage
      setStats(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // ── Derived chart data from localStorage ──────────────────────────────
  const pdCount      = history.filter((h) => h.prediction?.includes("Parkinson")).length;
  const healthyCount = history.length - pdCount;

  const pieData = [
    { name: "PD Detected", value: pdCount,      color: "#f43f5e" },
    { name: "Healthy",     value: healthyCount, color: "#10b981" },
  ];

  const confBuckets = {
    "0–50": 0, "50–60": 0, "60–70": 0,
    "70–80": 0, "80–90": 0, "90–100": 0,
  };
  history.forEach((h) => {
    const c = h.confidence || 0;
    if      (c < 50)  confBuckets["0–50"]++;
    else if (c < 60)  confBuckets["50–60"]++;
    else if (c < 70)  confBuckets["60–70"]++;
    else if (c < 80)  confBuckets["70–80"]++;
    else if (c < 90)  confBuckets["80–90"]++;
    else              confBuckets["90–100"]++;
  });
  const confData = Object.entries(confBuckets).map(([range, count]) => ({
    range,
    count,
  }));

  const timelineData = [...history]
    .reverse()
    .slice(-12)
    .map((h, i) => ({
      n:          i + 1,
      confidence: h.confidence || 0,
      isPD:       h.prediction?.includes("Parkinson") ? 1 : 0,
    }));

  // Feature averages — prefer backend stats, fall back to computing from localStorage
  const featureAvgData = (() => {
    // Use backend stats if available
    if (stats?.feature_averages) {
      return Object.entries(stats.feature_averages)
        .slice(0, 8)
        .map(([name, val]) => ({
          name:  name.replace(/_/g, " ").replace(/mean$/, "").trim(),
          value: parseFloat(val.toFixed(4)),
        }));
    }
    // Compute from localStorage history
    if (history.length === 0) return [];
    const keys = [
      "jitter_local", "shimmer_local", "hnr",
      "pitch_mean", "zcr_mean", "rpde",
    ];
    return keys
      .map((key) => {
        const vals = history
          .map((h) => h.features?.[key])
          .filter((v) => v !== undefined && v !== null);
        if (vals.length === 0) return null;
        const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
        return {
          name:  key.replace(/_/g, " ").replace(/mean$/, "").trim(),
          value: parseFloat(avg.toFixed(4)),
        };
      })
      .filter(Boolean);
  })();

  const avgConf = history.length
    ? history.reduce((s, h) => s + (h.confidence || 0), 0) / history.length
    : 0;

  const summaryCards = [
    {
      label: "Total Analyses",
      value: history.length,
      icon:  Database,
      color: "text-violet-400",
      glow:  "from-violet-600/20 to-cyan-600/20",
      border: "border-violet-500/20",
    },
    {
      label: "PD Detected",
      value: pdCount,
      icon:  ShieldAlert,
      color: "text-rose-400",
      glow:  "from-rose-600/20 to-orange-600/20",
      border: "border-rose-500/20",
    },
    {
      label: "Healthy",
      value: healthyCount,
      icon:  ShieldCheck,
      color: "text-emerald-400",
      glow:  "from-emerald-600/20 to-cyan-600/20",
      border: "border-emerald-500/20",
    },
    {
      label: "Avg Confidence",
      value: history.length ? `${avgConf.toFixed(1)}%` : "—",
      icon:  TrendingUp,
      color: "text-amber-400",
      glow:  "from-amber-600/20 to-orange-600/20",
      border: "border-amber-500/20",
    },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12">
      {/* Header */}
      <div className="flex items-center justify-between mb-10 animate-fade-up">
        <div>
          <p className="label mb-2">Analytics Overview</p>
          <h1 className="text-4xl font-black font-display text-white flex items-center gap-4">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-violet-600/30 to-cyan-600/30 border border-violet-500/20 flex items-center justify-center">
              <BarChart2 className="w-6 h-6 text-violet-400" />
            </div>
            Dashboard
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={async () => {
              setSeeding(true);
              try {
                const data = await seedDemoData();
                toast.success(`${data.seeded} demo analyses loaded!`);
                setHistory(getLocalHistory());
                const s = await getStats();
                setStats(s);
              } catch {
                toast.error("Failed to load demo data. Is the server running?");
              } finally {
                setSeeding(false);
              }
            }}
            disabled={seeding}
            className="btn-secondary-glass flex items-center gap-2 text-sm px-4 py-2.5"
          >
            {seeding ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> Loading...</>
            ) : (
              <><FlaskConical className="w-4 h-4" /> Load Demo Data</>
            )}
          </button>
          <button
            onClick={load}
            disabled={loading}
            className="btn-secondary-glass flex items-center gap-2 text-sm px-4 py-2.5"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8 animate-fade-up delay-100">
        {summaryCards.map(({ label, value, icon: Icon, color, glow, border }, i) => (
          <div
            key={label}
            className={`glass-dark rounded-2xl p-6 bg-gradient-to-br ${glow} border ${border} animate-fade-up`}
            style={{ animationDelay: `${i * 80}ms` }}
          >
            <div className="flex items-center justify-between mb-4">
              <p className="label">{label}</p>
              <div
                className={`w-9 h-9 rounded-xl glass flex items-center justify-center border ${border}`}
              >
                <Icon className={`w-4 h-4 ${color}`} />
              </div>
            </div>
            <p className={`text-4xl font-black font-display ${color}`}>{value}</p>
          </div>
        ))}
      </div>

      {/* Empty state */}
      {!loading && history.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-24 gap-5 glass-dark rounded-3xl animate-fade-up">
          <BarChart2 className="w-16 h-16 text-white/10" />
          <div className="text-center">
            <p className="text-white/40 font-bold text-lg">No data yet</p>
            <p className="text-sm text-white/20 mt-1">
              Run some analyses on the Analyze tab.
            </p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-up delay-200">

          {/* Pie — Result Distribution */}
          <div className="glass-dark rounded-3xl p-6">
            <h3 className="font-bold text-white mb-5 flex items-center gap-2.5">
              <Activity className="w-4 h-4 text-violet-400" />
              Result Distribution
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={65}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((e) => (
                    <Cell key={e.name} fill={e.color} stroke="transparent" />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={TIP.contentStyle}
                  formatter={(v, n) => [`${v} analyses`, n]}
                />
                <Legend
                  formatter={(v) => (
                    <span style={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}>
                      {v}
                    </span>
                  )}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Bar — Confidence Distribution */}
          <div className="glass-dark rounded-3xl p-6">
            <h3 className="font-bold text-white mb-5 flex items-center gap-2.5">
              <TrendingUp className="w-4 h-4 text-cyan-400" />
              Confidence Distribution
            </h3>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={confData} barSize={28}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(255,255,255,0.04)"
                  vertical={false}
                />
                <XAxis dataKey="range" tick={TICK} />
                <YAxis tick={TICK} allowDecimals={false} />
                <Tooltip
                  contentStyle={TIP.contentStyle}
                  cursor={TIP.cursor}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]} name="Analyses">
                  {confData.map((_, i) => (
                    <Cell
                      key={i}
                      fill={`hsl(${250 - i * 12},70%,60%)`}
                      opacity={0.85}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Line — Confidence Timeline */}
          <div className="glass-dark rounded-3xl p-6 lg:col-span-2">
            <h3 className="font-bold text-white mb-5 flex items-center gap-2.5">
              <Brain className="w-4 h-4 text-violet-400" />
              Confidence Trend (Last {Math.min(12, history.length)})
            </h3>
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={timelineData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(255,255,255,0.04)"
                  vertical={false}
                />
                <XAxis
                  dataKey="n"
                  tick={TICK}
                  label={{
                    value: "Analysis #",
                    position: "insideBottom",
                    fill: "rgba(255,255,255,0.2)",
                    fontSize: 10,
                    offset: -2,
                  }}
                />
                <YAxis domain={[0, 100]} tick={TICK} unit="%" />
                <defs>
                  <linearGradient id="lineGrad" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%"   stopColor="#8b5cf6" />
                    <stop offset="100%" stopColor="#06b6d4" />
                  </linearGradient>
                </defs>
                <Tooltip
                  contentStyle={TIP.contentStyle}
                  cursor={TIP.cursor}
                  formatter={(v) => [`${v.toFixed(1)}%`, "Confidence"]}
                />
                <Line
                  type="monotone"
                  dataKey="confidence"
                  stroke="url(#lineGrad)"
                  strokeWidth={3}
                  dot={{ fill: "#8b5cf6", r: 5, strokeWidth: 0 }}
                  activeDot={{ r: 7, fill: "#fff" }}
                  name="Confidence"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Bar — Feature Averages */}
          {featureAvgData.length > 0 && (
            <div className="glass-dark rounded-3xl p-6 lg:col-span-2">
              <h3 className="font-bold text-white mb-5 flex items-center gap-2.5">
                <Zap className="w-4 h-4 text-amber-400" />
                Average Feature Values Across All Analyses
              </h3>
              <ResponsiveContainer width="100%" height={230}>
                <BarChart
                  data={featureAvgData}
                  layout="vertical"
                  barSize={16}
                  margin={{ left: 120, right: 20 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(255,255,255,0.04)"
                    horizontal={false}
                  />
                  <XAxis type="number" tick={TICK} />
                  <YAxis
                    dataKey="name"
                    type="category"
                    tick={{ ...TICK, fill: "rgba(255,255,255,0.45)" }}
                  />
                  <defs>
                    <linearGradient id="barGrad" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%"   stopColor="#8b5cf6" stopOpacity={0.9} />
                      <stop offset="100%" stopColor="#06b6d4" stopOpacity={0.9} />
                    </linearGradient>
                  </defs>
                  <Tooltip
                    contentStyle={TIP.contentStyle}
                    cursor={TIP.cursor}
                  />
                  <Bar
                    dataKey="value"
                    radius={[0, 6, 6, 0]}
                    fill="url(#barGrad)"
                    name="Avg Value"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
