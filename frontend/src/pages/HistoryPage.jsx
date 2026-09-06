import React, { useState, useEffect, useCallback } from "react";
import {
  History, RefreshCw, Trash2, ChevronDown, ChevronUp,
  ShieldAlert, ShieldCheck, Calendar, Loader2, ClipboardX,
  FileDown, Search, X, Hash, FlaskConical,
} from "lucide-react";
import toast from "react-hot-toast";
import {
  getLocalHistory,
  clearLocalHistory,
  deleteLocalEntry,
  downloadReport,
  seedDemoData,
} from "../api/client.js";

function HistoryItem({ item, onDelete }) {
  const [expanded,    setExpanded]    = useState(false);
  const [downloading, setDownloading] = useState(false);
  const isPD = item.prediction?.includes("Parkinson");
  const f    = item.features || {};

  const metrics = [
    { l: "Jitter",     v: f.jitter_local,  d: 5 },
    { l: "Shimmer",    v: f.shimmer_local, d: 5 },
    { l: "HNR (dB)",   v: f.hnr,           d: 2 },
    { l: "Pitch (Hz)", v: f.pitch_mean,    d: 1 },
    { l: "ZCR",        v: f.zcr_mean,      d: 4 },
    { l: "RPDE",       v: f.rpde,          d: 4 },
  ];

  const handleDownload = async (e) => {
    e.stopPropagation();
    setDownloading(true);
    try {
      await downloadReport(item);
    } catch {
      toast.error("PDF download failed.");
    } finally {
      setDownloading(false);
    }
  };

  const handleDelete = (e) => {
    e.stopPropagation();
    if (!window.confirm("Delete this record?")) return;
    deleteLocalEntry(item.id);
    onDelete(item.id);
    toast.success("Deleted.");
  };

  return (
    <div
      className="glass-dark rounded-2xl overflow-hidden transition-all duration-300 hover:bg-white/4"
      style={{
        border: isPD
          ? "1px solid rgba(244,63,94,0.2)"
          : "1px solid rgba(16,185,129,0.2)",
      }}
    >
      {/* Summary row */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-4 p-4 sm:p-5 text-left"
      >
        {/* Icon */}
        <div
          className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${
            isPD
              ? "bg-rose-500/15 border border-rose-500/25"
              : "bg-emerald-500/15 border border-emerald-500/25"
          }`}
        >
          {isPD ? (
            <ShieldAlert className="w-5 h-5 text-rose-400" />
          ) : (
            <ShieldCheck className="w-5 h-5 text-emerald-400" />
          )}
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <p
            className={`font-bold text-sm ${
              isPD ? "text-rose-300" : "text-emerald-300"
            }`}
          >
            {item.prediction}
          </p>
          <p className="text-xs text-white/30 flex items-center gap-2 mt-0.5">
            <Calendar className="w-3 h-3" />
            {item.timestamp}
            <span className="text-white/15">·</span>
            <Hash className="w-3 h-3" />
            {item.id}
          </p>
        </div>

        {/* Right badges */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <span
            className={`text-sm font-black px-2.5 py-1 rounded-xl ${
              isPD
                ? "bg-rose-500/15 text-rose-300"
                : "bg-emerald-500/15 text-emerald-300"
            }`}
          >
            {item.confidence?.toFixed(1)}%
          </span>
          <span
            className={`hidden sm:inline badge text-[10px] ${
              item.risk_level === "High"
                ? "badge-pd"
                : item.risk_level === "Medium"
                ? "badge-warning"
                : "badge-healthy"
            }`}
          >
            {item.risk_level}
          </span>
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-white/25" />
          ) : (
            <ChevronDown className="w-4 h-4 text-white/25" />
          )}
        </div>
      </button>

      {/* Expanded */}
      {expanded && (
        <div className="border-t border-white/6 px-4 sm:px-5 py-4 animate-fade-up">
          <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 mb-4">
            {metrics.map((m) => (
              <div key={m.l} className="glass rounded-xl p-3 text-center">
                <p className="text-[9px] font-bold uppercase tracking-widest text-white/25 mb-1">
                  {m.l}
                </p>
                <p className="text-xs font-bold text-white/80">
                  {m.v !== undefined ? m.v.toFixed(m.d) : "—"}
                </p>
              </div>
            ))}
          </div>

          {item.recommendations?.length > 0 && (
            <div className="mb-4">
              <p className="label mb-2">Recommendations</p>
              <ul className="space-y-1">
                {item.recommendations.slice(0, 3).map((r, i) => (
                  <li
                    key={i}
                    className="text-xs text-white/40 flex items-start gap-2"
                  >
                    <span className="text-violet-400 mt-0.5">›</span>
                    {r}
                  </li>
                ))}
                {item.recommendations.length > 3 && (
                  <li className="text-xs text-white/20">
                    +{item.recommendations.length - 3} more…
                  </li>
                )}
              </ul>
            </div>
          )}

          <p className="text-xs text-white/20 mb-3">
            Model: {item.model_used || "ensemble"}
          </p>

          <div className="flex items-center gap-2">
            <button
              onClick={handleDownload}
              disabled={downloading}
              className="btn-secondary-glass text-xs px-3 py-2 rounded-xl flex items-center gap-1.5"
            >
              {downloading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Generating…
                </>
              ) : (
                <>
                  <FileDown className="w-3.5 h-3.5" />
                  PDF
                </>
              )}
            </button>
            <button
              onClick={handleDelete}
              className="text-xs px-3 py-2 rounded-xl flex items-center gap-1.5 text-rose-400/60 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Delete
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function HistoryPage() {
  const [history,  setHistory]  = useState([]);
  const [loading,  setLoading]  = useState(true);
  const [clearing, setClearing] = useState(false);
  const [filter,   setFilter]   = useState("all");
  const [search,   setSearch]   = useState("");
  const [seeding,  setSeeding]  = useState(false);

  // Load from localStorage
  const load = useCallback(() => {
    setLoading(true);
    try {
      setHistory(getLocalHistory());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleClearAll = () => {
    if (!window.confirm("Delete ALL history? This cannot be undone.")) return;
    setClearing(true);
    try {
      clearLocalHistory();
      setHistory([]);
      toast.success("History cleared.");
    } catch {
      toast.error("Failed to clear history.");
    } finally {
      setClearing(false);
    }
  };

  const handleDelete = (id) =>
    setHistory((h) => h.filter((x) => x.id !== id));

  const filtered = history.filter((item) => {
    const isPD = item.prediction?.includes("Parkinson");
    // Apply tab filter first
    if (filter === "pd"      && !isPD) return false;
    if (filter === "healthy" &&  isPD) return false;
    // Then apply search on top
    if (search) {
      const q = search.toLowerCase();
      return (
        item.id?.toLowerCase().includes(q) ||
        item.prediction?.toLowerCase().includes(q) ||
        item.timestamp?.toLowerCase().includes(q)
      );
    }
    return true;
  });

  const pdCount      = history.filter((h) => h.prediction?.includes("Parkinson")).length;
  const healthyCount = history.length - pdCount;
  const avgConf      = history.length
    ? (
        history.reduce((s, h) => s + (h.confidence || 0), 0) /
        history.length
      ).toFixed(1)
    : null;

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-12">
      {/* Header */}
      <div className="mb-10 animate-fade-up">
        <p className="label mb-2">Browser Storage Records</p>
        <h1 className="text-4xl font-black font-display text-white flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-violet-600/30 to-cyan-600/30 border border-violet-500/20 flex items-center justify-center">
            <History className="w-6 h-6 text-violet-400" />
          </div>
          Analysis History
        </h1>
        <p className="text-xs text-white/30 mt-3">
          Saved locally in your browser — persists across sessions on this device.
        </p>
      </div>

      {/* Stats */}
      {history.length > 0 && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8 animate-fade-up delay-100">
          {[
            {
              label: "Total",
              value: history.length,
              color: "text-white",
              glow: "from-violet-600/20 to-cyan-600/20",
              border: "border-violet-500/20",
            },
            {
              label: "PD Positive",
              value: pdCount,
              color: "text-rose-400",
              glow: "from-rose-600/20 to-orange-600/20",
              border: "border-rose-500/20",
            },
            {
              label: "Healthy",
              value: healthyCount,
              color: "text-emerald-400",
              glow: "from-emerald-600/20 to-cyan-600/20",
              border: "border-emerald-500/20",
            },
            {
              label: "Avg Confidence",
              value: avgConf ? `${avgConf}%` : "—",
              color: "text-amber-400",
              glow: "from-amber-600/20 to-orange-600/20",
              border: "border-amber-500/20",
            },
          ].map(({ label, value, color, glow, border }) => (
            <div
              key={label}
              className={`glass-dark rounded-2xl p-5 text-center bg-gradient-to-br ${glow} border ${border}`}
            >
              <p className={`text-3xl font-black font-display ${color}`}>
                {value}
              </p>
              <p className="text-xs text-white/30 font-medium mt-1">{label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 mb-6 animate-fade-up delay-200">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-white/25" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search records…"
            className="input-glass pl-10 py-2.5 text-sm"
          />
          {search && (
            <button
              onClick={() => setSearch("")}
              className="absolute right-3.5 top-1/2 -translate-y-1/2"
            >
              <X className="w-4 h-4 text-white/25 hover:text-white transition-colors" />
            </button>
          )}
        </div>

        <div className="flex items-center gap-1 glass rounded-xl p-1">
          {[
            ["all", "All"],
            ["pd", "PD"],
            ["healthy", "Healthy"],
          ].map(([v, l]) => (
            <button
              key={v}
              onClick={() => setFilter(v)}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all duration-200 ${
                filter === v
                  ? "bg-gradient-to-r from-violet-600 to-cyan-600 text-white shadow"
                  : "text-white/40 hover:text-white"
              }`}
            >
              {l}
            </button>
          ))}
        </div>

        <button
          onClick={load}
          disabled={loading}
          className="btn-secondary-glass flex items-center gap-2 text-sm px-4 py-2.5"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>

        {history.length > 0 && (
          <button
            onClick={handleClearAll}
            disabled={clearing}
            className="btn-danger-glow flex items-center gap-2 text-sm px-4 py-2.5"
          >
            {clearing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Clearing…
              </>
            ) : (
              <>
                <Trash2 className="w-4 h-4" />
                Clear All
              </>
            )}
          </button>
        )}

        <button
          onClick={async () => {
            setSeeding(true);
            try {
              const data = await seedDemoData();
              toast.success(`${data.seeded} demo analyses loaded!`);
              setHistory(getLocalHistory());
            } catch {
              toast.error("Failed to load demo data.");
            } finally {
              setSeeding(false);
            }
          }}
          disabled={seeding}
          className="btn-secondary-glass flex items-center gap-2 text-sm px-4 py-2.5"
        >
          {seeding ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading…
            </>
          ) : (
            <>
              <FlaskConical className="w-4 h-4" />
              Demo Data
            </>
          )}
        </button>
      </div>

      {/* Loading skeleton */}
      {loading && (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="skeleton h-20 rounded-2xl" />
          ))}
        </div>
      )}

      {/* Empty state */}
      {!loading && filtered.length === 0 && (
        <div className="flex flex-col items-center justify-center py-32 gap-5 animate-fade-up">
          <div className="w-20 h-20 rounded-3xl glass flex items-center justify-center">
            <ClipboardX className="w-9 h-9 text-white/20" />
          </div>
          <div className="text-center">
            <p className="text-white/50 font-bold text-lg">
              {history.length === 0
                ? "No analyses yet"
                : "No matching records"}
            </p>
            <p className="text-sm text-white/25 mt-1">
              {history.length === 0
                ? "Go to Analyze to start a new voice screening."
                : "Try changing the filter or search query."}
            </p>
          </div>
        </div>
      )}

      {/* List */}
      {!loading && filtered.length > 0 && (
        <div className="space-y-3">
          {filtered.map((item) => (
            <HistoryItem key={item.id} item={item} onDelete={handleDelete} />
          ))}
        </div>
      )}
    </div>
  );
}
