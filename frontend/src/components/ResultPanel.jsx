import React, { useState } from "react";
import {
  ShieldAlert, ShieldCheck, Activity, TrendingUp,
  CheckCircle2, AlertTriangle, FileDown, Loader2,
  BarChart2, Hash, Clock, Cpu, Info,
} from "lucide-react";
import toast from "react-hot-toast";
import ConfidenceRing from "./ConfidenceRing.jsx";
import AcousticCharts from "./AcousticCharts.jsx";
import { downloadReport } from "../api/client.js";

const NORMAL_RANGES = {
  jitter_local:            [0.001, 0.25],  // browser WebM jitter range
  shimmer_local:           [0.01,  0.12],
  hnr:                     [5,     35  ],
  pitch_mean:              [75,    300 ],
  pitch_std:               [0.5,   100 ],  // browser pitch variation
  zcr_mean:                [0.01,  0.25],
  spectral_centroid_mean:  [300,   5000],
  rms_mean:                [0.005, 0.5 ],
};

function MetricTile({ label, value, unit, rangeKey, delay = 0 }) {
  const v       = parseFloat(value);
  const range   = NORMAL_RANGES[rangeKey];
  const inRange = range ? (v >= range[0] && v <= range[1]) : null;
  const display = isNaN(v) ? "—"
    : v < 0.0001 && v !== 0 ? v.toExponential(2)
    : v.toFixed(v < 1 ? 4 : 2);

  return (
    <div
      className="glass-dark rounded-2xl p-4 animate-fade-up hover:bg-white/6 transition-colors duration-200"
      style={{ animationDelay: `${delay}ms` }}
    >
      <p className="text-[10px] font-bold uppercase tracking-widest text-white/30 mb-2">{label}</p>
      <p className={`text-xl font-black font-display ${
        inRange === false ? "text-rose-400"
        : inRange === true ? "text-emerald-400"
        : "text-white"
      }`}>
        {display}
        {unit && <span className="text-xs font-normal text-white/30 ml-1">{unit}</span>}
      </p>
      {inRange !== null && (
        <p className={`text-[10px] font-bold mt-1.5 flex items-center gap-1 ${
          inRange ? "text-emerald-500" : "text-rose-500"
        }`}>
          <span className={`w-1.5 h-1.5 rounded-full ${inRange ? "bg-emerald-500" : "bg-rose-500"}`} />
          {inRange ? "Normal" : "Abnormal"}
        </p>
      )}
    </div>
  );
}

export default function ResultPanel({ result }) {
  const [downloading, setDownloading] = useState(false);
  const isPD = result.prediction?.includes("Parkinson");
  const f    = result.features || {};

  const handleDownload = async () => {
    setDownloading(true);
    try { await downloadReport(result); toast.success("PDF downloaded."); }
    catch { toast.error("PDF failed. Is the backend running?"); }
    finally { setDownloading(false); }
  };

  const keyMetrics = [
    { label: "Jitter Local",      value: f.jitter_local,           rangeKey: "jitter_local"          },
    { label: "Shimmer Local",     value: f.shimmer_local,          rangeKey: "shimmer_local"         },
    { label: "HNR",               value: f.hnr,          unit:"dB", rangeKey: "hnr"                  },
    { label: "Pitch Mean",        value: f.pitch_mean,   unit:"Hz", rangeKey: "pitch_mean"           },
    { label: "Pitch Std",         value: f.pitch_std,    unit:"Hz", rangeKey: "pitch_std"            },
    { label: "ZCR Mean",          value: f.zcr_mean,               rangeKey: "zcr_mean"             },
    { label: "Spectral Centroid", value: f.spectral_centroid_mean, unit:"Hz", rangeKey: "spectral_centroid_mean" },
    { label: "RMS Energy",        value: f.rms_mean,               rangeKey: "rms_mean"             },
    { label: "RPDE",              value: f.rpde                                                      },
    { label: "DFA",               value: f.dfa                                                       },
    { label: "Jitter (Abs)",      value: f.jitter_absolute                                           },
    { label: "Shimmer (dB)",      value: f.shimmer_db,   unit:"dB"                                   },
  ];

  return (
    <div className="space-y-6">
      {/* ── Hero result card ── */}
      <div
        className="rounded-3xl overflow-hidden animate-scale-in"
        style={{
          background: isPD
            ? "linear-gradient(135deg, rgba(244,63,94,0.08) 0%, rgba(15,23,42,0.9) 50%)"
            : "linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(15,23,42,0.9) 50%)",
          border: isPD ? "1px solid rgba(244,63,94,0.25)" : "1px solid rgba(16,185,129,0.25)",
          boxShadow: isPD
            ? "0 0 60px rgba(244,63,94,0.12), 0 8px 32px rgba(0,0,0,0.5)"
            : "0 0 60px rgba(16,185,129,0.12), 0 8px 32px rgba(0,0,0,0.5)",
        }}
      >
        {/* Top section */}
        <div className="p-6 sm:p-8">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-6">
            {/* Prediction info */}
            <div className="flex items-start gap-5">
              <div
                className="w-16 h-16 rounded-2xl flex items-center justify-center flex-shrink-0"
                style={{
                  background: isPD ? "rgba(244,63,94,0.15)" : "rgba(16,185,129,0.15)",
                  border: isPD ? "1px solid rgba(244,63,94,0.35)" : "1px solid rgba(16,185,129,0.35)",
                  boxShadow: isPD ? "0 0 20px rgba(244,63,94,0.2)" : "0 0 20px rgba(16,185,129,0.2)",
                }}
              >
                {isPD
                  ? <ShieldAlert className="w-8 h-8 text-rose-400" />
                  : <ShieldCheck className="w-8 h-8 text-emerald-400" />
                }
              </div>

              <div>
                <p className="label mb-2">Screening Result</p>
                <h2
                  className={`text-2xl sm:text-3xl font-black font-display leading-none mb-3 ${
                    isPD ? "text-gradient-rose" : "text-gradient-emerald"
                  }`}
                >
                  {result.prediction}
                </h2>

                <div className="flex flex-wrap items-center gap-2">
                  <span className={isPD ? "badge badge-pd" : "badge badge-healthy"}>
                    {result.risk_level} Risk
                  </span>
                  <span className="badge badge-info">
                    <Cpu className="w-3 h-3" />
                    {result.model_used?.includes("Ensemble") ? "Ensemble Model" : result.model_used || "ensemble"}
                  </span>
                  {/* Recording quality badge */}
                  {result.quality_score !== undefined && (
                    <span style={{
                      display: "inline-flex", alignItems: "center", gap: "5px",
                      padding: "3px 10px", borderRadius: "20px", fontSize: "11px", fontWeight: 700,
                      background: result.quality_score >= 70 ? "rgba(16,185,129,0.15)" : result.quality_score >= 40 ? "rgba(251,191,36,0.15)" : "rgba(244,63,94,0.15)",
                      border: `1px solid ${result.quality_score >= 70 ? "rgba(16,185,129,0.4)" : result.quality_score >= 40 ? "rgba(251,191,36,0.4)" : "rgba(244,63,94,0.4)"}`,
                      color: result.quality_score >= 70 ? "#34d399" : result.quality_score >= 40 ? "#fbbf24" : "#f87171",
                    }}>
                      🎙 Quality: {result.quality_score}%
                    </span>
                  )}
                  {/* SNR badge */}
                  {result.snr_db !== undefined && result.snr_db > 0 && (
                    <span style={{
                      display: "inline-flex", alignItems: "center", gap: "5px",
                      padding: "3px 10px", borderRadius: "20px", fontSize: "11px", fontWeight: 700,
                      background: "rgba(139,92,246,0.12)", border: "1px solid rgba(139,92,246,0.3)", color: "#a78bfa",
                    }}>
                      SNR: {result.snr_db} dB
                    </span>
                  )}
                </div>

                {/* Corrupted recording warning */}
                {result.is_corrupted && (
                  <div style={{ marginTop: "12px", background: "rgba(251,191,36,0.1)", border: "1px solid rgba(251,191,36,0.3)", borderRadius: "12px", padding: "10px 14px", display: "flex", alignItems: "flex-start", gap: "8px" }}>
                    <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                    <p style={{ fontSize: "12px", color: "#fbbf24", margin: 0, lineHeight: 1.5 }}>
                      <strong>Recording Quality Warning:</strong> High background noise detected. Please re-record in a quieter environment for most accurate results.
                    </p>
                  </div>
                )}

                <div className="flex items-center gap-4 mt-3">
                  <span className="text-xs text-white/25 flex items-center gap-1.5">
                    <Hash className="w-3 h-3" />{result.id}
                  </span>
                  <span className="text-xs text-white/25 flex items-center gap-1.5">
                    <Clock className="w-3 h-3" />{result.timestamp}
                  </span>
                </div>
              </div>
            </div>

            {/* Confidence ring */}
            <div className="flex-shrink-0">
              <ConfidenceRing confidence={result.confidence ?? 50} isPD={isPD} size={150} />
            </div>
          </div>
        </div>

        {/* Divider */}
        <div className="divider mx-6" />

        {/* Metrics grid */}
        <div className="p-6 sm:p-8">
          <p className="label flex items-center gap-2 mb-4">
            <Activity className="w-3.5 h-3.5 text-violet-400" />Acoustic Metrics
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {keyMetrics.map((m, i) => (
              <MetricTile key={m.label} {...m} delay={i * 40} />
            ))}
          </div>
        </div>

        {/* Recommendations */}
        <div className="px-6 sm:px-8 pb-6">
          <p className="label flex items-center gap-2 mb-4">
            <TrendingUp className="w-3.5 h-3.5 text-cyan-400" />Recommendations
          </p>
          <div className="space-y-2.5">
            {(result.recommendations || []).map((rec, i) => (
              <div
                key={i}
                className="flex items-start gap-3 glass-dark rounded-xl p-3.5 animate-slide-right"
                style={{ animationDelay: `${i * 60}ms` }}
              >
                <CheckCircle2 className={`w-4 h-4 mt-0.5 flex-shrink-0 ${isPD ? "text-amber-400" : "text-emerald-400"}`} />
                <p className="text-sm text-white/70 leading-relaxed">{rec}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Disclaimer + Download */}
        <div className="px-6 sm:px-8 pb-6 sm:pb-8 space-y-3">
          <div className="flex items-start gap-3 rounded-2xl bg-amber-500/8 border border-amber-500/20 p-4">
            <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-amber-400/70 leading-relaxed">
              This result is <strong className="text-amber-400">for research purposes only</strong> and does not constitute
              a medical diagnosis. Please consult a qualified neurologist for professional evaluation.
            </p>
          </div>

          <button
            onClick={handleDownload}
            disabled={downloading}
            className="w-full btn-secondary-glass flex items-center justify-center gap-2 py-3 rounded-2xl font-semibold"
          >
            {downloading
              ? <><Loader2 className="w-4 h-4 animate-spin" />Generating PDF…</>
              : <><FileDown className="w-4 h-4" />Download PDF Report</>
            }
          </button>
        </div>
      </div>

      {/* ── Charts card ── */}
      <div className="glass-dark rounded-3xl p-6 sm:p-8 animate-fade-up delay-200">
        <h3 className="text-xl font-black font-display text-white mb-6 flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-600/30 to-cyan-600/30 border border-violet-500/20 flex items-center justify-center">
            <BarChart2 className="w-4 h-4 text-violet-400" />
          </div>
          Acoustic Feature Analysis
        </h3>
        <AcousticCharts features={f} featureImportance={result.feature_importance} isPD={isPD} />
      </div>

      {/* ── MFCC grid ── */}
      <div className="glass-dark rounded-3xl p-6 sm:p-8 animate-fade-up delay-300">
        <h3 className="text-lg font-bold font-display text-white mb-5 flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-cyan-600/30 to-violet-600/30 border border-cyan-500/20 flex items-center justify-center">
            <Info className="w-4 h-4 text-cyan-400" />
          </div>
          MFCC Feature Coefficients
        </h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
          {Array.from({ length: 13 }, (_, i) => i + 1).map((n, i) => {
            const val = f[`mfcc_${n}_mean`];
            return (
              <div key={n} className="glass rounded-xl p-3 text-center animate-fade-up" style={{ animationDelay: `${i*30}ms` }}>
                <p className="text-[9px] font-bold uppercase tracking-widest text-white/25 mb-1">MFCC {n}</p>
                <p className="text-sm font-bold text-white/70">
                  {val !== undefined ? val.toFixed(1) : "—"}
                </p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
