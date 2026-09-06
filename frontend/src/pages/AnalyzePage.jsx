import React, { useState } from "react";
import { Zap, Mic, Upload, Activity, WifiOff } from "lucide-react";
import AudioRecorder from "../components/AudioRecorder.jsx";
import ResultPanel   from "../components/ResultPanel.jsx";

const STATS = [
  { value: "92.3%",  label: "Model Accuracy (CV)",  color: "text-emerald-400" },
  { value: "68+",    label: "Acoustic Features",     color: "text-violet-400"  },
  { value: "4",      label: "ML Models Ensemble",    color: "text-cyan-400"    },
  { value: "<10s",   label: "Analysis Speed",        color: "text-amber-400"   },
];

export default function AnalyzePage({ apiStatus }) {
  const [result,      setResult]      = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleComplete = (data) => {
    setResult(data);
    setIsAnalyzing(false);
    setTimeout(() => document.getElementById("results")?.scrollIntoView({ behavior: "smooth" }), 150);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-12">
      {/* ── Hero ── */}
      {!result && (
        <section className="text-center mb-14 animate-fade-up">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 glass px-4 py-2 rounded-full text-xs font-bold uppercase tracking-widest text-violet-300 mb-6 border border-violet-500/20">
            <Zap className="w-3.5 h-3.5" />
            AI-Powered Voice Biomarker Screening
          </div>

          {/* Headline */}
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-black font-display leading-[1.05] tracking-tight mb-6">
            <span className="text-white">Detect </span>
            <span className="text-gradient-violet">Parkinson's</span>
            <br />
            <span className="text-white">Through Your </span>
            <span className="text-gradient-violet">Voice</span>
          </h1>

          <p className="text-lg text-white/50 max-w-2xl mx-auto leading-relaxed mb-10">
            Our ensemble machine learning model analyzes 60+ acoustic biomarkers — jitter, shimmer, HNR, MFCCs
            — extracted from a short voice recording to screen for Parkinson's Disease indicators.
          </p>

          {/* Stats */}
          <div className="flex flex-wrap justify-center gap-6 mb-10">
            {STATS.map(({ value, label, color }) => (
              <div key={label} className="text-center animate-fade-up">
                <p className={`text-3xl font-black font-display ${color}`}>{value}</p>
                <p className="text-xs text-white/40 font-medium mt-0.5">{label}</p>
              </div>
            ))}
          </div>

          {/* Feature pills */}
          <div className="flex flex-wrap justify-center gap-2">
            {["Live Waveform", "Audio Quality Check", "SNR Analysis", "Noise Reduction", "68+ Features", "4-Model Ensemble", "PDF Report"].map((f) => (
              <span key={f} className="glass px-3 py-1.5 rounded-full text-xs text-white/50 font-medium" style={{ border: "1px solid rgba(255,255,255,0.08)" }}>
                {f}
              </span>
            ))}
          </div>
        </section>
      )}

      {/* ── Offline warning ── */}
      {apiStatus === "offline" && (
        <div className="max-w-2xl mx-auto mb-6 animate-fade-up">
          <div className="glass border-rose-500/30 rounded-2xl p-4 flex items-start gap-3" style={{ borderColor: "rgba(244,63,94,0.3)" }}>
            <WifiOff className="w-5 h-5 text-rose-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-bold text-rose-300 text-sm">Backend API is offline</p>
              <p className="text-rose-400/70 text-xs mt-1">
                Run: <code className="bg-white/5 px-2 py-0.5 rounded font-mono">cd backend && venv\Scripts\uvicorn main:app --reload --port 8000</code>
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── Recorder ── */}
      {!result && (
        <div className="animate-fade-up delay-300">
          <AudioRecorder
            onComplete={handleComplete}
            onStart={() => setIsAnalyzing(true)}
            isAnalyzing={isAnalyzing}
          />
        </div>
      )}

      {/* ── Results ── */}
      {result && (
        <div id="results" className="animate-fade-up">
          <div className="flex items-center justify-between mb-8">
            <div>
              <p className="label mb-1">Analysis Complete</p>
              <h2 className="text-2xl font-black font-display text-white">Voice Analysis Results</h2>
            </div>
            <button
              onClick={() => { setResult(null); window.scrollTo({ top: 0, behavior: "smooth" }); }}
              className="btn-secondary-glass flex items-center gap-2 text-sm font-semibold"
            >
              ← New Analysis
            </button>
          </div>
          <ResultPanel result={result} />
        </div>
      )}
    </div>
  );
}
