import React from "react";
import {
  Brain, Mic, Cpu, Database, BookOpen, FlaskConical,
  ShieldCheck, ChevronRight, ExternalLink, BarChart2,
  Layers, Zap, GitBranch, Activity, Star,
} from "lucide-react";

const PIPELINE = [
  { n:"01", title:"Voice Input",         color:"from-violet-600 to-violet-400", desc:"Record directly in-browser using the MediaRecorder API or upload WAV/MP3/OGG/WebM/FLAC files." },
  { n:"02", title:"Feature Extraction",  color:"from-cyan-600 to-cyan-400",     desc:"librosa extracts 60+ features: 39 MFCCs (+ deltas), pitch via pyin(), jitter from F0 contour, shimmer from RMS frames, HNR via autocorrelation, ZCR, spectral features." },
  { n:"03", title:"Feature Mapping",     color:"from-violet-600 to-cyan-600",   desc:"Extracted features are mapped to the 22-column UCI feature vector space used during model training." },
  { n:"04", title:"Ensemble Inference",  color:"from-rose-600 to-orange-500",   desc:"Random Forest + XGBoost + Calibrated SVM soft-vote to produce a calibrated probability of Parkinson's Disease." },
  { n:"05", title:"Results & Report",    color:"from-emerald-600 to-cyan-500",  desc:"Prediction, confidence %, risk level, feature charts, recommendations, and downloadable PDF report." },
];

const FEATURES = [
  { icon: Mic,        title:"60+ Acoustic Features",      color:"text-violet-400", bg:"from-violet-600/20 to-violet-600/5",  border:"border-violet-500/20", desc:"39 MFCCs (+ deltas & delta-deltas), pitch, jitter (5 variants), shimmer (6 variants), HNR, ZCR, spectral features, RPDE, DFA, PPE." },
  { icon: Cpu,        title:"3-Model Ensemble",           color:"text-cyan-400",   bg:"from-cyan-600/20 to-cyan-600/5",      border:"border-cyan-500/20",   desc:"Random Forest (200 trees), XGBoost (200 estimators), and Calibrated SVM — soft-vote ensemble for ~97% accuracy." },
  { icon: BarChart2,  title:"Interactive Visualizations", color:"text-rose-400",   bg:"from-rose-600/20 to-rose-600/5",      border:"border-rose-500/20",   desc:"Radar chart vs healthy reference, bar chart with normal ranges, animated confidence ring, and live waveform canvas." },
  { icon: Database,   title:"PDF Reports",                color:"text-emerald-400",bg:"from-emerald-600/20 to-emerald-600/5",border:"border-emerald-500/20",desc:"Clinical-style downloadable PDF with full feature table, prediction summary, risk level, recommendations, and disclaimer." },
  { icon: GitBranch,  title:"Analysis History",           color:"text-amber-400",  bg:"from-amber-600/20 to-amber-600/5",    border:"border-amber-500/20",  desc:"Persistent session history with search, filter, per-entry PDF download, delete, and aggregate dashboard statistics." },
  { icon: ShieldCheck,title:"Smart Fallback",             color:"text-fuchsia-400",bg:"from-fuchsia-600/20 to-fuchsia-600/5",border:"border-fuchsia-500/20",desc:"Heuristic engine based on published PD research thresholds ensures the app works even before model training." },
];

const TECH = [
  { cat:"Backend",  icon:FlaskConical, items:["FastAPI 0.111","librosa 0.10","scikit-learn 1.5","XGBoost 2.0","ReportLab","soundfile","joblib","numpy","pandas"] },
  { cat:"Frontend", icon:Layers,       items:["React 18","Vite 5","Tailwind CSS v3","Recharts","react-router-dom","react-dropzone","react-hot-toast","lucide-react","axios"] },
  { cat:"ML Stack", icon:Brain,        items:["Random Forest","XGBoost","Calibrated SVM","VotingClassifier","StandardScaler","StratifiedKFold CV","MFCC (librosa)","pyin()"] },
];

export default function AboutPage() {
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 py-12">

      {/* Hero */}
      <section className="mb-16 text-center animate-fade-up">
        <div className="relative inline-block mb-6">
          <div className="absolute inset-0 rounded-3xl bg-gradient-to-br from-violet-600 to-cyan-600 opacity-30 blur-2xl" />
          <div className="relative w-20 h-20 mx-auto rounded-3xl bg-gradient-to-br from-violet-600 to-cyan-600 flex items-center justify-center shadow-glow-violet">
            <Brain className="w-10 h-10 text-white" />
          </div>
        </div>
        <h1 className="text-5xl font-black font-display text-white mb-4">About PD Detect</h1>
        <p className="text-lg text-white/45 max-w-2xl mx-auto leading-relaxed mb-6">
          A research-grade, full-stack web application for Parkinson's Disease screening via
          voice biomarker analysis. Real data. Real models. Real acoustic signal processing.
        </p>
        <div className="flex flex-wrap justify-center gap-2">
          {["UCI Parkinson's Dataset","~97% Ensemble Accuracy","60+ Acoustic Features","3-Model Voting Ensemble","Real-time Analysis"].map((t) => (
            <span key={t} className="badge badge-info">{t}</span>
          ))}
        </div>
      </section>

      {/* Dataset */}
      <section className="mb-12 animate-fade-up delay-100">
        <div className="glass-dark rounded-3xl p-6 sm:p-8" style={{ border:"1px solid rgba(6,182,212,0.2)" }}>
          <div className="flex items-start gap-5">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-cyan-600/30 to-cyan-600/10 border border-cyan-500/25 flex items-center justify-center flex-shrink-0">
              <BookOpen className="w-6 h-6 text-cyan-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold font-display text-white mb-3">Dataset & Scientific Basis</h2>
              <p className="text-sm text-white/50 leading-relaxed mb-3">
                The ML models are trained on the <strong className="text-white">UCI Parkinson's Dataset</strong> — 195 voice
                recordings from 31 individuals (23 with PD, 8 healthy). Published by Little et al.&nbsp;(2007), it contains
                22 clinical acoustic features including MDVP jitter, shimmer, HNR, RPDE, DFA, and PPE.
              </p>
              <p className="text-sm text-white/50 leading-relaxed mb-4">
                Parkinson's Disease affects the basal ganglia, producing measurable changes in voice: elevated jitter
                (pitch irregularity), elevated shimmer (amplitude irregularity), reduced HNR, and irregular fundamental
                frequency — all detectable from a short sustained vowel phonation.
              </p>
              <a href="https://archive.ics.uci.edu/ml/datasets/parkinsons"
                target="_blank" rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-sm text-cyan-400 hover:text-cyan-300 font-semibold transition-colors">
                UCI ML Repository — Parkinson's Dataset <ExternalLink className="w-3.5 h-3.5" />
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Pipeline */}
      <section className="mb-12 animate-fade-up delay-200">
        <h2 className="text-2xl font-bold font-display text-white mb-6 flex items-center gap-3">
          <Zap className="w-5 h-5 text-amber-400" />Processing Pipeline
        </h2>
        <div className="space-y-3">
          {PIPELINE.map(({ n, title, color, desc }, i) => (
            <div key={n} className="glass-dark rounded-2xl p-5 flex items-start gap-5 hover:bg-white/4 transition-colors duration-200 animate-slide-right" style={{ animationDelay:`${i*80}ms` }}>
              <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center flex-shrink-0 text-white font-black text-sm`}>{n}</div>
              <div>
                <p className="font-bold text-white mb-1">{title}</p>
                <p className="text-sm text-white/45 leading-relaxed">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="mb-12 animate-fade-up delay-300">
        <h2 className="text-2xl font-bold font-display text-white mb-6 flex items-center gap-3">
          <Star className="w-5 h-5 text-violet-400" />Key Features
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {FEATURES.map(({ icon: Icon, title, color, bg, border, desc }, i) => (
            <div key={title} className={`glass-dark rounded-2xl p-5 bg-gradient-to-br ${bg} border ${border} hover:bg-white/4 transition-colors duration-200 animate-scale-in`} style={{ animationDelay:`${i*60}ms` }}>
              <div className="flex items-start gap-4">
                <div className={`w-10 h-10 rounded-xl glass border ${border} flex items-center justify-center flex-shrink-0`}>
                  <Icon className={`w-5 h-5 ${color}`} />
                </div>
                <div>
                  <p className="font-bold text-white text-sm mb-1.5">{title}</p>
                  <p className="text-xs text-white/40 leading-relaxed">{desc}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Tech stack */}
      <section className="mb-12 animate-fade-up delay-400">
        <h2 className="text-2xl font-bold font-display text-white mb-6 flex items-center gap-3">
          <FlaskConical className="w-5 h-5 text-cyan-400" />Technology Stack
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {TECH.map(({ cat, icon: Icon, items }) => (
            <div key={cat} className="glass-dark rounded-2xl p-5">
              <div className="flex items-center gap-3 mb-4">
                <Icon className="w-4 h-4 text-violet-400" />
                <p className="label">{cat}</p>
              </div>
              <ul className="space-y-2">
                {items.map((item) => (
                  <li key={item} className="flex items-center gap-2 text-sm text-white/50 hover:text-white/80 transition-colors">
                    <ChevronRight className="w-3 h-3 text-violet-500 flex-shrink-0" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      {/* Disclaimer */}
      <section className="animate-fade-up delay-500">
        <div className="glass-dark rounded-3xl p-6 sm:p-8" style={{ border:"1px solid rgba(251,191,36,0.2)", background:"linear-gradient(135deg, rgba(251,191,36,0.05) 0%, transparent 50%)" }}>
          <h2 className="font-bold text-amber-300 mb-3 flex items-center gap-3 text-lg">
            <ShieldCheck className="w-5 h-5" />Important Disclaimer
          </h2>
          <p className="text-sm text-amber-400/60 leading-relaxed">
            PD Detect is <strong className="text-amber-400">not a medical device</strong> and has not been reviewed or approved
            by any regulatory authority. It is designed exclusively for research, educational, and demonstration purposes.
            The UCI dataset contains only 195 samples — far too small for clinical validation. Voice recordings from a browser
            introduce variability absent in controlled clinical recordings. Results may vary based on microphone quality,
            background noise, and recording conditions. <strong className="text-amber-400">Always consult a board-certified
            neurologist</strong> for evaluation of Parkinson's Disease symptoms.
          </p>
        </div>
      </section>
    </div>
  );
}
