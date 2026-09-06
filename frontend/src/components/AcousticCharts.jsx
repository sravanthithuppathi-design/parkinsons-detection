import React, { useMemo } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend,
} from "recharts";

const norm = (v, min, max) =>
  Math.min(100, Math.max(0, ((v - min) / (max - min)) * 100));

const RADAR_CFG = [
  { key: "mfcc_1_mean",           label: "MFCC",       min: -600, max: 50   },
  { key: "jitter_local",          label: "Jitter",     min: 0,    max: 0.04 },
  { key: "shimmer_local",         label: "Shimmer",    min: 0,    max: 0.12 },
  { key: "hnr",                   label: "HNR",        min: 0,    max: 35   },
  { key: "pitch_mean",            label: "Pitch",      min: 50,   max: 350  },
  { key: "zcr_mean",              label: "ZCR",        min: 0,    max: 0.2  },
  { key: "spectral_centroid_mean",label: "S.Centroid", min: 200,  max: 5000 },
  { key: "rms_mean",              label: "RMS",        min: 0,    max: 0.3  },
];

const HEALTHY_REF = {
  mfcc_1_mean:           norm(-200, -600, 50),
  jitter_local:          norm(0.003, 0, 0.04),
  shimmer_local:         norm(0.025, 0, 0.12),
  hnr:                   norm(22, 0, 35),
  pitch_mean:            norm(150, 50, 350),
  zcr_mean:              norm(0.06, 0, 0.2),
  spectral_centroid_mean:norm(1500, 200, 5000),
  rms_mean:              norm(0.05, 0, 0.3),
};

const BAR_FEATURES = [
  { name: "Jitter",    key: "jitter_local",  min: 0.001, max: 0.08,  digits: 5 },
  { name: "Shimmer",   key: "shimmer_local", min: 0.01,  max: 0.12,  digits: 5 },
  { name: "HNR (dB)",  key: "hnr",           min: 5,     max: 35,    digits: 2 },
  { name: "Pitch",     key: "pitch_mean",    min: 75,    max: 300,   digits: 1 },
  { name: "Pitch Std", key: "pitch_std",     min: 0.5,   max: 20,    digits: 2 },
  { name: "ZCR",       key: "zcr_mean",      min: 0.01,  max: 0.25,  digits: 4 },
  { name: "RPDE",      key: "rpde",          min: 0.4,   max: 0.65,  digits: 4 },
  { name: "DFA",       key: "dfa",           min: 0.5,   max: 0.8,   digits: 4 },
  { name: "Spread1",   key: "spread1",       min: -7,    max: -4,    digits: 3 },
  { name: "Spread2",   key: "spread2",       min: 0.1,   max: 0.3,   digits: 3 },
];

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
};

function CustomBarTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const { value, inRange, normalMin, normalMax } = payload[0].payload;
  return (
    <div style={TIP.contentStyle}>
      <p className="font-bold text-white mb-1.5">{label}</p>
      <p className="text-white/70 text-xs">Value: <strong className="text-white">{value}</strong></p>
      {normalMin !== undefined && (
        <p className="text-white/40 text-xs mt-1">Normal: {normalMin} – {normalMax}</p>
      )}
      <p className={`text-xs font-bold mt-1.5 ${inRange ? "text-emerald-400" : "text-rose-400"}`}>
        {inRange ? "✓ Within normal range" : "✗ Outside normal range"}
      </p>
    </div>
  );
}

export default function AcousticCharts({ features, featureImportance, isPD }) {
  const patientColor = isPD ? "#f43f5e" : "#10b981";
  const patientFill  = isPD ? "rgba(244,63,94,0.25)" : "rgba(16,185,129,0.2)";

  const radarData = useMemo(() => {
    if (!features) return [];
    return RADAR_CFG.map((cfg) => ({
      subject: cfg.label,
      Patient: Math.round(norm(features[cfg.key] ?? 0, cfg.min, cfg.max)),
      Healthy: Math.round(HEALTHY_REF[cfg.key] ?? 50),
      fullMark: 100,
    }));
  }, [features]);

  const barData = useMemo(() => {
    if (!features) return [];
    return BAR_FEATURES.map(({ name, key, min, max, digits }) => {
      const val     = features[key];
      const inRange = val !== undefined ? (val >= min && val <= max) : null;
      return {
        name,
        value:     parseFloat((val ?? 0).toFixed(digits)),
        inRange,
        normalMin: min,
        normalMax: max,
      };
    });
  }, [features]);

  const importanceData = useMemo(() => {
    if (!featureImportance) return [];
    return Object.entries(featureImportance)
      .slice(0, 10)
      .map(([name, info]) => ({
        name: name.replace(/MDVP:/g, "").replace(/_/g, " ").slice(0, 14),
        value: parseFloat(
          (typeof info === "object" ? info.importance * 100 : info * 100).toFixed(2)
        ),
      }));
  }, [featureImportance]);

  const tickStyle = { fill: "rgba(255,255,255,0.3)", fontSize: 10 };

  return (
    <div className="space-y-10">
      {/* ── Radar ── */}
      <div>
        <p className="label mb-4">Feature Profile vs Healthy Reference</p>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
            <PolarGrid stroke="rgba(255,255,255,0.06)" />
            <PolarAngleAxis dataKey="subject" tick={{ fill: "rgba(255,255,255,0.45)", fontSize: 11, fontWeight: 600 }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: "rgba(255,255,255,0.2)", fontSize: 9 }} />
            <Radar name="Patient" dataKey="Patient"
              stroke={patientColor} fill={patientColor} fillOpacity={0.3} strokeWidth={2} />
            <Radar name="Healthy Ref." dataKey="Healthy"
              stroke="#10b981" fill="#10b981" fillOpacity={0.08} strokeWidth={1.5} strokeDasharray="5 3" />
            <Legend formatter={(v) => <span style={{ color: "rgba(255,255,255,0.5)", fontSize: 11 }}>{v}</span>} />
            <Tooltip contentStyle={TIP.contentStyle}
              formatter={(v, n) => [`${v} (norm.)`, n]} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Bar: feature values ── */}
      <div>
        <p className="label mb-2">Key Feature Values vs Normal Ranges</p>
        <div className="flex items-center gap-4 text-xs text-white/40 mb-4">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-emerald-500/70 inline-block" />Within range
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-rose-500/70 inline-block" />Outside range
          </span>
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={barData} barSize={18} margin={{ top: 5, right: 10, bottom: 75, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
            <XAxis dataKey="name" tick={tickStyle} angle={-40} textAnchor="end" interval={0} height={80} />
            <YAxis tick={tickStyle}
              tickFormatter={(v) => Math.abs(v) > 100 ? v.toFixed(0) : v.toFixed(3)} width={65} />
            <Tooltip content={<CustomBarTooltip />} />
            <Bar dataKey="value" radius={[6, 6, 0, 0]}>
              {barData.map((entry, i) => (
                <Cell key={i}
                  fill={entry.inRange === false ? "#f43f5e" : entry.inRange === true ? "#10b981" : "#8b5cf6"}
                  opacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* ── Feature importance ── */}
      {importanceData.length > 0 && (
        <div>
          <p className="label mb-4">Model Feature Importances</p>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={importanceData} layout="vertical" barSize={12} margin={{ top: 0, right: 30, left: 110, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
              <XAxis type="number" tick={tickStyle} unit="%" />
              <YAxis dataKey="name" type="category" tick={{ ...tickStyle, fill: "rgba(255,255,255,0.45)" }} />
              <Tooltip contentStyle={TIP.contentStyle}
                formatter={(v) => [`${v}%`, "Importance"]} />
              <Bar dataKey="value" radius={[0, 6, 6, 0]}
                fill="url(#importGrad)" />
              <defs>
                <linearGradient id="importGrad" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.9} />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity={0.9} />
                </linearGradient>
              </defs>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
