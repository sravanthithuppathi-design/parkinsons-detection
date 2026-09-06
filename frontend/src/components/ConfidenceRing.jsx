import React, { useEffect, useRef, useState } from "react";

export default function ConfidenceRing({ confidence, isPD, size = 160 }) {
  const [animated, setAnimated] = useState(0);
  const rafRef = useRef(null);

  const stroke = 10;
  const r      = (size - stroke) / 2;
  const circ   = 2 * Math.PI * r;

  useEffect(() => {
    let start = null;
    const dur = 1600;
    const anim = (ts) => {
      if (!start) start = ts;
      const t    = Math.min((ts - start) / dur, 1);
      const ease = 1 - Math.pow(1 - t, 4); // ease-out-quart
      setAnimated(ease * confidence);
      if (t < 1) rafRef.current = requestAnimationFrame(anim);
    };
    rafRef.current = requestAnimationFrame(anim);
    return () => cancelAnimationFrame(rafRef.current);
  }, [confidence]);

  const offset = circ - (animated / 100) * circ;

  // Dynamic color
  const color = isPD
    ? animated > 75 ? "#f43f5e" : animated > 50 ? "#fb923c" : "#fbbf24"
    : "#10b981";

  const glowColor = isPD
    ? "rgba(244,63,94,0.4)"
    : "rgba(16,185,129,0.4)";

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative" style={{ width: size, height: size }}>
        {/* Outer glow ring */}
        <div
          className="absolute inset-0 rounded-full opacity-30 blur-xl animate-pulse-glow"
          style={{ background: `radial-gradient(circle, ${glowColor}, transparent 70%)` }}
        />

        <svg width={size} height={size} className="relative z-10 -rotate-90">
          {/* Track */}
          <circle cx={size/2} cy={size/2} r={r} fill="none"
            stroke="rgba(255,255,255,0.06)" strokeWidth={stroke} />
          {/* Glow track */}
          <circle cx={size/2} cy={size/2} r={r} fill="none"
            stroke={color} strokeWidth={stroke + 4}
            strokeLinecap="round"
            strokeDasharray={circ}
            strokeDashoffset={offset}
            opacity="0.15"
            style={{ transition: "stroke-dashoffset 0.04s linear" }}
          />
          {/* Main arc */}
          <circle cx={size/2} cy={size/2} r={r} fill="none"
            stroke={color} strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={circ}
            strokeDashoffset={offset}
            style={{ transition: "stroke-dashoffset 0.04s linear, stroke 0.4s ease" }}
          />
        </svg>

        {/* Center */}
        <div className="absolute inset-0 flex flex-col items-center justify-center z-20">
          <span className="text-3xl font-black font-display" style={{ color }}>
            {Math.round(animated)}%
          </span>
          <span className="text-[10px] text-white/40 font-bold uppercase tracking-widest mt-0.5">
            Confidence
          </span>
        </div>
      </div>
    </div>
  );
}
