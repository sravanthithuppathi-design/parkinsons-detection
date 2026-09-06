import React, { useRef, useEffect, useCallback } from "react";

const BAR_COUNT = 64;
const GAP = 2;

export default function WaveformVisualizer({ isRecording, analyserNode }) {
  const canvasRef    = useRef(null);
  const rafRef       = useRef(null);
  const phaseRef     = useRef(0);

  const drawIdle = useCallback((ctx, W, H) => {
    ctx.clearRect(0, 0, W, H);
    const bw  = (W - GAP * (BAR_COUNT - 1)) / BAR_COUNT;
    const mid = H / 2;
    const t   = phaseRef.current;

    for (let i = 0; i < BAR_COUNT; i++) {
      const wave = Math.sin(t + i * 0.28) * 3 + Math.sin(t * 0.7 + i * 0.5) * 1.5;
      const bh   = Math.max(2, 3 + Math.abs(wave));
      const x    = i * (bw + GAP);
      const frac = i / BAR_COUNT;

      // gradient bar colour
      const r = Math.floor(100 + frac * 40);
      const g = Math.floor(60  + frac * 20);
      const b = Math.floor(200 - frac * 40);
      ctx.fillStyle = `rgba(${r},${g},${b},0.35)`;

      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x, mid - bh / 2, bw, bh, 2);
      else ctx.rect(x, mid - bh / 2, bw, bh);
      ctx.fill();
    }
    phaseRef.current += 0.03;
  }, []);

  const drawActive = useCallback((ctx, W, H, data) => {
    ctx.clearRect(0, 0, W, H);
    const bw  = (W - GAP * (BAR_COUNT - 1)) / BAR_COUNT;
    const mid = H / 2;

    for (let i = 0; i < BAR_COUNT; i++) {
      const idx = Math.floor((i / BAR_COUNT) * data.length);
      const raw = (data[idx] - 128) / 128;          // -1 … 1
      const amp = Math.abs(raw);
      const bh  = Math.max(3, amp * H * 0.90);
      const x   = i * (bw + GAP);

      // Cyan → violet gradient based on bar position
      const frac = i / BAR_COUNT;
      const r = Math.floor(6   + frac * 133);
      const g = Math.floor(182 - frac * 90);
      const b = Math.floor(212 + frac * 44);
      const alpha = 0.5 + amp * 0.5;

      // Glow effect on tall bars
      if (amp > 0.4) {
        ctx.shadowColor  = `rgba(${r},${g},${b},0.5)`;
        ctx.shadowBlur   = 8;
      } else {
        ctx.shadowBlur = 0;
      }

      ctx.fillStyle = `rgba(${r},${g},${b},${alpha.toFixed(2)})`;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x, mid - bh / 2, bw, bh, 2);
      else ctx.rect(x, mid - bh / 2, bw, bh);
      ctx.fill();
    }
    ctx.shadowBlur = 0;
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const loop = () => {
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0) { rafRef.current = requestAnimationFrame(loop); return; }
      canvas.width  = rect.width;
      canvas.height = rect.height;

      if (isRecording && analyserNode) {
        const buf = new Uint8Array(analyserNode.frequencyBinCount);
        analyserNode.getByteTimeDomainData(buf);
        drawActive(ctx, canvas.width, canvas.height, buf);
      } else {
        drawIdle(ctx, canvas.width, canvas.height);
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [isRecording, analyserNode, drawIdle, drawActive]);

  return (
    <div
      className="w-full h-24 rounded-2xl overflow-hidden transition-all duration-500"
      style={{
        background: isRecording
          ? "linear-gradient(135deg, rgba(6,182,212,0.08) 0%, rgba(139,92,246,0.08) 100%)"
          : "rgba(255,255,255,0.02)",
        border: isRecording
          ? "1px solid rgba(139,92,246,0.3)"
          : "1px solid rgba(255,255,255,0.06)",
        boxShadow: isRecording ? "0 0 20px rgba(139,92,246,0.15), inset 0 0 20px rgba(6,182,212,0.05)" : "none",
      }}
    >
      <canvas ref={canvasRef} className="w-full h-full" />
    </div>
  );
}
