import React, { useEffect, useRef } from "react";

const PARTICLE_COUNT = 60;

function randomBetween(a, b) { return a + Math.random() * (b - a); }

export default function ParticleField() {
  const canvasRef = useRef(null);
  const particles = useRef([]);
  const raf       = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const resize = () => {
      canvas.width  = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener("resize", resize);

    // Init particles
    const COLORS = [
      "rgba(139,92,246,",   // violet
      "rgba(6,182,212,",    // cyan
      "rgba(244,63,94,",    // rose
      "rgba(16,185,129,",   // emerald
    ];
    particles.current = Array.from({ length: PARTICLE_COUNT }, () => ({
      x:    Math.random() * canvas.width,
      y:    Math.random() * canvas.height,
      r:    randomBetween(0.5, 2.5),
      vx:   randomBetween(-0.2, 0.2),
      vy:   randomBetween(-0.3, -0.05),
      color:COLORS[Math.floor(Math.random() * COLORS.length)],
      alpha:randomBetween(0.2, 0.6),
      life: Math.random(),
    }));

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      particles.current.forEach((p) => {
        // Move
        p.x  += p.vx;
        p.y  += p.vy;
        p.life = (p.life + 0.002) % 1;
        const opacity = Math.sin(p.life * Math.PI) * p.alpha;

        // Wrap
        if (p.y < -5)             p.y = canvas.height + 5;
        if (p.x < -5)             p.x = canvas.width  + 5;
        if (p.x > canvas.width+5) p.x = -5;

        // Draw glow dot
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `${p.color}${opacity.toFixed(2)})`;
        ctx.fill();

        // Soft outer glow
        const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 6);
        grd.addColorStop(0, `${p.color}${(opacity * 0.4).toFixed(2)})`);
        grd.addColorStop(1, `${p.color}0)`);
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r * 6, 0, Math.PI * 2);
        ctx.fillStyle = grd;
        ctx.fill();
      });

      // Draw thin connections between nearby particles
      particles.current.forEach((a, i) => {
        particles.current.slice(i + 1).forEach((b) => {
          const dx   = a.x - b.x;
          const dy   = a.y - b.y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 120) {
            const alpha = (1 - dist / 120) * 0.06;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y);
            ctx.lineTo(b.x, b.y);
            ctx.strokeStyle = `rgba(139,92,246,${alpha})`;
            ctx.lineWidth   = 0.5;
            ctx.stroke();
          }
        });
      });

      raf.current = requestAnimationFrame(draw);
    };

    raf.current = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf.current);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 z-0 pointer-events-none"
      style={{ opacity: 0.7 }}
    />
  );
}
