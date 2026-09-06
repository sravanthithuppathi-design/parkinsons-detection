import React, { useState, useEffect } from "react";
import { Routes, Route, NavLink, useLocation } from "react-router-dom";
import {
  Brain, Activity, History, BookOpen, BarChart2,
  Menu, X, AlertTriangle, WifiOff, Loader2, Sun, Moon,
} from "lucide-react";
import AnalyzePage   from "./pages/AnalyzePage.jsx";
import HistoryPage   from "./pages/HistoryPage.jsx";
import DashboardPage from "./pages/DashboardPage.jsx";
import AboutPage     from "./pages/AboutPage.jsx";
import ParticleField from "./components/ParticleField.jsx";
import { checkHealth } from "./api/client.js";

const NAV = [
  { to: "/",          label: "Analyze",   icon: Activity,  end: true },
  { to: "/dashboard", label: "Dashboard", icon: BarChart2           },
  { to: "/history",   label: "History",   icon: History            },
  { to: "/about",     label: "About",     icon: BookOpen           },
];

export default function App() {
  const [apiStatus,   setApiStatus]   = useState("checking");
  const [mobileOpen,  setMobileOpen]  = useState(false);
  const [scrolled,    setScrolled]    = useState(false);
  const [darkMode,    setDarkMode]    = useState(true);
  const location = useLocation();

  useEffect(() => { setMobileOpen(false); }, [location]);

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", fn);
    return () => window.removeEventListener("scroll", fn);
  }, []);

  // Apply theme to body
  useEffect(() => {
    if (darkMode) {
      document.body.style.background = "";
      document.body.style.color = "";
      document.documentElement.classList.remove("light-mode");
    } else {
      document.body.style.background = "#f1f5f9";
      document.body.style.color = "#0f172a";
      document.documentElement.classList.add("light-mode");
    }
  }, [darkMode]);

  useEffect(() => {
    const ping = async () => {
      try { await checkHealth(); setApiStatus("online"); }
      catch { setApiStatus("offline"); }
    };
    ping();
    const id = setInterval(ping, 20000);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column", position: "relative", overflow: "hidden" }}>

      {/* Particle background */}
      <ParticleField />

      {/* Ambient orbs */}
      <div style={{ position: "fixed", inset: 0, pointerEvents: "none", zIndex: 0, overflow: "hidden" }}>
        <div className="animate-float-slow" style={{ position: "absolute", width: 600, height: 600, borderRadius: "50%", opacity: 0.07, filter: "blur(120px)", background: "radial-gradient(circle, #8b5cf6, transparent)", top: "-10%", left: "-10%" }} />
        <div className="animate-float-med"  style={{ position: "absolute", width: 500, height: 500, borderRadius: "50%", opacity: 0.06, filter: "blur(100px)", background: "radial-gradient(circle, #06b6d4, transparent)", top: "30%", right: "-10%", animationDelay: "2s" }} />
        <div className="animate-float-slow" style={{ position: "absolute", width: 400, height: 400, borderRadius: "50%", opacity: 0.05, filter: "blur(80px)",  background: "radial-gradient(circle, #f43f5e, transparent)", bottom: "10%", left: "30%", animationDelay: "4s" }} />
        {/* Grid */}
        <div style={{ position: "absolute", inset: 0, opacity: 0.025, backgroundImage: "linear-gradient(rgba(255,255,255,0.4) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.4) 1px, transparent 1px)", backgroundSize: "60px 60px" }} />
      </div>

      {/* ── Navbar ── */}
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 50,
          transition: "all 0.4s",
          background: scrolled ? "rgba(3,7,18,0.85)" : "transparent",
          backdropFilter: scrolled ? "blur(24px)" : "none",
          borderBottom: scrolled ? "1px solid rgba(255,255,255,0.07)" : "1px solid transparent",
        }}
      >
        <div style={{ maxWidth: 1280, margin: "0 auto", padding: "0 24px", height: 64, display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}>

          {/* Logo */}
          <NavLink to="/" style={{ display: "flex", alignItems: "center", gap: 12, textDecoration: "none", flexShrink: 0 }}>
            <div style={{ position: "relative", width: 40, height: 40 }}>
              <div style={{ position: "absolute", inset: 0, borderRadius: 12, background: "linear-gradient(135deg, #7c3aed, #0891b2)", opacity: 0.75, filter: "blur(6px)" }} />
              <div style={{ position: "relative", width: 40, height: 40, borderRadius: 12, background: "linear-gradient(135deg, #7c3aed, #0891b2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <Brain size={20} color="#fff" />
              </div>
            </div>
            <div className="hidden sm:block">
              <p style={{ fontSize: 15, fontWeight: 800, background: "linear-gradient(135deg, #a78bfa, #22d3ee)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", lineHeight: 1 }}>PD Detect</p>
              <p style={{ fontSize: 10, color: "rgba(255,255,255,0.3)", marginTop: 2, fontWeight: 500, letterSpacing: "0.05em" }}>Voice Analysis AI</p>
            </div>
          </NavLink>

          {/* Desktop nav */}
          <nav
            className="hidden md:flex"
            style={{ alignItems: "center", padding: 4, background: "rgba(255,255,255,0.05)", backdropFilter: "blur(12px)", border: "1px solid rgba(255,255,255,0.09)", borderRadius: 18, gap: 2 }}
          >
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                style={({ isActive }) => ({
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "8px 14px",
                  borderRadius: 12,
                  fontSize: 13,
                  fontWeight: 600,
                  textDecoration: "none",
                  transition: "all 0.2s",
                  background: isActive ? "linear-gradient(135deg, rgba(124,58,237,0.8), rgba(8,145,178,0.8))" : "transparent",
                  color: isActive ? "#fff" : "rgba(255,255,255,0.5)",
                  boxShadow: isActive ? "0 2px 12px rgba(139,92,246,0.25)" : "none",
                })}
                onMouseEnter={(e) => {
                  if (!e.currentTarget.style.background.includes("gradient")) {
                    e.currentTarget.style.background = "rgba(255,255,255,0.07)";
                    e.currentTarget.style.color = "#fff";
                  }
                }}
                onMouseLeave={(e) => {
                  const isAct = e.currentTarget.getAttribute("aria-current") === "page";
                  if (!isAct) {
                    e.currentTarget.style.background = "transparent";
                    e.currentTarget.style.color = "rgba(255,255,255,0.5)";
                  }
                }}
              >
                <Icon size={14} />
                {label}
              </NavLink>
            ))}
          </nav>

          {/* Right side */}
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            {/* Dark/Light toggle */}
            <button
              onClick={() => setDarkMode(!darkMode)}
              title={darkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
              style={{
                background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.12)",
                borderRadius: 10, padding: "6px 10px", cursor: "pointer",
                color: "rgba(255,255,255,0.7)", display: "flex", alignItems: "center",
                gap: 5, fontSize: 12, fontWeight: 600, transition: "all 0.2s",
              }}
            >
              {darkMode ? <Sun size={14} /> : <Moon size={14} />}
              <span className="hidden sm:inline">{darkMode ? "Light" : "Dark"}</span>
            </button>
            {/* API status badge */}
            <div
              className="hidden sm:flex"
              style={{ alignItems: "center", gap: 6, padding: "6px 12px", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.09)", borderRadius: 12, fontSize: 12, fontWeight: 600 }}
            >
              {apiStatus === "checking" && <><Loader2 size={12} color="#fbbf24" className="animate-spin" /><span style={{ color: "#fbbf24" }}>Connecting</span></>}
              {apiStatus === "online"   && <><div style={{ width: 7, height: 7, borderRadius: "50%", background: "#10b981", boxShadow: "0 0 6px #10b981" }} /><span style={{ color: "#10b981" }}>API Online</span></>}
              {apiStatus === "offline"  && <><WifiOff size={12} color="#f43f5e" /><span style={{ color: "#f43f5e" }}>API Offline</span></>}
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="md:hidden"
              style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.09)", borderRadius: 10, padding: 8, cursor: "pointer", color: "rgba(255,255,255,0.6)", display: "flex", alignItems: "center", justifyContent: "center" }}
            >
              {mobileOpen ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>

        {/* Mobile nav drawer */}
        {mobileOpen && (
          <div
            className="md:hidden animate-fade-up"
            style={{ borderTop: "1px solid rgba(255,255,255,0.07)", background: "rgba(3,7,18,0.97)", backdropFilter: "blur(24px)", padding: "12px 16px", display: "flex", flexDirection: "column", gap: 4 }}
          >
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                style={({ isActive }) => ({
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: "12px 16px",
                  borderRadius: 14,
                  fontSize: 14,
                  fontWeight: 600,
                  textDecoration: "none",
                  transition: "all 0.2s",
                  background: isActive ? "linear-gradient(135deg, rgba(124,58,237,0.25), rgba(8,145,178,0.25))" : "transparent",
                  border: isActive ? "1px solid rgba(139,92,246,0.3)" : "1px solid transparent",
                  color: isActive ? "#c4b5fd" : "rgba(255,255,255,0.45)",
                })}
              >
                <Icon size={16} />
                {label}
              </NavLink>
            ))}
          </div>
        )}
      </header>

      {/* ── Disclaimer banner ── */}
      <div style={{ position: "relative", zIndex: 10, borderBottom: "1px solid rgba(251,191,36,0.15)", background: "rgba(251,191,36,0.05)", padding: "8px 24px" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", display: "flex", alignItems: "center", gap: 8 }}>
          <AlertTriangle size={13} color="rgba(251,191,36,0.8)" style={{ flexShrink: 0 }} />
          <span style={{ fontSize: 12, color: "rgba(251,191,36,0.7)" }}>
            <strong style={{ color: "rgba(251,191,36,0.9)", fontWeight: 700 }}>Research Only.</strong>{" "}
            Not a medical diagnosis. Always consult a qualified neurologist for health concerns.
          </span>
        </div>
      </div>

      {/* ── Main content ── */}
      <main style={{ flex: 1, position: "relative", zIndex: 10 }}>
        <Routes>
          <Route path="/"          element={<AnalyzePage   apiStatus={apiStatus} />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/history"   element={<HistoryPage   />} />
          <Route path="/about"     element={<AboutPage     />} />
        </Routes>
      </main>

      {/* ── Footer ── */}
      <footer style={{ position: "relative", zIndex: 10, borderTop: "1px solid rgba(255,255,255,0.07)", background: "rgba(3,7,18,0.7)", backdropFilter: "blur(20px)", padding: "28px 24px" }}>
        <div style={{ maxWidth: 1280, margin: "0 auto", display: "flex", flexDirection: "column", gap: 16, alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 32, height: 32, borderRadius: 10, background: "linear-gradient(135deg, #7c3aed, #0891b2)", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <Brain size={16} color="#fff" />
            </div>
            <span style={{ fontWeight: 800, fontSize: 15, background: "linear-gradient(135deg, #a78bfa, #22d3ee)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>PD Detect</span>
            <span style={{ fontSize: 11, color: "rgba(255,255,255,0.2)" }}>v2.0 · Python + ML · UCI Dataset</span>
          </div>
          <p style={{ fontSize: 11, color: "rgba(255,255,255,0.2)", textAlign: "center", maxWidth: 600, lineHeight: 1.7 }}>
            Built with Python (FastAPI + scikit-learn + XGBoost + librosa) and React.
            Trained on the UCI Parkinson's Dataset. For research and educational purposes only.
            Not FDA approved. Results do not constitute a medical diagnosis.
          </p>
        </div>
      </footer>
    </div>
  );
}
