import axios from "axios";

const BASE = import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD ? "https://pd-detect-api.onrender.com" : "http://localhost:8000");

const api = axios.create({ baseURL: BASE, timeout: 120000 });

api.interceptors.response.use(
  (r) => r,
  (err) => {
    const isNet =
      err.code === "ERR_NETWORK" ||
      err.code === "ECONNREFUSED" ||
      err.message?.toLowerCase().includes("network");
    err.userMessage = isNet
      ? "Server is waking up — please wait 30 seconds and try again."
      : err.response?.data?.detail || err.message || "An unexpected error occurred.";
    return Promise.reject(err);
  }
);

// ── localStorage persistence ──────────────────────────────────────────────
const HISTORY_KEY = "pd_detect_history_v2";

export const saveToLocalHistory = (result) => {
  try {
    const existing = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    const updated = [result, ...existing].slice(0, 100);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
  } catch {}
};

export const getLocalHistory = () => {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
};

export const clearLocalHistory = () => {
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch {}
};

export const deleteLocalEntry = (id) => {
  try {
    const existing = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
    localStorage.setItem(
      HISTORY_KEY,
      JSON.stringify(existing.filter((h) => h.id !== id))
    );
  } catch {}
};

// ── API calls ─────────────────────────────────────────────────────────────
export const checkHealth = () => api.get("/health").then((r) => r.data);
export const getStats    = () => api.get("/stats").then((r) => r.data);

export const seedDemoData = () =>
  api.post("/seed-demo").then((r) => {
    // Store each result in localStorage history
    const results = r.data.results || [];
    results.forEach((entry) => saveToLocalHistory(entry));
    return r.data;
  });

export const analyzeAudio = (blob, filename, onProgress) => {
  const form = new FormData();
  form.append("file", blob, filename);
  return api
    .post("/analyze", form, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (e) => {
        if (onProgress && e.total)
          onProgress(Math.round((e.loaded / e.total) * 100));
      },
    })
    .then((r) => {
      saveToLocalHistory(r.data); // persist immediately to localStorage
      return r.data;
    });
};

export const downloadReport = async (result) => {
  const response = await api.post("/report", result, { responseType: "blob" });
  const url = URL.createObjectURL(
    new Blob([response.data], { type: "application/pdf" })
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = `pd_report_${result.id || Date.now()}.pdf`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

export default api;
