import React, { useState, useRef, useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import toast from "react-hot-toast";
import {
  Mic, Square, Upload, Play, Pause, Trash2, Zap,
  Loader2, CheckCircle2, AlertCircle, Info, Clock, FileAudio,
} from "lucide-react";
import WaveformVisualizer from "./WaveformVisualizer.jsx";
import { analyzeAudio } from "../api/client.js";

export default function AudioRecorder({ onComplete, onStart, isAnalyzing }) {
  const [recState,     setRecState]     = useState("idle");
  const [blob,         setBlob]         = useState(null);
  const [audioUrl,     setAudioUrl]     = useState(null);
  const [filename,     setFilename]     = useState("");
  const [isPlaying,    setIsPlaying]    = useState(false);
  const [duration,     setDuration]     = useState(0);
  const [progress,     setProgress]     = useState(0);
  const [error,        setError]        = useState(null);
  const [analyserNode, setAnalyserNode] = useState(null);

  const mediaRecRef = useRef(null);
  const chunksRef   = useRef([]);
  const audioCtxRef = useRef(null);
  const streamRef   = useRef(null);
  const audioRef    = useRef(null);
  const timerRef    = useRef(null);

  useEffect(() => {
    if (recState === "recording") {
      timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000);
    } else {
      clearInterval(timerRef.current);
      if (recState !== "ready") setDuration(0);
    }
    return () => clearInterval(timerRef.current);
  }, [recState]);

  const fmt = (s) =>
    `${Math.floor(s / 60).toString().padStart(2, "0")}:${(s % 60).toString().padStart(2, "0")}`;

  const reset = useCallback(() => {
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (audioCtxRef.current?.state !== "closed") audioCtxRef.current?.close();
    setAnalyserNode(null);
    setBlob(null);
    setAudioUrl(null);
    setFilename("");
    setIsPlaying(false);
    setDuration(0);
    setProgress(0);
    setError(null);
    setRecState("idle");
  }, [audioUrl]);

  const startRecording = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      audioCtxRef.current = ctx;
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      src.connect(analyser);
      setAnalyserNode(analyser);

      const mime =
        ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus"].find((m) =>
          MediaRecorder.isTypeSupported(m)
        ) || "";
      const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
      mediaRecRef.current = rec;
      chunksRef.current = [];

      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      rec.onstop = () => {
        const b = new Blob(chunksRef.current, { type: mime || "audio/webm" });
        const url = URL.createObjectURL(b);
        setBlob(b);
        setAudioUrl(url);
        setFilename("recording.webm");
        setRecState("ready");
        stream.getTracks().forEach((t) => t.stop());
        if (ctx.state !== "closed") ctx.close();
        setAnalyserNode(null);
      };
      rec.start(100);
      setRecState("recording");
    } catch (e) {
      const msg =
        e.name === "NotAllowedError"
          ? "Microphone access denied. Please allow permissions."
          : e.name === "NotFoundError"
          ? "No microphone found. Connect a microphone and retry."
          : `Recording error: ${e.message}`;
      setError(msg);
      toast.error(msg);
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecRef.current?.state !== "inactive") mediaRecRef.current.stop();
  }, []);

  const onDrop = useCallback(
    (files) => {
      const file = files[0];
      if (!file) return;
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setBlob(file);
      setAudioUrl(URL.createObjectURL(file));
      setFilename(file.name);
      setRecState("ready");
      setError(null);
      toast.success(`Loaded: ${file.name}`);
    },
    [audioUrl]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "audio/*": [".wav", ".mp3", ".ogg", ".webm", ".flac", ".m4a"] },
    maxFiles: 1,
    disabled: recState !== "idle" || isAnalyzing,
  });

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleAnalyze = useCallback(async () => {
    if (!blob) return;
    setError(null);
    setProgress(0);
    onStart?.();
    try {
      const result = await analyzeAudio(blob, filename, setProgress);
      onComplete?.(result);
      toast.success("Analysis complete!");
    } catch (e) {
      const msg = e.userMessage || "Analysis failed. Is the backend running on port 8000?";
      setError(msg);
      toast.error(msg);
      onStart?.();
    }
  }, [blob, filename, onComplete, onStart]);

  /* ─────────────────────────────────── RENDER ─────────────────────────── */
  return (
    <div className="max-w-2xl mx-auto">
      <div
        style={{
          background: "rgba(5, 8, 22, 0.85)",
          backdropFilter: "blur(24px)",
          border: "1px solid rgba(139,92,246,0.25)",
          borderRadius: "24px",
          overflow: "hidden",
        }}
      >
        {/* ── Header ── */}
        <div
          style={{ borderBottom: "1px solid rgba(255,255,255,0.07)", padding: "20px 24px 16px" }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <div
              style={{
                width: 40,
                height: 40,
                borderRadius: 14,
                background: "linear-gradient(135deg, rgba(139,92,246,0.3), rgba(6,182,212,0.3))",
                border: "1px solid rgba(139,92,246,0.35)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
              }}
            >
              <Mic size={18} color="#a78bfa" />
            </div>
            <div style={{ flex: 1 }}>
              <p style={{ color: "#fff", fontWeight: 700, fontSize: 16, margin: 0 }}>
                Voice Recorder
              </p>
              <p style={{ color: "rgba(255,255,255,0.35)", fontSize: 12, marginTop: 2 }}>
                Record or upload — sustain "ahhh" for 5–10 seconds
              </p>
            </div>
            {recState === "recording" && (
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div
                  className="recording-ring"
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: "#f43f5e",
                    flexShrink: 0,
                  }}
                />
                <span
                  style={{
                    fontFamily: "monospace",
                    color: "#f43f5e",
                    fontWeight: 700,
                    fontSize: 14,
                  }}
                >
                  {fmt(duration)}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* ── Waveform ── */}
        <div style={{ padding: "20px 24px" }}>
          <WaveformVisualizer
            isRecording={recState === "recording"}
            analyserNode={analyserNode}
          />
        </div>

        {/* ── Controls ── */}
        <div style={{ padding: "0 24px 24px", display: "flex", flexDirection: "column", gap: 12 }}>

          {/* IDLE */}
          {recState === "idle" && (
            <>
              {/* Record button */}
              <button
                onClick={startRecording}
                disabled={isAnalyzing}
                className="recording-ring"
                style={{
                  width: "100%",
                  padding: "16px",
                  borderRadius: 16,
                  background: "linear-gradient(135deg, #e11d48, #ea580c)",
                  border: "none",
                  color: "#fff",
                  fontWeight: 700,
                  fontSize: 16,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 12,
                  transition: "all 0.3s",
                  opacity: isAnalyzing ? 0.4 : 1,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.boxShadow =
                    "0 0 40px rgba(244,63,94,0.4), 0 0 80px rgba(244,63,94,0.2)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = "none";
                }}
              >
                <div
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: "50%",
                    border: "2px solid rgba(255,255,255,0.6)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Mic size={16} />
                </div>
                Start Recording
              </button>

              {/* Divider — NO border-white/8, plain inline style */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: "4px 0",
                }}
              >
                <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,0.08)" }} />
                <span
                  style={{
                    fontSize: 11,
                    fontWeight: 600,
                    color: "rgba(255,255,255,0.25)",
                    padding: "4px 10px",
                    background: "rgba(255,255,255,0.05)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: 20,
                    whiteSpace: "nowrap",
                    letterSpacing: "0.05em",
                  }}
                >
                  or upload a file
                </span>
                <div style={{ flex: 1, height: 1, background: "rgba(255,255,255,0.08)" }} />
              </div>

              {/* Drop zone */}
              <div
                {...getRootProps()}
                style={{
                  borderRadius: 16,
                  border: isDragActive
                    ? "2px dashed #8b5cf6"
                    : "2px dashed rgba(255,255,255,0.10)",
                  padding: "32px 24px",
                  textAlign: "center",
                  cursor: "pointer",
                  background: isDragActive ? "rgba(139,92,246,0.08)" : "rgba(255,255,255,0.02)",
                  transition: "all 0.25s",
                }}
                onMouseEnter={(e) => {
                  if (!isDragActive) {
                    e.currentTarget.style.borderColor = "rgba(255,255,255,0.22)";
                    e.currentTarget.style.background = "rgba(255,255,255,0.04)";
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isDragActive) {
                    e.currentTarget.style.borderColor = "rgba(255,255,255,0.10)";
                    e.currentTarget.style.background = "rgba(255,255,255,0.02)";
                  }
                }}
              >
                <input {...getInputProps()} />
                <Upload
                  size={28}
                  color={isDragActive ? "#8b5cf6" : "rgba(255,255,255,0.22)"}
                  style={{ margin: "0 auto 10px" }}
                />
                <p
                  style={{
                    fontWeight: 600,
                    fontSize: 14,
                    color: isDragActive ? "#a78bfa" : "rgba(255,255,255,0.35)",
                    marginBottom: 4,
                  }}
                >
                  {isDragActive ? "Drop audio file here" : "Drag & drop audio file"}
                </p>
                <p style={{ fontSize: 11, color: "rgba(255,255,255,0.18)" }}>
                  WAV · MP3 · OGG · WebM · FLAC
                </p>
              </div>
            </>
          )}

          {/* RECORDING */}
          {recState === "recording" && (
            <button
              onClick={stopRecording}
              style={{
                width: "100%",
                padding: "16px",
                borderRadius: 16,
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.12)",
                color: "#fff",
                fontWeight: 700,
                fontSize: 16,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 12,
                transition: "all 0.2s",
              }}
            >
              <Square size={20} fill="#fff" />
              Stop Recording
            </button>
          )}

          {/* READY */}
          {recState === "ready" && (
            <>
              {/* Audio player */}
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: 12,
                  borderRadius: 16,
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(255,255,255,0.09)",
                }}
              >
                <button
                  onClick={togglePlay}
                  style={{
                    width: 40,
                    height: 40,
                    borderRadius: 12,
                    background: "linear-gradient(135deg, #7c3aed, #0891b2)",
                    border: "none",
                    cursor: "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flexShrink: 0,
                    transition: "all 0.2s",
                  }}
                >
                  {isPlaying ? (
                    <Pause size={16} color="#fff" />
                  ) : (
                    <Play size={16} color="#fff" />
                  )}
                </button>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <p
                    style={{
                      color: "#fff",
                      fontWeight: 600,
                      fontSize: 13,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                    }}
                  >
                    <FileAudio size={13} color="#a78bfa" />
                    {filename || "Audio file"}
                  </p>
                  {duration > 0 && (
                    <p
                      style={{
                        color: "rgba(255,255,255,0.28)",
                        fontSize: 11,
                        marginTop: 2,
                        display: "flex",
                        alignItems: "center",
                        gap: 4,
                      }}
                    >
                      <Clock size={11} /> {fmt(duration)}
                    </p>
                  )}
                </div>
                <CheckCircle2 size={18} color="#10b981" />
                <audio
                  ref={audioRef}
                  src={audioUrl}
                  onEnded={() => setIsPlaying(false)}
                  style={{ display: "none" }}
                />
              </div>

              {/* Analyze button */}
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                style={{
                  width: "100%",
                  padding: "16px",
                  borderRadius: 16,
                  background: isAnalyzing
                    ? "rgba(139,92,246,0.4)"
                    : "linear-gradient(135deg, #7c3aed, #0891b2)",
                  border: "none",
                  color: "#fff",
                  fontWeight: 700,
                  fontSize: 16,
                  cursor: isAnalyzing ? "not-allowed" : "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 10,
                  transition: "all 0.3s",
                }}
                onMouseEnter={(e) => {
                  if (!isAnalyzing)
                    e.currentTarget.style.boxShadow =
                      "0 0 40px rgba(139,92,246,0.45), 0 0 80px rgba(139,92,246,0.2)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.boxShadow = "none";
                }}
              >
                {isAnalyzing ? (
                  <>
                    <Loader2 size={20} className="animate-spin" />
                    Analyzing voice patterns…
                    {progress > 0 && progress < 100 && ` (${progress}%)`}
                  </>
                ) : (
                  <>
                    <Zap size={20} />
                    Analyze Voice
                  </>
                )}
              </button>

              {/* Progress bar */}
              {isAnalyzing && progress > 0 && (
                <div
                  style={{
                    width: "100%",
                    height: 4,
                    background: "rgba(255,255,255,0.06)",
                    borderRadius: 99,
                    overflow: "hidden",
                  }}
                >
                  <div
                    style={{
                      height: "100%",
                      width: `${progress}%`,
                      background: "linear-gradient(90deg, #8b5cf6, #06b6d4)",
                      borderRadius: 99,
                      transition: "width 0.3s ease",
                    }}
                  />
                </div>
              )}

              {/* Discard */}
              <button
                onClick={reset}
                disabled={isAnalyzing}
                style={{
                  width: "100%",
                  padding: "10px",
                  borderRadius: 12,
                  background: "transparent",
                  border: "none",
                  color: "rgba(255,255,255,0.25)",
                  fontSize: 13,
                  cursor: "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 6,
                  transition: "color 0.2s",
                }}
                onMouseEnter={(e) => (e.currentTarget.style.color = "rgba(255,255,255,0.55)")}
                onMouseLeave={(e) => (e.currentTarget.style.color = "rgba(255,255,255,0.25)")}
              >
                <Trash2 size={14} /> Discard &amp; try again
              </button>

              {/* Quality tip */}
              <div style={{
                padding: "10px 14px", borderRadius: "12px",
                background: "rgba(139,92,246,0.08)", border: "1px solid rgba(139,92,246,0.2)",
                display: "flex", alignItems: "flex-start", gap: "8px",
              }}>
                <span style={{ fontSize: "14px", flexShrink: 0 }}>💡</span>
                <p style={{ fontSize: "11px", color: "rgba(255,255,255,0.45)", margin: 0, lineHeight: 1.6 }}>
                  <strong style={{ color: "rgba(139,92,246,0.9)" }}>For best results:</strong> Record in a quiet room · Say &quot;Aaah&quot; steadily for 5–10 seconds · Keep mic ~15cm from mouth · Avoid background noise
                </p>
              </div>
            </>
          )}

          {/* Error */}
          {error && (
            <div
              style={{
                borderRadius: 14,
                background: "rgba(244,63,94,0.1)",
                border: "1px solid rgba(244,63,94,0.3)",
                padding: "12px 14px",
                display: "flex",
                alignItems: "flex-start",
                gap: 10,
              }}
            >
              <AlertCircle size={15} color="#f87171" style={{ flexShrink: 0, marginTop: 1 }} />
              <p style={{ fontSize: 13, color: "#fca5a5", lineHeight: 1.5 }}>{error}</p>
            </div>
          )}
        </div>

        {/* ── Tips ── */}
        {recState === "idle" && (
          <div style={{ padding: "0 24px 24px" }}>
            <div
              style={{
                borderRadius: 16,
                background: "rgba(255,255,255,0.025)",
                border: "1px solid rgba(255,255,255,0.06)",
                padding: "14px 16px",
              }}
            >
              <p
                style={{
                  fontSize: 10,
                  fontWeight: 700,
                  textTransform: "uppercase",
                  letterSpacing: "0.12em",
                  color: "rgba(255,255,255,0.28)",
                  marginBottom: 10,
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                <Info size={12} /> Recording Tips
              </p>
              <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
                {[
                  "Record in a quiet room — minimize background noise.",
                  'Sustain a vowel "ahhh" steadily for 5–10 seconds.',
                  "Hold the microphone 10–15 cm from your mouth.",
                  "Speak at a natural, comfortable volume.",
                ].map((tip, i) => (
                  <li
                    key={i}
                    style={{
                      fontSize: 12,
                      color: "rgba(255,255,255,0.32)",
                      display: "flex",
                      alignItems: "flex-start",
                      gap: 8,
                      marginBottom: i < 3 ? 6 : 0,
                      lineHeight: 1.5,
                    }}
                  >
                    <span style={{ color: "#8b5cf6", marginTop: 1, flexShrink: 0 }}>›</span>
                    {tip}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
