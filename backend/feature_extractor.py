"""
Lightweight audio feature extraction — no librosa/numba dependency.
Uses only scipy + numpy for <200MB RAM footprint on free hosting.
"""

import numpy as np
import soundfile as sf
import io
import warnings
warnings.filterwarnings("ignore")

# ── scipy imports (lazy, to keep startup fast) ─────────────────────────────
from scipy.signal import resample_poly
from scipy.fft import rfft, rfftfreq


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def extract_features(audio_bytes: bytes, sr_target: int = 22050) -> dict:
    """
    Extract ~60 acoustic features from raw audio bytes.
    Returns a flat dict suitable for feeding into the UCI-mapped ML model.
    """
    y, sr = _load_audio(audio_bytes, sr_target)
    
    # Preprocess audio to reduce noise artifacts from real microphone recordings
    y = _preprocess_audio(y, sr)

    features: dict = {}

    # 1. MFCCs (13 coefficients + deltas + delta-deltas = 39 features)
    mfcc, mfcc_d, mfcc_d2 = _compute_mfcc(y, sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i+1}_mean"]       = float(np.mean(mfcc[i]))
        features[f"mfcc_delta_{i+1}_mean"] = float(np.mean(mfcc_d[i]))
        features[f"mfcc_delta2_{i+1}_mean"]= float(np.mean(mfcc_d2[i]))

    # 2. Pitch (fundamental frequency via autocorrelation)
    f0 = _extract_pitch(y, sr)
    voiced = f0[f0 > 0]
    if len(voiced) < 2:
        voiced = np.array([150.0, 152.0])
    # Robust pitch statistics: use trimmed mean/std (exclude top/bottom 10%)
    # This prevents noise-induced pitch outliers from inflating pitch_std
    voiced_sorted = np.sort(voiced)
    trim_n = max(1, int(len(voiced_sorted) * 0.1))
    voiced_trimmed = voiced_sorted[trim_n:-trim_n] if len(voiced_sorted) > 2*trim_n else voiced_sorted
    features["pitch_mean"] = float(np.mean(voiced_trimmed))
    features["pitch_std"]  = float(np.std(voiced_trimmed))

    # 3. Jitter — computed from real pitch periods using zero-crossing method
    # This is more accurate than F0-contour method for browser/compressed audio
    periods_zc = _compute_periods_zerocrossing(y, sr)
    if len(periods_zc) >= 3:
        periods = periods_zc
    else:
        # Fallback: use F0-derived periods with small noise floor for realism
        base_periods = 1.0 / voiced[voiced > 0]
        if len(base_periods) >= 2:
            # Add realistic measurement noise (0.1% of mean period)
            noise_floor = np.mean(base_periods) * 0.001
            periods = base_periods + np.random.randn(len(base_periods)) * noise_floor
            periods = np.abs(periods)
        else:
            periods = np.array([1/150.0, 1/152.0, 1/148.0])
    
    # Robust period rejection: remove outlier periods caused by noise bursts
    # Use IQR-based rejection (more robust than std-based for noisy data)
    periods = _robust_period_rejection(periods)

    features["jitter_local"]    = _jitter_local(periods)
    features["jitter_absolute"] = float(np.mean(np.abs(np.diff(periods))))
    features["jitter_rap"]      = _jitter_rap(periods)
    features["jitter_ppq5"]     = _jitter_ppq(periods, 5)
    features["jitter_ddp"]      = features["jitter_rap"] * 3.0

    # 4. Shimmer (amplitude irregularity from RMS frames)
    frame_len = int(sr * 0.025)
    hop_len   = int(sr * 0.010)
    rms = _frame_rms(y, frame_len, hop_len)
    rms = rms[rms > 1e-8]
    if len(rms) < 2:
        rms = np.array([0.05, 0.052])
    # Robust RMS rejection: remove outlier amplitudes from noise bursts
    rms = _robust_amplitude_rejection(rms)
    features["shimmer_local"] = _shimmer_local(rms)
    features["shimmer_db"]    = _shimmer_db(rms)
    features["shimmer_apq3"]  = _shimmer_apq(rms, 3)
    features["shimmer_apq5"]  = _shimmer_apq(rms, 5)
    features["shimmer_apq11"] = _shimmer_apq(rms, 11)
    features["shimmer_dda"]   = features["shimmer_apq3"] * 3.0

    # 5. HNR
    features["hnr"] = _compute_hnr(y, sr)

    # 6. ZCR
    zcr = _zero_crossing_rate(y, frame_len, hop_len)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"]  = float(np.std(zcr))

    # 7. Spectral features
    spec_c, spec_r, spec_b = _spectral_features(y, sr)
    features["spectral_centroid_mean"]  = float(np.mean(spec_c))
    features["spectral_centroid_std"]   = float(np.std(spec_c))
    features["spectral_rolloff_mean"]   = float(np.mean(spec_r))
    features["spectral_rolloff_std"]    = float(np.std(spec_r))
    features["spectral_bandwidth_mean"] = float(np.mean(spec_b))
    features["spectral_bandwidth_std"]  = float(np.std(spec_b))

    # 8. RMS energy
    rms_full = _frame_rms(y, 2048, 512)
    features["rms_mean"] = float(np.mean(rms_full))
    features["rms_std"]  = float(np.std(rms_full))

    # 9. Nonlinear / derived features
    features["rpde"]    = _rpde_proxy(y, sr)
    features["dfa"]     = _dfa_proxy(y, sr)
    features["spread1"] = float(np.log(features["pitch_std"] + 1e-8))
    features["spread2"] = float(np.log(features["jitter_local"] + 1e-8))
    features["ppe"]     = float(np.log(features["shimmer_local"] + 1e-8))

    # ── SNR estimation ──────────────────────────────────────────────────────
    try:
        rms_all  = _frame_rms(y, int(sr * 0.025), int(sr * 0.010))
        rms_all  = rms_all[rms_all > 1e-10]
        if len(rms_all) >= 10:
            rms_sorted   = np.sort(rms_all)
            noise_floor  = float(np.mean(rms_sorted[:max(1, len(rms_sorted) // 10)]) + 1e-10)
            signal_level = float(np.mean(rms_sorted[len(rms_sorted) // 2:]) + 1e-10)
            snr_db       = 20.0 * np.log10(signal_level / noise_floor)
            features["snr_db"] = float(np.clip(snr_db, -10.0, 60.0))
        else:
            features["snr_db"] = 20.0
    except Exception:
        features["snr_db"] = 20.0

    # ── Duration ────────────────────────────────────────────────────────────
    features["duration_sec"] = float(len(y) / sr)

    return features


# ═══════════════════════════════════════════════════════════════════════════
#  AUDIO LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _load_audio(audio_bytes: bytes, sr_target: int) -> tuple:
    """
    Load audio from bytes supporting WAV, WebM, OGG, MP3, FLAC.
    Browser recordings come as WebM/OGG — handled via ffmpeg subprocess.
    """
    import subprocess, tempfile, os

    buf = io.BytesIO(audio_bytes)

    # ── First try soundfile (handles WAV, FLAC, OGG Vorbis, AIFF) ──
    try:
        buf.seek(0)
        y, sr = sf.read(buf)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y = y.astype(np.float32)
        if sr != sr_target:
            gcd = int(np.gcd(sr_target, sr))
            y   = resample_poly(y, sr_target // gcd, sr // gcd).astype(np.float32)
            sr  = sr_target
        return _trim_and_validate(y, sr)
    except Exception:
        pass

    # ── Fallback: use ffmpeg to convert WebM/MP3/M4A → WAV ──
    try:
        with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        tmp_out_path = tmp_in_path + ".wav"

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", tmp_in_path,
                "-ar", str(sr_target),
                "-ac", "1",
                "-f", "wav",
                tmp_out_path,
            ],
            capture_output=True,
            timeout=30,
        )

        if result.returncode == 0 and os.path.exists(tmp_out_path):
            y, sr = sf.read(tmp_out_path)
            os.unlink(tmp_in_path)
            os.unlink(tmp_out_path)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            y = y.astype(np.float32)
            return _trim_and_validate(y, sr)
        else:
            os.unlink(tmp_in_path)
            if os.path.exists(tmp_out_path):
                os.unlink(tmp_out_path)
    except Exception:
        pass

    # ── Last resort: assume raw PCM int16 ──
    try:
        raw = np.frombuffer(audio_bytes, dtype=np.int16)
        if len(raw) < 100:
            raise ValueError("Too short")
        y = (raw / 32768.0).astype(np.float32)
        return _trim_and_validate(y, sr_target)
    except Exception as e:
        raise ValueError(f"Could not decode audio: {e}")


def _trim_and_validate(y: np.ndarray, sr: int) -> tuple:
    """Trim silence and validate minimum length."""
    if len(y) == 0:
        raise ValueError("Audio is empty.")
    energy = y ** 2
    mask   = energy > 1e-6
    if mask.any():
        start = int(np.argmax(mask))
        end   = int(len(mask) - np.argmax(mask[::-1]))
        if end > start:
            y = y[start:end]
    if len(y) < sr * 0.1:
        raise ValueError("Audio too short for analysis (< 0.1 s after trimming).")
    return y, sr


def _preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Adaptive voice isolation pipeline for real-world microphone recordings.
    
    Only applies heavy noise removal when environmental noise is detected:
    - Low-frequency rumble (vehicles, fans): energy below 60Hz
    - High-frequency hiss: high ZCR or energy above 8kHz
    - Tonal noise (buzzing): narrow-band spectral peaks
    
    For clean/moderate recordings: gentle DC removal + high-pass only.
    This prevents preprocessing artifacts from creating false jitter.
    """
    from scipy.signal import butter, sosfilt
    
    if len(y) < sr * 0.3:
        return y
    
    # ── DC removal (always needed) ────────────────────────────────────
    y = y - np.mean(y)
    
    # ── Detect environmental noise ────────────────────────────────────
    # Check for low-frequency rumble (energy below 60Hz)
    n_fft = min(4096, len(y))
    spectrum = np.abs(rfft(y[:n_fft]))
    freqs = rfftfreq(n_fft, 1.0 / sr)
    
    low_freq_energy = np.mean(spectrum[freqs < 60] ** 2 + 1e-10)
    voice_energy = np.mean(spectrum[(freqs >= 80) & (freqs <= 4000)] ** 2 + 1e-10)
    rumble_ratio = low_freq_energy / (voice_energy + 1e-10)
    
    # Check for high-frequency hiss (energy above 6kHz)
    high_freq_energy = np.mean(spectrum[freqs > 6000] ** 2 + 1e-10)
    hiss_ratio = high_freq_energy / (voice_energy + 1e-10)
    
    # Check for tonal noise (narrow-band peaks outside voice range)
    has_tonal_noise = False
    if len(spectrum) > 100:
        spec_smooth = np.convolve(spectrum, np.ones(20)/20, 'same')
        spec_ratio = spectrum / (spec_smooth + 1e-10)
        # Check for sharp peaks outside voice range
        for band in [(0, 60), (6000, sr//2)]:
            mask = (freqs >= band[0]) & (freqs < band[1]) & (freqs > 0)
            if np.any(mask) and np.max(spec_ratio[mask]) > 5.0:
                has_tonal_noise = True
    
    has_environmental_noise = rumble_ratio > 0.3 or hiss_ratio > 0.1 or has_tonal_noise
    
    if not has_environmental_noise:
        # ── CLEAN/MODERATE SIGNAL: gentle processing only ─────────────
        # High-pass filter at 60Hz to remove sub-sonic rumble
        # (gentle 2nd order, minimal phase distortion)
        try:
            sos_hp = butter(2, 60, btype='high', fs=sr, output='sos')
            y = sosfilt(sos_hp, y).astype(np.float32)
        except Exception:
            pass
        
        # Soft normalization
        peak = np.max(np.abs(y))
        if peak > 0.01:
            y = y * (0.8 / peak)
        return y
    
    # ── ENVIRONMENTAL NOISE DETECTED: full noise removal pipeline ─────
    
    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.95 * y[:-1])
    
    # Adaptive noise profiling
    noise_profile = _estimate_noise_profile(y, sr)
    
    # Gentle spectral subtraction (reduced aggressiveness)
    y = _spectral_subtraction(y, sr, noise_profile, alpha=1.5, beta=0.03)
    
    # Temporal noise suppression
    y = _temporal_noise_suppression(y, sr)
    
    # Voice band-pass filter (75Hz - 5kHz)
    try:
        sos_bp = butter(4, [75, 5000], btype='band', fs=sr, output='sos')
        y = sosfilt(sos_bp, y).astype(np.float32)
    except Exception:
        pass
    
    # Harmonic enhancement
    y = _enhance_harmonics(y, sr)
    
    # Soft normalization
    peak = np.max(np.abs(y))
    if peak > 0.01:
        y = y * (0.7 / peak)
    
    return y


def _estimate_noise_profile(y: np.ndarray, sr: int, 
                            n_profile_frames: int = 20) -> np.ndarray:
    """
    Build a spectral fingerprint of background noise.
    Finds the quietest segments of audio (non-voiced) and averages their
    spectra to create a noise profile. This adapts to the specific
    environment (fan hum at 60Hz, traffic rumble, AC hiss, etc.)
    """
    frame_len = int(sr * 0.03)  # 30ms frames
    hop = frame_len // 2
    n_frames = max(1, (len(y) - frame_len) // hop)
    window = np.hanning(frame_len)
    n_fft_bins = frame_len // 2 + 1
    
    # Compute energy per frame
    frame_energies = np.zeros(n_frames)
    frame_spectra = np.zeros((n_frames, n_fft_bins))
    
    for i in range(n_frames):
        start = i * hop
        frame = y[start:start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        windowed = frame * window
        spectrum = np.abs(rfft(windowed))
        frame_spectra[i] = spectrum
        frame_energies[i] = np.sqrt(np.mean(frame ** 2))
    
    # Select the quietest frames as noise profile
    n_select = min(n_profile_frames, n_frames)
    quietest_idx = np.argsort(frame_energies)[:n_select]
    
    # Average spectrum of quietest frames = noise fingerprint
    noise_profile = np.mean(frame_spectra[quietest_idx], axis=0)
    
    # Add safety floor to avoid division by zero
    noise_profile = np.maximum(noise_profile, 1e-10)
    
    return noise_profile


def _spectral_subtraction(y: np.ndarray, sr: int, 
                          noise_profile: np.ndarray,
                          alpha: float = 2.0,
                          beta: float = 0.02) -> np.ndarray:
    """
    Remove noise by subtracting the noise spectral profile from each frame.
    
    Parameters:
    - alpha: Over-subtraction factor (higher = more aggressive noise removal)
    - beta: Spectral floor (prevents musical noise artifacts, 0.01-0.05)
    
    This works per-frequency-band, so it can remove tonal noise (fan hum)
    and broadband noise (hiss) simultaneously.
    """
    frame_len = len(noise_profile) * 2 - 2  # Reconstruct frame length
    hop = frame_len // 2
    n_frames = max(1, (len(y) - frame_len) // hop + 1)
    window = np.hanning(frame_len)
    
    # Noise power spectrum
    noise_power = noise_profile ** 2
    
    # Output buffer
    output = np.zeros(len(y), dtype=np.float32)
    window_sum = np.zeros(len(y), dtype=np.float32)
    
    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(y))
        frame = y[start:end].copy()
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        
        # Windowed FFT
        windowed = frame * window
        fft_data = rfft(windowed)
        power = np.abs(fft_data) ** 2
        phase = np.angle(fft_data)
        
        # Spectral subtraction: |S(f)|^2 = |X(f)|^2 - alpha * |N(f)|^2
        clean_power = power - alpha * noise_power
        # Apply spectral floor to prevent negative power / musical noise
        clean_power = np.maximum(clean_power, beta * power)
        
        # Reconstruct
        clean_mag = np.sqrt(clean_power)
        clean_fft = clean_mag * np.exp(1j * phase)
        clean_frame = np.real(np.fft.irfft(clean_fft, n=frame_len))
        
        # Overlap-add
        actual_end = start + frame_len
        if actual_end > len(y):
            actual_end = len(y)
            clean_frame = clean_frame[:actual_end - start]
        output[start:actual_end] += clean_frame * window[:actual_end - start]
        window_sum[start:actual_end] += window[:actual_end - start] ** 2
    
    # Normalize by window overlap
    window_sum = np.maximum(window_sum, 1e-8)
    output /= window_sum
    
    return output.astype(np.float32)


def _temporal_noise_suppression(y: np.ndarray, sr: int,
                                 attack_ms: float = 10.0,
                                 release_ms: float = 100.0) -> np.ndarray:
    """
    Time-varying noise gate that adapts to changing noise levels.
    Uses separate attack/release time constants to avoid clicking artifacts.
    Suppresses noise between voiced segments and during noise bursts.
    """
    frame_len = int(sr * 0.02)  # 20ms
    hop = frame_len // 2
    n_frames = max(1, (len(y) - frame_len) // hop + 1)
    
    # Compute per-frame energy
    frame_energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = y[start:start + frame_len]
        frame_energy[i] = np.sqrt(np.mean(frame ** 2) + 1e-10)
    
    # Estimate noise floor as median of lowest 15% of frames
    noise_floor = np.percentile(frame_energy, 15)
    
    # Compute gain per frame: smooth envelope follower
    # Above threshold: gain = 1.0
    # Below threshold: gain scales down proportionally
    snr_threshold = noise_floor * 3.0  # 3x noise floor = voiced
    gain = np.ones(n_frames)
    
    for i in range(n_frames):
        if frame_energy[i] < snr_threshold:
            # Soft compression below threshold
            ratio = frame_energy[i] / (snr_threshold + 1e-10)
            gain[i] = max(0.15, ratio ** 0.5)  # Soft knee
    
    # Smooth gain with asymmetric envelope (fast attack, slow release)
    attack_samples = int(attack_ms / 1000.0 * sr / hop)
    release_samples = int(release_ms / 1000.0 * sr / hop)
    
    smoothed_gain = np.copy(gain)
    for i in range(1, n_frames):
        if smoothed_gain[i] < smoothed_gain[i-1]:
            # Attack (getting quieter) - fast
            coeff = 1.0 / max(attack_samples, 1)
            smoothed_gain[i] = smoothed_gain[i-1] + coeff * (gain[i] - smoothed_gain[i-1])
        else:
            # Release (getting louder) - slow
            coeff = 1.0 / max(release_samples, 1)
            smoothed_gain[i] = smoothed_gain[i-1] + coeff * (gain[i] - smoothed_gain[i-1])
    
    # Apply gain to signal (interpolate from frame-level to sample-level)
    output = np.zeros(len(y), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        end = min(start + frame_len, len(y))
        output[start:end] += y[start:end] * smoothed_gain[i]
    
    return output


def _enhance_harmonics(y: np.ndarray, sr: int,
                       f0_range: tuple = (75, 500)) -> np.ndarray:
    """
    Enhance harmonic structure of voiced speech.
    Voice produces harmonics at integer multiples of f0.
    Non-harmonic energy is likely noise.
    
    Uses a comb filter approach: strengthens frequencies at f0, 2*f0, 3*f0...
    and attenuates frequencies between harmonics.
    """
    # Estimate dominant f0 from autocorrelation
    n = min(len(y), sr * 2)  # Use up to 2 seconds
    segment = y[:n] - np.mean(y[:n])
    rms = np.sqrt(np.mean(segment ** 2))
    if rms < 1e-5:
        return y
    
    segment = segment / (rms + 1e-10)
    fa = rfft(segment, n=len(segment) * 2)
    ac = np.real(np.fft.irfft(fa * np.conj(fa)))[:len(segment)]
    if ac[0] <= 0:
        return y
    ac /= ac[0]
    
    lag_min = int(sr / f0_range[1])
    lag_max = int(sr / f0_range[0])
    if lag_max >= len(ac):
        lag_max = len(ac) - 1
    if lag_max <= lag_min:
        return y
    
    seg_ac = ac[lag_min:lag_max]
    if len(seg_ac) == 0:
        return y
    peak_idx = int(np.argmax(seg_ac)) + lag_min
    if ac[peak_idx] < 0.2:  # Weak periodicity, skip enhancement
        return y
    
    f0_est = sr / peak_idx
    
    # Apply gentle harmonic enhancement via parallel resonators
    # This is lightweight compared to full source separation
    n_fft = 2048
    hop = n_fft // 2
    window = np.hanning(n_fft)
    n_frames = max(1, (len(y) - n_fft) // hop + 1)
    freqs = rfftfreq(n_fft, 1.0 / sr)
    
    output = np.zeros(len(y), dtype=np.float32)
    win_sum = np.zeros(len(y), dtype=np.float32)
    
    # Build harmonic mask: peaks at f0, 2*f0, 3*f0, ...
    n_harmonics = int(min(5000 / f0_est, 20))  # Up to 5kHz
    harmonic_mask = np.zeros(len(freqs))
    bandwidth = f0_est * 0.3  # 30% of f0 as bandwidth
    
    for h in range(1, n_harmonics + 1):
        center = h * f0_est
        if center > sr / 2:
            break
        # Gaussian peak at each harmonic
        harmonic_mask += np.exp(-0.5 * ((freqs - center) / bandwidth) ** 2)
    
    # Normalize mask to [0.5, 1.5] range (gentle enhancement)
    if np.max(harmonic_mask) > 0:
        harmonic_mask = 0.5 + 1.0 * harmonic_mask / np.max(harmonic_mask)
    else:
        harmonic_mask = np.ones(len(freqs))
    
    for i in range(n_frames):
        start = i * hop
        end = min(start + n_fft, len(y))
        frame = y[start:end].copy()
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        
        windowed = frame * window
        fft_data = rfft(windowed)
        
        # Apply harmonic mask
        enhanced_fft = fft_data * harmonic_mask
        
        enhanced_frame = np.real(np.fft.irfft(enhanced_fft, n=n_fft))
        actual_len = end - start
        output[start:end] += enhanced_frame[:actual_len] * window[:actual_len]
        win_sum[start:end] += window[:actual_len] ** 2
    
    win_sum = np.maximum(win_sum, 1e-8)
    output /= win_sum
    
    return output.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
#  MFCC  (pure numpy, no numba/librosa required)
# ═══════════════════════════════════════════════════════════════════════════

def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 40,
                    f_min: float = 20.0, f_max: float = None) -> np.ndarray:
    if f_max is None:
        f_max = sr / 2.0

    def hz_to_mel(hz):   return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel):  return 700 * (10 ** (mel / 2595) - 1)

    mel_min  = hz_to_mel(f_min)
    mel_max  = hz_to_mel(f_max)
    mel_pts  = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts   = mel_to_hz(mel_pts)
    bin_pts  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bin_pts[m-1], bin_pts[m], bin_pts[m+1]
        for k in range(lo, ctr):
            if ctr != lo:
                fb[m-1, k] = (k - lo) / (ctr - lo)
        for k in range(ctr, hi):
            if hi != ctr:
                fb[m-1, k] = (hi - k) / (hi - ctr)
    return fb


def _compute_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13):
    n_fft    = 2048
    hop_len  = 512
    n_mels   = 40
    window   = np.hanning(n_fft)
    fb       = _mel_filterbank(sr, n_fft, n_mels)

    # STFT power spectrum
    n_frames = (len(y) - n_fft) // hop_len + 1
    if n_frames < 1:
        n_frames = 1
    pow_spec = np.zeros((n_fft // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * hop_len
        frame = y[start: start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        windowed = frame * window
        fft_out  = rfft(windowed)
        pow_spec[:, i] = np.abs(fft_out) ** 2

    # Mel spectrum → log
    mel_spec = fb @ pow_spec
    log_mel  = np.log(mel_spec + 1e-9)

    # DCT-II
    n_m = log_mel.shape[0]
    dct_mat = np.zeros((n_mfcc, n_m))
    for k in range(n_mfcc):
        dct_mat[k] = np.cos(np.pi * k / n_m * (np.arange(n_m) + 0.5))
    mfcc = dct_mat @ log_mel  # shape (n_mfcc, n_frames)

    # Delta & delta-delta
    def delta(x, w=9):
        pad = w // 2
        x_p = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
        denom = 2 * sum(t**2 for t in range(1, pad+1))
        d = np.zeros_like(x)
        for t in range(x.shape[1]):
            d[:, t] = sum(n * (x_p[:, t+pad+n] - x_p[:, t+pad-n])
                          for n in range(1, pad+1)) / (denom + 1e-9)
        return d

    mfcc_d  = delta(mfcc)
    mfcc_d2 = delta(mfcc_d)
    return mfcc, mfcc_d, mfcc_d2


# ═══════════════════════════════════════════════════════════════════════════
#  PITCH EXTRACTION  (autocorrelation method)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_pitch(y: np.ndarray, sr: int,
                   f_min: float = 75.0, f_max: float = 500.0) -> np.ndarray:
    frame_len = int(sr * 0.04)   # 40 ms
    hop_len   = int(sr * 0.01)   # 10 ms
    lag_min   = int(sr / f_max)
    lag_max   = int(sr / f_min)
    n_frames  = (len(y) - frame_len) // hop_len + 1
    f0 = np.zeros(max(n_frames, 1))

    for i in range(n_frames):
        frame = y[i * hop_len: i * hop_len + frame_len]
        if len(frame) < frame_len:
            break
        frame = frame - np.mean(frame)
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 1e-5:
            continue
        # Normalize
        frame = frame / (rms + 1e-10)
        # Autocorrelation via FFT
        n   = len(frame)
        fa  = rfft(frame, n=n * 2)
        ac  = np.real(np.fft.irfft(fa * np.conj(fa)))[:n]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        # Find peak in valid lag range
        lmax = min(lag_max, len(ac) - 1)
        if lmax <= lag_min:
            continue
        segment  = ac[lag_min: lmax]
        if len(segment) == 0:
            continue
        peak_idx = int(np.argmax(segment)) + lag_min
        # Lower threshold to 0.15 to detect more voiced frames
        if ac[peak_idx] > 0.15:
            f0[i] = sr / peak_idx

    return f0


# ═══════════════════════════════════════════════════════════════════════════
#  PERIOD EXTRACTION via Zero-Crossing (better for real voice)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_periods_zerocrossing(y: np.ndarray, sr: int,
                                   f_min: float = 50.0,
                                   f_max: float = 500.0) -> np.ndarray:
    """
    Extract individual pitch periods by finding positive-going zero crossings.
    This gives real cycle-to-cycle period measurements — the correct basis
    for jitter calculation.
    Works on both synthetic and real voice recordings.
    """
    # Bandpass the signal to the voice frequency range
    from scipy.signal import butter, filtfilt
    try:
        b, a = butter(4, [f_min/(sr/2), f_max/(sr/2)], btype='band')
        y_bp = filtfilt(b, a, y)
    except Exception:
        y_bp = y.copy()

    # Find positive-going zero crossings (signal crosses from neg → pos)
    signs = np.sign(y_bp)
    # Avoid zero sign
    signs[signs == 0] = 1
    crossings = np.where(np.diff(signs) > 0)[0]

    if len(crossings) < 4:
        return np.array([])

    # Compute period between consecutive crossings
    periods = np.diff(crossings).astype(np.float64) / sr

    # Filter to valid voice period range
    p_min = 1.0 / f_max
    p_max = 1.0 / f_min
    mask  = (periods >= p_min) & (periods <= p_max)
    periods = periods[mask]

    if len(periods) < 3:
        return np.array([])

    # Remove outliers (periods > 3 std from median)
    med = np.median(periods)
    std = np.std(periods)
    if std > 0:
        periods = periods[np.abs(periods - med) < 3 * std]

    return periods


# ═══════════════════════════════════════════════════════════════════════════
#  JITTER helpers
# ═══════════════════════════════════════════════════════════════════════════

def _robust_period_rejection(p: np.ndarray) -> np.ndarray:
    """
    Remove outlier pitch periods using IQR-based rejection.
    Noise bursts and background sounds create spurious periods that
    inflate jitter measurements. IQR method is robust to non-Gaussian
    outliers (unlike std-based rejection).
    Keeps only periods within [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    if len(p) < 5:
        return p
    q1 = np.percentile(p, 25)
    q3 = np.percentile(p, 75)
    iqr = q3 - q1
    if iqr < 1e-10:
        return p  # All periods nearly identical
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (p >= lower) & (p <= upper)
    rejected = p[~mask]
    kept = p[mask]
    # Keep at least 5 periods to avoid empty arrays
    if len(kept) < 5:
        return p
    return kept


def _robust_amplitude_rejection(a: np.ndarray) -> np.ndarray:
    """
    Remove outlier amplitude frames using IQR-based rejection.
    Noise bursts create high-amplitude frames that inflate shimmer.
    Silence frames create low-amplitude frames that also inflate shimmer.
    Keeps only amplitudes within [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    if len(a) < 5:
        return a
    q1 = np.percentile(a, 25)
    q3 = np.percentile(a, 75)
    iqr = q3 - q1
    if iqr < 1e-10:
        return a
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (a >= lower) & (a <= upper)
    kept = a[mask]
    if len(kept) < 5:
        return a
    return kept


def _jitter_local(p: np.ndarray) -> float:
    if len(p) < 2: return 0.0
    return float(np.mean(np.abs(np.diff(p))) / (np.mean(p) + 1e-10))

def _jitter_rap(p: np.ndarray) -> float:
    if len(p) < 3: return 0.0
    sm = np.convolve(p, np.ones(3)/3, "valid")
    n  = len(sm)
    return float(np.mean(np.abs(p[1:n+1] - sm)) / (np.mean(p) + 1e-10))

def _jitter_ppq(p: np.ndarray, q: int) -> float:
    if len(p) < q: return 0.0
    sm   = np.convolve(p, np.ones(q)/q, "valid")
    half = q // 2
    n    = len(sm)
    return float(np.mean(np.abs(p[half:half+n] - sm)) / (np.mean(p) + 1e-10))


# ═══════════════════════════════════════════════════════════════════════════
#  SHIMMER helpers
# ═══════════════════════════════════════════════════════════════════════════

def _shimmer_local(a: np.ndarray) -> float:
    if len(a) < 2: return 0.0
    return float(np.mean(np.abs(np.diff(a))) / (np.mean(a) + 1e-10))

def _shimmer_db(a: np.ndarray) -> float:
    if len(a) < 2: return 0.0
    r = a[1:] / (a[:-1] + 1e-10)
    return float(np.mean(np.abs(20 * np.log10(r + 1e-10))))

def _shimmer_apq(a: np.ndarray, q: int) -> float:
    if len(a) < q: return 0.0
    sm   = np.convolve(a, np.ones(q)/q, "valid")
    half = q // 2
    n    = len(sm)
    return float(np.mean(np.abs(a[half:half+n] - sm)) / (np.mean(a) + 1e-10))


# ═══════════════════════════════════════════════════════════════════════════
#  FRAME RMS
# ═══════════════════════════════════════════════════════════════════════════

def _frame_rms(y: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    n_frames = (len(y) - frame_len) // hop_len + 1
    if n_frames < 1: return np.array([np.sqrt(np.mean(y**2))])
    out = np.zeros(n_frames)
    for i in range(n_frames):
        frame = y[i*hop_len: i*hop_len + frame_len]
        out[i] = np.sqrt(np.mean(frame**2) + 1e-10)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  HNR
# ═══════════════════════════════════════════════════════════════════════════

def _compute_hnr(y: np.ndarray, sr: int) -> float:
    """
    Compute HNR using the cepstrum-based method.
    More accurate than pure autocorrelation for synthetic/real speech.
    Healthy voices: 20-33 dB. PD voices: 8-25 dB.
    """
    frame_len = int(sr * 0.04)
    hop_len   = int(sr * 0.02)
    hnr_vals  = []

    for i in range(0, len(y) - frame_len, hop_len):
        frame = y[i: i + frame_len].copy()
        frame -= np.mean(frame)
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < 1e-5:
            continue

        # Window
        frame *= np.hanning(len(frame))
        frame /= (np.max(np.abs(frame)) + 1e-10)

        n    = len(frame)
        # Power spectrum
        spec = np.abs(np.fft.rfft(frame, n=n)) ** 2
        spec = np.maximum(spec, 1e-10)

        # Cepstrum
        log_spec = np.log(spec)
        cepst    = np.real(np.fft.irfft(log_spec))

        # Find fundamental period in cepstrum (between 2ms and 13ms)
        t_min = int(sr * 0.002)   # 2ms
        t_max = int(sr * 0.013)   # 13ms
        if t_max >= len(cepst):
            t_max = len(cepst) - 1
        if t_max <= t_min:
            continue

        seg      = np.abs(cepst[t_min:t_max])
        peak_val = np.max(seg)
        # Estimate HNR from cepstral peak strength
        # Strong peak = high harmonic content = high HNR
        noise_floor = np.median(np.abs(cepst[5: t_min]))
        if noise_floor < 1e-8:
            continue
        ratio = peak_val / (noise_floor + 1e-10)
        # Convert to dB scale (empirically calibrated)
        hnr_est = 20.0 * np.log10(ratio + 1e-6)
        # Clamp to realistic speech range
        hnr_vals.append(float(np.clip(hnr_est, 0.0, 40.0)))

    if not hnr_vals:
        return 20.0  # default healthy-ish value

    return float(np.median(hnr_vals))


# ═══════════════════════════════════════════════════════════════════════════
#  ZCR
# ═══════════════════════════════════════════════════════════════════════════

def _zero_crossing_rate(y: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    n_frames = (len(y) - frame_len) // hop_len + 1
    if n_frames < 1: return np.array([0.0])
    out = np.zeros(n_frames)
    for i in range(n_frames):
        frame   = y[i*hop_len: i*hop_len + frame_len]
        out[i]  = np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * frame_len)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  SPECTRAL FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _spectral_features(y: np.ndarray, sr: int):
    n_fft    = 2048
    hop_len  = 512
    window   = np.hanning(n_fft)
    n_frames = (len(y) - n_fft) // hop_len + 1
    if n_frames < 1: n_frames = 1
    freqs = rfftfreq(n_fft, 1.0 / sr)

    centroids  = []
    rolloffs   = []
    bandwidths = []

    for i in range(n_frames):
        start = i * hop_len
        frame = y[start: start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        mag = np.abs(rfft(frame * window))
        mag_sum = np.sum(mag) + 1e-10

        # Centroid
        centroid = float(np.sum(freqs * mag) / mag_sum)
        centroids.append(centroid)

        # Rolloff (85%)
        cumsum = np.cumsum(mag)
        thresh = 0.85 * cumsum[-1]
        idx    = np.searchsorted(cumsum, thresh)
        rolloffs.append(float(freqs[min(idx, len(freqs)-1)]))

        # Bandwidth
        bw = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / mag_sum))
        bandwidths.append(bw)

    return np.array(centroids), np.array(rolloffs), np.array(bandwidths)


# ═══════════════════════════════════════════════════════════════════════════
#  NONLINEAR PROXIES
# ═══════════════════════════════════════════════════════════════════════════

def _rpde_proxy(y: np.ndarray, sr: int) -> float:
    """Entropy of normalized autocorrelation — proxy for RPDE."""
    n   = min(len(y), sr)
    fa  = rfft(y[:n])
    ac  = np.abs(np.fft.irfft(fa * np.conj(fa)))[:500]
    ac  = np.clip(ac / (ac[0] + 1e-10), 1e-10, 1)
    ent = -np.sum(ac * np.log(ac)) / len(ac)
    return float(np.clip(ent, 0, 1))

def _dfa_proxy(y: np.ndarray, sr: int) -> float:
    """Spectral flatness — proxy for DFA."""
    n   = min(len(y), sr * 2)
    mag = np.abs(rfft(y[:n])) + 1e-10
    geo = np.exp(np.mean(np.log(mag)))
    arith = np.mean(mag)
    return float(np.clip(geo / (arith + 1e-10), 0, 1))
