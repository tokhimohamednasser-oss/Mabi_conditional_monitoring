"""
MABI Robot Condition Monitoring — Simulation & Signal Processing
================================================================
Original work: Fraunhofer IPK Berlin (MSc Thesis, 2025)
Author       : Mohamed Nasser

This module simulates the full vibration monitoring pipeline developed
for the MABI robot at Fraunhofer IPK. The real system used:
  - 4x ICP vibration sensors (PCB Piezotronics)
  - 4-channel ICP DAQ USB
  - Raspberry Pi 4B (edge computing)
  - MQTT → ThingsBoard Cloud

As the sensor data is proprietary to Fraunhofer IPK, this module
provides a faithful simulation of the signal processing and KPI
computation pipeline with synthetic vibration signals that replicate
real operational behaviour including baseline noise, periodic components
(motor frequencies, gear mesh), and injected fault signatures.

Run this to understand the full data pipeline before connecting real hardware.
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class SensorConfig:
    sampling_rate: int   = 10_000   # Hz — 4-ch ICP DAQ typical rate
    duration:      float = 2.0      # seconds per acquisition window
    num_channels:  int   = 4
    sensitivity:   float = 100.0    # mV/g (typical ICP accelerometer)


@dataclass
class RobotState:
    """Simulates robot controller state variables."""
    joint_speed_rpm:   float = 120.0   # main drive motor
    payload_kg:        float = 5.0
    movement_type:     str   = "sweep"  # sweep | hold | rapid
    fault_injected:    bool  = False
    fault_type:        Optional[str] = None   # bearing | looseness | None


# ── Signal simulation ─────────────────────────────────────────────────────────

def simulate_vibration(cfg: SensorConfig, state: RobotState) -> np.ndarray:
    """
    Generate synthetic vibration data that mimics ICP sensor output.
    
    Frequency components included:
    - Motor fundamental and harmonics
    - Gear mesh frequency (assuming 30-tooth gear)
    - Structural resonance (~800 Hz)
    - Gaussian noise floor
    - Optional fault signature (bearing BPFO, looseness sub-harmonics)
    """
    n      = int(cfg.sampling_rate * cfg.duration)
    t      = np.linspace(0, cfg.duration, n, endpoint=False)
    f_rot  = state.joint_speed_rpm / 60.0          # rotation Hz
    f_gear = f_rot * 30                             # gear mesh (30 teeth)

    data = np.zeros((n, cfg.num_channels))

    for ch in range(cfg.num_channels):
        # Baseline noise
        signal = np.random.randn(n) * 0.05

        # Motor harmonics (amplitude scales with payload)
        load_factor = 1.0 + state.payload_kg * 0.05
        for h in range(1, 4):
            amp = (0.3 / h) * load_factor
            signal += amp * np.sin(2 * np.pi * f_rot * h * t + np.random.uniform(0, 2*np.pi))

        # Gear mesh
        signal += 0.15 * np.sin(2 * np.pi * f_gear * t)

        # Structural resonance
        signal += 0.05 * np.sin(2 * np.pi * 800 * t) * np.exp(-t * 2)

        # Fault injection
        if state.fault_injected:
            if state.fault_type == "bearing":
                # BPFO — bearing pass frequency outer race (~3.5x rotation)
                bpfo = f_rot * 3.5
                signal += 0.4 * np.sin(2 * np.pi * bpfo * t) * (
                    1 + 0.3 * np.sin(2 * np.pi * f_rot * t))
            elif state.fault_type == "looseness":
                # Sub-harmonics and impulsive events
                signal += 0.25 * np.sin(2 * np.pi * (f_rot * 0.5) * t)
                # Add random impulses
                impulse_times = np.random.choice(n, size=int(n * 0.01), replace=False)
                signal[impulse_times] += np.random.randn(len(impulse_times)) * 0.6

        data[:, ch] = signal

    return data


# ── Signal processing ─────────────────────────────────────────────────────────

def bandpass_filter(data: np.ndarray, fs: int, low: float, high: float) -> np.ndarray:
    """Apply a 4th order Butterworth bandpass filter."""
    nyq = fs / 2
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def compute_rms(data: np.ndarray) -> np.ndarray:
    """Overall RMS per channel — primary health indicator."""
    return np.sqrt(np.mean(data ** 2, axis=0))


def compute_crest_factor(data: np.ndarray) -> np.ndarray:
    """Peak / RMS — sensitive to impulsive faults (bearing spalling)."""
    rms  = compute_rms(data)
    peak = np.max(np.abs(data), axis=0)
    return np.where(rms > 0, peak / rms, 0.0)


def compute_kurtosis(data: np.ndarray) -> np.ndarray:
    """4th statistical moment — flags impulsive events."""
    n = data.shape[0]
    mu   = np.mean(data, axis=0)
    std  = np.std(data, axis=0)
    return np.where(std > 0,
                    np.mean(((data - mu) / std) ** 4, axis=0),
                    0.0)


def compute_fft(data: np.ndarray, fs: int) -> tuple:
    """
    FFT for all channels.
    Returns positive frequencies and scaled amplitude spectrum.
    """
    n      = data.shape[0]
    freqs  = fftfreq(n, d=1/fs)
    pos    = freqs >= 0
    freqs  = freqs[pos]

    spectrum = np.zeros((pos.sum(), data.shape[1]))
    for ch in range(data.shape[1]):
        yf = fft(data[:, ch])
        spectrum[:, ch] = (2.0 / n) * np.abs(yf[pos])

    return freqs, spectrum


def compute_kpis(data: np.ndarray, cfg: SensorConfig, state: RobotState) -> dict:
    """
    Compute all KPIs from raw vibration data.
    These are the values transmitted to ThingsBoard in the real system.
    """
    freqs, spectrum = compute_fft(data, cfg.sampling_rate)
    rms    = compute_rms(data)
    cf     = compute_crest_factor(data)
    kurt   = compute_kurtosis(data)

    kpis = {
        "timestamp":     time.time(),
        "joint_speed_rpm": state.joint_speed_rpm,
        "movement_type": state.movement_type,
        "payload_kg":    state.payload_kg,
    }

    for ch in range(cfg.num_channels):
        prefix = f"ch{ch}"
        dom_idx = np.argmax(spectrum[1:, ch]) + 1   # skip DC
        kpis[f"{prefix}_rms"]               = round(float(rms[ch]), 6)
        kpis[f"{prefix}_crest_factor"]      = round(float(cf[ch]), 4)
        kpis[f"{prefix}_kurtosis"]          = round(float(kurt[ch]), 4)
        kpis[f"{prefix}_dominant_freq_hz"]  = round(float(freqs[dom_idx]), 2)
        kpis[f"{prefix}_dominant_amp"]      = round(float(spectrum[dom_idx, ch]), 6)

    return kpis


# ── Anomaly detection ─────────────────────────────────────────────────────────

def check_anomalies(kpis: dict, thresholds: Optional[dict] = None) -> list:
    """
    Rule-based anomaly detection mirroring ThingsBoard rule engine logic.
    Returns a list of alert strings (empty = healthy).
    """
    if thresholds is None:
        thresholds = {
            "rms_warning":     0.8,
            "rms_critical":    1.2,
            "crest_warning":   4.0,
            "kurtosis_warning": 4.5,
        }

    alerts = []
    for ch in range(4):
        rms  = kpis.get(f"ch{ch}_rms", 0)
        cf   = kpis.get(f"ch{ch}_crest_factor", 0)
        kurt = kpis.get(f"ch{ch}_kurtosis", 0)

        if rms > thresholds["rms_critical"]:
            alerts.append(f"CRITICAL — Ch{ch}: RMS={rms:.3f}g exceeds critical threshold")
        elif rms > thresholds["rms_warning"]:
            alerts.append(f"WARNING  — Ch{ch}: RMS={rms:.3f}g exceeds warning threshold")

        if cf > thresholds["crest_warning"]:
            alerts.append(f"WARNING  — Ch{ch}: Crest Factor={cf:.2f} — possible impulsive fault")

        if kurt > thresholds["kurtosis_warning"]:
            alerts.append(f"WARNING  — Ch{ch}: Kurtosis={kurt:.2f} — possible bearing defect")

    return alerts


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_results(data: np.ndarray, cfg: SensorConfig, state: RobotState):
    """Plot time domain and FFT for channel 0."""
    t      = np.linspace(0, cfg.duration, data.shape[0])
    freqs, spectrum = compute_fft(data, cfg.sampling_rate)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle(
        f"MABI Robot Vibration Monitor  —  {state.movement_type}  "
        f"{'[FAULT: ' + state.fault_type + ']' if state.fault_injected else '[Healthy]'}",
        fontsize=13, fontweight='bold'
    )

    # Time domain
    axes[0].plot(t, data[:, 0], lw=0.5, color='#1A6B7C')
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Acceleration (g)")
    axes[0].set_title("Channel 0 — Time domain")
    axes[0].grid(True, alpha=0.3)

    # Frequency domain (0 – 1000 Hz)
    mask = freqs <= 1000
    axes[1].plot(freqs[mask], spectrum[mask, 0], lw=0.7, color='#1A6B7C')
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_title("Channel 0 — Frequency spectrum (FFT)")
    axes[1].grid(True, alpha=0.3)

    # Annotate rotation frequency
    f_rot = state.joint_speed_rpm / 60.0
    axes[1].axvline(f_rot, color='orange', linestyle='--', alpha=0.7,
                    label=f"Motor freq: {f_rot:.1f} Hz")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.show()
    plt.close()


# ── Main demo ─────────────────────────────────────────────────────────────────

def demo():
    cfg   = SensorConfig()
    print("=" * 60)
    print("  MABI Robot Condition Monitoring — Simulation Demo")
    print("=" * 60)

    scenarios = [
        RobotState(joint_speed_rpm=120, movement_type="sweep",  fault_injected=False),
        RobotState(joint_speed_rpm=180, movement_type="rapid",  fault_injected=True, fault_type="bearing"),
        RobotState(joint_speed_rpm=90,  movement_type="hold",   fault_injected=True, fault_type="looseness"),
    ]

    for state in scenarios:
        label = f"[{state.fault_type}]" if state.fault_injected else "[Healthy]"
        print(f"\nScenario: {state.movement_type} at {state.joint_speed_rpm} RPM  {label}")
        data  = simulate_vibration(cfg, state)
        kpis  = compute_kpis(data, cfg, state)
        alerts = check_anomalies(kpis)

        print(f"  Ch0 RMS     : {kpis['ch0_rms']:.4f} g")
        print(f"  Ch0 Crest   : {kpis['ch0_crest_factor']:.2f}")
        print(f"  Ch0 Kurtosis: {kpis['ch0_kurtosis']:.2f}")
        print(f"  Dominant freq: {kpis['ch0_dominant_freq_hz']:.1f} Hz")

        if alerts:
            for a in alerts:
                print(f"  ⚠  {a}")
        else:
            print("  ✓  All KPIs within normal range")

        print(f"  ThingsBoard payload: {json.dumps({k: v for k, v in kpis.items() if k.startswith('ch0')}, indent=2)[:200]}...")

    # Show plots for first scenario
    data  = simulate_vibration(cfg, scenarios[0])
    plot_results(data, cfg, scenarios[0])
    data  = simulate_vibration(cfg, scenarios[1])
    plot_results(data, cfg, scenarios[1])


if __name__ == "__main__":
    demo()
