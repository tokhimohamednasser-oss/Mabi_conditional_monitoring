# MABI Robot Condition Monitoring — IIoT Vibration Analysis

> Real-time vibration-based condition monitoring for industrial robots using edge computing and cloud IoT.

Developed as MSc thesis at **Fraunhofer IPK Berlin** (Mar–Oct 2025). Contributes to the IEEE MetroInd 2026 publication on industrial digitization.

---

## System architecture

```
┌─────────────────────┐     USB     ┌──────────────────┐    MQTT    ┌─────────────────┐
│  4x ICP Vibration   │────────────▶│  Raspberry Pi 4B │───────────▶│  ThingsBoard    │
│  Sensors (PCB)      │             │  (Edge Computing)│            │  Cloud          │
│  4-ch ICP DAQ USB   │             │                  │            │  - Dashboards   │
└─────────────────────┘             │  ┌────────────┐  │            │  - Alerts       │
                                    │  │ FFT / RMS  │  │            │  - History      │
┌─────────────────────┐   Trigger   │  │ KPI Compute│  │            └─────────────────┘
│  MABI Robot         │────────────▶│  │ Anomaly Det│  │
│  Controller         │             │  └────────────┘  │
│  (Siemens SINUMERIK)│             └──────────────────┘
└─────────────────────┘
```

**Hardware used at Fraunhofer IPK:**
- MABI industrial robot arm (Siemens SINUMERIK 840D sl controller)
- 4x ICP piezoelectric accelerometers
- 4-channel ICP DAQ USB (10 kHz sampling per channel)
- Raspberry Pi 4B (4GB RAM)
- ThingsBoard Cloud

---

## What this repo contains

| File | Description |
|------|-------------|
| `mabi_simulation.py` | Full pipeline simulation with synthetic vibration signals |
| `signal_processing.py` | FFT, RMS, crest factor, kurtosis — works with real DAQ data |
| `thingsboard_client.py` | MQTT client template for ThingsBoard telemetry |
| `anomaly_detection.py` | Rule-based and statistical anomaly detection |

> **Note:** Proprietary sensor data and ThingsBoard credentials belong to Fraunhofer IPK Berlin and are not included. The simulation generates realistic synthetic signals that replicate the operational behaviour observed during the internship.

---

## KPIs computed

| KPI | Method | Detects |
|-----|--------|---------|
| RMS (per channel) | `√(mean(x²))` | General vibration severity |
| Crest Factor | `peak / RMS` | Impulsive faults (bearing spalling) |
| Kurtosis | 4th statistical moment | Localised defects |
| Dominant frequency | FFT peak | Motor / gear mesh anomalies |
| Band energy | FFT integration | Specific fault frequencies |

---

## Quick start

```bash
git clone https://github.com/tokhimohamednasser/mabi-condition-monitoring.git
cd mabi-condition-monitoring
pip install -r requirements.txt

# Run the full simulation demo (3 scenarios: healthy, bearing fault, looseness)
python mabi_simulation.py
```

### Example output

```
Scenario: sweep at 120 RPM  [Healthy]
  Ch0 RMS      : 0.3309 g
  Ch0 Crest    : 2.74
  Ch0 Kurtosis : 2.37
  ✓  All KPIs within normal range

Scenario: rapid at 180 RPM  [bearing]
  Ch0 RMS      : 0.8712 g
  Ch0 Crest    : 5.83
  Ch0 Kurtosis : 6.14
  ⚠  WARNING — Ch0: RMS=0.871g exceeds warning threshold
  ⚠  WARNING — Ch0: Crest Factor=5.83 — possible impulsive fault
  ⚠  WARNING — Ch0: Kurtosis=6.14 — possible bearing defect
```

### Connect to real DAQ hardware

```python
from signal_processing import compute_kpis, check_anomalies
from thingsboard_client import send_telemetry
import your_daq_library as daq  # vendor SDK

cfg  = SensorConfig(sampling_rate=10_000, duration=2.0, num_channels=4)
data = daq.read(channels=4, samples=20_000)   # replace with your DAQ call
kpis = compute_kpis(data, cfg)
send_telemetry(kpis)                          # → ThingsBoard
```

---

## Signal processing pipeline

```python
raw_data (n_samples × 4_channels)
    ↓
bandpass_filter(100–5000 Hz)       # remove low-freq structural noise
    ↓
FFT → frequency spectrum            # identify motor / gear / bearing frequencies
    ↓
RMS, Crest Factor, Kurtosis         # time-domain health indicators
    ↓
KPI dict → JSON → MQTT publish      # → ThingsBoard Cloud
    ↓
Rule-based anomaly detection        # threshold comparison + alerting
```

---

## Fault signatures detected

| Fault type | Indicator | Typical pattern |
|------------|-----------|----------------|
| Bearing outer race (BPFO) | High kurtosis + crest factor | 3.5× rotation freq amplitude spike |
| Joint looseness | Sub-harmonic + impulsive | 0.5× rotation freq + RMS increase |
| General wear | Increasing RMS trend | Gradual broadband energy increase |
| Gear mesh defect | FFT sideband growth | Sidebands ± rotation freq around gear mesh |

---

## Related work

**Publication:** M. Nasser et al., *"Integration of Measurements for Digitation: Challenges and Opportunities for the Industry of Tomorrow"*, IEEE International Workshop on Metrology for Industry 4.0 & IoT (MetroInd4.0&IoT), 2026.

**Related repos in this profile:**
- [crack-detection](https://github.com/YOUR_USERNAME/crack-detection) — Computer vision pipeline for surface defect detection
- [smart-factory-monitor](https://github.com/YOUR_USERNAME/smart-factory-monitor) — End-to-end IIoT dashboard (in progress)

---

## Author

**Mohamed Nasser** — Digital Manufacturing Engineer  
Fraunhofer IPK Berlin · Politecnico di Bari  
[LinkedIn](https://linkedin.com/in/YOUR_USERNAME) · [Website](https://YOUR_USERNAME.github.io)
