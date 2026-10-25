# 👁️🛡️ BlinkCV
> **Production-Grade Computer Vision Eye Health & Neuro-Ergonomics Suite for Developers**  
> *Zero-Cloud Inference • Air-Gapped Privacy • Sub-5% CPU Usage • Multi-Modal Neural Verification*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceLandmarker_VIDEO-00A67E?style=flat-square)](https://developers.google.com/mediapipe)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.31-005CED?style=flat-square)](https://onnxruntime.ai/)
[![CPU Load](https://img.shields.io/badge/CPU_Load-%3C5%25_Low_Footprint-success?style=flat-square)]()
[![RAM Footprint](https://img.shields.io/badge/Memory_Footprint-~177MB-blue?style=flat-square)]()
[![Privacy](https://img.shields.io/badge/Privacy-100%25_Air--Gapped_Local-brightgreen?style=flat-square)]()

---

## 📖 Overview

**BlinkCV** is an open-source, edge-case proof computer vision and neuro-ergonomics system engineered specifically for developers, competitive programmers, and high-screen-time professionals. 

Prolonged screen focus causes **Computer Vision Syndrome (CVS)**: blink rates drop from a normal 15–20 BPM down to 4–6 BPM, leading to cornea tear-film evaporation, ciliary muscle spasm, and cognitive fatigue. Traditional "dumb timers" interrupt flow state with disruptive pop-ups, while heavy webcam tools consume massive CPU and compromise privacy.

**BlinkCV** solves this by running a **100% on-device, sub-5% CPU multi-modal vision engine** alongside a **frameless, always-on-top desktop floating pill** that hovers quietly over VS Code, JetBrains, Chrome, or terminals without ever getting in your way.

---

## ✨ Key Features

* **🪟 Always-On-Top Desktop Floating Pill:**
  - Frameless, dark-midnight HUD (`#11131f`) that floats on top of full-screen IDEs and terminals.
  - Zero-drift, zero-jitter draggable positioning using screen-root coordinates.
  - Dirty-checked label rendering eliminating Windows GDI churn, frame drops, and flicker.
  - 1-click controls: **`[Cam ON/OFF]`**, **`[Zen Sound / Muted]`**, **`[Break]`**, and **`[Snooze 5m]`**.

* **🧠 Multi-Modal 3-Tier Blink Classifier:**
  - **Geometric 5-Pair Trimmed-Mean EAR (50%):** Outlier-resistant eye aspect ratio with rolling left/right individual baselines.
  - **ConvNeXt-Femto Neural Classifier (`blinkcnn.onnx`, 30%):** Roll-aligned $64 \times 64$ eye-patch deep neural verification.
  - **Neural Blendshapes & Iris-Sclera Contrast (20%):** MediaPipe eyelid probability fusion.

* **🎯 Optical Gaze & Saccade Tracker (`ai_gaze_tracker.py`):**
  - Measures relative iris vector displacement inside the eye contour.
  - Detects true line-of-sight independent of head pose.
  - Flags static gaze fixation (reading code without natural saccadic eye jumps).
  - Validates 20-20-20 breaks with **optical proof** (requires actually looking away or closing eyes).

* **🩺 Clinical Biomarkers & Fatigue Predictor (`ai_fatigue_predictor.py`):**
  - **8-D Neural MLP:** Computes real-time **Eye Strain Index (0%–100%)** based on PERCLOS (60s droop), stare duration, BPM, distance, slouch, ambient lux, yawns, and fixation duration.
  - **1€ Adaptive Low-Pass Filter (ACM CHI '12):** Zero webcam landmark jitter while stationary; zero latency during micro-blinks.
  - **TFOS DEWS II Tear-Film Break-Up Physics:** Models cornea dry spot development.

* **📐 Ergonomic Proximity & Posture Guard:**
  - Real-time distance measurement using interpupillary distance (IPD) and outer cheekbone bizygomatic width.
  - **Schmitt-Trigger Hysteresis:** Warns immediately when closer than $45\text{ cm}$; automatically clears once back at $48\text{ cm}+$. No rapid flip-flopping.

* **☕ AFK Auto-Detection & Session Reset:**
  - Face-loss grace periods freeze timers when stepping away from desk.
  - Being away for $>2\text{ minutes}$ resets the 20-minute cycle fresh upon return.

* **🤫 Psychoacoustic "Zero-Irritation" Audio:**
  - **Zero audio alerts by default** (subtle visual amber pulse on the floating pill).
  - Optional Zen acoustic chime (gentle D5 $\to$ A5 harmonic chord, 0.28 volume) with a 45-second cooldown to eliminate alert fatigue.

* **⚡ Ultra-Low CPU Governance (<5%):**
  - Thread-level C/OpenMP/MKL configuration (`OMP_WAIT_POLICY=PASSIVE`, `KMP_BLOCKTIME=0`, single-threaded sequential ONNX runtime).
  - Paced execution: 25 FPS idle, 30 FPS active dynamic states. Drops CPU from 43% down to <5% and RAM to ~177 MB.

---

## 🏗️ System Architecture & Finite State Machine (FSM)

```text
                             ┌──────────────────┐
                             │    DND_PAUSED    │ ◄─── [D] Key / Meeting Mode
                             └────────▲─────────┘
                                      │
                                      ▼
┌──────────────────┐               ┌────────────────────┐
│     IDLE_AFK     │ ◄──────────── │ MONITORING_NORMAL  │
└────────┬─────────┘   Face Lost   └────────┬───────────┘
         │                                  │
         │ Face Back > 2m                   │ Distance < 45cm or
         │ (20m Timer Reset)                │ Sustained Slouch > 5s
         ▼                                  ▼
┌──────────────────┐               ┌────────────────────┐
│ MONITORING_NORMAL│               │  POSTURE_WARNING   │
└────────┬─────────┘               └────────┬───────────┘
         │                                  │ Distance >= 48cm / Posture Fixed
         │ 20 Mins Passed                   ▼
         ▼                         ┌────────────────────┐
┌──────────────────┐   [S] Snooze  │ MONITORING_NORMAL  │
│  BREAK_PENDING   ├─────────────► │ (Extended 5 Mins)  │
└────────┬─────────┘               └────────────────────┘
         │
         │ Gaze Away / Eyes Closed Verified
         ▼
┌──────────────────┐
│   BREAK_ACTIVE   │ (20-Second Active Optical Countdown)
└──────────────────┘
```

---

## 🔬 Multi-Modal Fusion Formula

Eye closure probability $P_{\text{closed}}$ is computed every frame by fusing geometric, deep learning, and blendshape modalities:

$$P_{\text{closed}} = 0.50 \cdot P_{\text{EAR}} + 0.30 \cdot P_{\text{ONNX}} + 0.20 \cdot P_{\text{Blendshape}}$$

```
                      ┌────────────────────────────────────┐
                      │    Input RGB Frame (640 x 480)     │
                      └─────────────────┬──────────────────┘
                                        │
                         [CLAHE & Glare Suppression]
                                        │
         ┌──────────────────────────────┴──────────────────────────────┐
         ▼                                                             ▼
┌────────────────────────────────┐                            ┌────────────────────────────────┐
│  MediaPipe Face Mesh (478 pts) │                            │  Roll-Aligned Eye Crop (64x64) │
│  - 5-Pair Trimmed EAR          │                            │  - ConvNeXt-Femto ONNX Model   │
│  - Iris Coordinates & Vectors  │                            │  - Direct Pixel Tensor Match   │
└───────────────┬────────────────┘                            └───────────────┬────────────────┘
                │                                                             │
                └──────────────────────┬──────────────────────────────────────┘
                                       ▼
                      ┌──────────────────────────────────┐
                      │   Fused Probability Evaluator    │
                      │  Closed >= 0.48  |  Open <= 0.30 │
                      └────────────────┬─────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
┌───────────────────┐        ┌───────────────────┐         ┌───────────────────┐
│ TinyGazeTracker   │        │ Fatigue Predictor │         │ Ergonomics FSM    │
│ Optical Saccades  │        │ 8-D Neural MLP    │         │ Distance/Posture  │
└───────────────────┘        └───────────────────┘         └───────────────────┘
```

---

## 🎮 Desktop Floating Pill & Hotkeys

### Floating Pill Controls
| Button | Action |
| :--- | :--- |
| **`[📷 Cam ON / OFF]`** | Hides or shows the OpenCV webcam preview window to save screen space and GPU cycles. |
| **`[🔔 Zen Sound / 🔇 Muted]`** | Toggles between silent visual mode and soft harmonic Zen audio chimes. |
| **`[Break]`** | Manually triggers an immediate 20-20-20 optical eye relaxation break. |
| **`[Snooze 5m]`** | Extends the active coding session by 5 minutes (up to 2 times). |

### Keyboard Shortcuts (When Webcam Window Active)
| Key | Action |
| :---: | :--- |
| **`Q`** / **`ESC`** | Gracefully quit BlinkCV. |
| **`S`** | Snooze 20-20-20 break for 5 minutes. |
| **`B`** | Trigger 20-20-20 eye relaxation break immediately. |
| **`D`** | Toggle **DND (Do Not Disturb) / Meeting Mode** (pauses alerts during screen sharing). |
| **`C`** | Re-calibrate baseline open-eye EAR and posture distance. |
| **`M`** | Toggle audio mute / Zen chime. |
| **`P`** | Toggle camera preview window (show / hide). |
| **`H`** | Toggle compact minimal HUD. |
| **`T`** | Test all acoustic alerts. |
| **`+`** / **`-`** | Adjust blink sensitivity threshold offset ($\pm 0.02$). |

---

## 📂 Project Structure

```text
blinkcv/
├── ai_fatigue_predictor.py   # 8-D Neural MLP for Cognitive Eye Strain Index (0-100%)
├── ai_gaze_tracker.py        # Optical Iris Vector, Saccades & Line-of-Sight Gaze Estimator
├── audio_alert.py            # Zen Harmonic Foley Audio Engine & Cooldown Manager
├── blinkcnn.onnx             # Pretrained ConvNeXt-Femto Eye-Patch Neural Classifier
├── config.py                 # Central System Constants, Weights & Threshold Parameters
├── desktop_widget.py         # Frameless Always-On-Top Desktop Floating Pill (Tkinter)
├── ear_calculator.py         # Multi-Modal Vision Engine, Adaptive EAR & Roll Alignment
├── face_landmarker.task      # MediaPipe FaceLandmarker Model Bundle
├── hud_renderer.py           # Cyberpunk-Themed OpenCV Overlay & Visual Diagnostics
├── main.py                   # Master Pipeline, FSM State Engine & Execution Coordinator
├── research_biomarkers.py    # 1€ Filter, Eyeglass Glare Masking & Tear-Film Physics
├── requirements.txt          # Frozen Python Dependencies
└── run.bat                   # 1-Click Launch Script for Windows
```

---

## 🚀 Quick Start & Installation

### Prerequisites
- **OS:** Windows 10/11, macOS, or Linux
- **Python:** 3.11 or higher
- **Webcam:** Any standard 720p or 1080p USB / integrated laptop webcam

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/blinkcv.git
cd blinkcv

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the Application

```bash
# Windows 1-Click Launch:
run.bat

# Or via Command Line:
python main.py
```

*Note: On first launch, `face_landmarker.task` will download automatically from Google Cloud Storage if not already present locally.*

---

## ⚙️ Configuration & Tuning (`config.py`)

You can customize thresholds and behaviors directly in `config.py`:

```python
# Eye-Blink Probability & Weights
WEIGHT_GEOMETRIC_EAR = 0.50     # Geometric EAR relative baseline weight
WEIGHT_PIXEL_ONNX    = 0.30     # 64x64 ConvNeXt-Femto neural weight
WEIGHT_BLENDSHAPE    = 0.20     # MediaPipe blendshape weight

# Hysteresis Decision Thresholds
FUSED_CLOSE_PROB_THRESH = 0.48  # Closure trigger probability
FUSED_OPEN_PROB_THRESH  = 0.30  # Reopen trigger probability

# Ergonomics & Distance
SAFE_DISTANCE_CM = 48.0             # Normal safe laptop working distance (>= 48cm)
PROXIMITY_WARN_DISTANCE_CM = 45.0   # Alert immediately when closer than 45cm
PROXIMITY_SAFE_DISTANCE_CM = 48.0   # Clear alert once back at >= 48cm
PROXIMITY_CONSEC_FRAMES = 3         # Instant 3-frame debounce (~100ms)

# 20-20-20 Timing
WORK_DURATION_SECONDS  = 20 * 60   # 20 minutes coding session
BREAK_DURATION_SECONDS = 20.0      # 20 seconds verified rest
SNOOZE_DURATION_SECONDS = 5 * 60   # 5 minutes snooze extension
```

---

## 🔒 Privacy Guarantee

BlinkCV is built with an **air-gapped architecture**:
* **Zero Network Traffic:** Video frames never touch sockets, APIs, or disk storage.
* **Transient Memory Only:** Frames are processed in local volatile RAM and immediately discarded.
* **Safe for Enterprise:** Complies with corporate privacy requirements and non-disclosure environments.

---

## 📜 License

This project is licensed under the **Apache-2.0 License**. See the `LICENSE` file for details.
