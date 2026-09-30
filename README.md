````markdown
# Rdtune---Attentive-Behavior-Estimator

# Rdtune
### Attentive-Behavior Estimator

**Real-time, on-device focus detection that responds with adaptive background sound, built for and tested on the Snapdragon® X2 Elite.**

> Submission for the **Snapdragon® AI Lab Build & Present Challenge**

![Rdtune running on Snapdragon X2 Elite at 92 FPS](Demo/fps.png)

| | |
|---|---|
| **Throughput** | **92 FPS** (in-app HUD) on Snapdragon® X2 Elite (SC8480X), Windows 11, via Qualcomm® Device Cloud |
| **Footprint** | **~213 MB RAM · ~2% CPU** (Windows Task Manager) |
| **Privacy** | 100% on-device. No cloud, no network calls, no video or images saved |
| **Face model** | MediaPipe **Face Landmarker** (`face_landmarker.task`, open source, Apache 2.0) |
| **Try it** | Run [`Application/attentive_behavior_estimator.exe`](Application/) |
| **Demo** | [`Demo/`](Demo/): demo video · how-to-run video · performance screenshots |

---

## 📌 Contents

1. [What it is](#1-what-it-is)
2. [How it works](#2-how-it-works)
3. [The sound engine: how the noises help](#3-the-sound-engine-how-the-noises-help)
4. [Features](#4-features)
5. [Optimization: 92 FPS at ~213 MB](#5-optimization-92-fps-at-213-mb)
6. [Scientific evidence](#6-scientific-evidence)
7. [Deployment & accessibility](#7-deployment--accessibility)
8. [Repository structure](#8-repository-structure)
9. [Limitations & future work](#9-limitations--future-work)
10. [Credits & licenses](#10-credits--licenses)

---

## 1. What it is

**Rdtune** (the Attentive-Behavior Estimator) watches your webcam, estimates **how likely you are to be off-task, relative to your own normal behavior**, and adjusts background sound to help you stay on track.

- 👁️ **Detects** off-task behavior from facial cues: where your gaze is, where your head is pointing, how long your eyes stay closed, and whether your face is visible.
- 📊 **Reports** three numbers live: *Off-task probability*, *Confidence*, and a *Data-quality* flag.
- 🔊 **Responds** with a calm steady sound (green noise by default), stepping up to pink or white noise when you drift, and going silent when you leave.
- 🔒 **Stays private**: everything runs locally.

**Built for:** students, remote workers, developers, and anyone who wants a gentle, non-intrusive focus aid rather than an app blocker or a strict timer.

### What makes it different

- **Personal, not generic.** It calibrates to *you* (your screen region, your head pose, your eye openness), so it works across faces, cameras and seating positions without labeled training data.
- **Honest about uncertainty.** Bad lighting, a small face, or an absent face lowers confidence instead of producing a confident guess.
- **Transparent.** A weighted, explainable rule set produces the score, and the HUD shows *why* (e.g. "gaze outside the calibrated screen region for 7.0 of 10 s").
- **Closed loop.** Detection directly drives the sound, with anti-flicker rules so audio never jumps around.

---

## 2. How it works

```mermaid
flowchart LR
  A["Webcam frame<br/>640×480"] --> B["MediaPipe Face Landmarker<br/>478 landmarks · blendshapes · head-pose matrix"]
  B --> C["Per-frame features<br/>gaze · head yaw/pitch · eye closure · quality"]
  E["Personal baseline<br/>12 s calibration"] --> D
  C --> D["10 s sliding window<br/>new estimate every 1 s"]
  D --> F["Transparent heuristic<br/>P(off-task)"]
  F --> G["Temporal smoothing"]
  D --> H["Data quality"]
  G --> I["Confidence"]
  H --> I
  G --> J["Adaptive audio controller"]
  I --> J
  J --> K["Green · Pink · White · Silence"]
````

### Step 1: Face landmarks (MediaPipe)

`FaceLandmarker` runs in **VIDEO mode** (tracking across frames) for one face and returns 478 landmarks (including both irises), 52 blendshape scores, and a facial transformation matrix (head pose).

### Step 2: Per-frame features (`FeatureExtractor`)

| Feature                          | How it is computed                                                                                            |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Gaze** (horizontal + vertical) | Iris centre relative to each eye's corners (and lid opening for vertical), averaged across both eyes          |
| **Head yaw / pitch**             | Euler angles from the transformation matrix                                                                   |
| **Eye closure**                  | Blink blendshape **or** Eye Aspect Ratio (EAR) dropping vs. your open-eye baseline (either can vote "closed") |
| **Quality metrics**              | Face size, brightness in the face region, landmark jitter                                                     |

### Step 3: Personal baseline (12-second calibration)

On launch you look at the screen and glance at each corner once. From that, the app learns *your*:

* on-screen **gaze region** (2nd–98th percentile of your gaze, plus a 1.25× margin),
* **neutral head pose** and its natural variation (robust SD via MAD),
* **open-eye EAR**.

Everything after this is measured *relative to you*. If calibration can't collect enough valid frames it falls back to a **provisional** baseline, and confidence is halved to reflect that. Press `c` to recalibrate anytime.

### Step 4: Window features (10 s window, updated every 1 s)

Time-weighted fractions of the window that show:

| Signal              | Definition                                                                      |
| ------------------- | ------------------------------------------------------------------------------- |
| `face_absent_frac`  | Face not visible                                                                |
| `gaze_off_frac`     | Gaze outside your calibrated region                                             |
| `head_away_frac`    | Head pose more than 3 of *your* SDs from neutral                                |
| `long_closure_frac` | Eyes closed for ≥ 0.5 s at a time (drowsiness proxy; normal blinks are ignored) |

### Step 5: Off-task probability (transparent heuristic)

```text
z = bias + Σ weight_i × clip((signal_i − deadzone_i) / (1 − deadzone_i), 0, 1)
P(off-task) = sigmoid(z)
```

| Rule             | Weight | Dead-zone |
| ---------------- | -----: | --------: |
| Face absent      |    6.0 |      0.10 |
| Gaze off-screen  |    4.0 |      0.10 |
| Head turned away |    3.0 |      0.10 |
| Long eye closure |    4.0 |      0.10 |

Bias = −3.0, so P ≈ 5% when nothing fires. Weights are **hand-set design choices, not fitted to data**, and are stated as such in the code.

### Step 6: Temporal smoothing

A two-state forward filter (`stay_prob = 0.9`, `evidence_weight = 0.3`) stops single noisy windows from flipping the state. It resets when the face is absent so an absence doesn't leave "off-task" belief behind after you return.

### Step 7: Confidence & data quality

```text
confidence = quality × baseline_factor × (0.4 + 0.6 × |2p − 1|)     (× 0.7 if face absent)
```

Quality is reduced for: `face_small`, `too_dark`, `overexposed`, `jittery_landmarks`, `extreme_pose`, `low_fps`. `baseline_factor` is 1.0 for a personal baseline and 0.5 for a provisional one. Estimates below **0.30 confidence never change the sound zone**.

### 🧪 Worked example: the HUD in the screenshot above

The Device Cloud camera feed contains no face, so the window is 100% `face_absent`:

* `z = −3.0 + 6.0 = 3.0` → **P(off-task) = sigmoid(3.0) ≈ 95%** ✅ matches the HUD
* quality = 0.60 (nobody visible) · provisional baseline ×0.5 · certainty term 0.94 · absent ×0.7 → **confidence ≈ 20%** ✅ matches the HUD
* Zone: **away** → audio: **silence** ✅ matches `Audio [AUTO]: silence`

This shows the design working as intended: with no face, it flags the situation, reports low confidence, and stays quiet.

### Rdtune with a face visible (focused state)

![Rdtune in the focused state with green noise playing](Demo/focused.png)

### Reading the HUD

| HUD element                          | Meaning                                                          |
| ------------------------------------ | ---------------------------------------------------------------- |
| **FPS**                              | Live loop rate                                                   |
| **Off-task p**                       | Smoothed probability of off-task behavior (bar coloured by zone) |
| **Confidence**                       | How much to trust the estimate                                   |
| **Data quality: `flag` (score)**     | Input reliability (`ok`, `face_absent`, `too_dark`, …)           |
| **Baseline: personal / PROVISIONAL** | Whether calibration succeeded fully                              |
| **Zone**                             | `focused` · `drifting` · `off_task` · `away`                     |
| **Explanation line**                 | Which rules fired and for how many seconds of the window         |
| **Audio [AUTO/MANUAL]**              | Sound mode, current noise and volume                             |

---

## 3. The sound engine: how the noises help

### The idea

Distractions and mind-wandering pull attention away. A steady, non-semantic sound can **mask sudden environmental noise** and, for some people, **raise arousal toward a level that suits concentration**. Rather than playing sound constantly, the app **matches sound to your detected state** and stays quiet when it shouldn't act.

### Noise types (synthesized on-device)

All tracks are generated in the frequency domain with random phases, which makes them **seamless loops**. They are 30 s, 44.1 kHz stereo (independent L/R), with equal loudness (RMS ≈ −18 dBFS) and no energy below 20 Hz. You can optionally drop your own tracks into `Noise/*.mp3`.

| Noise     | Spectrum                                                                               | Character                                                                                                        |
| --------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Green** | Smooth mid-band hump centred at **~500 Hz** (~220 Hz–1.15 kHz at ±1σ) over a low floor | Calm, natural, unobtrusive. *There is no standard definition of green noise; this is one common interpretation.* |
| **Pink**  | Power ∝ **1/f** (−3 dB/octave)                                                         | Softer, balanced broadband masking                                                                               |
| **White** | Flat power spectrum, **20 Hz–22 kHz**                                                  | Strongest broadband masking of sudden sounds                                                                     |
| **Brown** | Power ∝ **1/f²** (−6 dB/octave)                                                        | Deep, low-frequency rumble (manual choice)                                                                       |

### Zone → sound mapping (the adaptive loop)

| Zone            | Trigger (smoothed P off-task) | Sound       | Volume* |
| --------------- | ----------------------------- | ----------- | ------: |
| ✅ **Focused**   | < 0.35                        | **Green**   |    0.15 |
| 🟡 **Drifting** | 0.35 – 0.65                   | **Pink**    |    0.25 |
| 🔴 **Off-task** | ≥ 0.65                        | **White**   |    0.30 |
| ⚪ **Away**      | Face not visible              | **Silence** |       0 |

*Relative playback volume (hard-capped at 0.5). Real loudness depends on your system volume.

**How this resolves lost focus:** as you drift, the sound steps up gradually from gentle green to broader-spectrum pink to white, increasing masking of distractions when you need it most. As you recover, it steps back down, so the sound never becomes a permanent crutch. When you're away, there's nothing to help, so it goes silent.

### Anti-annoyance rules (what makes it usable)

| Rule                                                                                                  | Setting     |
| ----------------------------------------------------------------------------------------------------- | ----------- |
| **Debounce**: a change must be wanted for this long before it happens                                 | 4 s         |
| **Distraction hold**: a distraction sound plays at least this long (never locked longer than 90 s)    | 20 s        |
| **Recovery confirm**: focused evidence needed before leaving a distraction state                      | 4 s         |
| **Hysteresis** on zone thresholds                                                                     | 0.05        |
| **Low-confidence hold**: unreliable estimates don't change the sound                                  | conf < 0.30 |
| **Away recovery**: after your face returns, ignore estimates while the window still holds the absence | 10 s        |
| **Crossfade** between sounds                                                                          | 1.5 s       |
| **Safety volume cap**                                                                                 | 0.5         |

> The zone→noise mapping is a **design hypothesis**, not a proven intervention, and the code says so. Manual override is always available and never locked.

---

## 4. Features

* ⚡ **Real-time on-device inference** at ~92 FPS on Snapdragon X2 Elite
* 🧍 **Personal 12-second calibration**: no dataset or training needed
* 🎯 **Multi-signal off-task estimate**: gaze, head pose, eye closure, face presence
* 🔎 **Explainable output**: the HUD states which rules fired
* 📈 **Confidence + data-quality scoring** with 6 quality checks
* 🕰️ **Temporal smoothing** for stable, non-flickering results
* 🔊 **Adaptive sound engine** with 4 synthesized noise colours, crossfades and anti-flicker logic
* 🎛️ **Manual controls & in-app settings menu** (window length, calibration time, audio delays, volumes, sound per zone)
* 🚶 **Away detection** that silences audio automatically
* 🖥️ **Headless mode** (`--headless`) for camera + audio without a window
* ⚙️ **Fully configurable** via optional YAML file
* 📦 **One-file Windows `.exe`**, works offline
* 🔒 **Private by design**: no recording, no upload

### Keyboard controls

| Key             | Action                                |
| --------------- | ------------------------------------- |
| `c`             | Recalibrate                           |
| `m`             | Settings menu                         |
| `a`             | Adaptive (auto) audio                 |
| `s`             | Silence                               |
| `1` `2` `3` `4` | Brown · Pink · Green · White (manual) |
| `q`             | Quit                                  |

---

## 5. Optimization: 92 FPS at ~213 MB

Measured on **Qualcomm® Device Cloud**: Snapdragon® X2 Elite (SC8480X), Windows 11, Python 3.12.10, pygame 2.6.1.

| Metric     | Result                                     | Evidence                                         |
| ---------- | ------------------------------------------ | ------------------------------------------------ |
| Throughput | **92 FPS** (≈ 11 ms per frame, end to end) | [`Demo/fps.png`](Demo/fps.png)                   |
| Memory     | **212.8 MB**                               | [`Demo/task_manager.png`](Demo/task_manager.png) |
| CPU        | **~2.0%**                                  | Task Manager screenshot                          |
| Delivery   | Single `.exe`                              | [`Application/`](Application/)                   |

![Task Manager showing \~213 MB and \~2% CPU](Demo/task_manager.png)

### How it's optimized

1. **Compact, purpose-built model.** `face_landmarker.task` (MediaPipe) is designed for real-time on-device use, with **no GPU or NPU required**.
2. **Tracking, not re-detecting.** `RunningMode.VIDEO` with `num_faces=1` tracks the face across frames, which is far cheaper than detecting from scratch every frame.
3. **Fixed 640×480 processing resolution.** Every frame is resized before inference, bounding per-frame cost regardless of camera resolution.
4. **One model, light maths.** The whole decision pipeline sits on the single landmark model's outputs: blendshapes and pose come from the same pass, and the rest is small NumPy geometry. There is no second neural network.
5. **Heavy work is throttled.** Per-frame features are cheap. Windowed aggregation, heuristic scoring and smoothing run **once per second** (`stride_seconds = 1.0`), not every frame.
6. **Cheap quality checks.** Brightness is computed only inside the face box, and jitter uses just 6 stable landmarks.
7. **Zero per-frame audio cost.** Noise is synthesized **once at start-up** and looped by the pygame mixer, and crossfades are handled by the mixer, not the main loop.
8. **Light rendering.** Only every 6th landmark is drawn, with a single overlay panel.
9. **Everything in memory, on-device.** No network, no disk writes in the loop, and a small footprint (~213 MB) that leaves the rest of the machine free for the user's real work.

**Result:** continuous focus tracking at ~92 FPS using only ~2% CPU. A background focus assistant should be nearly invisible to the system, and this one is.

> ℹ️ On a physical webcam, the displayed rate is naturally capped by the camera's frame rate (typically 30 FPS). The ~11 ms per-frame figure shows the pipeline has substantial headroom beyond that.

---

## 6. Scientific evidence

### 6.1 Attention can be inferred from facial and head features

| Finding                                                                          | Reference                                                                               |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Student attention level can be predicted from facial and head/body features      | Zaletelj & Košir (2017), *EURASIP Journal on Image and Video Processing*                |
| Eye Aspect Ratio from facial landmarks reliably detects eye closure and blinks   | Soukupová & Čech (2016), *Computer Vision Winter Workshop*                              |
| Prolonged eye closure (PERCLOS) is a validated drowsiness / vigilance indicator  | Dinges & Grace (1998), *FHWA-MCRT-98-006*                                               |
| Head pose is a strong cue to visual focus of attention                           | Ba & Odobez (2011), *IEEE Transactions on Pattern Analysis and Machine Intelligence*    |
| Mind-wandering has measurable behavioral signatures                              | Smallwood & Schooler (2015), *Annual Review of Psychology*                              |
| Real-time iris tracking from monocular video (basis of MediaPipe iris landmarks) | Ablavatski et al. (2020), *arXiv:2006.11341*                                            |
| Real-time face-mesh landmarks on mobile-class devices                            | Kartynnik et al. (2019), *arXiv:1907.06724*; Lugaresi et al. (2019), *arXiv:1906.08172* |

*Implemented here as: gaze from iris landmarks, head pose from the transformation matrix, and eye closure via EAR + blendshapes with a ≥ 0.5 s "long closure" rule inspired by PERCLOS.*

### 6.2 Background noise can improve focus

| Finding                                                                                                                       | Reference                                                                        |
| ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Moderate-arousal model: noise can benefit attention, especially in people with lower baseline arousal or attention difficulty | Sikström & Söderlund (2007), *Psychological Review*                              |
| White noise improved cognitive performance in children with attention problems                                                | Söderlund, Sikström & Smart (2007), *Journal of Child Psychology and Psychiatry* |
| White noise improved learning and modulated dopaminergic midbrain activity                                                    | Rausch, Bauch & Bunzeck (2014), *Journal of Cognitive Neuroscience*              |
| Moderate ambient noise (~70 dB) enhanced creative cognition versus quieter conditions                                         | Mehta, Zhu & Cheema (2012), *Journal of Consumer Research*                       |
| Auditory-beat stimulation: review of effects on cognition and mood                                                            | Chaieb, Wilpert, Reber & Fell (2015), *Frontiers in Psychiatry*                  |
| Binaural beats: small but significant benefits for attention and memory                                                       | Garcia-Argibay, Santed & Reales (2019), *Psychological Research*                 |

### How strong is the evidence for *this* design?

| Design choice                                                                            | Evidence                                                                                       |
| ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Steady noise can aid focus for some people; white / moderate-level noise is best studied | ✅ Reasonable                                                                                   |
| Broadband masking is helpful when the environment is distracting                         | ✅ Reasonable                                                                                   |
| Pink, brown and "green" noise as focus aids                                              | ⚠️ Limited direct evidence; chosen as calmer, lower-harshness alternatives across the spectrum |
| The specific *zone → noise* mapping                                                      | 🧪 Design hypothesis; the code and this README say so explicitly                               |

> Effects vary by person and level. That is exactly why the app **adapts, keeps sound optional, caps volume, and defaults to quiet when you're focused or away** rather than playing noise nonstop.

> *Please verify each citation's details before final submission.*

---

## 7. Deployment & accessibility

### Option A: Run the `.exe` (recommended, no setup)

1. Download `attentive_behavior_estimator.exe` from [`Application/`](Application/).
2. For the first ~12 s, look at your screen and glance at each corner once (calibration).
3. The HUD appears and adaptive sound starts automatically (green while you're focused).

📹 Step-by-step video: [`Demo/how_to_run_exe.mp4`](Demo/how_to_run_exe.mp4)

### Option B: Run from source

```bash
# 1. Install Python 3.12
winget install --id Python.Python.3.12 --source winget

# 2. Verify Python 3.12
py -3.12 --version

# 3. Create a virtual environment using Python 3.12
py -3.12 -m venv .venv

# 4. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 5. Upgrade pip
py -3.12 -m pip install --upgrade pip

# 6. Install dependencies from requirements.txt
py -3.12 -m pip install -r requirements.txt

# Optional
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Optional flags

```bash
python attentive_behavior_estimator.py --camera 1
# pick another webcam

python attentive_behavior_estimator.py --headless
# no window, audio + camera only

python attentive_behavior_estimator.py --config my.yaml
# override any setting

python attentive_behavior_estimator.py --landmarker PATH
# custom model location
```

**Dependencies:** `numpy`, `pyyaml`, `opencv-python`, `mediapipe`, `pygame`

### Requirements

* EXE file: **Windows 10/11** (tested on **Windows 11, Snapdragon X2 Elite**)
* Source code: run on any OS (macOS, Linux) — tested on **Mac M3 Air**
* Webcam and speakers or headphones
* Roughly 250 MB of free RAM

### Accessibility & usability

* **Zero-install** `.exe`, and no account or internet needed
* **Private**: safe for schools and workplaces, since nothing leaves the device
* **Non-intrusive**: sound, not pop-ups or blocking
* **User control**: keyboard shortcuts, live settings menu, silence at any time
* **Adapts to different users, cameras and seating** via personal calibration
* **Fails gracefully**: no audio device → runs silently; missing audio file → synthesizes it; bad input → lowers confidence instead of guessing

---

## 8. Repository structure

```text
Rdtune/
├── source_code/
│   ├── attentive_behavior_estimator.py
│   ├── face_landmarker.task
│   ├── requirements.txt
│   └── Noise/
│
├── Application/
│   └── attentive_behavior_estimator.exe
│
├── Demo/
│   ├── demo.mov
│   ├── how_to_run_exe.mp4
│   ├── fps.png
│   ├── focused.png
│   └── task_manager.png
│
├── README.md
├── .gitignore
├── .gitattributes
└── LICENSE
```

---

## 9. Limitations & future work

### Limitations (stated openly)

* The output is a **behavioral proxy**, the probability of *off-task behavior relative to your baseline*, not a measure of what you are thinking.
* Heuristic **weights are hand-set, not fitted to data**, and have not been validated on a labeled dataset. Transparency was prioritised over a black-box classifier.
* Legitimate off-screen activity (e.g. reading a paper notebook) can look like "gaze off-screen".
* Accuracy drops in poor lighting, with occlusion or with extreme camera angles. The app surfaces this via data quality and confidence.
* Benefits of background noise vary between individuals, and the zone→noise mapping is a hypothesis.

### Future work

* Accelerate on the Snapdragon **NPU** (e.g. via Qualcomm AI tooling) and ship a native ARM64 build
* Validate and fit the weights on labeled sessions, then compare against the heuristic
* Session summaries and focus trends
* Learn each user's preferred sound and volume
* Optional user-defined "work zones" (multi-monitor, notes area)

---

## 10. Credits & licenses

* **MediaPipe Face Landmarker** (`face_landmarker.task`): Google, open source, Apache 2.0

* **OpenCV**, **NumPy**, **PyYAML**, **pygame**: their respective open-source licenses

* Intellectual Property Rights in submissions and final submissions remain solely with participants

* Project license: MIT (see `LICENSE`)

**Author:** Rudram Dindorkar
**GitHub:** [rudramdindorkar](https://github.com/rudramdindorkar)


