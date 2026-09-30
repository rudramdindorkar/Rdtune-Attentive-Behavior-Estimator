# #############################################################################
# ATTENTIVE-BEHAVIOR ESTIMATOR
#
# Estimates a probability that a person is showing OFF-TASK behaviour in short
# webcam windows, RELATIVE TO THEIR OWN BASELINE, with a confidence value and a
# data-quality flag. Runs fully on-device. Optionally plays adaptive background
# noise (green by default, pink/white while distracted, silence on demand).

# Keys in the video window:
#   c recalibrate | m settings | a adaptive audio | s silence
#   1 brown | 2 pink | 3 green | 4 white | q quit
# #############################################################################
import argparse
import copy
import os
from pathlib import Path
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

#configuration
SCRIPT_DIR = Path(__file__).resolve().parent

CONFIG_YAML = """
paths:
  landmarker: "face_landmarker.task"

window:
  seconds: 10.0               # analysis window
  stride_seconds: 1.0         # how often a new estimate is produced
  min_frames: 15              # fewer frames than this -> no estimate

calibration:
  seconds: 12.0               # "work normally, glance at all screen corners once"
  min_valid_frames: 60        # frames with a face needed for a personal baseline
  max_wait_multiplier: 3.0    # give up (use a provisional baseline) after N x seconds
  gaze_percentiles: [2, 98]   # extent of the person's on-screen gaze region
  region_margin: 1.25         # tolerance added around that region
  gaze_hw_floor: {h: 0.05, v: 0.08}   # min half-width of gaze region
  head_sd_floor_deg: {yaw: 4.0, pitch: 4.0}

features:
  gaze_off_excess: 1.0        # gaze excess > 1.0 == outside calibrated region
  head_away_z: 3.0            # |deviation| / person's own SD
  eye_closed_blendshape: 0.5  # eyeBlink >= this counts as "closed" ...
  eye_closed_ear_ratio: 0.55  # ... or EAR / baseline EAR <= this (either signal)
  long_closure_seconds: 0.5   # closures >= this are "long" (not blinks)
  max_frame_gap_seconds: 0.5  # larger gaps are clipped (camera stalls)

quality:
  min_face_height_frac: 0.18  # landmark bbox height / frame height
  min_luminance: 60           # mean gray level (0-255) in the face region
  max_luminance: 215
  max_jitter: 0.012           # 2nd-difference landmark motion / face height
  extreme_yaw_deg: 60
  extreme_pitch_deg: 45
  max_extreme_frac: 0.5
  min_fps: 8

confidence:
  no_baseline_factor: 0.5     # multiplier when only a provisional baseline exists
  face_absent_factor: 0.7     # absence can also mean a camera problem

# Calibrated heuristic:  p = sigmoid(bias + sum weight * clip((frac - dz)/(1 - dz), 0, 1))
# Weights are hand-set design choices, NOT fitted to data.
heuristic:
  bias: -3.0                  # p(off-task) ~ 5% when nothing fires
  explain_min_contribution: 0.3
  rules:
    face_absent:      {weight: 6.0, deadzone: 0.10}
    gaze_off_screen:  {weight: 4.0, deadzone: 0.10}
    head_away:        {weight: 3.0, deadzone: 0.10}
    long_eye_closure: {weight: 4.0, deadzone: 0.10}   # drowsiness proxy

# Temporal smoothing of window probabilities (windows overlap ~90%)
temporal:
  stay_prob: 0.90             # probability the on/off-task state persists per stride
  evidence_weight: 0.3

# ---- Adaptive sound ----
# DEFAULT while focused = calm GREEN noise. Press 's' for silence.
# When a distraction (drifting / off_task) is detected, that noise plays for
# distraction_hold_seconds; only after that does it follow behaviour again.
# The mapping is a design hypothesis, not a proven intervention.
audio:
  enabled: true
  source: files_if_present      # files_if_present | synth_only
  files:                        # your own tracks; missing files are synthesized
    brown: "Noise/brown.mp3"
    pink: "Noise/pink.mp3"
    green: "Noise/green.mp3"
    white: "Noise/white.mp3"
  synth_seconds: 30
  mapping:                      # zone -> noise (null = silence) and volume 0..1
    focused:   {noise: green, volume: 0.15}   # DEFAULT
    drifting:  {noise: pink,  volume: 0.25}
    off_task:  {noise: white, volume: 0.30}
    away:      {noise: null,  volume: 0.0}
  zone_thresholds: {drifting: 0.35, off_task: 0.65, hysteresis: 0.05}
  min_confidence: 0.30          # below this the audio zone is held, not changed
  switch_delay_seconds: 4.0     # debounce before reacting to any change
  distraction_hold_seconds: 20.0  # minimum dwell time for a distraction noise
  max_distraction_hold_seconds: 90.0 # never lock a distraction sound longer than this
  recovery_confirm_seconds: 4.0    # focused evidence required before leaving distraction
  away_recovery_seconds: 10.0   # after the face returns, ignore estimates this long (window still holds the absence)
  crossfade_ms: 1500
  max_volume: 0.5               # safety cap
  manual_volume: 0.25           # volume for keys 1-4

display:
  enabled: true                  # set false for headless camera/audio operation
  mirror: true
  width: 640
  height: 480
"""


def _resolve_path(value, base_dir):
    if not value:
        return value
    path = Path(os.path.expanduser(str(value)))
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def load_config(path=None, landmarker=None):
    """Load config and resolve relative model/audio paths from its source directory."""
    config_path = Path(path).expanduser().resolve() if path else None
    if config_path:
        with config_path.open() as fh:
            cfg = yaml.safe_load(fh) or {}
        base_dir = config_path.parent
    else:
        cfg = yaml.safe_load(CONFIG_YAML)
        base_dir = SCRIPT_DIR
    cfg.setdefault("paths", {})
    cfg.setdefault("audio", {}).setdefault("files", {})
    marker_value = landmarker or cfg["paths"].get("landmarker", "face_landmarker.task")
    marker_base = Path.cwd() if landmarker else base_dir
    cfg["paths"]["landmarker"] = _resolve_path(marker_value, marker_base)
    for name, value in list(cfg["audio"]["files"].items()):
        cfg["audio"]["files"][name] = _resolve_path(value, base_dir)
    cfg["_config_path"] = str(config_path) if config_path else None
    return cfg


def validate_paths(cfg):
    landmarker = cfg["paths"]["landmarker"]
    if not landmarker or not Path(landmarker).is_file():
        raise RuntimeError(
            f"Face landmarker model not found: {landmarker!r}. "
            "Pass --landmarker PATH or set paths.landmarker in --config.")


# ==============================================================================
# UTILS
# ==============================================================================
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def wrap_deg(a):
    """Wrap angle(s) to [-180, 180)."""
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def circ_median(deg):
    """Median of angles that may straddle the +/-180 seam."""
    deg = np.asarray(deg, dtype=float)
    deg = deg[~np.isnan(deg)]
    if deg.size == 0:
        return 0.0
    ref = deg[0]
    return float(wrap_deg(ref + np.median(wrap_deg(deg - ref))))


def robust_sd(deg, center):
    """1.4826 * MAD around `center`, seam-safe."""
    deg = np.asarray(deg, dtype=float)
    deg = deg[~np.isnan(deg)]
    if deg.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(wrap_deg(deg - center))))


def euler_from_matrix(M):
    """(yaw, pitch) in degrees from a 4x4 / 3x3 pose matrix. Axis signs differ between
    libraries, but everything downstream uses deviations from the person's own baseline."""
    R = np.asarray(M, dtype=float)[:3, :3]
    norms = np.linalg.norm(R, axis=0)
    norms[norms < 1e-9] = 1.0
    R = R / norms
    sy = np.hypot(R[0, 0], R[1, 0])
    yaw = np.degrees(np.arctan2(-R[2, 0], sy))
    pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return float(yaw), float(pitch)


def frame_dts(t, max_gap):
    """Per-frame durations (seconds), gaps clipped to `max_gap`."""
    t = np.asarray(t, dtype=float)
    if t.size == 0:
        return t
    if t.size == 1:
        return np.array([min(0.033, max_gap)])
    dt = np.diff(t)
    last = float(np.median(dt)) if dt.size else 0.033
    return np.clip(np.append(dt, last), 0.0, max_gap)


def true_runs(mask):
    """List of (start, end_exclusive) index pairs for consecutive True values."""
    mask = np.asarray(mask, dtype=bool)
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _nm(x):
    """nanmean that returns 0.0 (no warning) for empty / all-NaN input."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(x.mean()) if x.size else 0.0


# ==============================================================================
# FEATURES  (per frame, from MediaPipe FaceLandmarker outputs only)
# ==============================================================================
NAN = float("nan")

# "A" = eye on the image-left, "B" = eye on the image-right (avoids left/right mirror confusion)
EYE_A = dict(outer=33, inner=133, iris=468, upper=159, lower=145, ear=(33, 160, 158, 133, 153, 144))
EYE_B = dict(outer=263, inner=362, iris=473, upper=386, lower=374, ear=(362, 385, 387, 263, 373, 380))
STABLE_PTS = [1, 33, 263, 61, 291, 152]          # for the jitter metric


@dataclass
class FrameFeatures:
    t: float
    present: bool = False
    gaze_h: float = NAN        # iris offset, eye-width units, image-right positive
    gaze_v: float = NAN        # iris offset, lid-opening units, image-down positive
    yaw: float = NAN
    pitch: float = NAN
    ear: float = NAN           # eye aspect ratio, mean of both eyes
    blink_bs: float = NAN      # mean(eyeBlinkLeft, eyeBlinkRight)
    bbox_h: float = NAN        # face landmark bbox height / frame height
    luminance: float = NAN     # mean gray level in face region
    jitter: float = NAN        # 2nd-difference landmark motion / face height


def _iris_ratios(pts, e):
    o, i, c = pts[e["outer"]], pts[e["inner"]], pts[e["iris"]]
    axis = i - o
    L = float(np.linalg.norm(axis))
    if L < 1e-6:
        return NAN, NAN
    h = float(np.dot(c - o, axis) / (L * L) - 0.5)
    up, lo = pts[e["upper"]], pts[e["lower"]]
    opening = lo[1] - up[1]
    # vertical ratio is unreliable when the lid is nearly closed
    v = float((c[1] - up[1]) / opening - 0.5) if opening > 0.12 * L else NAN
    return h, v


def _ear(pts, idx):
    p1, p2, p3, p4, p5, p6 = (pts[k] for k in idx)
    w = np.linalg.norm(p1 - p4)
    if w < 1e-6:
        return NAN
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * w))


def _bs(bs, *names):
    vals = [bs[n] for n in names if n in bs]
    return float(np.mean(vals)) if vals else NAN


class FeatureExtractor:
    def __init__(self):
        self._hist = []   # last two stable-landmark arrays, for the jitter metric

    def reset(self):
        self._hist = []

    def extract(self, t, result, gray, size):
        """result: a FaceLandmarkerResult. size: (width, height) of the frame in px."""
        if not result.face_landmarks:
            self._hist = []
            return FrameFeatures(t=t)                      # absent frame
        w, h = size
        pts = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], dtype=float)
        bs = {c.category_name: c.score for c in result.face_blendshapes[0]} \
            if result.face_blendshapes else {}
        mats = getattr(result, "facial_transformation_matrixes", None)
        f = FrameFeatures(t=t, present=True)

        # ---- gaze (iris relative to eye corners) ----
        hA, vA = _iris_ratios(pts, EYE_A)
        hB, vB = _iris_ratios(pts, EYE_B)
        # eye A's outer->inner axis points image-right, eye B's points image-left,
        # so subtract to get one image-right-positive horizontal gaze.
        f.gaze_h = float((hA - hB) / 2.0)
        vs = [v for v in (vA, vB) if not np.isnan(v)]
        f.gaze_v = float(np.mean(vs)) if vs else NAN

        # ---- head pose ----
        if mats:
            f.yaw, f.pitch = euler_from_matrix(mats[0])

        # ---- eyes ----
        ears = [e for e in (_ear(pts, EYE_A["ear"]), _ear(pts, EYE_B["ear"])) if not np.isnan(e)]
        f.ear = float(np.mean(ears)) if ears else NAN
        f.blink_bs = _bs(bs, "eyeBlinkLeft", "eyeBlinkRight")

        # ---- quality metrics ----
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        face_h = float(y1 - y0)
        f.bbox_h = face_h / h
        if gray is not None and face_h > 1:
            ys, ye = int(max(y0, 0)), int(min(y1, gray.shape[0]))
            xs, xe = int(max(x0, 0)), int(min(x1, gray.shape[1]))
            if ye > ys and xe > xs:
                f.luminance = float(gray[ys:ye, xs:xe].mean())
        stable = pts[STABLE_PTS]
        if len(self._hist) == 2 and face_h > 1:
            second_diff = stable - 2 * self._hist[1] + self._hist[0]
            f.jitter = float(np.mean(np.linalg.norm(second_diff, axis=1)) / face_h)
        self._hist = (self._hist + [stable])[-2:]
        return f


# ==============================================================================
# BASELINE  (per-person, from the calibration segment)
# Everything downstream is relative to this, so the same code works for different
# faces, cameras and seating positions. No labeled data needed.
# ==============================================================================
def eye_closed_mask(blink_bs, ear, ear0, fcfg):
    """Either signal may vote 'closed' (blendshape blink OR EAR drop vs baseline)."""
    with np.errstate(invalid="ignore"):
        by_bs = np.asarray(blink_bs, float) >= fcfg["eye_closed_blendshape"]
        by_ear = (np.asarray(ear, float) / max(ear0, 1e-6)) <= fcfg["eye_closed_ear_ratio"]
    return by_bs | by_ear


@dataclass
class Baseline:
    gaze_center: tuple = (0.0, 0.0)
    gaze_hw: tuple = (0.12, 0.15)      # half-width of the person's on-screen gaze region
    yaw0: float = 0.0
    pitch0: float = 0.0
    yaw_sd: float = 4.0
    pitch_sd: float = 4.0
    ear0: float = 0.28
    is_personal: bool = False          # True only with enough valid calibration frames

    @classmethod
    def default(cls, cfg):
        c = cfg["calibration"]
        return cls(gaze_hw=(c["gaze_hw_floor"]["h"] * 2, c["gaze_hw_floor"]["v"] * 2),
                   yaw_sd=c["head_sd_floor_deg"]["yaw"], pitch_sd=c["head_sd_floor_deg"]["pitch"])

    @classmethod
    def from_frames(cls, frames, cfg):
        """With too few valid frames the result is a *provisional* baseline
        (is_personal=False) that lowers confidence."""
        cal, fcfg = cfg["calibration"], cfg["features"]
        pres = [f for f in frames if f.present]
        n = len(pres)
        if n < 5:
            return cls.default(cfg)

        def col(name):
            return np.array([getattr(f, name) for f in pres], float)

        def region(x, floor):
            x = x[~np.isnan(x)]
            if x.size < 5:
                return 0.0, floor * 2
            lo, hi = np.percentile(x, cal["gaze_percentiles"])
            return float((lo + hi) / 2), float(max((hi - lo) / 2 * cal["region_margin"], floor))

        gh, hh = region(col("gaze_h"), cal["gaze_hw_floor"]["h"])
        gv, hv = region(col("gaze_v"), cal["gaze_hw_floor"]["v"])

        fl = cal["head_sd_floor_deg"]
        yaw, pitch = col("yaw"), col("pitch")
        y0, p0 = circ_median(yaw), circ_median(pitch)

        # open-eye EAR: frames where the blink blendshape says the eye is open
        ear, bs = col("ear"), col("blink_bs")
        with np.errstate(invalid="ignore"):
            open_ear = ear[bs < 0.3]
        open_ear = open_ear[~np.isnan(open_ear)]
        if open_ear.size == 0:
            open_ear = ear[~np.isnan(ear)]
        ear0 = float(np.median(open_ear)) if open_ear.size else 0.28

        dt = frame_dts(np.array([f.t for f in pres]), fcfg["max_frame_gap_seconds"])
        return cls(
            gaze_center=(gh, gv), gaze_hw=(hh, hv),
            yaw0=y0, pitch0=p0,
            yaw_sd=max(robust_sd(yaw, y0), fl["yaw"]),
            pitch_sd=max(robust_sd(pitch, p0), fl["pitch"]),
            ear0=ear0,
            # personal = enough frames AND the face was visible for >= half the calibration time
            is_personal=n >= cal["min_valid_frames"] and dt.sum() >= 0.5 * cal["seconds"],
        )


# ==============================================================================
# WINDOWS
# Aggregate per-frame features into baseline-normalized values for a short window.
# Fractions are time-weighted. 'Face absent' is its own category: it is measured
# explicitly and never treated as missing data.
# ==============================================================================
@dataclass
class WindowFeatures:
    values: dict
    quality: float
    flag: str                     # single data-quality flag ("ok" if none)
    is_personal: bool
    window_s: float
    t_end: float


def aggregate_window(frames, baseline: Baseline, cfg):
    fc, qc = cfg["features"], cfg["quality"]
    n = len(frames)
    t = np.array([f.t for f in frames], float)
    dt = frame_dts(t, fc["max_frame_gap_seconds"])
    total = max(dt.sum(), 1e-6)
    present = np.array([f.present for f in frames], bool)
    pres_t = dt[present].sum()

    def col(name):
        return np.array([getattr(f, name) for f in frames], float)

    def wfrac(mask, valid):
        d = dt[valid].sum()
        return float(dt[mask & valid].sum() / d) if d > 1e-6 else 0.0

    v = {"face_absent_frac": float(dt[~present].sum() / total)}

    with np.errstate(invalid="ignore"):
        # ---- gaze relative to the person's calibrated on-screen region ----
        gh, gv = baseline.gaze_center
        hh, hv = baseline.gaze_hw
        excess = np.fmax(np.abs(col("gaze_h") - gh) / hh, np.abs(col("gaze_v") - gv) / hv)
        gvalid = present & ~np.isnan(excess)
        v["gaze_off_frac"] = wfrac(excess > fc["gaze_off_excess"], gvalid)

        # ---- head pose (deviation from baseline, in the person's own SDs) ----
        dyaw = wrap_deg(col("yaw") - baseline.yaw0)
        dpit = wrap_deg(col("pitch") - baseline.pitch0)
        zhead = np.fmax(np.abs(dyaw) / baseline.yaw_sd, np.abs(dpit) / baseline.pitch_sd)
        hvalid = present & ~np.isnan(zhead)
        v["head_away_frac"] = wfrac(zhead > fc["head_away_z"], hvalid)

        # ---- long eye closures (drowsiness PROXY, not a diagnosis) ----
        closed = eye_closed_mask(col("blink_bs"), col("ear"), baseline.ear0, fc) & present
        long_t = 0.0
        for a, b in true_runs(closed):
            d = dt[a:b].sum()
            if d >= fc["long_closure_seconds"]:
                long_t += d
        v["long_closure_frac"] = float(long_t / pres_t) if pres_t > 1e-6 else 0.0

    # ---- data quality ----
    q, issues = 1.0, []
    if pres_t > 1e-6:
        with np.errstate(invalid="ignore"):
            bbox = _nm(col("bbox_h")[present])
            lum = _nm(col("luminance")[present])
            jit = _nm(col("jitter")[present])
            extreme = wfrac((np.abs(dyaw) > qc["extreme_yaw_deg"]) | (np.abs(dpit) > qc["extreme_pitch_deg"]),
                            hvalid) if hvalid.any() else 0.0
        span = max(t[-1] - t[0], 1e-6)
        fps = (n - 1) / span if n > 1 else 0.0
        if 0 < bbox < qc["min_face_height_frac"]:
            q *= 0.5; issues.append("face_small")
        if 0 < lum < qc["min_luminance"]:
            q *= 0.6; issues.append("too_dark")
        if lum > qc["max_luminance"]:
            q *= 0.7; issues.append("overexposed")
        if jit > qc["max_jitter"]:
            q *= 0.7; issues.append("jittery_landmarks")
        if extreme > qc["max_extreme_frac"]:
            q *= 0.6; issues.append("extreme_pose")
        if fps < qc["min_fps"]:
            q *= 0.6; issues.append("low_fps")
    else:
        q = 0.6   # cannot verify the camera when nobody is visible

    flag = "face_absent" if v["face_absent_frac"] >= 0.9 else (issues[0] if issues else "ok")
    return WindowFeatures(values=v, quality=float(q), flag=flag, is_personal=baseline.is_personal,
                          window_s=float(total), t_end=float(t[-1]))


# ==============================================================================
# CONFIDENCE = calibration certainty x window data quality x baseline availability
# ==============================================================================
def compute_confidence(p, quality, is_personal, flag, cfg):
    cc = cfg["confidence"]
    certainty = abs(2.0 * p - 1.0)                       # 0 at p=0.5, 1 at p in {0,1}
    baseline_factor = 1.0 if is_personal else cc["no_baseline_factor"]
    c = quality * baseline_factor * (0.4 + 0.6 * certainty)
    if flag == "face_absent":
        c *= cc["face_absent_factor"]
    return float(np.clip(c, 0.0, 1.0))


# ==============================================================================
# HEURISTIC DETECTOR
# Transparent weighted sum of baseline-normalized signals through a logistic squash:
#     z = bias + sum_i weight_i * clip((signal_i - deadzone_i) / (1 - deadzone_i), 0, 1)
#     p(off-task) = sigmoid(z)
# ==============================================================================
# rule name -> (window-feature name, human description)
RULES = {
    "face_absent": ("face_absent_frac", "face not visible"),
    "gaze_off_screen": ("gaze_off_frac", "gaze outside the calibrated screen region"),
    "head_away": ("head_away_frac", "head turned away from the calibrated pose"),
    "long_eye_closure": ("long_closure_frac", "eyes closed for long stretches (drowsiness proxy)"),
}


class HeuristicDetector:
    def __init__(self, cfg):
        h = cfg["heuristic"]
        self.bias = float(h["bias"])
        self.weights = {k: float(h["rules"][k]["weight"]) for k in RULES}
        self.deadzones = {k: float(h["rules"][k]["deadzone"]) for k in RULES}
        self.min_contrib = float(h["explain_min_contribution"])

    def contributions(self, feats):
        out = {}
        for rule, (sig, _) in RULES.items():
            dz = self.deadzones[rule]
            act = float(np.clip((float(feats[sig]) - dz) / max(1.0 - dz, 1e-6), 0.0, 1.0))
            out[rule] = self.weights[rule] * act
        return out

    def predict_proba(self, feats):
        return float(sigmoid(self.bias + sum(self.contributions(feats).values())))

    def explain(self, feats, window_s):
        """e.g. 'gaze outside the calibrated screen region for 7.0 of 10 s; ...'"""
        fired = []
        for rule, c in sorted(self.contributions(feats).items(), key=lambda kv: -kv[1]):
            if c >= self.min_contrib:
                frac = float(feats[RULES[rule][0]])
                fired.append(f"{RULES[rule][1]} for {frac * window_s:.1f} of {window_s:.0f} s")
        return "; ".join(fired) if fired else "no off-task rule fired"


# ==============================================================================
# TEMPORAL SMOOTHING
# Two-state forward filter with sticky transitions. Window probabilities are used
# as tempered pseudo-likelihoods (windows overlap ~90%). A heuristic smoother, not
# a calibrated model: the raw p is reported alongside it.
# ==============================================================================
class TemporalSmoother:
    def __init__(self, stay_prob=0.9, evidence_weight=0.3):
        self.stay, self.w, self.belief = stay_prob, evidence_weight, 0.5

    def reset(self):
        self.belief = 0.5

    def update(self, p):
        p = float(np.clip(p, 1e-3, 1 - 1e-3))
        prior = self.stay * self.belief + (1 - self.stay) * (1 - self.belief)
        num = prior * p ** self.w
        den = num + (1 - prior) * (1 - p) ** self.w
        self.belief = float(num / den)
        return self.belief


# ==============================================================================
# ESTIMATOR
#   calibration -> Baseline -> sliding window -> heuristic -> smoothing -> Estimate
# The estimate is the probability of *off-task behaviour relative to this person's
# own baseline*. It is not a measure of what the person is thinking.
# ==============================================================================
@dataclass
class Estimate:
    t: float
    p_offtask: float            # smoothed (used for UI / audio)
    p_raw: float                # heuristic output
    confidence: float
    quality: float
    quality_flag: str
    baseline_personal: bool = False
    explanation: str = ""


class AttentionEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.heuristic = HeuristicDetector(cfg)
        t = cfg["temporal"]
        self.smoother = TemporalSmoother(t["stay_prob"], t["evidence_weight"])
        self.reset_calibration()

    # ---- calibration ----
    def reset_calibration(self):
        self.baseline: Optional[Baseline] = None
        self._calib, self._calib_start = [], None
        self._buf = deque()
        self._last_emit = -1e9
        self.smoother.reset()

    @property
    def calibrating(self):
        return self.baseline is None

    def calib_progress(self, now):
        if self._calib_start is None:
            return 0.0
        return float(min((now - self._calib_start) / self.cfg["calibration"]["seconds"], 1.0))

    def _maybe_finish_calibration(self, t):
        c = self.cfg["calibration"]
        elapsed = t - self._calib_start
        if elapsed < c["seconds"]:
            return
        b = Baseline.from_frames(self._calib, self.cfg)
        if b.is_personal or elapsed >= c["seconds"] * c["max_wait_multiplier"]:
            self.baseline = b        # otherwise keep waiting; after the cap use a provisional baseline

    # ---- streaming ----
    def update(self, ff) -> Optional[Estimate]:
        if self.calibrating:
            if self._calib_start is None:
                self._calib_start = ff.t
            self._calib.append(ff)
            self._maybe_finish_calibration(ff.t)
            return None

        W = self.cfg["window"]
        self._buf.append(ff)
        while self._buf and ff.t - self._buf[0].t > W["seconds"]:
            self._buf.popleft()
        span = self._buf[-1].t - self._buf[0].t
        ready = len(self._buf) >= W["min_frames"] and span >= 0.6 * W["seconds"]
        due = ff.t - self._last_emit >= W["stride_seconds"]
        if not (ready and due):
            return None
        self._last_emit = ff.t
        return self.estimate_window(list(self._buf))

    def estimate_window(self, frames) -> Estimate:
        wf = aggregate_window(frames, self.baseline, self.cfg)
        p_raw = self.heuristic.predict_proba(wf.values)
        if wf.flag == "face_absent":
            # Don't let an absence build up "off-task" belief that would linger after the
            # person returns; the audio layer reacts to absence separately (away zone).
            self.smoother.reset()
            p = p_raw
        else:
            p = self.smoother.update(p_raw)
        return Estimate(
            t=wf.t_end, p_offtask=float(p), p_raw=float(p_raw),
            confidence=compute_confidence(p_raw, wf.quality, wf.is_personal, wf.flag, self.cfg),
            quality=wf.quality, quality_flag=wf.flag, baseline_personal=wf.is_personal,
            explanation=self.heuristic.explain(wf.values, wf.window_s))


# ==============================================================================
# AUDIO
# Zones: focused -> GREEN (default) | drifting -> pink | off_task -> white | away -> silence.
# Rules:
#   * a change must be wanted for switch_delay_seconds before it happens (debounce)
#   * once a distraction noise (drifting / off_task) starts, it plays for
#     distraction_hold_seconds; only then does it follow behaviour again
#   * low-confidence estimates never change the zone
#   * after the face returns from 'away', estimates are ignored for away_recovery_seconds
#     (the 10 s window still contains the absence and would look like a distraction)
#   * manual keys ('s' silence, 1-4 a noise) act immediately and are never locked
# Any track whose file is missing is synthesized, so no audio assets are required.
# ==============================================================================
NOISES = ("brown", "pink", "green", "white")
SR = 44100


def synth_noise(kind, seconds=30.0, sr=SR, seed=0):
    """Synthesize a seamlessly-looping stereo int16 noise array of shape (N, 2).

    white : flat power spectrum      pink : power ~ 1/f (-3 dB/octave)
    brown : power ~ 1/f^2 (-6 dB/octave)
    green : white noise shaped to a ~500 Hz mid-band hump (no standard definition;
            this is one common interpretation)
    Built in the frequency domain with random phases, so the loop point is seamless.
    """
    n = int(seconds * sr)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    f = np.maximum(freqs, 20.0)
    if kind == "white":
        amp = np.ones_like(f)
    elif kind == "pink":
        amp = 1.0 / np.sqrt(f)
    elif kind == "brown":
        amp = 1.0 / f
    elif kind == "green":
        amp = 0.05 + np.exp(-0.5 * (np.log2(f / 500.0) / 1.2) ** 2)
    else:
        raise ValueError(f"unknown noise kind: {kind}")
    amp = amp.copy()
    amp[freqs < 20.0] = 0.0                       # no DC / subsonic energy
    channels = []
    for ch in range(2):
        rng = np.random.default_rng(seed * 2 + ch)
        spec = amp * np.exp(1j * rng.uniform(0, 2 * np.pi, amp.size))
        x = np.fft.irfft(spec, n)
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-12) * 0.12       # equal RMS (~-18 dBFS) for all colours
        channels.append(np.clip(x, -0.98, 0.98))
    return (np.stack(channels, axis=1) * 32767).astype(np.int16)


class AdaptiveAudio:
    """Turns Estimates into sound decisions. Pure logic + an injected backend, so it
    can be tested without an audio device.

    backend needs: play(noise, volume, fade_ms), set_volume(volume), fade_out(fade_ms), stop()
    """

    def __init__(self, cfg, backend):
        self.a = cfg["audio"]
        self.backend = backend
        self.zone = "focused"
        self.manual = "auto"                    # "auto" | "off" | one of NOISES
        self.current = (None, 0.0)              # (noise, volume) actually playing
        self._want, self._want_since = self.current, 0.0
        self._hold_until = 0.0                  # distraction noise is locked in until this time
        self._ignore_until = 0.0                # estimates ignored until this time (after 'away')
        self._focused_since = None

    def reset(self):
        """Call on recalibration: back to the default zone."""
        self.zone = "focused"
        self._ignore_until = 0.0
        self._focused_since = None

    def set_manual(self, choice):
        """choice: 'auto', 'off', or a noise name. Manual choices bypass the adaptive logic."""
        self.manual = choice
        self._hold_until = 0.0                  # manual choices never leave a stale lock behind
        self._focused_since = None

    def _update_zone(self, p):
        """focused / drifting / off_task with hysteresis."""
        t = self.a["zone_thresholds"]
        thr = {1: t["drifting"], 2: t["off_task"]}
        h = t["hysteresis"]
        level = {"focused": 0, "drifting": 1, "off_task": 2}.get(self.zone, 0)
        while level < 2 and p >= thr[level + 1]:
            level += 1
        while level > 0 and p < thr[level] - h:
            level -= 1
        self.zone = ("focused", "drifting", "off_task")[level]

    def _mapped(self, zone):
        m = self.a["mapping"][zone]
        noise, vol = m.get("noise"), float(m.get("volume", 0.0))
        return (noise, min(vol, self.a["max_volume"])) if noise else (None, 0.0)

    def _manual_target(self):
        if self.manual == "off":
            return (None, 0.0)
        return (self.manual, min(self.a["manual_volume"], self.a["max_volume"]))

    def update(self, est, now):
        """Call every frame (cheap). `est` may be None (e.g. during calibration)."""
        if self.manual != "auto":
            self._apply(self._manual_target(), now, immediate=True)
            return
        if est is not None:
            if est.quality_flag == "face_absent":
                self.zone = "away"
                self._focused_since = None
            elif self.zone == "away":
                self.zone = "focused"
                self._ignore_until = now + float(self.a["away_recovery_seconds"])
                self._focused_since = now
            elif now >= self._ignore_until and est.confidence >= self.a["min_confidence"]:
                previous = self.zone
                self._update_zone(est.p_offtask)
                if self.zone == "focused":
                    self._focused_since = self._focused_since or now
                else:
                    self._focused_since = None
                # Do not immediately leave a distraction state on one good window.
                # This prevents an overlapping window or a transient gaze estimate
                # from causing rapid audio oscillation.
                if (previous in ("drifting", "off_task") and self.zone == "focused"
                        and now - self._focused_since < self.a["recovery_confirm_seconds"]):
                    self.zone = previous
                elif previous in ("drifting", "off_task") and self.zone == "focused":
                    self._focused_since = None
            # else: low confidence / recovering -> hold the current zone
        # est None (calibrating) -> the current zone applies, so green starts right away
        self._apply(self._mapped(self.zone), now, immediate=self.zone == "away")

    def _apply(self, want, now, immediate=False):
        if want != self._want:
            self._want, self._want_since = want, now
        if want == self.current:
            return
        if not immediate:
            if now - self._want_since < self.a["switch_delay_seconds"]:
                return                                     # debounce
            if now < self._hold_until and want[0] is not None:
                return                                     # distraction noise still in its hold time
        fade = self.a["crossfade_ms"]
        if want[0] is None:
            self.backend.fade_out(fade)
        elif want[0] == self.current[0]:
            self.backend.set_volume(want[1])
        else:
            try:
                self.backend.play(want[0], want[1], fade)
            except RuntimeError as exc:
                print(f"Audio: {exc}; falling back to silence")
                want = (None, 0.0)
                self.backend.fade_out(fade)
        self.current = want
        if not immediate and self.zone in ("drifting", "off_task"):
            hold = float(self.a["distraction_hold_seconds"])
            maximum = float(self.a["max_distraction_hold_seconds"])
            self._hold_until = min(now + hold, now + maximum)

    def stop(self):
        self.backend.stop()


class PygameBackend:
    """Crossfading playback over two channels."""

    def __init__(self, cfg):
        import pygame
        self.pg = pygame
        a = cfg["audio"]
        pygame.mixer.init(frequency=SR, size=-16, channels=2)
        pygame.mixer.set_num_channels(8)
        self.sounds = {}
        for name in NOISES:
            path = a["files"].get(name)
            try:
                if a["source"] == "files_if_present" and path and os.path.exists(path):
                    self.sounds[name] = pygame.mixer.Sound(path)
                else:
                    arr = np.ascontiguousarray(synth_noise(name, a["synth_seconds"], SR))
                    self.sounds[name] = pygame.sndarray.make_sound(arr)
            except Exception as e:                                # keep running without this track
                print(f"Audio: could not load/synthesize '{name}' ({e})")
                self.sounds[name] = None
        self.active = pygame.mixer.Channel(0)
        self.standby = pygame.mixer.Channel(1)

    def play(self, noise, volume, fade_ms):
        sound = self.sounds.get(noise)
        if sound is None:
            raise RuntimeError(f"Audio track is unavailable: {noise}")
        next_channel = self.standby
        next_channel.set_volume(volume)
        next_channel.play(sound, loops=-1, fade_ms=fade_ms)
        old_channel = self.active
        self.active, self.standby = next_channel, old_channel
        if old_channel.get_busy():
            old_channel.fadeout(fade_ms)

    def set_volume(self, volume):
        self.active.set_volume(volume)

    def fade_out(self, fade_ms):
        self.active.fadeout(fade_ms)

    def stop(self):
        self.active.fadeout(300)
        self.standby.fadeout(300)
        self.pg.mixer.quit()


# ==============================================================================
# LIVE APP (webcam). Everything stays on this device.
# ==============================================================================
FONT = cv2.FONT_HERSHEY_SIMPLEX
ZONE_COLORS = {"focused": (0, 200, 0), "drifting": (0, 200, 255), "off_task": (60, 60, 255), "away": (160, 160, 160)}
AUDIO_KEYS = {ord("s"): "off", ord("1"): "brown", ord("2"): "pink",
              ord("3"): "green", ord("4"): "white", ord("a"): "auto"}
SETTING_NOISES = ("green", "pink", "white", "brown", "off")


def txt(img, s, x, y, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_bar(img, label, value, x, y, color, w=170):
    txt(img, label, x, y + 12)
    bx = x + 95
    cv2.rectangle(img, (bx, y), (bx + w, y + 14), (70, 70, 70), -1)
    cv2.rectangle(img, (bx, y), (bx + int(w * float(np.clip(value, 0, 1))), y + 14), color, -1)
    txt(img, f"{value * 100:3.0f}%", bx + w + 6, y + 12)


def draw_panel(img, est, estimator, audio, now):
    x0, y0, pw, ph = 10, 45, 330, 225
    ov = img.copy()
    cv2.rectangle(ov, (x0, y0), (x0 + pw, y0 + ph), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, img, 0.45, 0, img)
    if estimator.calibrating:
        p = estimator.calib_progress(now)
        txt(img, "Calibrating: look at your screen,", x0 + 10, y0 + 28, 0.55, (0, 255, 255))
        txt(img, "glance at each corner once, then relax", x0 + 10, y0 + 50, 0.55, (0, 255, 255))
        cv2.rectangle(img, (x0 + 10, y0 + 65), (x0 + 260, y0 + 79), (70, 70, 70), -1)
        cv2.rectangle(img, (x0 + 10, y0 + 65), (x0 + 10 + int(250 * p), y0 + 79), (0, 255, 255), -1)
        return
    if est is None:
        txt(img, "Warming up first window...", x0 + 10, y0 + 30, 0.55, (0, 255, 255))
        return
    z = audio.zone if audio else "focused"
    y = y0 + 10
    draw_bar(img, "Off-task p", est.p_offtask, x0 + 10, y, ZONE_COLORS[z]); y += 24
    draw_bar(img, "Confidence", est.confidence, x0 + 10, y, (200, 200, 200)); y += 24
    q_col = (0, 220, 0) if est.quality_flag == "ok" else (0, 165, 255)
    txt(img, f"Data quality: {est.quality_flag} ({est.quality:.2f})", x0 + 10, y + 12, 0.5, q_col); y += 22
    base = "personal" if est.baseline_personal else "PROVISIONAL (low confidence)"
    txt(img, f"Baseline: {base}", x0 + 10, y + 12, 0.45); y += 22
    txt(img, f"Zone: {z}", x0 + 10, y + 12, 0.6, ZONE_COLORS[z], 2); y += 26
    s = est.explanation
    for i in range(0, min(len(s), 138), 46):                      # wrap the explanation (3 lines max)
        txt(img, s[i:i + 46], x0 + 10, y + 12, 0.42, (220, 220, 220)); y += 16
    y += 6
    if audio:
        n, v = audio.current
        mode = "AUTO" if audio.manual == "auto" else f"MANUAL:{audio.manual}"
        sound = f"{n.capitalize()} @{v:.2f}" if n else "silence"
        txt(img, f"Audio [{mode}]: {sound}", x0 + 10, y + 12, 0.5, (0, 255, 100), 1)


def _setting_rows(cfg):
    audio = cfg["audio"]
    return [
        ("Analysis window (s)", "window", "seconds", 1.0, 60.0, 1.0),
        ("Calibration duration (s)", "calibration", "seconds", 3.0, 120.0, 1.0),
        ("Audio switch delay (s)", "audio", "switch_delay_seconds", 0.0, 30.0, 1.0),
        ("Distraction hold (s)", "audio", "distraction_hold_seconds", 0.0, 300.0, 5.0),
        ("Manual/max volume", "audio", "manual_volume", 0.0, audio["max_volume"], 0.05),
        ("Focused sound", "audio.mapping.focused", "noise", None, None, None),
        ("Drifting sound", "audio.mapping.drifting", "noise", None, None, None),
        ("Off-task sound", "audio.mapping.off_task", "noise", None, None, None),
        ("Audio enabled", "audio", "enabled", None, None, None),
        ("Apply & close (current run)", None, None, None, None, None),
    ]


def _setting_value(cfg, row):
    _, section, key, *_ = row
    if section is None:
        return "select Enter"
    target = cfg
    for part in section.split("."):
        target = target[part]
    value = target.get(key)
    if key == "noise":
        return str(value or "off")
    if key == "enabled":
        return "ON" if value else "OFF"
    return f"{float(value):.1f}"


def _adjust_setting(cfg, row, direction):
    label, section, key, low, high, step = row
    if section is None:
        return True
    target = cfg
    for part in section.split("."):
        target = target[part]
    if key == "noise":
        current = target.get(key)
        options = SETTING_NOISES
        target[key] = options[(options.index(current or "off") + direction) % len(options)]
    elif key == "enabled":
        target[key] = not target.get(key, True)
    else:
        target[key] = float(np.clip(float(target.get(key, low)) + direction * step, low, high))
    return False


def settings_menu(frame, cfg, audio):
    """Edit runtime settings; return (audio, applied), preserving cancel semantics."""
    snapshot = copy.deepcopy(cfg)
    rows = _setting_rows(cfg)
    selected = 0
    while True:
        menu = frame.copy()
        overlay = menu.copy()
        cv2.rectangle(overlay, (25, 20), (615, 455), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.88, menu, 0.12, 0, menu)
        txt(menu, "SETTINGS  (Up/Down select, Left/Right change, Enter apply)", 40, 50, 0.55, (0, 255, 255), 1)
        txt(menu, "Esc cancels | m closes without applying | changes affect this run only", 40, 72, 0.42, (210, 210, 210))
        for i, row in enumerate(rows):
            y = 105 + i * 30
            color = (0, 255, 255) if i == selected else (220, 220, 220)
            marker = ">" if i == selected else " "
            txt(menu, f"{marker} {row[0]}: {_setting_value(cfg, row)}", 48, y, 0.48, color, 1)
        txt(menu, "Sound choices: green, pink, white, brown, off", 40, 435, 0.4, (180, 180, 180))
        cv2.imshow("Attentive-Behavior Estimator", menu)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("m")):
            cfg.clear()
            cfg.update(snapshot)
            return audio, False
        if key in (ord("w"), 13):
            if selected == len(rows) - 1 or key == ord("w"):
                return audio, True
            if selected in (5, 6, 7):
                _adjust_setting(cfg, rows[selected], 1)
            continue
        if key == 255:
            continue
        if key in (ord("k"), 82):
            selected = (selected - 1) % len(rows)
        elif key in (ord("j"), 84):
            selected = (selected + 1) % len(rows)
        elif key in (ord("h"), 81):
            _adjust_setting(cfg, rows[selected], -1)
        elif key in (ord("l"), 83):
            _adjust_setting(cfg, rows[selected], 1)


def run_live(camera=0, cfg=None, display=None):
    cfg = cfg or load_config()
    validate_paths(cfg)
    display_cfg = cfg.get("display", {})
    show_window = display_cfg.get("enabled", True) if display is None else bool(display)
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=cfg["paths"]["landmarker"]),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        min_face_detection_confidence=0.5, min_face_presence_confidence=0.5, min_tracking_confidence=0.6,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open the camera")

    extractor = FeatureExtractor()
    estimator = AttentionEstimator(cfg)
    audio = None
    if cfg["audio"]["enabled"]:
        try:
            audio = AdaptiveAudio(cfg, PygameBackend(cfg))
        except Exception as e:
            print(f"Audio disabled ({e})")

    start, last_ts, est, prev = time.time(), -1, None, time.time()

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (int(display_cfg.get("width", 640)),
                                           int(display_cfg.get("height", 480))))
                if display_cfg.get("mirror", True):
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                t = time.time() - start
                ts_ms = max(int(t * 1000), last_ts + 1)          # MediaPipe needs strictly increasing timestamps
                last_ts = ts_ms
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(mp_img, ts_ms)

                ff = extractor.extract(t, result, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (w, h))
                new_est = estimator.update(ff)
                if new_est is not None:
                    est = new_est
                if audio:
                    audio.update(est, t)

                if show_window and ff.present:
                    pts = (np.array([(p.x, p.y) for p in result.face_landmarks[0]]) * (w, h)).astype(np.int32)
                    for pt in pts[::6]:
                        cv2.circle(frame, tuple(pt), 1, (110, 110, 110), -1)
                    for i in (468, 473):
                        cv2.circle(frame, tuple(pts[i]), 2, (0, 255, 255), -1)
                elif show_window:
                    txt(frame, "No face detected", 10, 300 if estimator.calibrating else 65, 0.6, (0, 0, 255), 2)

                if show_window:
                    now = time.time()
                    draw_panel(frame, est, estimator, audio, t)
                    txt(frame, f"FPS: {1.0 / max(now - prev, 1e-6):.0f}", 10, 30, 0.7, (0, 255, 255), 2)
                    prev = now
                    txt(frame, "c recal | m settings | a auto | s silence | 1-4 sound | q quit",
                        10, h - 12, 0.4, (200, 200, 200))
                    cv2.imshow("Attentive-Behavior Estimator", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("c"):
                        estimator.reset_calibration(); extractor.reset(); est = None
                        if audio:
                            audio.reset()
                    elif key == ord("m"):
                        audio, applied = settings_menu(frame, cfg, audio)
                        if audio:
                            audio.a = cfg["audio"]
                        if applied and cfg["audio"]["enabled"] and audio is None:
                            try:
                                audio = AdaptiveAudio(cfg, PygameBackend(cfg))
                            except Exception as exc:
                                print(f"Audio disabled ({exc})")
                        elif applied and not cfg["audio"]["enabled"] and audio:
                            audio.stop()
                            audio = None
                    elif key in AUDIO_KEYS and audio:
                        audio.set_manual(AUDIO_KEYS[key])
        finally:
            cap.release()
            if show_window:
                cv2.destroyAllWindows()
            if audio:
                audio.stop()
            if show_window:
                for _ in range(4):
                    cv2.waitKey(1)


# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="On-device relative attention estimator (webcam + optional adaptive audio)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--config", help="Optional YAML configuration file")
    parser.add_argument("--landmarker", help="Face landmarker .task file; overrides config")
    parser.add_argument("--headless", action="store_true",
                        help="Run live mode without opening an OpenCV window")
    # Jupyter launches scripts with additional kernel arguments (for example
    # ``--f=...``); ignore those while still validating our own options.
    args, _ = parser.parse_known_args()
    cfg = load_config(args.config, landmarker=args.landmarker)
    run_live(camera=args.camera, cfg=cfg, display=not args.headless)


if __name__ == "__main__":
    main()