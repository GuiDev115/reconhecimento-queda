# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install opencv-python numpy pyrealsense2
pip install mediapipe  # optional, only needed for webcam/mediapipe mode
```

## Running the System

```bash
# Auto-detect camera and detector
python src/fall_detection.py

# Intel RealSense with depth visualization
python src/fall_detection.py --camera-source realsense --detector skeleton --show-depth --show-mask

# Webcam with MediaPipe pose estimation
python src/fall_detection.py --camera-source webcam --detector mediapipe
```

Press `q` to quit. There is no build step, test suite, or linter configured.

## Architecture

The system is a **real-time rule-based fall detection pipeline** — no machine learning models are used. All detection is geometry and kinematics heuristics over depth data.

### Detection Pipeline

```
Camera input (RealSense depth+RGB or webcam RGB)
    → Person segmentation via depth mask or MediaPipe
    → Skeleton estimation: head, hip, foot positions
    → Metric extraction: aspect_ratio, height_ratio, body angle
    → EMA smoothing (alpha=0.35) to reduce sensor noise
    → Fall rule evaluation (all must be true):
         lying_posture (aspect > 1.20 OR angle > 58°)
         AND low_center (hip_y > 0.62)
         AND transition_from_upright (≥8 upright frames prior)
         AND height_dropped (current < 72% of standing_height_ref)
    → Temporal confirmation: 5+ consecutive frames
    → On confirmed fall: save snapshot + 5s clip + JSON event
```

### Module Responsibilities

| Module | Role |
|--------|------|
| `src/fall_detection.py` | Main loop: frame acquisition, display, orchestration |
| `fall_core/camera.py` | Camera abstraction — auto-detects RealSense, falls back to webcam; returns `capture_ctx` dict |
| `fall_core/vision.py` | `depth_person_metrics()` segments person from depth map; `estimate_depth_skeleton()` extracts head/hip/foot; `ema()` smoothing |
| `fall_core/processing.py` | `process_depth_mode()` and `process_mediapipe_mode()` implement each detector; `update_fall_state()` handles temporal confirmation and cooldown |
| `fall_core/state.py` | `RuntimeState` dataclass — holds all mutable state: counters, deque histories, smoothed metrics, standing reference values |
| `fall_core/events.py` | `handle_confirmed_fall()` creates `capturas_quedas/queda_*/`, saves snapshot + clip, appends to `relatorio_quedas.json` |
| `fall_core/args.py` | All CLI argument definitions with defaults |

### Key Design Decisions

- **Detector modes**: `skeleton` (depth-based, primary) uses `depth_person_metrics` + `estimate_depth_skeleton`; `mediapipe` (2D pose fallback, less reliable) uses MediaPipe landmarks. `auto` selects based on camera source.
- **Standing reference**: `standing_height_ref` and `standing_angle_ref` are set dynamically when the person is detected as upright (angle < 25°, aspect < 0.95), not from a calibration step.
- **Risk detection**: Instability warnings (early fall-risk alert) use a 4-vote system over a 30-frame window — sway, horizontal speed, hip jitter, angle jitter. Triggers if ≥2 votes exceed thresholds for ≥4 frames.
- **Frame buffer**: A `deque` in `fall_detection.py` keeps the last N seconds of frames for clip capture on event.
- **Cooldown**: `cooldown_time` (default 3s) prevents duplicate events for the same fall.

### Output

- `relatorio_quedas.json` — append-only event log with timestamp, image path, and score
- `capturas_quedas/queda_YYYYMMDD-HHMMSS-mmm_N/` — per-event directories with `snapshot.jpg` and `clip_ultimos_5s.mp4`

### Configurable Thresholds

All detection thresholds are CLI flags (see `fall_core/args.py`). Key ones to adjust per environment:

- `--depth-aspect-threshold` (1.20) — aspect ratio for lying posture
- `--depth-center-threshold` (0.62) — normalized Y for "low center"
- `--depth-height-drop-ratio` (0.72) — fraction of standing height that triggers alarm
- `--depth-upright-frames` (8) — frames standing before a fall can be detected
- `--fall-clip-seconds` (5.0) — duration of saved video clip
- `--disable-imu` — skip IMU even when using RealSense
- `--imu-gyro-threshold` (0.08) — gyroscope magnitude (rad/s) that flags camera movement
