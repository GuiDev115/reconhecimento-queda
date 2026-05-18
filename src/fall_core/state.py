from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RuntimeState:
    fall_detected: bool = False
    fall_counter: int = 0
    fall_frames: int = 0
    fall_frame_threshold: int = 5
    cooldown_time: float = 3.0
    last_fall_time: float = 0.0

    risk_detected: bool = False
    risk_frames: int = 0

    x_history: deque = field(default_factory=lambda: deque(maxlen=30))
    hip_motion_history: deque = field(default_factory=lambda: deque(maxlen=30))
    angle_risk_history: deque = field(default_factory=lambda: deque(maxlen=30))

    risk_sway: float = np.nan
    risk_speed: float = np.nan
    risk_hip_jitter: float = np.nan
    risk_angle_jitter: float = np.nan

    center_y_hist: deque = field(default_factory=lambda: deque(maxlen=20))
    height_hist: deque = field(default_factory=lambda: deque(maxlen=20))
    angle_hist: deque = field(default_factory=lambda: deque(maxlen=20))

    standing_height_ref: float = None
    standing_angle_ref: float = None
    upright_frames: int = 0

    smooth_hip_center_y: float = None
    smooth_aspect_ratio: float = None
    smooth_height_ratio: float = None

    # IMU (acelerômetro + giroscópio D435i)
    camera_pitch_deg: float = 0.0
    camera_roll_deg: float = 0.0
    camera_moving: bool = False
    camera_suppress_frames: int = 0
    gyro_history: deque = field(default_factory=lambda: deque(maxlen=10))
    accel_smooth: Optional[np.ndarray] = None

    # Referência 3D (altura real em metros via intrínsecas)
    standing_height_m: Optional[float] = None
