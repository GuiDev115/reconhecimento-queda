import numpy as np

from fall_core.vision import depth_person_metrics, ema, estimate_depth_skeleton


def resolve_detector_mode(args, capture_mode):
    if args.detector == "auto":
        return "skeleton" if capture_mode == "realsense" else "mediapipe"
    return args.detector


def process_depth_mode(frame, depth_frame, depth_scale, args, state, cv2):
    center_y = np.nan
    center_x = np.nan
    aspect_ratio = np.nan
    height_ratio = np.nan
    person_depth_m = np.nan
    depth_text = "n/d"
    is_falling_this_frame = False
    depth_mask = None

    metrics, depth_mask = depth_person_metrics(depth_frame, depth_scale)
    if metrics is None:
        return {
            "center_y": center_y,
            "center_x": center_x,
            "aspect_ratio": aspect_ratio,
            "height_ratio": height_ratio,
            "person_depth_m": person_depth_m,
            "depth_text": depth_text,
            "is_falling_this_frame": is_falling_this_frame,
            "depth_mask": depth_mask,
        }

    x, y, bw, bh = metrics["bbox"]
    center_y = metrics["center_y"]
    center_x = metrics["center_x"]
    aspect_ratio = metrics["aspect_ratio"]
    height_ratio = metrics["height_ratio"]
    person_depth_m = metrics["median_depth_m"]

    if not np.isnan(person_depth_m):
        depth_text = f"{person_depth_m:.2f} m"

    skeleton = estimate_depth_skeleton(metrics, depth_mask)
    angle_deg = np.nan
    hip_center_y = center_y
    if skeleton is not None:
        angle_deg = skeleton["angle_deg"]
        hip_center_y = skeleton["hip"][1] / max(frame.shape[0], 1)

        cv2.circle(frame, skeleton["head"], 4, (0, 255, 255), -1)
        cv2.circle(frame, skeleton["hip"], 4, (0, 255, 255), -1)
        cv2.circle(frame, skeleton["foot"], 4, (0, 255, 255), -1)
        cv2.line(frame, skeleton["head"], skeleton["hip"], (0, 255, 255), 2)
        cv2.line(frame, skeleton["hip"], skeleton["foot"], (0, 255, 255), 2)

    state.smooth_hip_center_y = ema(state.smooth_hip_center_y, hip_center_y)
    state.smooth_aspect_ratio = ema(state.smooth_aspect_ratio, aspect_ratio)
    state.smooth_height_ratio = ema(state.smooth_height_ratio, height_ratio)

    state.center_y_hist.append(state.smooth_hip_center_y if state.smooth_hip_center_y is not None else hip_center_y)
    state.height_hist.append(state.smooth_height_ratio if state.smooth_height_ratio is not None else height_ratio)
    if not np.isnan(angle_deg):
        state.angle_hist.append(angle_deg)
        state.angle_risk_history.append(angle_deg)

    if not np.isnan(center_x):
        state.x_history.append(center_x)
    if state.smooth_hip_center_y is not None and not np.isnan(state.smooth_hip_center_y):
        state.hip_motion_history.append(state.smooth_hip_center_y)

    current_aspect = state.smooth_aspect_ratio if state.smooth_aspect_ratio is not None else aspect_ratio
    current_hip_y = state.smooth_hip_center_y if state.smooth_hip_center_y is not None else hip_center_y
    current_height_ratio = state.smooth_height_ratio if state.smooth_height_ratio is not None else height_ratio

    is_upright_now = (
        (np.isnan(angle_deg) or angle_deg < 25.0)
        and current_aspect < 0.95
        and current_hip_y < 0.62
        and current_height_ratio > 0.38
    )

    if is_upright_now:
        state.upright_frames = min(state.upright_frames + 1, 120)
        if state.standing_height_ref is None:
            state.standing_height_ref = current_height_ratio
        else:
            state.standing_height_ref = (0.9 * state.standing_height_ref) + (0.1 * current_height_ratio)
        if not np.isnan(angle_deg):
            if state.standing_angle_ref is None:
                state.standing_angle_ref = angle_deg
            else:
                state.standing_angle_ref = (0.9 * state.standing_angle_ref) + (0.1 * angle_deg)
    else:
        state.upright_frames = max(state.upright_frames - 1, 0)

    prev_center_y = state.center_y_hist[-2] if len(state.center_y_hist) >= 2 else current_hip_y
    vertical_speed = current_hip_y - prev_center_y

    lying_posture = (current_aspect > args.depth_aspect_threshold) or (not np.isnan(angle_deg) and angle_deg > 58.0)
    low_center = current_hip_y > args.depth_center_threshold
    sudden_drop = vertical_speed > args.depth_vertical_speed_threshold
    transition_from_upright = state.upright_frames >= args.depth_upright_frames

    if state.standing_height_ref is not None and state.standing_height_ref > 1e-6:
        height_ratio_vs_standing = current_height_ratio / state.standing_height_ref
    else:
        height_ratio_vs_standing = np.nan

    height_dropped = not np.isnan(height_ratio_vs_standing) and (height_ratio_vs_standing < args.depth_height_drop_ratio)

    is_falling_this_frame = lying_posture and low_center and transition_from_upright and (height_dropped or sudden_drop)

    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 0), 2)
    cv2.putText(
        frame,
        f"AR:{aspect_ratio:.2f} HIPY:{hip_center_y:.2f} ANG:{angle_deg:.1f}",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )

    if len(state.x_history) >= args.risk_min_frames:
        x_array = np.array(state.x_history, dtype=np.float32)
        x_diff = np.diff(x_array)
        state.risk_sway = float(np.max(x_array) - np.min(x_array))
        state.risk_speed = float(np.mean(np.abs(x_diff))) if x_diff.size > 0 else 0.0
    else:
        state.risk_sway = np.nan
        state.risk_speed = np.nan

    if len(state.hip_motion_history) >= args.risk_min_frames:
        state.risk_hip_jitter = float(np.std(np.array(state.hip_motion_history, dtype=np.float32)))
    else:
        state.risk_hip_jitter = np.nan

    if len(state.angle_risk_history) >= args.risk_min_frames:
        state.risk_angle_jitter = float(np.std(np.array(state.angle_risk_history, dtype=np.float32)))
    else:
        state.risk_angle_jitter = np.nan

    instability_votes = 0
    instability_votes += int(not np.isnan(state.risk_sway) and state.risk_sway > args.risk_sway_threshold)
    instability_votes += int(not np.isnan(state.risk_speed) and state.risk_speed > args.risk_horizontal_speed_threshold)
    instability_votes += int(not np.isnan(state.risk_hip_jitter) and state.risk_hip_jitter > args.risk_hip_jitter_threshold)
    instability_votes += int(not np.isnan(state.risk_angle_jitter) and state.risk_angle_jitter > args.risk_angle_jitter_threshold)

    instability_now = instability_votes >= 2 and not is_falling_this_frame
    if instability_now:
        state.risk_frames = min(state.risk_frames + 1, 120)
    else:
        state.risk_frames = max(state.risk_frames - 1, 0)
    state.risk_detected = state.risk_frames >= args.risk_confirm_frames

    return {
        "center_y": center_y,
        "center_x": center_x,
        "aspect_ratio": aspect_ratio,
        "height_ratio": height_ratio,
        "person_depth_m": person_depth_m,
        "depth_text": depth_text,
        "is_falling_this_frame": is_falling_this_frame,
        "depth_mask": depth_mask,
    }


def process_mediapipe_mode(frame, pose, mp_pose, mp_drawing, args, state, cv2):
    center_y = np.nan
    center_x = np.nan
    aspect_ratio = np.nan
    height_ratio = np.nan
    person_depth_m = np.nan
    depth_text = "n/d"
    depth_mask = None
    is_falling_this_frame = False

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return {
            "center_y": center_y,
            "center_x": center_x,
            "aspect_ratio": aspect_ratio,
            "height_ratio": height_ratio,
            "person_depth_m": person_depth_m,
            "depth_text": depth_text,
            "is_falling_this_frame": is_falling_this_frame,
            "depth_mask": depth_mask,
        }

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    landmarks = results.pose_landmarks.landmark

    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
    shoulder_avg_x = (left_shoulder.x + right_shoulder.x) / 2
    hip_avg_y = (left_hip.y + right_hip.y) / 2
    vertical_distance = hip_avg_y - shoulder_avg_y

    center_y = shoulder_avg_y
    center_x = shoulder_avg_x
    height_ratio = vertical_distance

    is_falling_this_frame = shoulder_avg_y > hip_avg_y or (vertical_distance < 0.15 and hip_avg_y > 0.5)

    state.x_history.append(center_x)
    if len(state.x_history) >= args.risk_min_frames and not is_falling_this_frame:
        x_array = np.array(state.x_history, dtype=np.float32)
        state.risk_sway = float(np.max(x_array) - np.min(x_array))
        state.risk_detected = state.risk_sway > args.risk_sway_threshold
    else:
        state.risk_sway = np.nan
        state.risk_detected = False

    return {
        "center_y": center_y,
        "center_x": center_x,
        "aspect_ratio": aspect_ratio,
        "height_ratio": height_ratio,
        "person_depth_m": person_depth_m,
        "depth_text": depth_text,
        "is_falling_this_frame": is_falling_this_frame,
        "depth_mask": depth_mask,
    }


def update_fall_state(state, is_falling_this_frame, current_time):
    if is_falling_this_frame:
        state.fall_frames += 1
    else:
        state.fall_frames = max(0, state.fall_frames - 1)

    new_fall_confirmed = False
    if state.fall_frames >= state.fall_frame_threshold:
        if (current_time - state.last_fall_time) > state.cooldown_time:
            state.fall_counter += 1
            state.last_fall_time = current_time
            new_fall_confirmed = True

        state.fall_detected = True
        state.risk_detected = False
        state.risk_frames = 0
    else:
        state.fall_detected = False

    return new_fall_confirmed


def build_hud_lines(capture_mode, detector_mode, depth_text, center_y_hist, angle_hist, aspect_ratio, height_ratio, state, show_fall_alert=False):
    if show_fall_alert or state.fall_detected:
        status_text = "ALERTA: QUEDA DETECTADA!"
        status_color = (0, 0, 255)
        status_scale = 1.0
        status_thickness = 3
    elif state.risk_detected:
        status_text = "RISCO: INSTABILIDADE CORPORAL!"
        status_color = (0, 165, 255)
        status_scale = 0.9
        status_thickness = 2
    else:
        status_text = "Status: Normal"
        status_color = (0, 255, 0)
        status_scale = 0.9
        status_thickness = 2

    return [
        (status_text, status_color, status_scale, status_thickness),
        (f"Quedas registradas: {state.fall_counter}", (255, 255, 0), 0.9, 2),
        (f"Fonte: {capture_mode}", (255, 255, 255), 0.8, 2),
        (f"Detector: {detector_mode}", (255, 255, 255), 0.8, 2),
        (f"Profundidade torso: {depth_text}", (255, 255, 255), 0.8, 2),
        (
            f"Skeleton dbg AR:{aspect_ratio:.2f} H:{height_ratio:.2f}",
            (200, 200, 200),
            0.62,
            1,
        ),
        (
            f"HipY:{(center_y_hist[-1] if len(center_y_hist) else np.nan):.2f} Ang:{(angle_hist[-1] if len(angle_hist) else np.nan):.1f}",
            (200, 200, 200),
            0.62,
            1,
        ),
        (
            f"Risco dbg sway:{state.risk_sway:.3f} spd:{state.risk_speed:.3f} hy:{state.risk_hip_jitter:.3f} ay:{state.risk_angle_jitter:.2f}",
            (200, 200, 200),
            0.58,
            1,
        ),
    ]
