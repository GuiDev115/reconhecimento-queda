import cv2
import numpy as np

DEPTH_MORPH_KERNEL = np.ones((5, 5), np.uint8)


def draw_hud_text(frame, text_lines, x=40, y=32, line_height=32):
    panel_w = 540
    panel_h = 12 + (line_height * len(text_lines))
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 14, y - 24), (x - 14 + panel_w, y - 24 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    for idx, (text, color, scale, thickness) in enumerate(text_lines):
        yy = y + (idx * line_height)
        cv2.putText(frame, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def ema(previous_value, current_value, alpha=0.35):
    if np.isnan(current_value):
        return previous_value
    if previous_value is None or np.isnan(previous_value):
        return float(current_value)
    return float((alpha * current_value) + ((1.0 - alpha) * previous_value))


def depth_person_metrics(depth_frame, depth_scale):
    depth_raw = np.asanyarray(depth_frame.get_data())
    depth_m = depth_raw.astype(np.float32) * depth_scale

    valid_depth = (depth_m > 0.35) & (depth_m < 4.0)
    if np.count_nonzero(valid_depth) < 400:
        return None, np.zeros_like(depth_raw, dtype=np.uint8)

    near_depth = float(np.percentile(depth_m[valid_depth], 15))
    far_depth = min(near_depth + 1.0, 4.0)
    mask = ((depth_m >= max(0.35, near_depth - 0.05)) & (depth_m <= far_depth)).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, DEPTH_MORPH_KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, DEPTH_MORPH_KERNEL)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask

    best_contour = None
    best_score = -1.0
    h, w = depth_raw.shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h * w * 0.01):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        roi = depth_m[y:y + bh, x:x + bw]
        roi_valid = roi[(roi > 0.35) & (roi < 4.0)]
        if roi_valid.size == 0:
            continue
        median_depth = float(np.median(roi_valid))
        score = area / (median_depth + 0.25)
        if score > best_score:
            best_score = score
            best_contour = cnt

    if best_contour is None:
        return None, mask

    area = cv2.contourArea(best_contour)
    x, y, bw, bh = cv2.boundingRect(best_contour)
    cx = x + bw // 2
    cy = y + bh // 2

    roi = depth_m[y:y + bh, x:x + bw]
    roi_valid = roi[(roi > 0.35) & (roi < 4.0)]
    median_depth = float(np.median(roi_valid)) if roi_valid.size > 0 else float("nan")

    metrics = {
        "bbox": (x, y, bw, bh),
        "contour": best_contour,
        "center_x": cx / max(w, 1),
        "center_y": cy / max(h, 1),
        "height_ratio": bh / max(h, 1),
        "width_ratio": bw / max(w, 1),
        "aspect_ratio": bw / max(bh, 1),
        "median_depth_m": median_depth,
        "area": area,
        "near_depth_m": near_depth,
    }
    return metrics, mask


def compute_camera_orientation(accel_smooth):
    """Calcula pitch e roll da câmera a partir do vetor de gravidade medido pelo acelerômetro.

    Convenção de eixos RealSense: Z aponta para a cena, Y aponta para baixo, X aponta para direita.
    Quando a câmera está nivelada e horizontal: accel ≈ [0, +g, 0].
    """
    ax, ay, az = float(accel_smooth[0]), float(accel_smooth[1]), float(accel_smooth[2])
    pitch_rad = np.arctan2(-az, ay)   # inclina para frente/trás (em torno de X)
    roll_rad  = np.arctan2( ax, ay)   # tomba para esquerda/direita (em torno de Z)
    return float(np.degrees(pitch_rad)), float(np.degrees(roll_rad))


def deproject_pixel(px, py, depth_m, intrinsics):
    """Converte um pixel + profundidade em ponto 3D (metros) usando modelo pinhole."""
    x = (px - intrinsics["ppx"]) * depth_m / intrinsics["fx"]
    y = (py - intrinsics["ppy"]) * depth_m / intrinsics["fy"]
    return np.array([x, y, depth_m], dtype=np.float32)


def estimate_depth_skeleton(metrics, mask, depth_frame=None, intrinsics=None, camera_pitch_deg=0.0):
    x, y, bw, bh = metrics["bbox"]
    roi = mask[y:y + bh, x:x + bw]
    points = np.column_stack(np.where(roi > 0))
    if points.size == 0:
        return None

    rows = points[:, 0]
    cols = points[:, 1]

    head_idx = np.argmin(rows)
    foot_idx = np.argmax(rows)

    head = (int(x + cols[head_idx]), int(y + rows[head_idx]))
    foot = (int(x + cols[foot_idx]), int(y + rows[foot_idx]))

    hip_target = np.percentile(rows, 55)
    hip_band = np.abs(rows - hip_target) <= 2
    if np.any(hip_band):
        hip_x = int(x + np.median(cols[hip_band]))
        hip_y = int(y + np.median(rows[hip_band]))
    else:
        hip_x = int(x + np.median(cols))
        hip_y = int(y + np.median(rows))
    hip = (hip_x, hip_y)

    dx = float(foot[0] - head[0])
    dy = float(foot[1] - head[1])
    angle_from_vertical_deg = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))

    # Ângulo corrigido pelo pitch da câmera (compensa inclinação da montagem)
    angle_corrected_deg = float(angle_from_vertical_deg) - camera_pitch_deg

    result = {
        "head": head,
        "hip": hip,
        "foot": foot,
        "angle_deg": float(angle_from_vertical_deg),
        "angle_corrected_deg": float(angle_corrected_deg),
        "height_m": None,
        "head_3d": None,
        "foot_3d": None,
    }

    # Deprojection 3D quando intrínsecas disponíveis
    if intrinsics is not None and depth_frame is not None:
        try:
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_scale = metrics.get("depth_scale", None)

            def _sample_depth(px, py):
                h, w = depth_raw.shape
                r = max(0, min(int(py), h - 1))
                c = max(0, min(int(px), w - 1))
                roi_s = depth_raw[max(0, r-2):r+3, max(0, c-2):c+3].astype(np.float32)
                valid = roi_s[roi_s > 0]
                return float(np.median(valid)) if valid.size > 0 else 0.0

            # depth_raw está em unidades do sensor; precisa do depth_scale (passado via metrics)
            ds = metrics.get("depth_scale", 0.001)
            head_d = _sample_depth(head[0], head[1]) * ds
            foot_d = _sample_depth(foot[0], foot[1]) * ds

            if head_d > 0.1 and foot_d > 0.1:
                head_3d = deproject_pixel(head[0], head[1], head_d, intrinsics)
                foot_3d = deproject_pixel(foot[0], foot[1], foot_d, intrinsics)
                height_m = float(np.linalg.norm(head_3d - foot_3d))
                result["height_m"] = height_m
                result["head_3d"] = head_3d
                result["foot_3d"] = foot_3d
        except Exception:
            pass  # deprojection falhou silenciosamente; usa métricas 2D normalmente

    return result
