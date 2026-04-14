import cv2
import numpy as np
import time
import csv
import os
import subprocess
import argparse
from datetime import datetime
from collections import deque

MEDIAPIPE_IMPORT_ERROR = None
mp_pose_module = None
mp_drawing = None

try:
    import mediapipe as mp
    if hasattr(mp, "solutions"):
        mp_pose_module = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
except Exception as exc:
    MEDIAPIPE_IMPORT_ERROR = exc

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def parse_args():
    parser = argparse.ArgumentParser(description="Monitoramento de quedas com RealSense e/ou MediaPipe")
    parser.add_argument(
        "--camera-source",
        choices=["auto", "webcam", "realsense"],
        default="auto",
        help="Fonte de captura de vídeo.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Índice da webcam no OpenCV.")
    parser.add_argument("--rs-width", type=int, default=640, help="Largura do stream da RealSense.")
    parser.add_argument("--rs-height", type=int, default=480, help="Altura do stream da RealSense.")
    parser.add_argument("--rs-fps", type=int, default=30, help="FPS do stream da RealSense.")
    parser.add_argument(
        "--detector",
        choices=["auto", "depth", "skeleton", "mediapipe"],
        default="auto",
        help="Tipo de detector de queda: depth/skeleton usa sensores da RealSense; mediapipe usa pose 2D.",
    )
    parser.add_argument("--show-depth", action="store_true", help="Exibe stream de profundidade da RealSense.")
    parser.add_argument("--show-mask", action="store_true", help="Exibe máscara de segmentação depth (modo depth).")
    parser.add_argument("--depth-aspect-threshold", type=float, default=1.20, help="Limiar de aspecto (largura/altura) para postura deitada.")
    parser.add_argument("--depth-center-threshold", type=float, default=0.62, help="Limiar do centro Y para considerar corpo próximo ao chão.")
    parser.add_argument("--depth-height-drop-ratio", type=float, default=0.72, help="Razão da altura atual vs altura em pé para indicar queda.")
    parser.add_argument("--depth-vertical-speed-threshold", type=float, default=0.035, help="Variação mínima por frame do centro Y para queda brusca.")
    parser.add_argument("--depth-upright-frames", type=int, default=8, help="Frames mínimos em postura ereta antes de confirmar transição de queda.")
    parser.add_argument("--disable-email-alert", action="store_true", help="Desativa envio de alerta por email.")
    return parser.parse_args()


def start_webcam_capture(camera_index):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a webcam no índice {camera_index}.")
    return {"mode": "webcam", "cap": cap}


def start_realsense_capture(width, height, fps):
    if rs is None:
        raise RuntimeError("Pacote 'pyrealsense2' não instalado. Instale para usar a RealSense D415.")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    return {
        "mode": "realsense",
        "pipeline": pipeline,
        "align": align,
        "depth_scale": depth_scale,
    }


def start_capture(args):
    if args.camera_source == "webcam":
        return start_webcam_capture(args.camera_index)
    if args.camera_source == "realsense":
        return start_realsense_capture(args.rs_width, args.rs_height, args.rs_fps)

    # auto: tenta RealSense primeiro; fallback para webcam
    try:
        return start_realsense_capture(args.rs_width, args.rs_height, args.rs_fps)
    except Exception as exc:
        print(f"RealSense indisponível ({exc}). Usando webcam índice {args.camera_index}.")
        return start_webcam_capture(args.camera_index)


def read_frame(capture_ctx):
    if capture_ctx["mode"] == "webcam":
        ret, frame = capture_ctx["cap"].read()
        return ret, frame, None

    frames = capture_ctx["pipeline"].wait_for_frames()
    aligned = capture_ctx["align"].process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame:
        return False, None, None
    frame = np.asanyarray(color_frame.get_data())
    return True, frame, depth_frame


def release_capture(capture_ctx):
    if capture_ctx["mode"] == "webcam":
        capture_ctx["cap"].release()
    else:
        capture_ctx["pipeline"].stop()


def draw_hud_text(frame, text_lines, x=40, y=32, line_height=32):
    # Desenha um painel translúcido para manter legibilidade em fundos claros/escuros.
    panel_w = 540
    panel_h = 12 + (line_height * len(text_lines))
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 14, y - 24), (x - 14 + panel_w, y - 24 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    for idx, (text, color, scale, thickness) in enumerate(text_lines):
        yy = y + (idx * line_height)
        cv2.putText(frame, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def resolve_detector_mode(args, capture_mode):
    if args.detector == "auto":
        return "skeleton" if capture_mode == "realsense" else "mediapipe"
    return args.detector


def depth_person_metrics(depth_frame, depth_scale):
    depth_raw = np.asanyarray(depth_frame.get_data())
    depth_m = depth_raw.astype(np.float32) * depth_scale

    valid_depth = (depth_m > 0.35) & (depth_m < 4.0)
    if np.count_nonzero(valid_depth) < 400:
        return None, np.zeros_like(depth_raw, dtype=np.uint8)

    # Recorta uma camada próxima da câmera para destacar o primeiro plano (pessoa mais próxima).
    near_depth = float(np.percentile(depth_m[valid_depth], 15))
    far_depth = min(near_depth + 1.0, 4.0)
    mask = ((depth_m >= max(0.35, near_depth - 0.05)) & (depth_m <= far_depth)).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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
        # Prioriza alvos maiores e mais próximos.
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


def estimate_depth_skeleton(metrics, mask):
    x, y, bw, bh = metrics["bbox"]
    roi = mask[y:y + bh, x:x + bw]
    points = np.column_stack(np.where(roi > 0))
    if points.size == 0:
        return None

    # points: (row=y, col=x) no ROI
    rows = points[:, 0]
    cols = points[:, 1]

    head_idx = np.argmin(rows)
    foot_idx = np.argmax(rows)

    head = (int(x + cols[head_idx]), int(y + rows[head_idx]))
    foot = (int(x + cols[foot_idx]), int(y + rows[foot_idx]))

    # Quadril aproximado: faixa central do corpo (percentil vertical ~55%)
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

    return {
        "head": head,
        "hip": hip,
        "foot": foot,
        "angle_deg": float(angle_from_vertical_deg),
    }


def main():
    args = parse_args()
    capture_ctx = start_capture(args)
    detector_mode = resolve_detector_mode(args, capture_ctx["mode"])

    if detector_mode in {"depth", "skeleton"} and capture_ctx["mode"] != "realsense":
        print("Detector depth requer câmera RealSense (stream de profundidade).")
        release_capture(capture_ctx)
        return

    pose = None
    mp_pose = None
    if detector_mode == "mediapipe":
        if mp_pose_module is None or mp_drawing is None:
            print("MediaPipe indisponível neste ambiente. Use detector depth com RealSense:")
            print("  python fall_detection.py --camera-source realsense --detector depth --show-depth")
            if MEDIAPIPE_IMPORT_ERROR is not None:
                print(f"Detalhe import: {MEDIAPIPE_IMPORT_ERROR}")
            release_capture(capture_ctx)
            return
        mp_pose = mp_pose_module
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Variáveis para auxiliar na detecção temporal
    fall_detected = False
    fall_counter = 0
    fall_frames = 0
    FALL_FRAME_THRESHOLD = 5
    cooldown_time = 3.0
    last_fall_time = 0.0

    # Variáveis para Detecção de Risco/Instabilidade
    risk_detected = False
    x_history = []
    SWAY_THRESHOLD = 0.10
    SWAY_FRAMES = 15

    # Memória temporal para detectar transição de queda no modo depth.
    center_y_hist = deque(maxlen=20)
    height_hist = deque(maxlen=20)
    angle_hist = deque(maxlen=20)
    standing_height_ref = None
    standing_angle_ref = None
    upright_frames = 0

    # Diretório para salvar evidências de quedas
    FALL_SNAPSHOT_DIR = "capturas_quedas"
    os.makedirs(FALL_SNAPSHOT_DIR, exist_ok=True)

    # Script externo de notificação por email
    NOTIFIER_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notificador-quedas", "send_alert.py"))

    # Configuração do arquivo de tabulação para a Iniciação Científica (CSV)
    csv_filename = "relatorio_quedas.csv"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Timestamp",
            "ID_Queda",
            "Detector",
            "Centro_Y",
            "Aspecto_Largura_Altura",
            "Altura_Relativa",
            "Margem_Frames",
            "Profundidade_m",
        ])

    print("Iniciando monitoramento de quedas. Pressione 'q' para sair.")
    print(f"Fonte de vídeo ativa: {capture_ctx['mode']}")
    print(f"Detector ativo: {detector_mode}")

    try:
        while True:
            ret, frame, depth_frame = read_frame(capture_ctx)
            if not ret:
                print("Não foi possível acessar a câmera.")
                break

            current_time = time.time()
            depth_text = "n/d"
            center_y = np.nan
            aspect_ratio = np.nan
            height_ratio = np.nan
            person_depth_m = np.nan
            is_falling_this_frame = False
            depth_mask = None

            if detector_mode in {"depth", "skeleton"} and depth_frame is not None:
                metrics, depth_mask = depth_person_metrics(depth_frame, capture_ctx["depth_scale"])
                if metrics is not None:
                    x, y, bw, bh = metrics["bbox"]
                    center_y = metrics["center_y"]
                    aspect_ratio = metrics["aspect_ratio"]
                    height_ratio = metrics["height_ratio"]
                    person_depth_m = metrics["median_depth_m"]

                    if not np.isnan(person_depth_m):
                        depth_text = f"{person_depth_m:.2f} m"

                    x_history.append(metrics["center_x"])
                    if len(x_history) > SWAY_FRAMES:
                        x_history.pop(0)

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

                    center_y_hist.append(hip_center_y)
                    height_hist.append(height_ratio)
                    if not np.isnan(angle_deg):
                        angle_hist.append(angle_deg)

                    is_upright_now = (
                        (np.isnan(angle_deg) or angle_deg < 25.0)
                        and aspect_ratio < 0.95
                        and hip_center_y < 0.62
                        and height_ratio > 0.38
                    )
                    if is_upright_now:
                        upright_frames = min(upright_frames + 1, 120)
                        if standing_height_ref is None:
                            standing_height_ref = height_ratio
                        else:
                            standing_height_ref = (0.9 * standing_height_ref) + (0.1 * height_ratio)
                        if not np.isnan(angle_deg):
                            if standing_angle_ref is None:
                                standing_angle_ref = angle_deg
                            else:
                                standing_angle_ref = (0.9 * standing_angle_ref) + (0.1 * angle_deg)
                    else:
                        upright_frames = max(upright_frames - 1, 0)

                    prev_center_y = center_y_hist[-2] if len(center_y_hist) >= 2 else hip_center_y
                    vertical_speed = hip_center_y - prev_center_y

                    lying_posture = (
                        aspect_ratio > args.depth_aspect_threshold
                        or (not np.isnan(angle_deg) and angle_deg > 58.0)
                    )
                    low_center = hip_center_y > args.depth_center_threshold
                    sudden_drop = vertical_speed > args.depth_vertical_speed_threshold
                    transition_from_upright = upright_frames >= args.depth_upright_frames

                    if standing_height_ref is not None and standing_height_ref > 1e-6:
                        height_ratio_vs_standing = height_ratio / standing_height_ref
                    else:
                        height_ratio_vs_standing = np.nan
                    height_dropped = (
                        not np.isnan(height_ratio_vs_standing)
                        and height_ratio_vs_standing < args.depth_height_drop_ratio
                    )

                    # Exige postura deitada + proximidade ao chão + transição (altura caiu ou queda brusca)
                    is_falling_this_frame = (
                        lying_posture
                        and low_center
                        and transition_from_upright
                        and (height_dropped or sudden_drop)
                    )

                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 255, 0), 2)
                    cv2.putText(frame, f"AR:{aspect_ratio:.2f} HIPY:{hip_center_y:.2f} ANG:{angle_deg:.1f}", (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

                    if len(x_history) == SWAY_FRAMES:
                        if (max(x_history) - min(x_history)) > SWAY_THRESHOLD and not is_falling_this_frame:
                            risk_detected = True
                        else:
                            risk_detected = False

            elif detector_mode == "mediapipe":
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    landmarks = results.pose_landmarks.landmark

                    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

                    shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
                    hip_avg_y = (left_hip.y + right_hip.y) / 2
                    vertical_distance = hip_avg_y - shoulder_avg_y
                    center_y = shoulder_avg_y
                    height_ratio = vertical_distance
                    is_falling_this_frame = shoulder_avg_y > hip_avg_y or (vertical_distance < 0.15 and hip_avg_y > 0.5)

            if is_falling_this_frame:
                fall_frames += 1
            else:
                fall_frames = max(0, fall_frames - 1)

            if fall_frames >= FALL_FRAME_THRESHOLD:
                if (current_time - last_fall_time) > cooldown_time:
                    fall_counter += 1
                    last_fall_time = current_time

                    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    file_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                    snapshot_path = os.path.join(FALL_SNAPSHOT_DIR, f"queda_{file_timestamp}_{fall_counter}.jpg")

                    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            timestamp_str,
                            fall_counter,
                            detector_mode,
                            f"{center_y:.4f}" if not np.isnan(center_y) else "",
                            f"{aspect_ratio:.4f}" if not np.isnan(aspect_ratio) else "",
                            f"{height_ratio:.4f}" if not np.isnan(height_ratio) else "",
                            fall_frames,
                            f"{person_depth_m:.3f}" if not np.isnan(person_depth_m) else "",
                        ])

                    cv2.imwrite(snapshot_path, frame)

                    if not args.disable_email_alert and os.path.exists(NOTIFIER_SCRIPT):
                        try:
                            subprocess.run([
                                "python",
                                NOTIFIER_SCRIPT,
                                "--image", snapshot_path,
                                "--subject", "Queda detectada",
                                "--body", "Foi detectada uma queda. Verifique o idoso."
                            ], check=True)
                        except Exception as exc:
                            print(f"Falha ao enviar alerta: {exc}")

                fall_detected = True
            else:
                fall_detected = False

            if fall_detected or (current_time - last_fall_time) < 2.0:
                status_text = "ALERTA: QUEDA DETECTADA!"
                status_color = (0, 0, 255)
                status_scale = 1.0
                status_thickness = 3
            elif risk_detected:
                status_text = "RISCO: INSTABILIDADE CORPORAL!"
                status_color = (0, 165, 255)
                status_scale = 0.9
                status_thickness = 2
            else:
                status_text = "Status: Normal"
                status_color = (0, 255, 0)
                status_scale = 0.9
                status_thickness = 2

            draw_hud_text(
                frame,
                [
                    (status_text, status_color, status_scale, status_thickness),
                    (f"Quedas registradas: {fall_counter}", (255, 255, 0), 0.9, 2),
                    (f"Fonte: {capture_ctx['mode']}", (255, 255, 255), 0.8, 2),
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
                ],
                x=40,
                y=40,
                line_height=32,
            )

            cv2.imshow('Monitoramento de Queda', frame)

            if args.show_depth and depth_frame is not None:
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow('Profundidade RealSense', depth_colormap)

            if args.show_mask and depth_mask is not None:
                cv2.imshow('Mascara Depth', depth_mask)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    finally:
        release_capture(capture_ctx)
        if pose is not None:
            pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
