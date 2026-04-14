import os
import time

import cv2
import numpy as np

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

from fall_core.args import parse_args
from fall_core.camera import read_frame, release_capture, start_capture
from fall_core.events import handle_confirmed_fall, initialize_csv
from fall_core.processing import (
    build_hud_lines,
    process_depth_mode,
    process_mediapipe_mode,
    resolve_detector_mode,
    update_fall_state,
)
from fall_core.state import RuntimeState
from fall_core.vision import draw_hud_text


def main():
    args = parse_args()
    capture_ctx = start_capture(args, cv2)
    detector_mode = resolve_detector_mode(args, capture_ctx["mode"])

    if detector_mode in {"depth", "skeleton"} and capture_ctx["mode"] != "realsense":
        print("Detector depth requer camera RealSense (stream de profundidade).")
        release_capture(capture_ctx)
        return

    pose = None
    mp_pose = None
    if detector_mode == "mediapipe":
        if mp_pose_module is None or mp_drawing is None:
            print("MediaPipe indisponivel neste ambiente. Use detector depth com RealSense:")
            print("  python src/fall_detection.py --camera-source realsense --detector depth --show-depth")
            if MEDIAPIPE_IMPORT_ERROR is not None:
                print(f"Detalhe import: {MEDIAPIPE_IMPORT_ERROR}")
            release_capture(capture_ctx)
            return
        mp_pose = mp_pose_module
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    state = RuntimeState()

    snapshot_dir = "capturas_quedas"
    os.makedirs(snapshot_dir, exist_ok=True)

    notifier_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notificador-quedas", "send_alert.py"))
    csv_filename = "relatorio_quedas.csv"
    initialize_csv(csv_filename)

    print("Iniciando monitoramento de quedas. Pressione 'q' para sair.")
    print(f"Fonte de video ativa: {capture_ctx['mode']}")
    print(f"Detector ativo: {detector_mode}")

    try:
        while True:
            ret, frame, depth_frame = read_frame(capture_ctx)
            if not ret:
                print("Nao foi possivel acessar a camera.")
                break

            current_time = time.time()

            result = {
                "center_y": np.nan,
                "center_x": np.nan,
                "aspect_ratio": np.nan,
                "height_ratio": np.nan,
                "person_depth_m": np.nan,
                "depth_text": "n/d",
                "is_falling_this_frame": False,
                "depth_mask": None,
            }

            if detector_mode in {"depth", "skeleton"} and depth_frame is not None:
                result = process_depth_mode(frame, depth_frame, capture_ctx["depth_scale"], args, state, cv2)
            elif detector_mode == "mediapipe":
                result = process_mediapipe_mode(frame, pose, mp_pose, mp_drawing, args, state, cv2)

            new_fall_confirmed = update_fall_state(state, result["is_falling_this_frame"], current_time)

            if new_fall_confirmed:
                handle_confirmed_fall(
                    frame=frame,
                    state=state,
                    detector_mode=detector_mode,
                    csv_filename=csv_filename,
                    snapshot_dir=snapshot_dir,
                    notifier_script=notifier_script,
                    disable_email_alert=args.disable_email_alert,
                    center_y=result["center_y"],
                    aspect_ratio=result["aspect_ratio"],
                    height_ratio=result["height_ratio"],
                    person_depth_m=result["person_depth_m"],
                )

            show_fall_alert = state.fall_detected or ((current_time - state.last_fall_time) < 2.0)
            hud_lines = build_hud_lines(
                capture_mode=capture_ctx["mode"],
                detector_mode=detector_mode,
                depth_text=result["depth_text"],
                center_y_hist=state.center_y_hist,
                angle_hist=state.angle_hist,
                aspect_ratio=result["aspect_ratio"],
                height_ratio=result["height_ratio"],
                state=state,
                show_fall_alert=show_fall_alert,
            )
            draw_hud_text(frame, hud_lines, x=40, y=40, line_height=32)

            cv2.imshow("Monitoramento de Queda", frame)

            if args.show_depth and depth_frame is not None:
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imshow("Profundidade RealSense", depth_colormap)

            if args.show_mask and result["depth_mask"] is not None:
                cv2.imshow("Mascara Depth", result["depth_mask"])

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    finally:
        release_capture(capture_ctx)
        if pose is not None:
            pose.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
