import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def start_webcam_capture(camera_index, cv2):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a webcam no indice {camera_index}.")
    return {"mode": "webcam", "cap": cap}


def start_realsense_capture(width, height, fps):
    if rs is None:
        raise RuntimeError("Pacote 'pyrealsense2' nao instalado. Instale para usar a RealSense.")

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


def start_capture(args, cv2):
    if args.camera_source == "webcam":
        return start_webcam_capture(args.camera_index, cv2)
    if args.camera_source == "realsense":
        return start_realsense_capture(args.rs_width, args.rs_height, args.rs_fps)

    try:
        return start_realsense_capture(args.rs_width, args.rs_height, args.rs_fps)
    except Exception as exc:
        print(f"RealSense indisponivel ({exc}). Usando webcam indice {args.camera_index}.")
        return start_webcam_capture(args.camera_index, cv2)


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
