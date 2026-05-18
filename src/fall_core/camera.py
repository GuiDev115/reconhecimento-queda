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


def _try_start_pipeline(pipeline, config):
    """Tenta iniciar o pipeline; retorna profile ou lança RuntimeError."""
    return pipeline.start(config)


def start_realsense_capture(width, height, fps):
    if rs is None:
        raise RuntimeError("Pacote 'pyrealsense2' nao instalado. Instale para usar a RealSense.")

    def _build_video_config(cfg):
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # Tenta primeiro com IMU (D435i); se falhar, usa apenas vídeo+depth
    has_imu = True
    pipeline = rs.pipeline()
    config = rs.config()
    _build_video_config(config)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    config.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f, 400)
    try:
        profile = _try_start_pipeline(pipeline, config)
    except RuntimeError:
        has_imu = False
        print("IMU nao disponivel neste dispositivo. Continuando sem acelerometro/giroscopio.")
        pipeline = rs.pipeline()
        config = rs.config()
        _build_video_config(config)
        profile = _try_start_pipeline(pipeline, config)

    align = rs.align(rs.stream.color)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    return {
        "mode": "realsense",
        "pipeline": pipeline,
        "align": align,
        "depth_scale": depth_scale,
        "profile": profile,
        "has_imu": has_imu,
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
        return ret, frame, None, None

    frames = capture_ctx["pipeline"].wait_for_frames()

    # Leitura IMU do frameset original (apenas se o dispositivo tem IMU habilitado)
    imu_data = None
    if capture_ctx.get("has_imu", False):
        accel_f = frames.first_or_default(rs.stream.accel, rs.format.motion_xyz32f)
        gyro_f  = frames.first_or_default(rs.stream.gyro,  rs.format.motion_xyz32f)
    else:
        accel_f = gyro_f = None
    if accel_f and gyro_f:
        a = accel_f.as_motion_frame().get_motion_data()
        g = gyro_f.as_motion_frame().get_motion_data()
        imu_data = {
            "accel": np.array([a.x, a.y, a.z], dtype=np.float32),
            "gyro":  np.array([g.x, g.y, g.z], dtype=np.float32),
        }

    aligned = capture_ctx["align"].process(frames)
    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    if not color_frame:
        return False, None, None, None
    frame = np.asanyarray(color_frame.get_data())
    return True, frame, depth_frame, imu_data


def get_depth_intrinsics(capture_ctx):
    """Retorna intrínsecas do stream de profundidade para deprojection 3D."""
    if capture_ctx["mode"] != "realsense" or "profile" not in capture_ctx:
        return None
    intr = (
        capture_ctx["profile"]
        .get_stream(rs.stream.depth)
        .as_video_stream_profile()
        .get_intrinsics()
    )
    return {"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy}


def release_capture(capture_ctx):
    if capture_ctx["mode"] == "webcam":
        capture_ctx["cap"].release()
    else:
        capture_ctx["pipeline"].stop()
