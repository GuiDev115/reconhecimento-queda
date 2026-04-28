import json
import os
from datetime import datetime

import cv2
import numpy as np



def initialize_json(json_filename):
    with open(json_filename, mode="w", encoding="utf-8") as file:
        json.dump([], file, ensure_ascii=False, indent=2)


def save_clip(clip_frames, clip_path, clip_fps):
    if not clip_frames:
        return

    height, width = clip_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        clip_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(1.0, float(clip_fps)),
        (width, height),
    )
    try:
        for clip_frame in clip_frames:
            writer.write(clip_frame)
    finally:
        writer.release()


def append_json_event(json_filename, event_data):
    if os.path.exists(json_filename):
        try:
            with open(json_filename, mode="r", encoding="utf-8") as file:
                events = json.load(file)
                if not isinstance(events, list):
                    events = []
        except json.JSONDecodeError:
            events = []
    else:
        events = []

    events.append(event_data)

    with open(json_filename, mode="w", encoding="utf-8") as file:
        json.dump(events, file, ensure_ascii=False, indent=2)


def handle_confirmed_fall(
    frame,
    state,
    detector_mode,
    json_filename,
    snapshot_dir,
    clip_frames,
    clip_fps,
    center_y,
    aspect_ratio,
    height_ratio,
    person_depth_m,
    notifier_script=None,
    disable_email_alert=False,
):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    file_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    fall_dir = os.path.join(snapshot_dir, f"queda_{file_timestamp}_{state.fall_counter}")
    os.makedirs(fall_dir, exist_ok=True)

    snapshot_path = os.path.join(fall_dir, "snapshot.jpg")
    clip_path = os.path.join(fall_dir, "clip_ultimos_5s.mp4")

    append_json_event(
        json_filename,
        {
            "data": datetime.now().strftime("%Y-%m-%d"),
            "hora": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "caminho_imagem": snapshot_path,
            "score": True,
        },
    )

    cv2.imwrite(snapshot_path, frame)
    save_clip(clip_frames, clip_path, clip_fps)

    return {
        "snapshot_path": snapshot_path,
        "clip_path": clip_path,
        "fall_dir": fall_dir,
        "timestamp": timestamp_str,
        "fall_id": state.fall_counter,
    }
