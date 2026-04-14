import csv
import os
import subprocess
from datetime import datetime

import cv2
import numpy as np


def initialize_csv(csv_filename):
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Timestamp",
                "ID_Queda",
                "Detector",
                "Centro_Y",
                "Aspecto_Largura_Altura",
                "Altura_Relativa",
                "Margem_Frames",
                "Profundidade_m",
            ]
        )


def handle_confirmed_fall(
    frame,
    state,
    detector_mode,
    csv_filename,
    snapshot_dir,
    notifier_script,
    disable_email_alert,
    center_y,
    aspect_ratio,
    height_ratio,
    person_depth_m,
):
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    file_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    snapshot_path = os.path.join(snapshot_dir, f"queda_{file_timestamp}_{state.fall_counter}.jpg")

    with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                timestamp_str,
                state.fall_counter,
                detector_mode,
                f"{center_y:.4f}" if not np.isnan(center_y) else "",
                f"{aspect_ratio:.4f}" if not np.isnan(aspect_ratio) else "",
                f"{height_ratio:.4f}" if not np.isnan(height_ratio) else "",
                state.fall_frames,
                f"{person_depth_m:.3f}" if not np.isnan(person_depth_m) else "",
            ]
        )

    cv2.imwrite(snapshot_path, frame)

    if not disable_email_alert and os.path.exists(notifier_script):
        try:
            subprocess.run(
                [
                    "python",
                    notifier_script,
                    "--image",
                    snapshot_path,
                    "--subject",
                    "Queda detectada",
                    "--body",
                    "Foi detectada uma queda. Verifique o idoso.",
                ],
                check=True,
            )
        except Exception as exc:
            print(f"Falha ao enviar alerta: {exc}")
