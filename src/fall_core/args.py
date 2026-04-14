import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Monitoramento de quedas com RealSense e/ou MediaPipe")
    parser.add_argument(
        "--camera-source",
        choices=["auto", "webcam", "realsense"],
        default="auto",
        help="Fonte de captura de video.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Indice da webcam no OpenCV.")
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
    parser.add_argument("--show-mask", action="store_true", help="Exibe mascara de segmentacao depth (modo depth).")
    parser.add_argument("--depth-aspect-threshold", type=float, default=1.20, help="Limiar de aspecto (largura/altura) para postura deitada.")
    parser.add_argument("--depth-center-threshold", type=float, default=0.62, help="Limiar do centro Y para considerar corpo proximo ao chao.")
    parser.add_argument("--depth-height-drop-ratio", type=float, default=0.72, help="Razao da altura atual vs altura em pe para indicar queda.")
    parser.add_argument("--depth-vertical-speed-threshold", type=float, default=0.035, help="Variacao minima por frame do centro Y para queda brusca.")
    parser.add_argument("--depth-upright-frames", type=int, default=8, help="Frames minimos em postura ereta antes de confirmar transicao de queda.")
    parser.add_argument("--risk-sway-threshold", type=float, default=0.08, help="Amplitude lateral minima para sinalizar instabilidade.")
    parser.add_argument("--risk-horizontal-speed-threshold", type=float, default=0.012, help="Velocidade lateral media minima para instabilidade.")
    parser.add_argument("--risk-hip-jitter-threshold", type=float, default=0.025, help="Jitter vertical minimo do quadril para instabilidade.")
    parser.add_argument("--risk-angle-jitter-threshold", type=float, default=8.0, help="Jitter angular minimo (graus) para instabilidade.")
    parser.add_argument("--risk-min-frames", type=int, default=12, help="Frames minimos para avaliar instabilidade.")
    parser.add_argument("--risk-confirm-frames", type=int, default=4, help="Frames consecutivos para confirmar estado de risco.")
    parser.add_argument("--disable-email-alert", action="store_true", help="Desativa envio de alerta por email.")
    return parser.parse_args()
