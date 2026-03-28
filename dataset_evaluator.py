import cv2
import mediapipe as mp
import numpy as np
import os
import glob
import csv

# --- CONFIGURACOES DE DIRETORIOS ---
# Pasta onde devem ficar os vídeos com quedas reais
DIR_QUEDAS = "datasets/quedas/"
# Pasta com vídeos de atividades normais (Andar, sentar, amarrar sapato)
DIR_ADL = "datasets/atividades_diarias/"

# Inicializa o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def analyze_video(video_path):
    """
    Roda a heurística de detecção em um vídeo.
    Retorna True se detectar uma queda no arquivo de vídeo, False caso contrário.
    """
    cap = cv2.VideoCapture(video_path)
    fall_frames = 0
    FALL_FRAME_THRESHOLD = 5
    fall_detected_in_video = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extração de pontos Y
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            shoulder_avg_y = (left_shoulder + right_shoulder) / 2

            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP].y
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y
            hip_avg_y = (left_hip + right_hip) / 2

            # Logica de detecção atual do seu projeto (modificada para o Avaliador)
            vertical_distance = hip_avg_y - shoulder_avg_y
            is_falling_this_frame = shoulder_avg_y > hip_avg_y or (vertical_distance < 0.15 and hip_avg_y > 0.5)

            if is_falling_this_frame:
                fall_frames += 1
            else:
                fall_frames = max(0, fall_frames - 1)

            if fall_frames >= FALL_FRAME_THRESHOLD:
                fall_detected_in_video = True
                break # Uma queda confirmada já classifica o vídeo

    cap.release()
    return fall_detected_in_video

def main():
    # Cria as pastas caso não existam
    os.makedirs(DIR_QUEDAS, exist_ok=True)
    os.makedirs(DIR_ADL, exist_ok=True)

    videos_queda = glob.glob(f"{DIR_QUEDAS}*.mp4") + glob.glob(f"{DIR_QUEDAS}*.avi")
    videos_adl = glob.glob(f"{DIR_ADL}*.mp4") + glob.glob(f"{DIR_ADL}*.avi")

    if not videos_queda and not videos_adl:
        print("IMPORTANTE: Para rodar a amostragem, insira videos de teste nas pastas:")
        print(f" -> {os.path.abspath(DIR_QUEDAS)}")
        print(f" -> {os.path.abspath(DIR_ADL)}")
        return

    # Contadores Matriz de Confusão
    VP = 0 # Verdadeiro Positivo (Era queda, detectou queda)
    FN = 0 # Falso Negativo (Era queda, não detectou)
    VN = 0 # Verdadeiro Negativo (Não era queda, não detectou)
    FP = 0 # Falso Positivo (Não era queda, detectou queda)

    print("--- INICIANDO AVALIACAO DO ALGORITMO COM DATASETS ---")
    
    # Processa vídeos POSITIVOS (Vídeos onde ocorrem quedas)
    for v in videos_queda:
        resultado = analyze_video(v)
        if resultado:
            VP += 1
            status = "[ACERTO - VP]"
        else:
            FN += 1
            status = "[ERRO   - FN]"
        print(f"Testando POSITIVO {os.path.basename(v)}: {status}")

    # Processa vídeos NEGATIVOS (Vídeos do dia a dia)
    for v in videos_adl:
        resultado = analyze_video(v)
        if resultado:
            FP += 1
            status = "[ERRO   - FP]"
        else:
            VN += 1
            status = "[ACERTO - VN]"
        print(f"Testando NEGATIVO {os.path.basename(v)}: {status}")

    total_arquivos = VP + FN + VN + FP
    
    # Calculando Métricas Acadêmicas
    try:
        acuracia = (VP + VN) / total_arquivos * 100
        precisao = VP / (VP + FP) * 100 if (VP + FP) > 0 else 0
        sensibilidade = VP / (VP + FN) * 100 if (VP + FN) > 0 else 0 # Recall
        especificidade = VN / (VN + FP) * 100 if (VN + FP) > 0 else 0
    except ZeroDivisionError:
        acuracia = precisao = sensibilidade = especificidade = 0.0

    print("\n--- MATRIZ DE CONFUSAO E RESULTADOS ---")
    print(f"Total de vídeos analisados: {total_arquivos}")
    print(f"Verdadeiros Positivos (VP): {VP}  | Falsos Positivos (FP): {FP}")
    print(f"Falsos Negativos (FN):      {FN}  | Verdadeiros Negativos (VN): {VN}\n")
    
    print(f"ACURACIA (Taxa geral de acertos): {acuracia:.2f}%")
    print(f"SENSIBILIDADE (Taxa de acerto em Quedas reais): {sensibilidade:.2f}%")
    print(f"ESPECIFICIDADE (Capacidade de ignorar movimentos falsos): {especificidade:.2f}%")

    # Salva relatório automatizado de amostragem
    with open("relatorio_amostragem.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Metrica", "Valor", "Descricao"])
        writer.writerow(["VP", VP, "Quedas corretamente detectadas"])
        writer.writerow(["FN", FN, "Quedas que o sistema perdeu"])
        writer.writerow(["VN", VN, "Atividades diarias (nao-queda) corretamente ignoradas"])
        writer.writerow(["FP", FP, "Atividades normais que causaram alarme falso"])
        writer.writerow(["Acuracia", f"{acuracia:.2f}%", "-"])
        writer.writerow(["Sensibilidade", f"{sensibilidade:.2f}%", "-"])

if __name__ == "__main__":
    main()
