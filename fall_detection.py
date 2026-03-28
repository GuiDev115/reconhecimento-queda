import cv2
import numpy as np
import time
import csv
import os
import subprocess
from datetime import datetime

# MediaPipe Solutions (A partir da versao 0.10.x os imports sao com tasks ou de forma diferente dependendo do OS.
# Mas importando via "mediapipe as mp" com o python3 do linux normalmente exige a importação do pacote sem o ".python".
try:
    import mediapipe as mp
    mp_pose_module = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except AttributeError:
    # Se mp.solutions falhar, tentar fazer fallback ou usar a estrutura direta
    import mediapipe.python.solutions.pose as mp_pose_module
    import mediapipe.python.solutions.drawing_utils as mp_drawing


# Inicializa o MediaPipe Pose
mp_pose = mp_pose_module
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inicia a captura da webcam (Se tiver mais de uma, tente trocar de 0 para 1 ou 2)
# O parâmetro 0 geralmente é a câmera do notebook (interna), 1 costuma ser a USB conectada.
cap = cv2.VideoCapture(1)

# Variáveis para auxiliar na detecção temporal
fall_detected = False
fall_counter = 0
fall_frames = 0
FALL_FRAME_THRESHOLD = 5  # Margem de erro: número de frames consecutivos para confirmar uma queda (falsos positivos)
cooldown_time = 3.0       # Tempo (em segundos) de espera até poder registrar outra queda (evita contagem múltipla)
last_fall_time = 0.0

# Variáveis para Detecção de Risco/Instabilidade (Tontura e Oscilação no Eixo X)
risk_detected = False
x_sway_history = []
SWAY_THRESHOLD = 0.08   # Variação máxima perdoável no eixo Horizontal
SWAY_FRAMES = 15      # Analisar a oscilação nos últimos X frames históricos

# Diretório para salvar evidências de quedas
FALL_SNAPSHOT_DIR = "capturas_quedas"
os.makedirs(FALL_SNAPSHOT_DIR, exist_ok=True)

# Script externo de notificação por email
NOTIFIER_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notificador-quedas", "send_alert.py"))

# Configuração do arquivo de tabulação para a Iniciação Científica (CSV)
csv_filename = "relatorio_quedas.csv"
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "ID_Queda", "Ombro_Y", "Quadril_Y", "Distancia_Vertical", "Margem_Frames"])

print("Iniciando monitoramento de quedas. Pressione 'q' para sair.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível acessar a câmera.")
        break

    # Pega as dimensões da imagem
    h, w, _ = frame.shape

    # Converte o frame para RGB (MediaPipe usa RGB)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Processa a imagem para encontrar a pose
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Desenha os pontos do corpo na imagem
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark

        # Extraindo pontos de interesse (y varia de 0 no topo a 1 na base da imagem)
        # Nariz (Head)
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        
        # Ombros
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2

        # Quadril
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        hip_avg_y = (left_hip.y + right_hip.y) / 2

        # Tornozelos
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
        ankle_avg_y = (left_ankle.y + right_ankle.y) / 2
        
        # Ponto Central (Gravidade/Oscilação Horizontal) - Extraindo o Eixo X
        shoulder_avg_x = (left_shoulder.x + right_shoulder.x) / 2
        x_sway_history.append(shoulder_avg_x)
        
        if len(x_sway_history) > SWAY_FRAMES:
            x_sway_history.pop(0) # Mantém janela limpa

        # Lógica Simples de Queda:
        # Se os ombros ou a cabeça estiverem muito próximos ou abaixo do quadril
        # E a distância vertical torço/pernas for comprimida
        
        # Y cresce para baixo na imagem. Se shoulder_y > hip_y, a pessoa está de cabeça para baixo ou deitada
        # Consideramos uma queda se a altura entre o ombro e o quadril for subitamente pequena ou invertida
        
        vertical_distance = hip_avg_y - shoulder_avg_y
        
        # Ajustando os parâmetros de detecção e calibrando a sensibilidade
        is_falling_this_frame = shoulder_avg_y > hip_avg_y or (vertical_distance < 0.15 and hip_avg_y > 0.5)

        current_time = time.time()

        # Janela de Margem de Erro temporal: exige X frames consecutivos confirmando a queda para evitar falsos positivos
        if is_falling_this_frame:
            fall_frames += 1
        else:
            fall_frames = max(0, fall_frames - 1)

        # Lógica Secundária Preditiva: Analisa Desequilíbrio/Tontura pré-queda
        if len(x_sway_history) == SWAY_FRAMES:
            sway_max = max(x_sway_history)
            sway_min = min(x_sway_history)
            
            # Se a pessoa mexe os ombros bruscamente pros lados sem andar (desequilíbrio)
            if (sway_max - sway_min) > SWAY_THRESHOLD and not is_falling_this_frame:
                risk_detected = True
            else:
                risk_detected = False

        # Confirma a queda se o limiar de frames for atingido
        if fall_frames >= FALL_FRAME_THRESHOLD:
            # Usa cooldown para não registrar a mesma queda várias vezes num curto período
            if (current_time - last_fall_time) > cooldown_time:
                fall_counter += 1
                last_fall_time = current_time

                timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                file_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                snapshot_path = os.path.join(FALL_SNAPSHOT_DIR, f"queda_{file_timestamp}_{fall_counter}.jpg")
                
                # Tabulação: Registra os dados métricos da queda no arquivo CSV
                with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp_str, fall_counter, f"{shoulder_avg_y:.4f}", f"{hip_avg_y:.4f}", f"{vertical_distance:.4f}", fall_frames])

                # Evidência visual da queda
                cv2.imwrite(snapshot_path, frame)

                # Dispara alerta por email em módulo separado
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

        # Exibição na tela (Prioridade: Queda > Risco > Normal)
        if fall_detected or (current_time - last_fall_time) < 2.0:
            cv2.putText(frame, "ALERTA: QUEDA DETECTADA!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
        elif risk_detected:
            cv2.putText(frame, "RISCO: INSTABILIDADE CORPORAL!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Status: Normal", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        # Mostra o contador de quedas na interface
        cv2.putText(frame, f"Quedas registradas: {fall_counter}", (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            
    cv2.imshow('Monitoramento de Queda', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
