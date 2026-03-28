# Reconhecimento de Quedas

Monitor de quedas em tempo real com MediaPipe Pose + OpenCV, com captura automática de evidências (frames) e geração de relatórios CSV. Inclui também um avaliador simples para testar a heurística em vídeos de dataset.

## Requisitos
- Python 3.9+ (testado em Linux)
- Webcam acessível pelo OpenCV (ajuste o índice da câmera em `fall_detection.py` se necessário)
- Dependências Python:
  - mediapipe
  - opencv-python
  - numpy

## Instalação
```bash
# 1) Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 2) Instalar dependências principais
pip install mediapipe opencv-python numpy
```

## Execução: detecção ao vivo
```bash
python fall_detection.py
```
- Pressione `q` para encerrar.
- Cada queda confirmada:
  - É registrada no CSV `relatorio_quedas.csv` (timestamp, métricas de pose, contagem).
  - Gera um frame salvo em `capturas_quedas/` com timestamp e ID da queda.
- Ajuste o índice da câmera (`cv2.VideoCapture(1)`) para `0`, `1`, `2` conforme o dispositivo disponível.

## Avaliação com datasets
Estrutura esperada de vídeos de teste:
```
datasets/
  quedas/             # vídeos com quedas reais
  atividades_diarias/ # vídeos sem queda (negativos)
```

Para rodar a avaliação e obter métricas (VP, FP, FN, VN, acurácia, sensibilidade, especificidade):
```bash
python dataset_evaluator.py
```
Um arquivo `relatorio_amostragem.csv` é gerado com os resultados.

## Saídas geradas
- `relatorio_quedas.csv`: log das quedas detectadas em tempo real.
- `capturas_quedas/`: frames JPG das quedas confirmadas.
- `relatorio_amostragem.csv`: métricas do avaliador de vídeos.

## Problemas comuns
- Webcam não abre: tente mudar `cv2.VideoCapture(1)` para `0` (câmera interna) ou outro índice.
- FPS baixo ou atraso: reduza a resolução da captura ou feche outros apps que usam a GPU.
- ImportError de mediapipe: garanta que o ambiente virtual está ativo e a instalação terminou sem erros.
