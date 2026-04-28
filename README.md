# Reconhecimento de Quedas

Monitor de quedas em tempo real com Intel RealSense D415 (profundidade) e OpenCV, com captura automática de evidências (frames) e geração de relatórios JSON. Inclui também modo opcional com MediaPipe e um avaliador simples para testar a heurística em vídeos de dataset.

## Requisitos
- Python 3.9+ (testado em Linux)
- Webcam acessível pelo OpenCV ou Intel RealSense D415
- Dependências Python:
  - opencv-python
  - numpy
  - pyrealsense2 (obrigatório para usar RealSense)
  - mediapipe (opcional, apenas no modo `--detector mediapipe`)

## Instalação
```bash
# 1) Criar e ativar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 2) Instalar dependências principais (detector por profundidade)
pip install opencv-python numpy pyrealsense2

# 3) Opcional: instalar MediaPipe para modo alternativo por pose
pip install mediapipe
```

## Estrutura do projeto
```
src/
  fall_detection.py
  fall_core/
docs/
  documento_realsense_quedas.md
  relatorio_funcoes_fall_detection.md
dataset_evaluator.py
datasets/
capturas_quedas/
```

## Execução: detecção ao vivo
```bash
python src/fall_detection.py
```
- Pressione `q` para encerrar.
- Por padrão o script usa `--camera-source auto` (tenta RealSense primeiro e faz fallback para webcam).
- Por padrão o detector usa `--detector auto` (usa `skeleton` quando a fonte é RealSense).

### Usando a RealSense D415
```bash
python src/fall_detection.py --camera-source realsense --detector skeleton --show-depth --show-mask
```

### Usando webcam (forçando OpenCV)
```bash
python src/fall_detection.py --camera-source webcam --camera-index 0 --detector mediapipe
```

### Parâmetros úteis
- `--camera-source {auto,webcam,realsense}`
- `--detector {auto,depth,skeleton,mediapipe}`
- `--camera-index N` (índice da webcam)
- `--rs-width`, `--rs-height`, `--rs-fps` (configuração da RealSense)
- `--show-depth` (abre janela da profundidade)
- `--show-mask` (abre janela da máscara segmentada do depth/skeleton)
- `--disable-email-alert` (desativa envio de alerta)
- `--depth-aspect-threshold`, `--depth-center-threshold`, `--depth-height-drop-ratio` (calibração da queda)
- `--depth-vertical-speed-threshold`, `--depth-upright-frames` (calibração temporal da queda)

- Cada queda confirmada:
  - É registrada no JSON `relatorio_quedas.json` com `data`, `hora`, `caminho_imagem` e `score`.
  - No modo `skeleton`, a queda é inferida por esqueleto estimado no depth (cabeça, quadril, pés e ângulo corporal).
  - Cria uma pasta de evento em `capturas_quedas/` com timestamp e ID da queda.
  - Salva nessa pasta um frame (`snapshot.jpg`) e um clipe MP4 (`clip_ultimos_5s.mp4`) com os últimos 5 segundos.

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
- `relatorio_quedas.json`: log JSON das quedas detectadas em tempo real para integração com outro microsserviço.
- `capturas_quedas/`: pasta por evento contendo snapshot e clipe de 5 segundos.
- `relatorio_amostragem.csv`: métricas do avaliador de vídeos.

## Problemas comuns
- RealSense não detectada: confira cabo USB 3.0, permissões e se o pacote `pyrealsense2` está instalado.
- Webcam não abre: rode com `--camera-source webcam --camera-index 0 --detector mediapipe`.
- FPS baixo ou atraso: reduza a resolução da captura ou feche outros apps que usam a GPU.
- Erro de MediaPipe no Python 3.12: prefira o modo `--detector depth` com RealSense, que não depende de MediaPipe.
