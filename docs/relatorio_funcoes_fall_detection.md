# Relatorio de Funcoes e Modularizacao do Projeto

Data: 14 de abril de 2026

## Visao geral
O projeto foi reorganizado para sair do formato monolitico e evoluir para arquitetura modular.
A execucao principal continua em fall_detection.py, mas as responsabilidades foram separadas no pacote fall_core.

## Modulos criados

### 1) fall_core/args.py
Responsabilidade: configuracao e parametros da aplicacao.

Funcoes:
- parse_args:
  - Define todos os argumentos de linha de comando.
  - Inclui parametros de camera, detector, limiares de queda e limiares de instabilidade.

### 2) fall_core/camera.py
Responsabilidade: inicializacao e acesso aos dispositivos de captura.

Funcoes:
- start_webcam_capture:
  - Inicia webcam OpenCV.
  - Valida se o dispositivo abriu corretamente.
- start_realsense_capture:
  - Inicia stream color + depth da RealSense.
  - Configura alinhamento depth-color e escala de profundidade.
- start_capture:
  - Escolhe automaticamente entre webcam e RealSense conforme argumentos.
  - Em modo auto, tenta RealSense e faz fallback para webcam.
- read_frame:
  - Le frame atual conforme tipo de captura.
  - Retorna frame RGB e, quando disponivel, frame de profundidade.
- release_capture:
  - Libera recursos de camera/pipeline.

### 3) fall_core/vision.py
Responsabilidade: funcoes de visao computacional e utilitarios numericos.

Funcoes:
- draw_hud_text:
  - Renderiza painel de texto com transparencia no frame.
- ema:
  - Suavizacao temporal por media movel exponencial.
  - Reduz ruido de medidas entre frames.
- depth_person_metrics:
  - Segmenta primeiro plano por profundidade.
  - Extrai bbox, centro, razao de aspecto, altura relativa e profundidade mediana do alvo.
- estimate_depth_skeleton:
  - Estima esqueleto simplificado (cabeca, quadril e pes) a partir da mascara depth.
  - Calcula angulo corporal em relacao ao eixo vertical.

### 4) fall_core/state.py
Responsabilidade: estado global de execucao em estrutura unica.

Estrutura:
- RuntimeState (dataclass):
  - Armazena estado de queda (contadores, cooldown, ultimo evento).
  - Armazena estado de risco/instabilidade.
  - Mantem historicos temporais para analise de movimento e suavizacao.

### 5) fall_core/events.py
Responsabilidade: eventos persistentes e notificacao externa.

Funcoes:
- initialize_csv:
  - Cria/inicializa relatorio CSV de quedas.
- handle_confirmed_fall:
  - Registra evento confirmado no CSV.
  - Salva snapshot da queda.
  - Dispara script de alerta por email (quando habilitado).

### 6) fall_core/processing.py
Responsabilidade: logica de deteccao por frame e regras de decisao.

Funcoes:
- resolve_detector_mode:
  - Escolhe detector em modo auto conforme fonte de camera.
- process_depth_mode:
  - Executa pipeline de deteccao com profundidade/esqueleto.
  - Aplica suavizacao temporal.
  - Detecta queda por transicao de postura + centro + variacao de altura/velocidade vertical.
  - Detecta instabilidade corporal com voto multi-metrica.
- process_mediapipe_mode:
  - Executa deteccao com pose 2D (MediaPipe).
  - Mantem estimativa de instabilidade lateral simplificada.
- update_fall_state:
  - Atualiza estado temporal de queda (janela de confirmacao e cooldown).
  - Indica quando um novo evento confirmado deve ser persistido.
- build_hud_lines:
  - Monta linhas de depuracao e status para HUD em tela.

### 7) fall_detection.py
Responsabilidade: orquestrador principal da aplicacao.

Fluxo:
- Le argumentos.
- Inicializa captura e detector.
- Inicializa estado e saídas (CSV/pasta de snapshots).
- Loop principal:
  - Le frame.
  - Chama processador conforme modo (depth/skeleton ou mediapipe).
  - Atualiza estado temporal de queda.
  - Em evento confirmado: persiste evidencias e dispara alerta.
  - Renderiza HUD e janelas auxiliares.

## Como o projeto ficou diversificado
A diversificacao foi feita em arquitetura e responsabilidades:
- Separacao por dominio (camera, visao, estado, eventos, processamento).
- Reuso de funcoes em diferentes modos (depth e mediapipe).
- Facil extensao para novos detectores/sensores sem inflar o arquivo principal.

## Beneficios da modularizacao
- Menor acoplamento e maior legibilidade.
- Facil manutencao e testes por componente.
- Melhor base para evolucao (multi-camera, fusao com IMU/radar, API de servico, avaliador unificado).

## Proximos passos recomendados
1. Criar testes unitarios para:
- process_depth_mode
- update_fall_state
- calculo de instabilidade

2. Unificar dataset_evaluator.py com fall_core/processing.py para avaliar o mesmo criterio usado ao vivo.

3. Criar arquivo de configuracao (ex.: YAML) para perfis de ambiente com limiares calibrados.
