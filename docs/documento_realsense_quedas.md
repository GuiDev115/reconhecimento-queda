# Documento Técnico: Uso da Intel RealSense D435i em Reconhecimento de Quedas

Data: 14 de abril de 2026
Projeto: Reconhecimento de Quedas

## 1. Objetivo
Este documento consolida a análise técnica sobre o uso da Intel RealSense D435i para detecção de quedas, incluindo:
- Adequação da câmera para a aplicação.
- Sensores recomendados para cenários reais.
- Avaliação do estado atual do projeto.
- Melhorias recomendadas para aumentar confiabilidade.

## 2. A Intel RealSense D435i é adequada para detecção de quedas?
Sim. A D435i é adequada para protótipos e projetos acadêmicos de detecção de quedas, principalmente em ambientes internos.

### Pontos fortes
- Captura de profundidade (depth) e imagem RGB no mesmo dispositivo.
- Presença de IMU (acelerômetro e giroscópio), útil para compensar movimentos da câmera.
- Bom desempenho em tempo real para resolução moderada.
- Capacidade de separar pessoa e fundo com mais robustez que câmera RGB comum.

### Limitações
- Menor robustez sob luz solar direta intensa.
- Sensível a superfícies reflexivas, muito escuras ou com baixa textura.
- Degradação de qualidade de profundidade em maiores distâncias.
- Oclusões (móveis, cobertores, outras pessoas) podem prejudicar a detecção.

## 3. Quais sensores são bons para esta aplicação?
A escolha depende do nível de criticidade do sistema.

### Nível 1: Protótipo acadêmico e pesquisa aplicada
- Intel RealSense D435i (boa opção).
- Intel RealSense D455 (alternativa com melhor estabilidade em distâncias maiores).

### Nível 2: Sistema mais robusto para operação contínua
- Fusão de sensores:
  - Câmera depth (RealSense) +
  - Radar mmWave +
  - Opcionalmente wearable com IMU.

Essa combinação reduz falsos positivos e falsos negativos em situações de oclusão e movimentos ambíguos.

## 4. Avaliação do projeto atual
A arquitetura atual está adequada para um sistema funcional de pesquisa.

### Pontos positivos no projeto
- Pipeline de captura com RealSense (color + depth).
- Modo de detecção por profundidade e modo por esqueleto estimado no depth.
- Uso de critérios temporais para validar queda (transição, velocidade vertical, postura final).
- Registro de eventos em CSV e captura de evidências (frames).

### Pontos de atenção
1. Inconsistência de documentação
- A documentação menciona D415, enquanto o contexto do uso é D435i.

2. Desalinhamento entre operação e avaliação
- O monitoramento principal usa fortemente profundidade/skeleton.
- O avaliador de dataset usa MediaPipe 2D.
- Isso pode gerar métricas que não representam o comportamento real do detector principal.

3. Dependência de limiares fixos
- Valores fixos podem funcionar em um ambiente e falhar em outro (altura da câmera, iluminação, perfil corporal e layout do cômodo).

4. Validação ainda limitada
- É necessário ampliar cenários de teste para distinguir quedas reais de movimentos normais rápidos (sentar rápido, ajoelhar, pegar objeto, deitar voluntariamente).

## 5. Recomendação técnica

### Curto prazo (alto impacto)
1. Atualizar documentação para refletir o uso real da D435i.
2. Adaptar o avaliador para usar a mesma lógica de detecção depth/skeleton utilizada no monitoramento ao vivo.
3. Criar protocolo de calibração por ambiente antes da coleta principal.

### Médio prazo
1. Construir benchmark por cenário com matriz de confusão por classe de movimento.
2. Medir latência de detecção e taxa de falsos alarmes por hora.
3. Introduzir lógica de pós-evento (janela temporal e confirmação contextual).

### Longo prazo
1. Implementar fusão sensorial (depth + mmWave, e opcional IMU vestível).
2. Evoluir de heurística para modelo temporal supervisionado com dados reais do domínio.

## 6. Conclusão
A Intel RealSense D435i é uma escolha tecnicamente válida para detecção de quedas em ambiente interno, especialmente para pesquisa e prototipagem. O projeto atual está bem direcionado, com base funcional consistente. O principal passo para elevar a confiabilidade é alinhar a avaliação de dataset com o mesmo detector de profundidade usado na operação real e estabelecer calibração sistemática por ambiente.

## 7. Resumo executivo
- A D435i é apropriada para o problema em cenário indoor.
- O projeto está adequado para fase acadêmica e protótipo funcional.
- A maior lacuna atual está no avaliador (MediaPipe 2D) não refletir o detector principal (depth/skeleton).
- A próxima evolução recomendada é alinhamento da avaliação + calibração + ampliação de cenários de teste.
