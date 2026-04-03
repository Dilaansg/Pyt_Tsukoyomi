# Arquitectura y Flujo de Tsukoyomi-IA (v3)

Este documento detalla el estado actual del flujo del sistema de Tsukoyomi-IA, así como el diseño de su Frontend, para facilitar futuras refactorizaciones o alineaciones con los objetivos del producto.

## 1. Flujo del Backend (API de Simulación)

El sistema opera bajo un pipeline modular que procesa cada turno de la conversación de forma secuencial. El endpoint principal es `/simular-friccion`.

### Bifurcación por Modos:

#### A) Modo "Simulador" (Flujo Completo)
1. **Fase A - Análisis NLP (Contexto Social):**
   - Usa `mDeBERTa-v3` mediante zero-shot classification (acelerado por caché MD5).
   - Extrae 4 ejes del `escenario`: Ansiedad, Poder (Autoridad vs Subordinado), Urgencia y Valencia (Confianza).
2. **Fase B - Inferencia Biométrica (Fricción):**
   - Toma los metadatos biométricos (teclas borradas, pausas, velocidad) y los ejes NLP.
   - Pasa por una **Red Neuronal MLP** (Multilayer Perceptron) que predice 4 métricas psicológicas reactivas: *Terquedad, Frialdad, Sarcasmo y Frustración*.
3. **Fase C - Traducción Vectorial (ADN de Tácticas):**
   - El vector predicho por el MLP se compara (similitud del coseno) contra un banco de 100 tácticas de fricción (`tacticas.json`).
   - Se selecciona la **táctica más afín** para dictar el comportamiento del antagonista.
4. **Fase D - Generación de IA (Gemini):**
   - El `EnsambladorPromptV5` infiere el rol de la IA a partir del escenario y genera el prompt final, forzando a la IA a obedecer la táctica sin mencionarla ("Show, don't tell").
   - Lógica de fallback: Intenta con modelos rápidos (Gemini Flash) y salta al siguiente si falla la cuota.

#### B) Modo "Consejo" (Flujo Corto / Fast-Track)
- **Fase A (NLP):** Sigue ejecutándose para entender el tono social.
- **Fases B y C (Desviadas):** *Se saltan*. El sistema no calcula fricción ni busca tácticas para el consejero, ya que esto causaba que el mentor se volviera sociópata.
- **Fase D (Gemini):** Se ensambla un `PROMPT_CONSEJO` autocontenido. El LLM asume el rol de amigo/analista callejero, lee el escenario y le aconseja al usuario cómo defenderse.

### Telemetría y Feedback (Fase 3 latente)
- **Latencias:** Cada paso se temporiza y se guarda localmente en `app/data/latency_logs.jsonl`.
- **Interacciones (`dataset_interacciones.jsonl`):** Al finalizar la sesión, la UI envía una puntuación y la efectividad de las tácticas usadas. Estos datos preparan el sistema para el ML Dinámico (auto-ajuste de vectores).

---

## 2. Paradigma del Frontend

El frontend actúa como la interfaz clínica de recolección de datos, diseñada para parecer un chat pero funcionar como un polígrafo pasivo.

### Composición Actual:
- **`index.html`:** Interfaz web básica inspirada en un flujo de mensajería asíncrona tipo Notion o WhatsApp limpio.
- **`style.css`:** Estilos minimalistas. Sin distracciones.
- **`biometria.js`:** El verdadero cerebro del cliente.

### Responsabilidades de `biometria.js`:
1. **Captura Silenciosa:** Intercepta eventos del teclado (`keyup`, `keydown`). Mide cuántas veces el usuario presiona "Backspace" (Teclas de Borrado).
2. **Ratio de Duda:** Calcula las pausas (> 1 segundo) durante la escritura. Mucha duda infiere inseguridad o "caminar sobre cáscaras de huevo".
3. **Copy/Paste Detection:** Si el usuario pega el texto, la matriz biométrica se neutraliza (peso biométrico decae al mínimo) porque la emoción no se traslada al teclado.
4. **Estado de Conversación:** Mantiene el historial de la sesión (`historial` array) que se envía en cada request al backend, ya que el backend es Stateless (no guarda memoria de sesiones activas en RAM).

### Consideraciones para Futuras Refactorizaciones del Frontend:
- **Experiencia de Usuario (UX):** El objetivo del sistema es "Entrenamiento de Habilidades Blandas". El frontend puede evolucionar para sentirse menos "clínico" y más como un simulador de RV en formato texto (con avatars, barras de estrés inferidas del oponente, etc).
- **Gamificación del Feedback:** Actualmente el usuario debe llenar un form/estrellas al terminar. Esto puede generar fricción. La UI podría inferir la "puntuación" basándose en si la charla fue resuelta en pocos turnos o en una evaluación final automática.
- **Visualización de la FASE B:** Aunque es útil para el desarrollador ver los vectores de *Terquedad* o *Frialdad* del oponente, en producción esto podría ocultarse para mantener al usuario concentrado en el diálogo, mostrándolo únicamente en un "Dashboard Post-Partida".
