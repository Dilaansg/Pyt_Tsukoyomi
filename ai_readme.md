🧠 TSUKOYOMI-IA: Technical System Architecture (v3.1)
===================================================

Este proyecto es un **Simulador de Fricción Social** basado en una arquitectura híbrida: Biometría de Teclado + NLP Zero-Shot + MLP (PyTorch) + ADN Vectorial + LLM (Gemini 3.1).

📁 Estructura de Archivos
-----------------------
- **app/main.py**: Orquestador FastAPI asíncrono. Gestiona el ciclo de vida de los modelos (lifespan), la conexión a MongoDB Atlas y los endpoints `/simular-friccion`, `/feedback` y `/health`.
- **app/core/rag_translator.py**: TraductorSemanticoV5. Selección de tácticas mediante Álgebra Lineal pura (Cosine Similarity) y Ensamblador de Prompts diferenciado (Simulador vs Consejo).
- **app/core/nlp_service.py**: Extracción de contexto social con mDeBERTa-v3 optimizado con caché MD5 y truncamiento de tokens.
- **app/db/mongo.py**: Módulo de persistencia asíncrona (Motor) para MongoDB Atlas.
- **static/js/biometria.js**: Frontend en Vanilla JS. Captura biometría (latencia, borrados, ratio de duda) y gestiona el estado de la comunicación.

🔄 Flujo de Datos (Pipeline Modular)
----------------------------------

### 1. Fase A: Ingesta y Contexto Social (NLP)
- **Input**: `texto_usuario` + `escenario`.
- **Modelo**: `mDeBERTa-v3-base-mnli-xnli`.
- **Proceso**: Clasificación Zero-shot (Ansiedad, Poder, Urgencia, Valencia).
- **Optimización**: Cacheado por hash del escenario (una sola inferencia por sesión).

### 2. Fase B: Motor de Fricción (PyTorch MLP)
- **Input**: Metadatos biométricos (JS por Pydantic) + Vector Fase A.
- **Modelo**: RedMediacionMLP con bloques residuales.
- **Output**: Predicción de 4 dimensiones: terquedad, frialdad, sarcasmo, frustración.
- **Seed Training**: Pesos iniciales pre-entrenados con `seed_trainer.py`.

### 3. Fase C: Selección de ADN Vectorial
- **Proceso**: El vector del MLP se compara contra el banco de 100 tácticas de fricción (`tacticas.json`).
- **Lógica**: En modo Simulador se recupera la táctica `top_k=1` para asegurar respuestas naturales no estructuradas.

### 4. Fase D: Generación con LLM (Multimodal Fallback)
- **Modelos**: Gemini 3.1 Flash / Pro / Lite (Rotación automática ante errores de cuota).
- **Prompt**:
  - **Simulador**: Adopta un rol inferido dinámicamente y aplica la táctica de forma "invisible".
  - **Consejo**: Salta las fases B y C para actuar como un confidente empático que deconstruye la situación.

⚙️ Persistencia y Cloud
----------------------
- **Base de Datos**: MongoDB Atlas (Cloud) + Backup local en `.jsonl`.
- **Infraestructura**: Preparado para despliegue en Render (ficheros `render.yaml` y `Procfile` incluidos).
- **Capa de Datos**: Repositorio Stateless. El historial reside en el cliente.

🛠 Especificaciones para Desarrolladores
--------------------------------------
- **Carga de Modelos**: Se realiza en el `lifespan` de FastAPI al arranque.
- **Requisitos**: `pip install -r requirements.txt` (incluye `motor`, `torch`, `transformers`, etc).
- **Entorno**: Requiere archivo `.env` con `GEMINI_API_KEY` y `MONGODB_URI`.