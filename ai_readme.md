🧠 TSUKOYOMI-IA: Technical System Architecture
Este proyecto es un Simulador de Fricción Social basado en una arquitectura híbrida: Biometría de Teclado + NLP Zero-Shot + MLP (PyTorch) + RAG Semántico + LLM (Gemini 2.5).

📁 Estructura de Archivos
api.py (o main.py): Orquestador FastAPI. Gestiona el ciclo de vida de los modelos (lifespan) y los endpoints REST.

motor_ia.py: Núcleo lógico. Contiene las clases de PyTorch, el banco de tácticas vectorizado y el ensamblador de prompts.

index.html: Frontend vanilla JS. Captura biometría (latencia, borrados, ratio de duda) y gestiona el estado de la simulación.

.env: Almacena la GEMINI_API_KEY.

🔄 Flujo de Datos (Pipeline)
1. Fase A: Ingesta y Contexto Social (NLP)
Input: texto_usuario.

Modelo: mDeBERTa-v3-base-mnli-xnli (Zero-shot classification).

Output: Vector soc_A (Ansiedad), soc_P (Poder), soc_U (Urgencia), soc_V (Valencia).

Lógica: Determina la jerarquía y el tono de la relación.

2. Fase B: Motor de Fricción (PyTorch MLP)
Input: Metadatos biométricos del JS + Vector Fase A.

Modelo: RedMediacionMLP (Multi-Layer Perceptron con bloques residuales y LayerNorm).

Output: Predicción de 4 dimensiones: terquedad, frialdad, sarcasmo, frustracion.

3. Fase C: Traducción Semántica (RAG)
Proceso: El ScoringAnalitico convierte las predicciones numéricas en una query de lenguaje natural.

Búsqueda: Se realiza una búsqueda de similitud coseno contra BANCO_TACTICAS usando all-MiniLM-L6-v2.

Output: Las 4 tácticas psicológicas más relevantes según el estado emocional detectado.

4. Fase D: Generación de Respuesta (LLM)
Modelo: Gemini 2.5 Flash.

Prompt Inyectado: Combina el Escenario Inicial, el historial de chat y las tácticas recuperadas por el RAG.

Rol: Inferencia dinámica de personalidad basada en el contexto provisto por el usuario.

🛠 Especificaciones Técnicas para Agentes (Aider/OpenCode)
Manejo de Estado: El servidor es stateless. El historial se mantiene en el cliente y se envía en cada request.

Normalización: Se utiliza LayerNorm en lugar de BatchNorm para soportar inferencia de batch_size=1 sin colapsar.

Biometría: El campo pulsaciones_totales incluye eventos keydown repetidos para capturar la intensidad física (ej: "aaaaa").

Seguridad: No hardcodear API Keys; usar load_dotenv().