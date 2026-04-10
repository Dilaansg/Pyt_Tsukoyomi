# Estructura del Proyecto: Tsukuyomi IA v3.0 (React/FastAPI)

Este documento detalla la arquitectura actual del sistema tras la migración a **React/Vite** y la integración con el backend de **FastAPI**, optimizando la respuesta visual y la captura biométrica.

---

## 1. Cambios en esta Versión (v3.0 - React Revamp)

### 🎨 Frontend (Modernización)
- **Migración a React 19 + Vite**: Eliminación del JS Vanilla por un sistema de componentes modular.
- **Tailwind CSS 4**: Implementación de una paleta basada en el rojo clásico (Red-600) con efectos de **Glassmorphism** (esmerilado).
- **Framer Motion**: Animaciones fluidas de entrada, transiciones entre pantallas y efectos de "typing" de la IA.
- **ThemeCustomizer**: Adición de un selector de color en tiempo real (basado en `react-color`) para personalizar la UI sin editar código.

### 🧠 Integración y Biometría
- **useBiometrics Hook**: El motor de captura biométrica se convirtió en un hook de React. Escucha:
    - `teclas_borrado`: Uso de Backspace/Delete.
    - `tiempo_escritura_segundos`: Duración desde que el usuario empieza a escribir hasta que envía.
    - `ratio_duda`: Relación entre pulsaciones totales y longitud del mensaje.
    - `copy_paste_detectado`: Booleano para respuestas pre-escritas.
- **Persistencia Local**: Uso de `localStorage` para recordar la Edad del usuario y el Modo (Oscuro/Claro) entre sesiones.

### 🔌 Backend (FastAPI Core)
- **Proxy de Desarrollo**: Vite redirige todas las peticiones `/simular-friccion`, `/feedback` y `/detectar-contexto-visual` al puerto 8000.
- **Pipeline Predictivo**: Se mantiene el flujo **NLP (Fase A) -> MLP (Fase B) -> RAG (Fase C) -> LLM (Fase D)**.

---

## 2. Estructura de Tsukuyomi IA

### Arquitectura de Carpetas
```text
/
├── app/                    # BACKEND (Python / FastAPI)
│   ├── api/                # Definición de Schemas (Pydantic) y Endpoints
│   ├── core/               # Motores de IA (NLP, MLP, RAG, Vision, Router)
│   ├── data/               # Logs de latencia y Datasets de interacciones (.jsonl)
│   ├── db/                 # Conexión a MongoDB Atlas
│   ├── models/             # Pesos del modelo MLP (.pt)
│   └── main.py             # Punto de entrada y orquestación del Lifespan
├── frontend/               # FRONTEND (React / Vite)
│   ├── src/
│   │   ├── components/     # Pantallas (Context, Modes, Chat) y UI (Toggle, Modal)
│   │   ├── hooks/          # useBiometrics y lógica reutilizable
│   │   ├── App.jsx         # Orquestador de pantallas y estado global
│   │   └── index.css       # Diseño con Tailwind v4 y Glassmorphism
└── static/                 # Archivos estáticos legacy (opcional)
```

### Funcionamiento del Pipeline
1. **Captura**: El usuario ingresa un escenario (o sube imagen).
2. **Visión (Opcional)**: `VisionService` utiliza Gemini 1.5 para describir la captura y extraer historial de chat visual.
3. **Fricción Biométrica**: El frontend envía los datos de typing del usuario.
4. **Cerebro (MLP)**: El modelo de mediación predice el nivel de fricción social (Atracción, Presión, Urgencia, Valor).
5. **RAG Vectorial**: Se busca en un dataset de tácticas psicológicas la respuesta más adecuada para ese nivel de fricción.
6. **LLM Router**: Se ensambla un System Prompt complejo y se genera la respuesta final con el modelo (Gemini Pro/Flash) con menor latencia disponible.

---

## 3. Viabilidad Técnica y Nube (Cloud Deployment)

### Viabilidad Actual: **Alta (Ready)**
El proyecto ya utiliza variables de entorno (`.env`) y tiene un archivo `render.yaml`, lo que lo hace compatible con plataformas como **Render**, **Railway** o **Google Cloud Run**.

### Desafíos para el Deployment:
1. **Inferencia de Torch**: El modelo MLP usa `torch`. Esto hace que el contenedor de Docker sea pesado (~1GB). Para subirlo a la nube gratis (tiers bajos), se recomienda usar `torch-cpu` para reducir el tamaño.
2. **Variables de Entorno**: Es CRÍTICO configurar `GEMINI_API_KEY` y `MONGODB_URI` en el dashboard del hosting elegido.
3. **CORS**: En producción, el `CORSMiddleware` debe limitarse al dominio final del frontend en lugar de usar `["*"]`.

---

## 4. Posibles Mejoras (Roadmap)

1. **Persistencia de Sesiones via DB**: Actualmente el historial de chat se pierde al refrescar (se guarda en local). Implementar un `sessionId` vinculado a MongoDB permitiría recuperar chats en múltiples dispositivos.
2. **Audio/Voice Feedback**: Implementar la API de Web Speech para que Tsukuyomi pueda "hablar" sus consejos, reforzando la inmersión.
3. **Optimización de Latencia (Fase A)**: El análisis NLP inicial toma tiempo. Se podría paralelizar la Fase A (NLP) con la Fase B (Biometría) para ganar ~0.5s por respuesta.
4. **Detección de Tácticas en Tiempo Real**: Mostrar al usuario un indicador tipo "HUD" que le diga qué táctica está usando la IA en ese momento (Ej. "Aplicando: Escasez Narrativa").
