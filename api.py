import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import pipeline
import google.generativeai as genai
from dotenv import load_dotenv

# Importamos tu lógica de motor_ia.py
from motor_ia import (
    RedMediacionMLP, TraductorSemanticoV3,
    predecir, PayloadFaseA, PayloadFaseB,
)

# ── Configuración y Env ────────────────────────────────────
load_dotenv()
_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("Falta la variable de entorno GEMINI_API_KEY en tu archivo .env")
genai.configure(api_key=_api_key)

ETIQUETAS_NLP = [
    "miedo al rechazo o ansiedad social",
    "la persona tiene autoridad o es el jefe",
    "la persona es un subordinado o novato",
    "situación urgente o crisis de tiempo",
    "son amigos cercanos o hay mucha confianza",
]

MAX_HISTORIAL = 20 

# ── Lifespan: Carga de modelos pesados ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n[TSUKOYOMI] Levantando motor de IA...")

    print("[API] 1/3 Cargando mDeBERTa (Análisis de Contexto)...")
    app.state.clasificador = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    )

    print("[API] 2/3 Cargando MLP (Red Neuronal de Fricción)...")
    mlp = RedMediacionMLP()
    mlp.eval() 
    app.state.modelo_mlp = mlp

    print("[API] 3/3 Cargando Traductor Semántico (RAG)...")
    app.state.traductor = TraductorSemanticoV3()

    print("[TSUKOYOMI] Todos los sistemas en línea.\n")
    yield

app = FastAPI(title="TSUKOYOMI-IA", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Modelos Pydantic ───────────────────────────────────────
class MetadatosJS(BaseModel):
    tiempo_escritura_segundos: float
    teclas_borrado:            int
    pulsaciones_totales:       int
    ratio_duda:                float
    copy_paste_detectado:      bool
    longitud_caracteres:       int
    edad_usuario:              int = Field(default=16, ge=10, le=100)

class MensajeHistorial(BaseModel):
    role:    str   # "user" o "model"
    content: str

class PeticionSimulacion(BaseModel):
    escenario:     str = Field(default="")
    texto_usuario: str = Field(min_length=1, max_length=2000)
    metadatos:     MetadatosJS
    historial:     list[MensajeHistorial] = []

class RespuestaSimulacion(BaseModel):
    respuesta_bot:         str
    friccion_calculada:    dict
    contexto_nlp_extraido: dict
    tacticas_usadas:       list[str]
    prompt_inyectado:      str

# ── Endpoint Principal ─────────────────────────────────────
@app.post("/simular-friccion", response_model=RespuestaSimulacion)
async def simular_friccion(peticion: PeticionSimulacion):
    try:
        # Recuperamos modelos del estado global
        clasificador = app.state.clasificador
        modelo_mlp   = app.state.modelo_mlp
        traductor    = app.state.traductor

        # ── Fase A: NLP (Contexto Social) ──────────────────
        # Si es el mensaje inicial, usamos valores neutrales para no sesgar
        if "Inicia la conversación" in peticion.texto_usuario:
            soc_a, soc_p, soc_u, soc_v = 0.1, 0.0, 0.1, 0.5
        else:
            resultado_nlp = clasificador(peticion.texto_usuario, ETIQUETAS_NLP, multi_label=True)
            scores = dict(zip(resultado_nlp["labels"], resultado_nlp["scores"]))
            soc_a = scores["miedo al rechazo o ansiedad social"]
            soc_p = scores["la persona tiene autoridad o es el jefe"] - scores["la persona es un subordinado o novato"]
            soc_u = scores["situación urgente o crisis de tiempo"]
            soc_v = scores["son amigos cercanos o hay mucha confianza"]

        fase_a = PayloadFaseA(soc_A=soc_a, soc_P=soc_p, soc_U=soc_u, soc_V=soc_v)

        # ── Fase B: MLP (Predicción de Fricción) ───────────
        fase_b = PayloadFaseB(**peticion.metadatos.model_dump())
        prediccion = predecir(modelo_mlp, fase_a, fase_b)

        # ── Fase C: Traducción Semántica (Tácticas) ────────
        prompt_final, resultados = traductor.traducir(prediccion, fase_a, peticion.escenario)
        tacticas_nombres = [t for t, _ in resultados]

        # ── Fase D: Generación con Gemini ──────────────────
        historial_reciente = peticion.historial[-MAX_HISTORIAL:]
        gemini_history = [{"role": m.role, "parts": [m.content]} for m in historial_reciente]

        modelo_gemini = genai.GenerativeModel(
            model_name="models/gemini-2.5-flash",
            system_instruction=prompt_final,
        )
        chat = modelo_gemini.start_chat(history=gemini_history)
        
        # Le enviamos el texto actual (sea el inicial o el del usuario)
        respuesta_llm = chat.send_message(peticion.texto_usuario)

        return RespuestaSimulacion(
            respuesta_bot=respuesta_llm.text,
            friccion_calculada=prediccion.to_dict(),
            contexto_nlp_extraido={
                "soc_A": round(fase_a.soc_A, 3),
                "soc_P": round(fase_a.soc_P, 3),
                "soc_U": round(fase_a.soc_U, 3),
                "soc_V": round(fase_a.soc_V, 3),
            },
            tacticas_usadas=tacticas_nombres,
            prompt_inyectado=prompt_final,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))