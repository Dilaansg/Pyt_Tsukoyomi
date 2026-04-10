import os
import json
import torch
from pathlib import Path
from datetime import datetime
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


from app.config.settings import config
from app.api.schemas import PeticionSimulacion, RespuestaSimulacion, FeedbackSession, VisionUploadRequest, MensajeHistorial
from app.core.nlp_service import NLPService
from app.core.friction_model import RedMediacionMLP, predecir
from app.core.rag_translator import TraductorSemanticoV4
from app.core.schemas import PayloadFaseA, PayloadFaseB, PrediccionFriccion
from app.core.llm_router import LLMRouter
from app.core.vision_service import VisionService
from app.core.tension_tracker import TensionTracker

from app.db.mongo import ping, col_sesiones, col_latencias

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona el ciclo de vida del servidor. 
    Carga modelos en memoria para optimizar la latencia en inferencia.
    """
    print("\n[TSUKOYOMI-IA] Inicializando servicios...")
    app.state.llm_router = LLMRouter()
    app.state.vision_service = VisionService()

    # MongoDB
    if config.mongodb_uri:
        mongo_ok = await ping()
        if mongo_ok:
            print("[TSUKOYOMI-IA] MongoDB Atlas conectado.")
        else:
            print("[TSUKOYOMI-IA] ADVERTENCIA: MongoDB no responde. Modo fallback (solo archivos locales).")
    else:
        print("[TSUKOYOMI-IA] mongodb_uri no configurado. Modo local.")

    app.state.mongo_disponible = config.mongodb_uri != "" and (await ping() if config.mongodb_uri else False)

    # Carga de motores de IA en el estado global
    app.state.nlp_service = NLPService()
    
    mlp_model = RedMediacionMLP()
    seed_path = Path("app/models/mlp_seed.pt")
    if seed_path.exists():
        mlp_model.load_state_dict(torch.load(str(seed_path), map_location="cpu"))
        print("[TSUKOYOMI-IA] Pesos seed cargados desde app/models/mlp_seed.pt")
    mlp_model.eval()
    app.state.modelo_mlp = mlp_model

    app.state.traductor = TraductorSemanticoV4()

    print("[TSUKOYOMI-IA] Todos los motores cargados correctamente.\n")
    yield

app = FastAPI(
    title="Tsukoyomi-IA", 
    description="Sistema Modular de Simulación de Fricción Social",
    version="3.0",
    lifespan=lifespan
)

# Permite que el navegador (Frontend) hable con esta API sin problemas de seguridad.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración para servir la SPA de React en /frontend/dist
dist_dir = Path("frontend/dist")
if dist_dir.exists():
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

@app.get("/health")
async def health():
    from app.db.mongo import ping
    mongo_ok = await ping() if config.mongodb_uri else False
    return {
        "status": "ok",
        "mongodb": "conectado" if mongo_ok else "no disponible",
        "version": "3.0"
    }

@app.post("/simular-friccion", response_model=RespuestaSimulacion)
async def simular_friccion(peticion: PeticionSimulacion):
    """
    Orquesta el pipeline de procesamiento A -> B -> C -> D.
    """
    try:
        nlp = app.state.nlp_service
        mlp = app.state.modelo_mlp
        rag = app.state.traductor

        latencias = {}

        # FASE A: Análisis NLP de Contexto Social
        t_nlp_start = time.time()
        # FIX: Antes ignorábamos el escenario en el primer mensaje. Ahora lo analizamos.
        texto_a_analizar = peticion.escenario if "Inicia la conversación." in peticion.texto_usuario else peticion.texto_usuario
        
        # Validación de seguridad: si el escenario está vacío, usamos valores neutros.
        if not texto_a_analizar.strip():
            soc_a, soc_p, soc_u, soc_v = 0.1, 0.0, 0.1, 0.5
        else:
            soc_a, soc_p, soc_u, soc_v = await nlp.extraer_metricas_sociales(app.state.llm_router, texto_a_analizar)
        
        fase_a = PayloadFaseA(soc_A=soc_a, soc_P=soc_p, soc_U=soc_u, soc_V=soc_v)
        latencias["FASE_A_NLP"] = round(time.time() - t_nlp_start, 3)

        # BIFURCACIÓN: modo consejo salta MLP y RAG (pipeline corto)
        if peticion.modo == "consejo":
            tracker = None
            prediccion = PrediccionFriccion(0.0, 0.0, 0.0, 0.0)
            tacticas_nombres = []
            tacticas_ids = []
            has_hist = len(peticion.historial) > 0
            prompt_final = rag.ensamblar_consejo(peticion.escenario, fase_a, tiene_historial=has_hist)
            latencias["modo_efectivo"] = "consejo_directo"
        else:
            # Recuperar estado de tensión del request
            tracker = TensionTracker.from_dict(peticion.tension_state)
            
            # Analizar el texto del usuario y actualizar tracker
            analisis_tension = tracker.analizar_respuesta_usuario(peticion.texto_usuario)
            tracker.actualizar_tension(analisis_tension)

            # FASE B: Predicción de Fricción vía MLP (Biometría)
            t_mlp_start = time.time()
            metadatos_dict = peticion.metadatos.model_dump()
            fase_b = PayloadFaseB(**metadatos_dict)
            prediccion_base = predecir(mlp, fase_a, fase_b)
            
            # Ajustar la predicción base escalando la fricción
            prediccion = tracker.modificar_prediccion(prediccion_base)
            
            latencias["FASE_B_MLP"] = round(time.time() - t_mlp_start, 3)
            print(f"\n[MLP-DEBUG] Input Biometría: {metadatos_dict}")
            print(f"[MLP-DEBUG] Fricción Base: {prediccion_base.to_dict()}")
            print(f"[MLP-DEBUG] Fricción Ajustada (Tensión {tracker.nivel_tension:.2f}): {prediccion.to_dict()}\n")

            # FASE C: Selección vectorial de táctica (top_k=1)
            t_rag_start = time.time()
            prompt_final, tacticas_nombres, tacticas_ids = rag.traducir(
                peticion.modo, prediccion, fase_a, peticion.escenario
            )
            latencias["FASE_C_VECTORIAL"] = round(time.time() - t_rag_start, 3)
            latencias["modo_efectivo"] = "simulador_completo"

        # FASE D: Generación de Respuesta con LLM ROUTER
        t_llm_start = time.time()
        
        # FIX: Uso de instancias reales MensajeHistorial en lugar de dicts
        historial_formateado = []
        if peticion.escenario and peticion.modo != "consejo":
            historial_formateado.append(MensajeHistorial(role="user", content=f"CONTEXTO DEL ESCENARIO (No respondas, solo tenlo en cuenta): {peticion.escenario}"))
            historial_formateado.append(MensajeHistorial(role="model", content="Entendido. Mantendré este contexto."))

        # Mantener los objetos MensajeHistorial Pydantic sin convertirlos a dict
        for msg in peticion.historial:
            historial_formateado.append(msg)

        try:
            respuesta_texto, modelo_usado = await app.state.llm_router.llamar_llm(
                sys_prompt=prompt_final,
                user_text=peticion.texto_usuario,
                historial=historial_formateado
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
            
        latencias["FASE_D_LLM"] = round(time.time() - t_llm_start, 3)
        # FIX: Evitamos sumar el nombre del modelo (str) a las latencias numéricas
        solo_numeros = [v for v in latencias.values() if isinstance(v, (int, float))]
        # Log of latencies (Solo tiempos y metadatos técnicos)
        try:
            log_file = Path("app/data/latency_logs.jsonl")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({"timestamp": datetime.now().isoformat(), "latencias": latencias, "modo": peticion.modo}, ensure_ascii=False) + "\n")
            
            if app.state.mongo_disponible:
                from app.db.mongo import col_latencias
                await col_latencias().insert_one({
                    "timestamp": datetime.now().isoformat(),
                    "latencias": latencias,
                    "modo": peticion.modo
                })
        except:
            pass

        return RespuestaSimulacion(
            respuesta_bot=respuesta_texto,
            friccion_calculada=prediccion.to_dict(),
            contexto_nlp_extraido={
                "soc_A": round(fase_a.soc_A, 3), "soc_P": round(fase_a.soc_P, 3),
                "soc_U": round(fase_a.soc_U, 3), "soc_V": round(fase_a.soc_V, 3),
            },
            tacticas_usadas=tacticas_nombres,
            id_tacticas_usadas=tacticas_ids,
            prompt_inyectado=prompt_final,
            latencias=latencias,
            modelo_utilizado=modelo_usado,
            tension_state=tracker.to_dict() if tracker else None
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def store_feedback(feedback: FeedbackSession):
    """
    Persiste el feedback del usuario en un dataset JSONL local y MongoDB Atlas.
    """
    try:
        feedback_dict = feedback.model_dump()
        feedback_dict["timestamp"] = datetime.now().isoformat()
        
        # Escritura en MongoDB si está disponible
        if app.state.mongo_disponible:
            from app.db.mongo import col_sesiones
            await col_sesiones().insert_one({**feedback_dict})
            
        # Escritura local siempre (backup y compatibilidad)
        data_dir = Path("app/data")
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / "dataset_interacciones.jsonl", "a", encoding="utf-8") as f:
            # Eliminar _id si mongo lo añadió al diccionario original mutándolo
            feedback_limpio = {k: v for k, v in feedback_dict.items() if k != "_id"}
            f.write(json.dumps(feedback_limpio, ensure_ascii=False) + "\n")
            
        return {"status": "success", "message": "Feedback recibido correctamente"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando feedback: {str(e)}")

@app.post("/detectar-contexto-visual")
async def detectar_contexto_visual(req: VisionUploadRequest):
    """ Toma un base64, lo pasa a Gemini Vision y retorna el JSON extraído. """
    try:
        t0 = time.time()
        vision: VisionService = app.state.vision_service
        router: LLMRouter = app.state.llm_router
        
        json_extraido = await vision.extraer_contexto_visual(router, req.imagen_base64)
        
        if "error" in json_extraido:
            # Propagamos el Fallback de Rate Limit limpiamente sin errores HTTP
            return json_extraido
            
        print(f"[VISION] Contexto extraído en {round(time.time() - t0, 2)}s.")
        return json_extraido
    except Exception as e:
        # Enviar aviso seguro al front para fallback natural
        return {"error": "fallo_critico", "fallback_sugerido": True, "detalle": str(e)}

# --- CATCH-ALL ROUTE PARA REACT FRONTEND ---
@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    if not dist_dir.exists():
        return {"message": "Frontend no compilado. Ejecuta 'npm run build' en /frontend."}
    
    # Primero intentar resolver el archivo exacto solicitado
    file_target = dist_dir / full_path
    if file_target.is_file():
        return FileResponse(file_target)
    
    # Fallback al index.html para que React Router (y Vite) manejen la ruta
    return FileResponse(dist_dir / "index.html")
