from pydantic import BaseModel, Field
from typing import List, Optional

class MetadatosJS(BaseModel):
    """ Metadatos biométricos y de comportamiento de escritura del cliente. """
    tiempo_escritura_segundos: float = 0.0
    teclas_borrado:            int = 0
    pulsaciones_totales:       int = 0
    ratio_duda:                float = 0.0
    copy_paste_detectado:      bool = False
    longitud_caracteres:       int = 0
    edad_usuario:              int = Field(default=16, ge=10, le=100)

class MensajeHistorial(BaseModel):
    """ Item individual del historial de chat (OpenAI/Gemini Format). """
    role:    str 
    content: str

class PeticionSimulacion(BaseModel):
    """ Esquema de entrada para el endpoint de simulación de fricción. """
    modo:          str = Field(default="simulador") # "simulador" o "consejo"
    escenario:     str = Field(default="")
    texto_usuario: str = Field(min_length=1, max_length=2000)
    metadatos:     MetadatosJS
    historial:     List[MensajeHistorial] = []

class RespuestaSimulacion(BaseModel):
    """ Datos proyectados de salida hacia el cliente frontend. """
    respuesta_bot:         str
    friccion_calculada:    dict 
    contexto_nlp_extraido: dict 
    tacticas_usadas:       List[str]
    id_tacticas_usadas:    List[str] = [] # IDs para mapear feedback a vectores
    prompt_inyectado:      str 
    latencias:             dict = {}
    modelo_utilizado:      str = ""

class FeedbackTactica(BaseModel):
    """ Feedback específico para una táctica para el ajuste de pesos. """
    id_tactica: str
    efectiva:   bool # ¿Se sintió natural/incómoda?

class FeedbackSession(BaseModel):
    """ Datos de retroalimentación enviados por el usuario al finalizar. """
    escenario:      str
    modo:           str
    historial:      List[MensajeHistorial]
    puntuacion:     int = Field(ge=1, le=5)
    comentario:     Optional[str] = ""
    # Feedback granular para la Fase 3 (ML Dinámico)
    tacticas_feedback: List[FeedbackTactica] = []
    timestamp:      Optional[str] = None

class VisionUploadRequest(BaseModel):
    """ Esquema de envío de captura de chat al endpoint de Visión. """
    imagen_base64: str = Field(description="Captura de chat comprimida en base64 pura sin cabecera MIME.")
