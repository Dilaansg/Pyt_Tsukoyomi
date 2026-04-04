import json
import re

class VisionService:
    """
    Orquestador para extracción de metadatos de capturas de chat.
    Usa el router existente (y por ende Gemini 2.5 Flash de forma asilada en RAM).
    """

    VISION_PROMPT = (
        "Estás actuando como un analizador forense de conversaciones sociales avanzado. "
        "Te estoy enviando una captura de pantalla de un chat (Tinder, WhatsApp, Instagram, etc.). "
        "Tu objetivo es leerla y extraer tres componentes clave en FORMATO JSON ESTRICTO:\n\n"
        "1. 'transcripcion_cronologica': Transcribe quién dice qué y en qué orden (Usa 'Usuario' para la persona que tomó la captura y 'El/Ella' para la contraparte).\n"
        "2. 'metadatos_inversion': Dime quién escribió el último mensaje, qué dice el último mensaje, "
        "si hay signos visuales de bajo interés (ej. respuestas frías de una sola palabra, minúsculas, falta de preguntas) "
        "o si detectaste los timestamps de separación entre mensajes.\n"
        "3. 'resumen_escenario': Sintetiza en un solo párrafo claro y coloquial cuál es la dinámica de poder actual, y qué está pasando.\n\n"
        "Responde EXCLUSIVAMENTE con el objeto JSON, usando las llaves 'transcripcion_cronologica', 'metadatos_inversion' y 'resumen_escenario'. Sin markdown inicial."
    )

    async def extraer_contexto_visual(self, router, imagen_base64: str) -> dict:
        """
        Orquesta la llamada al LLMRouter pasando la imagen. Retorna un diccionario.
        """
        respuesta_cruda, _ = await router.llamar_llm(
            sys_prompt=self.VISION_PROMPT,
            user_text="Analiza esta captura siguiendo estrictamente las instrucciones del sistema.",
            historial=[],
            imagen_base64=imagen_base64
        )

        # Manejador de Fallback (Error capturado del Router)
        try:
            datos = json.loads(respuesta_cruda)
            if "error" in datos and datos.get("error") == "rate_limit":
                return datos
        except:
            pass

        # Parseo exitoso
        match = re.search(r'\{.*\}', respuesta_cruda, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                raise ValueError("El JSON extraído por visión es inválido.")
        else:
            raise ValueError("No se encontró estructura JSON en la respuesta visual.")
