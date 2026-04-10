import json
from app.api.schemas import MensajeHistorial
from app.core.llm_router import LLMRouter

class SessionAnalyzer:
    @staticmethod
    async def analizar_sesion(
        router: LLMRouter, 
        historial: list[MensajeHistorial],
        escenario: str,
        tension_final: float,
        tacticas_usadas: list[str]
    ) -> dict:
        historial_texto = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in historial])
        tacticas_texto = ", ".join(tacticas_usadas) if tacticas_usadas else "Ninguna registrada explícitamente"
        
        prompt = f"""Eres un psicólogo analista experto en dinámica social e interacciones de fricción.
Debes deconstruir la siguiente sesión completa de simulación entre un USUARIO y un ANTAGONISTA (Tsukuyomi).

ESCENARIO DE LA SESIÓN:
{escenario}

TÁCTICAS VECTORIALES DISPARADAS (por IDs):
{tacticas_texto}

NIVEL DE TENSIÓN FINAL DE LA SIMULACIÓN (0.0=Relajado, 1.0=Máxima Agresión):
{tension_final:.2f}

TRANSCRIPCIÓN COMPLETA:
{historial_texto}

Genera un análisis en formato JSON estricto con exactamente esta estructura y claves:
{{
  "momentos_criticos": ["Describe brevemente los turnos donde el usuario cedió sus límites muy rápido o escaló mal de forma pasivo-agresiva. Lista de strings."],
  "patron_dominante": "Un string que resuma en máx 15 palabras el patrón psicológico principal del usuario (ej. Tendencia a disculparse innecesariamente).",
  "fortalezas_observadas": ["Momentos o interacciones donde el usuario mantuvo sus límites con asertividad y claridad. Lista de strings (puede estar vacía)."],
  "tactica_que_mas_afecto": "Analiza qué táctica o aproximación del antagonista hizo tambalear al usuario (string corto).",
  "recomendacion_principal": "Un consejo táctico o de comunicación muy concreto para la próxima iteración del usuario.",
  "puntuacion_asertividad": 7.5
}}

Nota sobre puntuacion_asertividad: El valor debe ser un float entre 0.0 y 10.0 donde 10 es asertividad perfecta y 0 es sumisión total o agresión destructiva.

REGLAS DE FORMATO:
- Responde ÚNICAMENTE con el objeto JSON.
- Asegúrate de que el formato JSON sea completamente sintácticamente válido.
- No uses sintaxis markdown de bloque de código como ```json. Devuelve el raw text del objeto abierto y cerrado con {{ y }}.
"""
        
        respuesta_llm, _ = await router.llamar_llm(
            sys_prompt=prompt,
            user_text="Genera el JSON final de análisis. Aplica todo tu conocimiento psicológico.",
            historial=[]
        )
        
        try:
            # Limpiar posible basura markdown si al LLM se le escapa
            limpio = respuesta_llm.replace("```json", "").replace("```", "").strip()
            # Buscar llave de inicio y fin para evitar texto libre periférico
            start_idx = limpio.find("{")
            end_idx = limpio.rfind("}")
            if start_idx != -1 and end_idx != -1:
                limpio = limpio[start_idx:end_idx+1]
                
            return json.loads(limpio)
        except Exception as e:
            print(f"[SESSION_ANALYZER] Error fatal procesando JSON de LLM: {e}\nRespuesta Cruda: {respuesta_llm}")
            return {
                "momentos_criticos": ["El análisis automático falló al generar la estructura de datos."],
                "patron_dominante": "Análisis Incompleto",
                "fortalezas_observadas": [],
                "tactica_que_mas_afecto": "Desconocida",
                "recomendacion_principal": "Intenta realizar un análisis manual con tu historial.",
                "puntuacion_asertividad": 0.0
            }
