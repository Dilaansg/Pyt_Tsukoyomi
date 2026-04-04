import json
import re
import hashlib

class NLPService:
    """
    Servicio de extracción de contexto social usando el LLMRouter (LLM Inference).
    Reemplaza al antiguo mDeBERTa para mejorar rendimiento de RAM y velocidad.
    Incluye caché simple en memoria.
    """

    def __init__(self):
        self._cache = {}

    def _hash_texto(self, texto: str) -> str:
        return hashlib.md5(texto.strip().lower().encode()).hexdigest()

    async def extraer_metricas_sociales(self, router, texto: str) -> tuple:
        """
        Calcula las variables sociales (A, P, U, V) extrayendo JSON desde el LLM.
        Retorna: (soc_A, soc_P, soc_U, soc_V).
        """
        if not texto.strip():
            return 0.1, 0.0, 0.1, 0.5

        clave = self._hash_texto(texto)
        if clave in self._cache:
            c = self._cache[clave]
            return c["soc_A"], c["soc_P"], c["soc_U"], c["soc_V"]

        prompt = f"""Analiza este texto y extrae 4 métricas de -1.0 a 1.0 basadas en la intención sutil, pragmatismo y hostilidad/empatía:
1. "soc_A" (Empatía/Hostilidad): -1.0 es extremo egoísmo/hostilidad, 1.0 es máxima empatía/apoyo.
2. "soc_P" (Autoridad/Sumisión): -1.0 es extrema sumisión/defensa, 1.0 es autoridad/jefe imbatible.
3. "soc_U" (Urgencia/Calma): -1.0 es extrema pausa/calma, 1.0 es extrema presión de tiempo/urgencia.
4. "soc_V" (Vulnerabilidad): -1.0 es lejanía/formalidad profesional, 1.0 es hermandad/máxima confianza y calidez.

Responde ÚNICAMENTE en JSON válido con esas 4 llaves numéricas.
Texto a clasificar: '{texto}'"""

        try:
            # Mandamos la carga al router. Responde con texto crudo pero se espera JSON.
            # Configuramos un sys_prompt simple asegurando el modo JSON.
            respuesta, _ = await router.llamar_llm(
                sys_prompt="Eres una máquina de extracción de métricas. Muestra exclusivamente un JSON sin markdown extra.",
                user_text=prompt,
                historial=[]
            )
            
            # Limpiar posible basura antes y despues del JSON
            match = re.search(r'\{.*\}', respuesta, re.DOTALL)
            if match:
                datos = json.loads(match.group(0))
                soc_a = float(datos.get("soc_A", 0.0))
                soc_p = float(datos.get("soc_P", 0.0))
                soc_u = float(datos.get("soc_U", 0.0))
                soc_v = float(datos.get("soc_V", 0.0))
            else:
                raise ValueError(f"No JSON found en la respuesta: {respuesta}")

            # Guardar en caché
            self._cache[clave] = {"soc_A": soc_a, "soc_P": soc_p, "soc_U": soc_u, "soc_V": soc_v}
            if len(self._cache) > 256:
                self._cache.pop(next(iter(self._cache)))
                
            return soc_a, soc_p, soc_u, soc_v

        except Exception as e:
            print(f"[NLPService] Error extrayendo métricas con LLM: {e}. Fallback a valores neutros.")
            return 0.1, 0.0, 0.1, 0.5
