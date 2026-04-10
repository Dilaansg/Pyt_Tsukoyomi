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
        El LLM devuelve valores en rangos variables; normalizamos A/U/V a 0-1.
        Retorna: (soc_A, soc_P, soc_U, soc_V).
        """
        if not texto.strip():
            return 0.1, 0.0, 0.1, 0.5

        clave = self._hash_texto(texto)
        if clave in self._cache:
            c = self._cache[clave]
            return c["soc_A"], c["soc_P"], c["soc_U"], c["soc_V"]

        prompt = f"""Analiza este texto y devuelve 4 métricas psicológicas en JSON.

Texto: '{texto[:400]}'

Métricas (valores float de 0.0 a 1.0 para A/U/V, de -1.0 a 1.0 para P):
- "soc_A": Ansiedad/hostilidad del autor (0=calmo/hostil, 1=ansioso/vulnerable)
- "soc_P": Poder relativo (-1=sumiso/subordinado, 0=igual, 1=autoridad/jefe)
- "soc_U": Urgencia temporal (0=relajado, 1=crisis/urgente)
- "soc_V": Confianza/cercanía (0=distante/formal, 1=amigo íntimo/máxima confianza)

Responde SOLO con el JSON, sin markdown, sin texto adicional."""

        try:
            respuesta, _ = await router.llamar_llm(
                sys_prompt="Eres una máquina de extracción de métricas. Muestra exclusivamente un JSON sin markdown extra.",
                user_text=prompt,
                historial=[]
            )
            
            # Limpiar posible basura antes y después del JSON
            match = re.search(r'\{.*\}', respuesta, re.DOTALL)
            if match:
                datos = json.loads(match.group(0))
                soc_a = float(datos.get("soc_A", 0.1))
                soc_p = float(datos.get("soc_P", 0.0))
                soc_u = float(datos.get("soc_U", 0.1))
                soc_v = float(datos.get("soc_V", 0.5))

                # FIX 4: Normalizar A/U/V de rango -1/+1 a 0/+1
                # Si el LLM devuelve en rango -1..1, lo convertimos.
                # Si ya está en 0..1, la normalización es idempotente (0→0.5 ya no es correcta).
                # Detectamos si es rango -1..1 comprobando si alguno es negativo.
                if soc_a < 0 or soc_u < 0 or soc_v < 0:
                    soc_a = (soc_a + 1.0) / 2.0
                    soc_u = (soc_u + 1.0) / 2.0
                    soc_v = (soc_v + 1.0) / 2.0

                # Clamping de seguridad
                soc_a = max(0.0, min(1.0, soc_a))
                soc_u = max(0.0, min(1.0, soc_u))
                soc_v = max(0.0, min(1.0, soc_v))
                soc_p = max(-1.0, min(1.0, soc_p))
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
