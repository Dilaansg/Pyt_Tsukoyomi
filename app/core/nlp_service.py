import hashlib
import torch
from transformers import pipeline
from functools import lru_cache


class NLPService:
    """
    Servicio de extracción de contexto social vía Zero-Shot Classification.
    Incluye caché LRU por hash del escenario para evitar re-inferencia en
    cada turno de conversación (la situación social no cambia turno a turno).
    """

    ETIQUETAS = [
        "miedo al rechazo o ansiedad social",
        "la persona tiene autoridad o es el jefe",
        "la persona es un subordinado o novato",
        "situación urgente o crisis de tiempo",
        "son amigos cercanos o hay mucha confianza",
    ]

    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"):
        self.clasificador = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1,              # CPU explícito — evita búsqueda de CUDA innecesaria
            torch_dtype=torch.float32,
        )
        # Pre-compila las etiquetas para evitar tokenización repetida
        self._etiquetas_tuple = tuple(self.ETIQUETAS)

    def _hash_texto(self, texto: str) -> str:
        return hashlib.md5(texto.strip().lower().encode()).hexdigest()

    def _analizar_interno(self, texto: str) -> dict:
        """Inferencia real contra el modelo."""
        resultado = self.clasificador(
            texto[:512],               # Limitar longitud → inferencia más rápida
            list(self._etiquetas_tuple),
            multi_label=True,
        )
        return dict(zip(resultado["labels"], resultado["scores"]))

    def analizar_contexto(self, texto: str) -> dict:
        """
        Clasifica el texto. Usa caché interna basada en hash MD5 del texto
        para evitar inferencias repetidas del mismo escenario.
        """
        clave = self._hash_texto(texto)
        return self._analizar_cacheado(clave, texto)

    def _analizar_cacheado(self, clave: str, texto: str) -> dict:
        """Wrapper con caché manual (lru_cache no funciona con métodos de instancia directamente)."""
        if not hasattr(self, "_cache"):
            self._cache = {}
        if clave not in self._cache:
            self._cache[clave] = self._analizar_interno(texto)
            # Evitar caché infinito: limitar a 128 entradas
            if len(self._cache) > 128:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[clave]

    def extraer_metricas_sociales(self, texto: str) -> tuple:
        """
        Calcula las variables sociales (A, P, U, V).
        Retorna: (Ansiedad, Poder, Urgencia, Valencia).
        Fuertemente cacheado: el mismo texto siempre produce el mismo vector.
        """
        scores = self.analizar_contexto(texto)

        soc_a = scores["miedo al rechazo o ansiedad social"]
        soc_p = scores["la persona tiene autoridad o es el jefe"] - scores["la persona es un subordinado o novato"]
        soc_u = scores["situación urgente o crisis de tiempo"]
        soc_v = scores["son amigos cercanos o hay mucha confianza"]

        return soc_a, soc_p, soc_u, soc_v
