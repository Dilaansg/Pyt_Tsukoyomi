from transformers import pipeline

class NLPService:
    """ Servicio de extracción de contexto social vía Zero-Shot Classification. """
    
    ETIQUETAS = [
        "miedo al rechazo o ansiedad social",
        "la persona tiene autoridad o es el jefe",
        "la persona es un subordinado o novato",
        "situación urgente o crisis de tiempo",
        "son amigos cercanos o hay mucha confianza",
    ]

    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"):
        self.clasificador = pipeline("zero-shot-classification", model=model_name)

    def analizar_contexto(self, texto: str) -> dict:
        """ Clasifica el texto según las etiquetas sociales definidas. """
        resultado = self.clasificador(texto, self.ETIQUETAS, multi_label=True)
        return dict(zip(resultado["labels"], resultado["scores"]))

    def extraer_metricas_sociales(self, texto: str) -> tuple:
        """
        Calcula las variables sociales (A, P, U, V).
        Retorna: (Ansiedad, Poder, Urgencia, Valencia).
        """
        scores = self.analizar_contexto(texto)
        
        soc_a = scores["miedo al rechazo o ansiedad social"]
        # El score de poder se normaliza restando la sumisión detectada.
        soc_p = scores["la persona tiene autoridad o es el jefe"] - scores["la persona es un subordinado o novato"]
        soc_u = scores["situación urgente o crisis de tiempo"]
        soc_v = scores["son amigos cercanos o hay mucha confianza"]
        
        return soc_a, soc_p, soc_u, soc_v
