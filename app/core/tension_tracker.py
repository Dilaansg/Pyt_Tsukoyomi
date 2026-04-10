import math
from app.core.schemas import PrediccionFriccion

class TensionTracker:
    def __init__(self, nivel_tension: float = 0.5, historial_cedidas: int = 0, historial_mantuvo: int = 0, turno_actual: int = 0):
        self.nivel_tension = float(nivel_tension)
        self.historial_cedidas = int(historial_cedidas)
        self.historial_mantuvo = int(historial_mantuvo)
        self.turno_actual = int(turno_actual)
        
        self.palabras_ceder = [
            "ok", "entiendo", "tienes razón", "tienes razon", "sí, claro", 
            "si, claro", "perdón", "perdon", "disculpa", "de acuerdo", 
            "lo siento", "vale"
        ]
        self.palabras_escalar = [
            "inaceptable", "exijo", "es un abuso", "basta", "harto", "injusto", 
            "maleducado", "no me importa", "cállate", "callate"
        ]

    def analizar_respuesta_usuario(self, texto: str) -> dict:
        texto_lower = texto.lower()
        cedio = any(p in texto_lower for p in self.palabras_ceder)
        
        # Heurística de escalada: signos de exclamación múltiples, mayúsculas o palabras agresivas
        escalo = any(p in texto_lower for p in self.palabras_escalar)
        exclamaciones = texto.count("!") >= 2
        
        # Verificar mayúsculas (ignorando los que no son caracteres del alfabeto)
        letras_mayus = sum(1 for c in texto if c.isupper())
        letras_minus = sum(1 for c in texto if c.islower())
        mayusculas = (letras_mayus > letras_minus) and len(texto) > 10
        
        if exclamaciones or mayusculas:
            escalo = True
            cedio = False # Si hay ambigüedad, la escalada toma precedencia
            
        cambio_tema = len(texto.strip()) < 20 or ("?" in texto and not cedio and not escalo)
        
        return {
            "cedio": cedio,
            "escalo": escalo,
            "cambio_tema": cambio_tema
        }

    def actualizar_tension(self, analisis: dict) -> float:
        self.turno_actual += 1
        
        if analisis["cedio"]:
            self.nivel_tension += 0.15
            self.historial_cedidas += 1
        elif analisis["escalo"]:
            self.nivel_tension -= 0.10
        else:
            self.nivel_tension -= 0.05
            self.historial_mantuvo += 1
            
        # Floor y Cap
        self.nivel_tension = max(0.0, min(1.0, self.nivel_tension))
        return self.nivel_tension

    def modificar_prediccion(self, pred: PrediccionFriccion) -> PrediccionFriccion:
        t = self.nivel_tension
        nueva_terquedad = pred.terquedad * (1 + t * 0.4)
        nueva_frialdad = pred.frialdad * (1 + t * 0.3)
        
        return PrediccionFriccion(
            terquedad=max(0.0, min(1.0, nueva_terquedad)),
            frialdad=max(0.0, min(1.0, nueva_frialdad)),
            sarcasmo=max(0.0, min(1.0, float(pred.sarcasmo))),    # Se mantienen sin modificar por tensión
            frustracion=max(0.0, min(1.0, float(pred.frustracion)))
        )

    def to_dict(self) -> dict:
        return {
            "nivel_tension": float(self.nivel_tension),
            "historial_cedidas": int(self.historial_cedidas),
            "historial_mantuvo": int(self.historial_mantuvo),
            "turno_actual": int(self.turno_actual)
        }

    @classmethod
    def from_dict(cls, data: dict):
        if not data:
            return cls()
        return cls(
            nivel_tension=data.get("nivel_tension", 0.5),
            historial_cedidas=data.get("historial_cedidas", 0),
            historial_mantuvo=data.get("historial_mantuvo", 0),
            turno_actual=data.get("turno_actual", 0)
        )
