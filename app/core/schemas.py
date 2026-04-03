from dataclasses import dataclass

@dataclass
class PayloadFaseA:
    """ Representación del contexto social (normalizado de 0.0 a 1.0). """
    soc_A: float # Ansiedad
    soc_P: float # Poder (-1.0 a 1.0)
    soc_U: float # Urgencia
    soc_V: float # Valencia (Confianza)

    def __post_init__(self):
        # Clip automático de rangos permitidos
        self.soc_A = float(max(0.0, min(1.0,  self.soc_A)))
        self.soc_P = float(max(-1.0, min(1.0, self.soc_P)))
        self.soc_U = float(max(0.0, min(1.0,  self.soc_U)))
        self.soc_V = float(max(0.0, min(1.0,  self.soc_V)))

@dataclass
class PayloadFaseB:
    """ Datos biométricos y telemétricos capturados (Input de Fase B). """
    tiempo_escritura_segundos: float
    teclas_borrado:            int
    pulsaciones_totales:       int
    ratio_duda:                float
    copy_paste_detectado:      bool
    longitud_caracteres:       int
    edad_usuario:              int

    def __post_init__(self):
        self.tiempo_escritura_segundos = max(0.0, float(self.tiempo_escritura_segundos))
        self.ratio_duda = max(0.0, float(self.ratio_duda))
        self.teclas_borrado = max(0, int(self.teclas_borrado))
        self.pulsaciones_totales = max(0, int(self.pulsaciones_totales))
        self.longitud_caracteres = max(1, int(self.longitud_caracteres))
        self.edad_usuario = max(10, min(100, int(self.edad_usuario)))

@dataclass
class PrediccionFriccion:
    """ Output del motor MLP (4 dimensiones psicológicas). """
    terquedad:   float
    frialdad:    float
    sarcasmo:    float
    frustracion: float

    def to_dict(self) -> dict:
        return {
            "terquedad":   round(self.terquedad,   4),
            "frialdad":    round(self.frialdad,    4),
            "sarcasmo":    round(self.sarcasmo,    4),
            "frustracion": round(self.frustracion, 4),
        }
