import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ============================================================
# CONTRATO DE DATOS
# ============================================================
@dataclass
class PayloadFaseA:
    soc_A: float  # Ansiedad social     [0.0 - 1.0]
    soc_P: float  # Diferencial poder   [0.0 - 1.0]
    soc_U: float  # Urgencia            [0.0 - 1.0]
    soc_V: float  # Valencia / afinidad [0.0 - 1.0]


@dataclass
class PayloadFaseB:
    tiempo_escritura_segundos: float
    teclas_borrado:            int
    pulsaciones_totales:       int
    ratio_duda:                float
    copy_paste_detectado:      bool
    longitud_caracteres:       int
    edad_usuario:              int    # ← añadido: factor de perfil demográfico


@dataclass
class PrediccionFriccion:
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

    def __str__(self) -> str:
        d = self.to_dict()
        lines = ["=== Predicción de fricción ==="]
        for k, v in d.items():
            bar = "█" * int(v * 20)
            lines.append(f"  {k:<12} {v:.4f}  {bar}")
        return "\n".join(lines)


# ============================================================
# NORMALIZADOR — 11 entradas, 3 dominios
# ============================================================
class NormalizadorEntrada(nn.Module):
    """
    Índices del tensor [batch, 11]:
        [0:4]  → NLP (soc_A, soc_P, soc_U, soc_V)           — BN(4)
        [4:10] → JS numérico (tiempo, borrados, pulsaciones,  — BN(6)
                              ratio_duda, longitud, edad)
        [10]   → copy_paste (booleano 0/1)                    — sin normalizar
    """
    def __init__(self):
        super().__init__()
        self.bn_nlp = nn.BatchNorm1d(4)
        self.bn_js  = nn.BatchNorm1d(6)   # era 5, ahora incluye edad_usuario

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nlp        = self.bn_nlp(x[:, 0:4])
        js_num     = self.bn_js(x[:, 4:10])
        copy_paste = x[:, 10:11]
        return torch.cat([nlp, js_num, copy_paste], dim=1)


# ============================================================
# BLOQUE RESIDUAL
# ============================================================
class BloqueResidual(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.bn1  = nn.BatchNorm1d(dim)
        self.fc2  = nn.Linear(dim, dim)
        self.bn2  = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuo = x
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = self.bn2(self.fc2(x))
        return F.relu(x + residuo)


# ============================================================
# RED NEURONAL DE MEDIACIÓN — v1.1
# ============================================================
class RedMediacionMLP(nn.Module):
    """
    Entradas (11 dimensiones):
        [0]  soc_A
        [1]  soc_P
        [2]  soc_U
        [3]  soc_V
        [4]  tiempo_escritura_segundos
        [5]  teclas_borrado
        [6]  pulsaciones_totales
        [7]  ratio_duda
        [8]  longitud_caracteres
        [9]  edad_usuario               ← factor demográfico imprevisto
        [10] copy_paste_detectado       ← circuit-breaker (booleano)
    """

    PESOS_COPY_PASTE = torch.tensor(
        [1.0, 1.0, 1.0, 1.0,   # NLP — intactos
         0.2, 0.1, 0.1, 0.1,   # JS métricas — contaminadas
         0.2, 1.0,              # longitud conserva algo; edad no está contaminada
         0.0],                  # copy_paste se anula a sí mismo
        dtype=torch.float32
    )

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.normalizador = NormalizadorEntrada()

        self.proyeccion = nn.Sequential(
            nn.Linear(11, 64),   # era 10
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.res_64 = BloqueResidual(64, dropout)

        self.compresion = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.res_32 = BloqueResidual(32, dropout)

        self.cuello = nn.Sequential(
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )

        self.salida = nn.Linear(16, 4)

        self._inicializar_pesos()

    def _inicializar_pesos(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _aplicar_circuit_breaker(
        self, x: torch.Tensor, flags: torch.Tensor
    ) -> torch.Tensor:
        pesos = self.PESOS_COPY_PASTE.to(x.device)
        mascara = flags.unsqueeze(1).bool()
        return torch.where(mascara, x * pesos.unsqueeze(0), x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        copy_paste_flags = x[:, 10]
        x = self._aplicar_circuit_breaker(x, copy_paste_flags)
        x = self.normalizador(x)
        x = self.proyeccion(x)
        x = self.res_64(x)
        x = self.compresion(x)
        x = self.res_32(x)
        x = self.cuello(x)
        x = torch.sigmoid(self.salida(x))

        return x.squeeze(0) if single else x


# ============================================================
# FUNCIÓN DE INFERENCIA
# ============================================================
def predecir(
    modelo: RedMediacionMLP,
    fase_a: PayloadFaseA,
    fase_b: PayloadFaseB,
    device: str = "cpu",
) -> PrediccionFriccion:
    modelo.eval()
    modelo.to(device)

    vector = [
        fase_a.soc_A,
        fase_a.soc_P,
        fase_a.soc_U,
        fase_a.soc_V,
        fase_b.tiempo_escritura_segundos,
        float(fase_b.teclas_borrado),
        float(fase_b.pulsaciones_totales),
        fase_b.ratio_duda,
        float(fase_b.longitud_caracteres),
        float(fase_b.edad_usuario),            # ← posición [9]
        1.0 if fase_b.copy_paste_detectado else 0.0,  # ← posición [10]
    ]

    tensor = torch.tensor(vector, dtype=torch.float32, device=device)

    with torch.no_grad():
        salida = modelo(tensor)

    return PrediccionFriccion(
        terquedad=   salida[0].item(),
        frialdad=    salida[1].item(),
        sarcasmo=    salida[2].item(),
        frustracion= salida[3].item(),
    )


# ============================================================
# FUNCIÓN DE PÉRDIDA
# ============================================================
def perdida_friccion(
    prediccion: torch.Tensor,
    objetivo:   torch.Tensor,
    pesos_dim:  Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy(prediccion, objetivo, reduction="none")
    if pesos_dim is not None:
        bce = bce * pesos_dim.to(prediccion.device)
    return bce.mean()


# ============================================================
# PRUEBA
# ============================================================
if __name__ == "__main__":

    modelo = RedMediacionMLP()
    print(f"Parámetros totales: {sum(p.numel() for p in modelo.parameters()):,}\n")

    casos = [
        {
            "desc": "Situación 1 — confrontación de pares (usuario 16 años)",
            "a": PayloadFaseA(soc_A=0.066, soc_P=0.30, soc_U=0.55, soc_V=0.40),
            "b": PayloadFaseB(38.2, 21, 190, 1.95, False, 97, edad_usuario=16),
        },
        {
            "desc": "Situación 14 — deuda entre amigos (usuario adulto)",
            "a": PayloadFaseA(soc_A=0.20, soc_P=0.654, soc_U=0.35, soc_V=0.55),
            "b": PayloadFaseB(55.0, 34, 280, 2.40, False, 116, edad_usuario=34),
        },
        {
            "desc": "Control — copy_paste activo (circuit-breaker)",
            "a": PayloadFaseA(soc_A=0.50, soc_P=0.50, soc_U=0.50, soc_V=0.50),
            "b": PayloadFaseB(2.1, 0, 3, 0.05, True, 60, edad_usuario=22),
        },
    ]

    for caso in casos:
        pred = predecir(modelo, caso["a"], caso["b"])
        print(f"--- {caso['desc']}")
        print(pred)
        print()