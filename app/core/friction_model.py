import torch
import torch.nn as nn
import torch.nn.functional as F
from .schemas import PayloadFaseA, PayloadFaseB, PrediccionFriccion

class NormalizadorEntrada(nn.Module):
    """ Escala las dimensiones de entrada para estabilizar la inferencia del MLP. """
    def __init__(self):
        super().__init__()
        self.ln_nlp = nn.LayerNorm(4)
        self.ln_js  = nn.LayerNorm(6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nlp        = self.ln_nlp(x[:, 0:4])
        js_num     = self.ln_js(x[:, 4:10])
        copy_paste = x[:, 10:11]
        return torch.cat([nlp, js_num, copy_paste], dim=1)

class BloqueResidual(nn.Module):
    """ Implementa bloques residuales para mitigar el desvanecimiento del gradiente. """
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.ln1  = nn.LayerNorm(dim)
        self.fc2  = nn.Linear(dim, dim)
        self.ln2  = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuo = x
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.drop(x)
        x = self.ln2(self.fc2(x))
        return F.relu(x + residuo)

class RedMediacionMLP(nn.Module):
    """ Arquitectura de red neuronal para la predicción de fricción social. """
    PESOS_COPY_PASTE = torch.tensor(
        [1.0, 1.0, 1.0, 1.0,
         0.2, 0.1, 0.1, 0.1, 0.2, 1.0,
         0.0],
        dtype=torch.float32
    )

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.normalizador = NormalizadorEntrada()
        self.proyeccion = nn.Sequential(
            nn.Linear(11, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(dropout),
        )
        self.res_64     = BloqueResidual(64, dropout)
        self.compresion = nn.Sequential(
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(dropout),
        )
        self.res_32  = BloqueResidual(32, dropout)
        self.cuello  = nn.Sequential(
            nn.Linear(32, 16), nn.LayerNorm(16), nn.ReLU(),
        )
        self.salida = nn.Linear(16, 4)
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 1
        if single: x = x.unsqueeze(0)
        
        flags = x[:, 10]
        pesos   = self.PESOS_COPY_PASTE.to(x.device)
        mascara = flags.unsqueeze(1).bool()
        x = torch.where(mascara, x * pesos.unsqueeze(0), x)

        x = self.normalizador(x)
        x = self.proyeccion(x)
        x = self.res_64(x)
        x = self.compresion(x)
        x = self.res_32(x)
        x = self.cuello(x)
        out = torch.sigmoid(self.salida(x))
        return out.squeeze(0) if single else out

def predecir(modelo: RedMediacionMLP, fase_a: PayloadFaseA, fase_b: PayloadFaseB, device: str = "cpu") -> PrediccionFriccion:
    """ Wrapper de inferencia para el modelo MLP. """
    tensor = torch.tensor([
        fase_a.soc_A, fase_a.soc_P, fase_a.soc_U, fase_a.soc_V,
        fase_b.tiempo_escritura_segundos, float(fase_b.teclas_borrado), float(fase_b.pulsaciones_totales),
        fase_b.ratio_duda, float(fase_b.longitud_caracteres), float(fase_b.edad_usuario),
        1.0 if fase_b.copy_paste_detectado else 0.0,
    ], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        salida = modelo(tensor)
    
    return PrediccionFriccion(
        terquedad=salida[0].item(), frialdad=salida[1].item(),
        sarcasmo=salida[2].item(), frustracion=salida[3].item(),
    )
