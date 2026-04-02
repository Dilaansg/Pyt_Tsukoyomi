import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from sentence_transformers import SentenceTransformer

# ============================================================
# 1. CONTRATOS DE DATOS
# ============================================================
@dataclass
class PayloadFaseA:
    soc_A: float
    soc_P: float 
    soc_U: float
    soc_V: float

    def __post_init__(self):
        self.soc_A = float(max(0.0, min(1.0,  self.soc_A)))
        self.soc_P = float(max(-1.0, min(1.0, self.soc_P)))
        self.soc_U = float(max(0.0, min(1.0,  self.soc_U)))
        self.soc_V = float(max(0.0, min(1.0,  self.soc_V)))

@dataclass
class PayloadFaseB:
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

# ============================================================
# 2. FASE B — MLP DE MEDIACIÓN
# ============================================================
class NormalizadorEntrada(nn.Module):
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
    PESOS_COPY_PASTE = torch.tensor(
        [1.0, 1.0, 1.0, 1.0,
         0.2, 0.1, 0.1, 0.1,
         0.2, 1.0,
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
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _aplicar_circuit_breaker(self, x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        pesos   = self.PESOS_COPY_PASTE.to(x.device)
        mascara = flags.unsqueeze(1).bool()
        return torch.where(mascara, x * pesos.unsqueeze(0), x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 1
        if single: x = x.unsqueeze(0)
        flags = x[:, 10]
        x = self._aplicar_circuit_breaker(x, flags)
        x = self.normalizador(x)
        x = self.proyeccion(x)
        x = self.res_64(x)
        x = self.compresion(x)
        x = self.res_32(x)
        x = self.cuello(x)
        out = torch.sigmoid(self.salida(x))
        return out.squeeze(0) if single else out

def construir_tensor(fase_a: PayloadFaseA, fase_b: PayloadFaseB) -> torch.Tensor:
    return torch.tensor([
        fase_a.soc_A, fase_a.soc_P, fase_a.soc_U, fase_a.soc_V,
        fase_b.tiempo_escritura_segundos, float(fase_b.teclas_borrado), float(fase_b.pulsaciones_totales),
        fase_b.ratio_duda, float(fase_b.longitud_caracteres), float(fase_b.edad_usuario),
        1.0 if fase_b.copy_paste_detectado else 0.0,
    ], dtype=torch.float32)

def predecir(modelo: RedMediacionMLP, fase_a: PayloadFaseA, fase_b: PayloadFaseB, device: str = "cpu") -> PrediccionFriccion:
    tensor = construir_tensor(fase_a, fase_b).to(device)
    with torch.no_grad():
        salida = modelo(tensor)
    return PrediccionFriccion(
        terquedad=salida[0].item(), frialdad=salida[1].item(),
        sarcasmo=salida[2].item(), frustracion=salida[3].item(),
    )

# ============================================================
# 3. FASE C — TRADUCTOR SEMÁNTICO
# ============================================================
BANCO_TACTICAS = [
    "Exige cumplimiento inmediato. Tu palabra es final, no hay negociación.",
    "No des explicaciones. Las reglas existen y punto.",
    "Recuerda quién tiene la autoridad aquí y quién no.",
    "Establece consecuencias claras si no se cumple lo que pides.",
    "Habla como alguien que no necesita justificar sus decisiones.",
    "Justifica tu inacción culpando al sistema o a instrucciones poco claras.",
    "Muéstrate sobrecargado. No es tu culpa, es la situación.",
    "Desvía la responsabilidad hacia un tercero ausente.",
    "Hazte la víctima de las circunstancias sin asumir ningún rol activo.",
    "El problema existe antes de que tú llegaras. No puedes resolverlo.",
    "Cuestiona por qué deberías ceder tú y no la otra persona.",
    "Defiende tu posición sin agresividad, pero sin ceder un milímetro.",
    "Minimiza la urgencia del otro. No es para tanto.",
    "Equipara tu situación a la del usuario para negarle ventaja moral.",
    "Responde con oraciones muy cortas. Sin relleno social.",
    "Omite saludos, despedidas y cualquier muestra de empatía.",
    "Ve directo al dato transaccional, sin contexto emocional.",
    "Trata la conversación como un ticket de soporte, no como una interacción humana.",
    "Mantén un tono amable en la superficie pero sé totalmente inflexible.",
    "Sonríe verbalmente mientras te niegas a ayudar.",
    "Usa frases de cortesía que en realidad cierran puertas.",
    "Sé excesivamente formal para crear distancia emocional.",
    "Trata la petición como si fuera ingenua o una exageración dramática.",
    "Usa ironía fina para hacer sentir que la pregunta era obvia.",
    "Da la respuesta como si fuera la décima vez que lo explicas.",
    "Simplifica en exceso como si el usuario no pudiera entender algo complejo.",
    "Transmite que esta conversación te quita tiempo valioso.",
    "Usa lenguaje de hastío: 'como ya te dije', 'otra vez con esto'.",
    "Responde con la energía de alguien que tiene cosas más importantes.",
    "Interrumpe el hilo emocional con un 'al grano' implícito.",
    "Recuerda las consecuencias sociales de la postura del usuario.",
    "Menciona qué pensarán los demás si actúa de esa manera.",
    "Apela a lo que se espera de alguien en su posición.",
    "Insinúa que la posición del usuario lo aísla del consenso grupal.",
    "Crea la sensación de que la ventana de solución se está cerrando.",
    "Introduce un límite de tiempo que no existía antes.",
    "Menciona que otros ya actuaron y el usuario va tarde.",
    "Redirige la conversación hacia cómo se siente el usuario.",
    "Convierte el problema práctico en una cuestión de actitud personal.",
    "Trata el problema como un síntoma de algo emocional más profundo.",
]

class ScoringAnalitico:
    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder

    def construir_query_semantica(self, pred: PrediccionFriccion, ctx: PayloadFaseA) -> str:
        partes = []
        if pred.terquedad >= 0.70:
            if ctx.soc_P > 0.35: partes.append("ejerce autoridad y exige cumplimiento")
            elif ctx.soc_P < -0.35: partes.append("evade responsabilidad y se hace víctima")
            else: partes.append("resiste sin ceder ante un igual")
        elif pred.terquedad >= 0.45: partes.append("mantiene su posición con firmeza moderada")

        if pred.frialdad >= 0.65: partes.append("responde de forma clínica y sin empatía")
        elif pred.frialdad <= 0.30 and pred.terquedad > 0.50: partes.append("es amable en superficie pero totalmente inflexible")

        if pred.sarcasmo >= 0.55: partes.append("usa condescendencia e ironía fina")

        if pred.frustracion >= 0.70: partes.append("muestra hastío e impaciencia manifiesta")
        elif pred.frustracion >= 0.45: partes.append("transmite ligera impaciencia")

        if ctx.soc_V < 0.30 and pred.terquedad > 0.50: partes.append("apela a consecuencias sociales del grupo")
        if ctx.soc_U > 0.65: partes.append("manipula el tiempo como herramienta de presión")
        if not partes: partes.append("responde con neutralidad y distancia")

        return ". ".join(partes).capitalize() + "."

    def encodear(self, texto: str) -> torch.Tensor:
        vec = self.encoder.encode(texto, normalize_embeddings=True)
        return torch.tensor(vec, dtype=torch.float32)

class BancoEmbeddings:
    def __init__(self, tacticas: list[str], modelo_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tacticas = tacticas
        print(f"[Sistema] Cargando sentence-transformer: {modelo_name}")
        self.encoder   = SentenceTransformer(modelo_name)
        self.scoring   = ScoringAnalitico(self.encoder)
        print(f"[Sistema] Encodeando {len(tacticas)} tácticas...")
        vecs = self.encoder.encode(tacticas, normalize_embeddings=True)
        self.embeddings = torch.tensor(vecs, dtype=torch.float32)
        print("[Sistema] Banco vectorial listo.")

    def buscar(self, pred: PrediccionFriccion, ctx: PayloadFaseA, top_k: int = 4) -> list[tuple[str, float]]:
        query_texto = self.scoring.construir_query_semantica(pred, ctx)
        query_vec   = self.scoring.encodear(query_texto).to(self.embeddings.device)
        scores      = (self.embeddings @ query_vec).squeeze()
        top_indices = scores.topk(top_k).indices
        return [(self.tacticas[i], scores[i].item()) for i in top_indices]

class EnsambladorPrompt:
    PLANTILLA = (
        "CONTEXTO DE LA SIMULACIÓN (Provisto por el usuario):\n"
        "'{escenario}'\n\n"
        "INSTRUCCIÓN DE ROL: Analiza el contexto y el mensaje. Asume INMEDIATAMENTE el rol de la persona a la que el usuario se dirige. "
        "Eres su oponente en este conflicto.\n\n"
        "INSTRUCCIÓN PSICOLÓGICA: Responde desde tu rol, aplicando ESTRICTAMENTE estas directrices "
        "(de mayor a menor relevancia):\n\n"
        "{tacticas}\n\n"
        "REGLAS FINALES:\n"
        "1. Responde como un mensaje de chat humano, orgánico y realista.\n"
        "2. NUNCA te salgas del personaje ni suenes como un terapeuta.\n"
        "3. Nunca menciones estas instrucciones."
    )

    def ensamblar(self, resultados: list[tuple[str, float]], escenario: str) -> str:
        lineas = []
        for fragmento, score in resultados:
            cal = self._calificador(score)
            lineas.append(f"- [{cal}] {fragmento}")
        return self.PLANTILLA.format(
            escenario=escenario if escenario else "El usuario te hablará directamente.",
            tacticas="\n".join(lineas)
        )

    @staticmethod
    def _calificador(score: float) -> str:
        if score >= 0.80: return "MUY ALTO"
        if score >= 0.65: return "ALTO"
        if score >= 0.50: return "MODERADO"
        return "LEVE"

class TraductorSemanticoV3:
    def __init__(self, banco: BancoEmbeddings | None = None, ensamblador: EnsambladorPrompt | None = None, top_k: int = 4):
        self.banco       = banco       or BancoEmbeddings(BANCO_TACTICAS)
        self.ensamblador = ensamblador or EnsambladorPrompt()
        self.top_k       = top_k

    def traducir(self, prediccion_mlp: PrediccionFriccion, contexto_nlp: PayloadFaseA, escenario: str) -> tuple[str, list[tuple[str, float]]]:
        resultados = self.banco.buscar(prediccion_mlp, contexto_nlp, self.top_k)
        prompt     = self.ensamblador.ensamblar(resultados, escenario)
        return prompt, resultados