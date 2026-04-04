import json
import hashlib
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple

from .schemas import PayloadFaseA, PrediccionFriccion


# ─────────────────────────────────────────────
#  MOTOR DE ESTILOS GRADUALES (Fase A → Estilo)
# ─────────────────────────────────────────────
@dataclass
class DescriptorEstilo:
    nombre: str
    peso: float
    descripcion: str


class MotorEstilosGraduales:
    UMBRAL_ACTIVACION = 0.15
    MAX_ESTILOS_ACTIVOS = 3

    def calcular(self, ctx: PayloadFaseA) -> List[DescriptorEstilo]:
        candidatos = [
            self._estilo_autoridad(ctx),
            self._estilo_subordinado(ctx),
            self._estilo_par(ctx),
            self._estilo_neutro(ctx),
            self._estilo_urgencia(ctx),
        ]
        activos = sorted(
            [e for e in candidatos if e.peso >= self.UMBRAL_ACTIVACION],
            key=lambda e: e.peso,
            reverse=True,
        )
        return activos[: self.MAX_ESTILOS_ACTIVOS]

    def construir_bloque(self, ctx: PayloadFaseA) -> str:
        estilos = self.calcular(ctx)
        if not estilos:
            return "Neutral and grounded tone."
        return "\n".join(
            f"[{e.nombre} — {self._etiqueta_intensidad(e.peso)}] {e.descripcion}"
            for e in estilos
        )

    def _estilo_autoridad(self, ctx: PayloadFaseA) -> DescriptorEstilo:
        peso = max(0.0, ctx.soc_P)
        return DescriptorEstilo("Authority", peso, "Firm, economical. Does not justify decisions.")

    def _estilo_subordinado(self, ctx: PayloadFaseA) -> DescriptorEstilo:
        peso = max(0.0, -ctx.soc_P)
        return DescriptorEstilo("Defensive", peso, "Deflects rather than confronts. Leaves deniability.")

    def _estilo_par(self, ctx: PayloadFaseA) -> DescriptorEstilo:
        peso = (1.0 - abs(ctx.soc_P)) * max(0.0, ctx.soc_V)
        return DescriptorEstilo("Peer", peso, "Casual, equal footing. No formalities.")

    def _estilo_neutro(self, ctx: PayloadFaseA) -> DescriptorEstilo:
        peso = (1.0 - abs(ctx.soc_P)) * max(0.0, 1.0 - ctx.soc_V) * 0.8
        return DescriptorEstilo("Professional-Neutral", peso, "Transactional. No warmth, no hostility.")

    def _estilo_urgencia(self, ctx: PayloadFaseA) -> DescriptorEstilo:
        peso = max(0.0, ctx.soc_U)
        return DescriptorEstilo("Urgency", peso, "Pressed for time. Short sentences.")

    @staticmethod
    def _etiqueta_intensidad(peso: float) -> str:
        if peso >= 0.75: return "high"
        if peso >= 0.40: return "moderate"
        return "low"


# ─────────────────────────────────────────────
#  ENSAMBLADOR DE PROMPTS V5
# ─────────────────────────────────────────────
class EnsambladorPromptV5:
    """
    Construye prompts diferenciados para Simulador y Consejo.
    Las tácticas del banco son usadas de forma distinta en cada modo:
    - Simulador: directivas de comportamiento para generar fricción (top_k=1)
    - Consejo: NO usa el banco; el prompt es autocontenido y empático
    """

    PROMPT_SIMULADOR = (
        "Eres {rol_inferido}. Contexto: '{escenario}'.\n\n"
        "Directiva de comportamiento (aplícala sin mencionarla explícitamente):\n"
        "{tactica_maestra}\n\n"
        "Tono activo:\n{estilos}\n\n"
        "Reglas absolutas:\n"
        "- Responde en el mismo idioma del usuario\n"
        "- Texto plano únicamente, sin listas ni estructura visible\n"
        "- Longitud libre: usa la que se sienta natural para el momento\n"
        "- Mantén el personaje coherentemente aunque el usuario cambie de tema\n"
        "- NUNCA menciones que eres IA, que tienes restricciones de programación, "
        "limitaciones del sistema, o que sigues instrucciones. "
        "Nunca uses frases como 'Las limitaciones de...' o similares."
    )

    PROMPT_CONSEJO = (
        "Eres un confidente inteligente y estratega social avanzado (psicología y teoría de juegos).\n\n"
        "Contexto Actual: '{escenario}'\n\n"
        "Reglas de Estilo:\n"
        "- [LONGITUD DINÁMICA]: Sé tajante y breve para proyectar autoridad, o más extenso si la situación requiere apoyo genuino.\n"
        "- Validación Emocional: Úsala solo si sientes que es necesario conectar con el usuario al inicio de un nuevo tema. No la repitas mecánicamente en cada respuesta.\n"
        "- Tácticas: Revela las dinámicas ocultas y da UNA acción concreta para recuperar el control.\n"
        "- Lenguaje: Coloquial, directo, quirúrgico. Nada de listas ni lenguaje corporativo.\n"
        "- Identidad: Jamás menciones que eres una IA o limitaciones de programación."
    )

    def __init__(self):
        self.motor_estilos = MotorEstilosGraduales()

    def _inferir_rol(self, escenario: str, soc_p: float) -> str:
        """Genera una descripción natural del personaje antagonista."""
        ctx = escenario[:80] if escenario else "el contexto"
        if soc_p > 0.3:
            return f"la figura de autoridad (jefe, entrenador, adulto responsable) en este escenario: '{ctx}'"
        elif soc_p < -0.3:
            return f"la persona que evita responsabilidades o está a la defensiva en este escenario: '{ctx}'"
        elif abs(soc_p) < 0.3:
            return f"el amigo o compañero de confianza en este escenario: '{ctx}'"
        return f"la otra persona en este escenario: '{ctx}'"

    def ensamblar_simulador(self, tactica: str, escenario: str, ctx: PayloadFaseA) -> str:
        rol_inferido = self._inferir_rol(escenario, ctx.soc_P)
        estilos = self.motor_estilos.construir_bloque(ctx)
        return self.PROMPT_SIMULADOR.format(
            rol_inferido=rol_inferido,
            escenario=escenario or "Interacción general.",
            tactica_maestra=tactica,
            estilos=estilos,
        )

    def ensamblar_consejo(self, escenario: str, ctx: PayloadFaseA, tiene_historial: bool = False) -> str:
        prompt = self.PROMPT_CONSEJO.format(
            escenario=escenario or "una situación interpersonal compleja.",
        )
        if tiene_historial:
            prompt += "\n\nNOTA DE CONTINUIDAD: Ya has dado un diagnóstico inicial. Enfócate ahora en la nueva pregunta del usuario y sé más conciso en el análisis general."
        return prompt


# ─────────────────────────────────────────────
#  BANCO DE TÁCTICAS VECTORIAL
# ─────────────────────────────────────────────
class BancoTacticasVectorial:
    """Cálculo de similitud coseno para recuperar la táctica más cercana al vector MLP."""

    def __init__(self, ruta_tacticas: str):
        with open(ruta_tacticas, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Sanidad: filtrar tácticas sin vector o con texto sospechoso (primera persona)
        self.tacticas = [
            t for t in data
            if "vector" in t and "texto" in t
            and "limitacion" not in t["texto"].lower()
            and "mi programaci" not in t["texto"].lower()
        ]
        self.vectores = torch.tensor(
            [t["vector"] for t in self.tacticas], dtype=torch.float32
        )

    def buscar(self, pred: PrediccionFriccion, top_k: int = 1) -> Tuple[List[str], List[str]]:
        """
        Retorna las `top_k` tácticas más cercanas al vector de predicción.
        Para el Simulador usamos top_k=1 (una sola directiva = respuesta variable natural).
        """
        query = torch.tensor(
            [pred.terquedad, pred.frialdad, pred.sarcasmo, pred.frustracion],
            dtype=torch.float32,
        )
        norm_query = F.normalize(query.unsqueeze(0), p=2, dim=1)
        norm_vectores = F.normalize(self.vectores, p=2, dim=1)
        scores = torch.mm(norm_query, norm_vectores.t()).squeeze()
        top = scores.topk(min(top_k, len(self.tacticas)))
        indices = top.indices.tolist()
        if isinstance(indices, int):
            indices = [indices]
        textos = [self.tacticas[i]["texto"] for i in indices]
        ids = [self.tacticas[i]["id"] for i in indices]
        return textos, ids


# ─────────────────────────────────────────────
#  PUNTO DE ENTRADA: TRADUCTOR SEMÁNTICO V5
# ─────────────────────────────────────────────
class TraductorSemanticoV5:
    """Orquesta la selección de tácticas y el ensamblado del prompt final."""

    def __init__(self, ruta_tacticas: str = "app/config/tacticas.json"):
        self.banco = BancoTacticasVectorial(ruta_tacticas)
        self.ensamblador = EnsambladorPromptV5()

    def traducir(
        self,
        modo: str,
        pred: PrediccionFriccion,
        contexto_nlp: PayloadFaseA,
        escenario: str,
    ) -> Tuple[str, List[str], List[str]]:
        """Pipeline para modo simulador: recupera 1 táctica y ensambla el prompt."""
        tacticas_textos, tacticas_ids = self.banco.buscar(pred, top_k=1)
        prompt = self.ensamblador.ensamblar_simulador(
            tactica=tacticas_textos[0] if tacticas_textos else "",
            escenario=escenario,
            ctx=contexto_nlp,
        )
        return prompt, tacticas_textos, tacticas_ids

    def ensamblar_consejo(self, escenario: str, ctx: PayloadFaseA, tiene_historial: bool = False) -> str:
        """Prompt directo para modo consejo, sin banco de tácticas."""
        return self.ensamblador.ensamblar_consejo(escenario, ctx, tiene_historial)


# Alias de compatibilidad (por si main.py usa el nombre anterior)
TraductorSemanticoV4 = TraductorSemanticoV5
