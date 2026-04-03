import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from .schemas import PayloadFaseA, PrediccionFriccion

@dataclass
class DescriptorEstilo:
    """ Encapsula la descripción de un estilo y su peso calculado. """
    nombre: str
    peso: float
    descripcion: str

class MotorEstilosGraduales:
    """ Calcula estilos con pesos continuos a partir de los ejes sociales. """
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
        if not estilos: return "Neutral and direct tone."
        lineas = []
        for e in estilos:
            intensidad = self._etiqueta_intensidad(e.peso)
            lineas.append(f"[{e.nombre} — {intensidad}] {e.descripcion}")
        return "\n".join(lineas)

    def _estilo_autoridad(self, ctx: PayloadFaseA) -> DescriptorEstilo:
        peso = max(0.0, ctx.soc_P)
        return DescriptorEstilo("Authority", peso, "You speak as someone who does not need to justify their decisions. Firm and economical.")

    def _estilo_subordinado(self, ctx: PayloadFaseA) -> DescriptorEstilo:
        peso = max(0.0, -ctx.soc_P)
        return DescriptorEstilo("Defensive", peso, "You deflect rather than confront. Leave room for plausible deniability.")

    def _estilo_par(self, ctx: PayloadFaseA) -> DescriptorEstilo:
        peso = (1.0 - abs(ctx.soc_P)) * max(0.0, ctx.soc_V)
        return DescriptorEstilo("Peer", peso, "Speak as an equal. Casual and direct. No formalities.")

    def _estilo_neutro(self, ctx: PayloadFaseA) -> DescriptorEstilo:
        peso = (1.0 - abs(ctx.soc_P)) * max(0.0, 1.0 - ctx.soc_V) * 0.8
        return DescriptorEstilo("Professional-Neutral", peso, "Standard, transactional tone. No warmth, no hostility.")

    def _estilo_urgencia(self, ctx: PayloadFaseA) -> DescriptorEstilo:
        peso = max(0.0, ctx.soc_U)
        return DescriptorEstilo("Urgency", peso, "Pressed for time. Short sentences. No pleasantries.")

    @staticmethod
    def _etiqueta_intensidad(peso: float) -> str:
        if peso >= 0.75: return "high"
        if peso >= 0.40: return "moderate"
        return "low"

class EnsambladorPromptV4:
    """ Construye el system prompt con jerarquía clara. """
    RESTRICCIONES_ABSOLUTAS = (
        "ABSOLUTE CONSTRAINTS:\n"
        "- Respond in the SAME language as the user (Spanish/English).\n"
        "- BE CONCISE but avoid being robotic or overly structured. No numbered lists unless necessary.\n"
        "- Natural flow is priority over sentence count. Avoid counting sentences.\n"
        "- Plain text only. No greetings or sign-offs."
    )

    ROL_SIMULADOR = "ROLE: Identify the other party in '{escenario}' and become them. React naturally to the user's input to generate realistic social friction. Do not provide a lecture, just stay in character."
    ROL_CONSEJO = "ROLE: Social Intelligence Mentor. Briefly deconstruct the other person's 'move' and give a concrete, human suggestion on how to respond. Be direct, street-smart, and empathetic."

    PLANTILLA = (
        "{restricciones}\n\n"
        "CONTEXT: '{escenario}'\n"
        "{rol}\n\n"
        "STYLE:\n{estilos}\n\n"
        "{bloque_tacticas}"
    )

    def __init__(self):
        self.motor_estilos = MotorEstilosGraduales()

    def ensamblar(self, modo: str, tacticas: List[str], escenario: str, ctx: PayloadFaseA) -> str:
        rol = self.ROL_CONSEJO if modo == "consejo" else self.ROL_SIMULADOR
        estilos = self.motor_estilos.construir_bloque(ctx)
        str_tacticas = "\n".join(f"- {t}" for t in tacticas)
        
        if modo == "consejo":
            bloque_tacticas = f"ANALYSIS (The other person is currently using these tactics):\n{str_tacticas}\n\nINSTRUCTION: Deconstruct these tactics for the user and advise them how to counter effectively."
        else:
            bloque_tacticas = f"DIRECTIVES (Follow these to generate friction):\n{str_tacticas}"

        return self.PLANTILLA.format(
            restricciones=self.RESTRICCIONES_ABSOLUTAS,
            escenario=escenario or "General interaction.",
            rol=rol,
            estilos=estilos,
            bloque_tacticas=bloque_tacticas,
        )

class BancoTacticasVectorial:
    """ Realiza el cálculo de distancias sobre los vectores ADN de las tácticas. """
    def __init__(self, ruta_tacticas: str):
        with open(ruta_tacticas, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.tacticas = data
        self.vectores = torch.tensor([t["vector"] for t in data], dtype=torch.float32)

    def buscar(self, pred: PrediccionFriccion, modo: str = "simulador", top_k: int = 3) -> Tuple[List[str], List[str]]:
        query = torch.tensor([pred.terquedad, pred.frialdad, pred.sarcasmo, pred.frustracion])
        
        # AJUSTE DE TUNING: Si es modo consejo, penalizamos el sarcasmo para evitar mentoría cínica
        if modo == "consejo":
            # Reducimos el peso del sarcasmo en la query para alejarnos de esas tácticas
            query[2] *= 0.3 
            # Potenciamos la terquedad (para que el consejo sea firme) pero bajamos la frialdad
            query[1] *= 0.7

        norm_query = F.normalize(query.unsqueeze(0), p=2, dim=1)
        norm_vectores = F.normalize(self.vectores, p=2, dim=1)
        scores = torch.mm(norm_query, norm_vectores.t()).squeeze()
        top = scores.topk(min(top_k, len(self.tacticas)))
        indices = top.indices.tolist()
        textos = [self.tacticas[i]["texto"] for i in indices]
        ids = [self.tacticas[i]["id"] for i in indices]
        return textos, ids

class TraductorSemanticoV4:
    """ Punto de entrada para la selección de tácticas vía Álgebra Lineal. """
    def __init__(self, ruta_tacticas: str = "app/config/tacticas.json"):
        self.banco = BancoTacticasVectorial(ruta_tacticas)
        self.ensamblador = EnsambladorPromptV4()

    def traducir(self, modo: str, pred: PrediccionFriccion, contexto_nlp: PayloadFaseA, escenario: str) -> Tuple[str, List[str], List[str]]:
        tacticas_textos, tacticas_ids = self.banco.buscar(pred, modo=modo)
        prompt = self.ensamblador.ensamblar(modo, tacticas_textos, escenario, contexto_nlp)
        return prompt, tacticas_textos, tacticas_ids
