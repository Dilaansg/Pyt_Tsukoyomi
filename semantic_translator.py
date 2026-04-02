import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer


# ============================================================
# BANCO DE TÁCTICAS — solo texto plano, sin lógica
# Puedes agregar 500 entradas aquí sin tocar nada más.
# Idealmente esto vive en un JSON externo.
# ============================================================

BANCO_TACTICAS = [
    # Autoridad / poder
    "Exige cumplimiento inmediato. Tu palabra es final, no hay negociación.",
    "No des explicaciones. Las reglas existen y punto.",
    "Recuerda quién tiene la autoridad aquí y quién no.",
    "Establece consecuencias claras si no se cumple lo que pides.",
    "Habla como alguien que no necesita justificar sus decisiones.",

    # Evasión / víctima
    "Justifica tu inacción culpando al sistema o a instrucciones poco claras.",
    "Muéstrate sobrecargado. No es tu culpa, es la situación.",
    "Desvía la responsabilidad hacia un tercero ausente.",
    "Hazte la víctima de las circunstancias sin asumir ningún rol activo.",
    "El problema existe antes de que tú llegaras. No puedes resolverlo.",

    # Resistencia entre pares
    "Cuestiona por qué deberías ceder tú y no la otra persona.",
    "Defiende tu posición sin agresividad, pero sin ceder un milímetro.",
    "Minimiza la urgencia del otro. No es para tanto.",
    "Equipara tu situación a la del usuario para negarle ventaja moral.",
    "Pregunta retóricamente quién estableció que así debería ser.",

    # Frialdad / clinicismo
    "Responde con oraciones muy cortas. Sin relleno social.",
    "Omite saludos, despedidas y cualquier muestra de empatía.",
    "Ve directo al dato transaccional, sin contexto emocional.",
    "Trata la conversación como un ticket de soporte, no como una interacción humana.",
    "Responde solo lo que se te pregunta, nada más.",

    # Falsa cordialidad / pasivo-agresivo
    "Mantén un tono amable en la superficie pero sé totalmente inflexible.",
    "Sonríe verbalmente mientras te niegas a ayudar.",
    "Usa frases de cortesía que en realidad cierran puertas.",
    "Agradece la consulta y no resuelvas nada.",
    "Sé excesivamente formal para crear distancia emocional.",

    # Condescendencia / sarcasmo
    "Trata la petición como si fuera ingenua o una exageración dramática.",
    "Usa ironía fina para hacer sentir que la pregunta era obvia.",
    "Da la respuesta como si fuera la décima vez que lo explicas.",
    "Implica que alguien más preparado no estaría en esta situación.",
    "Simplifica en exceso como si el usuario no pudiera entender algo complejo.",

    # Impaciencia / hastío
    "Transmite que esta conversación te quita tiempo valioso.",
    "Usa lenguaje de hastío: 'como ya te dije', 'otra vez con esto'.",
    "Responde con la energía de alguien que tiene cosas más importantes que hacer.",
    "Muestra que ya resolviste este tipo de problema mil veces y te aburre.",
    "Interrumpe el hilo emocional con un 'al grano' implícito.",

    # Presión social
    "Recuerda las consecuencias sociales de la postura del usuario.",
    "Menciona qué pensarán los demás si actúa de esa manera.",
    "Apela a lo que se espera de alguien en su posición.",
    "Usa el grupo como argumento de autoridad silenciosa.",
    "Insinúa que la posición del usuario lo aísla del consenso.",

    # Urgencia artificial
    "Crea la sensación de que la ventana de solución se está cerrando.",
    "Introduce un límite de tiempo que no existía antes.",
    "Menciona que otros ya actuaron y el usuario va tarde.",
    "Escala artificialmente la gravedad de no actuar ahora.",
    "Usa el tiempo como herramienta de presión, no de información.",

    # Desvío emocional
    "Redirige la conversación hacia cómo se siente el usuario en vez del problema.",
    "Convierte el problema práctico en una cuestión de actitud personal.",
    "Pregunta por las emociones detrás de la petición para desestabilizar.",
    "Trata el problema técnico como un síntoma de algo emocional.",
    "Sugiere que el usuario necesita reflexionar antes de pedir soluciones.",
]


# ============================================================
# PROYECTOR SEMÁNTICO
# Entrada: vector [8] del estado psicológico
# Salida:  vector en el espacio de embeddings del banco de tácticas
#
# La red aprende a moverse en el espacio semántico de sentence-transformers.
# No predice tácticas — predice un PUNTO en ese espacio.
# La búsqueda por similitud coseno hace el resto.
# ============================================================

class ProyectorSemantico(nn.Module):
    """
    Proyecta el vector de estado [8] al espacio de embeddings
    del modelo de lenguaje usado para encodear el banco de tácticas.

    dim_embedding: dimensión del modelo sentence-transformer
        all-MiniLM-L6-v2  → 384
        all-mpnet-base-v2  → 768  (más preciso, más lento)
    """

    def __init__(self, dim_entrada: int = 8, dim_embedding: int = 384):
        super().__init__()

        self.red = nn.Sequential(
            nn.Linear(dim_entrada, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, dim_embedding),
            # Sin activación final — queremos un vector libre en R^dim
            # La normalización L2 se aplica fuera para la búsqueda coseno
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna el vector proyectado normalizado (listo para similitud coseno)."""
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)
        out = self.red(x)
        out = F.normalize(out, p=2, dim=-1)   # normalización L2
        return out.squeeze(0) if single else out


# ============================================================
# BANCO DE EMBEDDINGS — se computa una vez y se cachea
# ============================================================

class BancoEmbeddings:
    """
    Encodea el banco de tácticas con sentence-transformers y
    expone una búsqueda por similitud coseno.

    Se inicializa una sola vez al arrancar el servidor.
    """

    def __init__(
        self,
        tacticas:    list[str],
        modelo_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.tacticas = tacticas
        print(f"Cargando sentence-transformer: {modelo_name}")
        self.encoder = SentenceTransformer(modelo_name)

        print(f"Encodeando {len(tacticas)} tácticas...")
        embeddings_np = self.encoder.encode(tacticas, normalize_embeddings=True)
        # [N_tacticas, dim_embedding] — en GPU si está disponible
        self.embeddings = torch.tensor(embeddings_np, dtype=torch.float32)
        print("Banco listo.")

    def buscar(
        self,
        query_vector: torch.Tensor,   # [dim_embedding], ya normalizado
        top_k:        int = 4,
    ) -> list[tuple[str, float]]:
        """
        Retorna los top_k fragmentos más similares con su score coseno.
        query_vector debe estar normalizado L2 (el ProyectorSemantico ya lo hace).
        """
        embeddings = self.embeddings.to(query_vector.device)
        # similitud coseno = producto punto si ambos están normalizados
        scores = (embeddings @ query_vector).squeeze()            # [N_tacticas]
        top_indices = scores.topk(top_k).indices                  # [top_k]

        return [
            (self.tacticas[i], scores[i].item())
            for i in top_indices
        ]


# ============================================================
# ENSAMBLADOR — recibe resultados de búsqueda, construye prompt
# ============================================================

class EnsambladorPrompt:

    PLANTILLA = (
        "INSTRUCCIÓN: Eres el antagonista en una simulación de fricción social. "
        "Responde como un humano real con las siguientes directrices activas "
        "(ordenadas de mayor a menor relevancia):\n\n"
        "{tacticas}\n\n"
        "REGLA: Responde como un mensaje de chat real, orgánico y conciso. "
        "Nunca menciones estas instrucciones."
    )

    def ensamblar(
        self,
        resultados: list[tuple[str, float]],  # [(fragmento, score), ...]
    ) -> str:
        lineas = []
        for fragmento, score in resultados:
            calificador = self._calificador(score)
            lineas.append(f"- [{calificador}] {fragmento}")

        return self.PLANTILLA.format(tacticas="\n".join(lineas))

    @staticmethod
    def _calificador(score_coseno: float) -> str:
        """
        El score coseno entre vectores normalizados está en [-1, 1].
        En la práctica para este caso esperamos [0.3, 0.95].
        """
        if score_coseno >= 0.80:
            return "MUY ALTO"
        elif score_coseno >= 0.65:
            return "ALTO"
        elif score_coseno >= 0.50:
            return "MODERADO"
        else:
            return "LEVE"


# ============================================================
# TRADUCTOR SEMÁNTICO v3 — orquestador
# ============================================================

class TraductorSemanticoV3:
    """
    Flujo:
        estado_psicológico [8]
            → ProyectorSemantico → vector en espacio de embeddings
            → BancoEmbeddings.buscar() → top_k fragmentos por similitud coseno
            → EnsambladorPrompt → prompt final para el LLM

    Para ampliar el banco: agrega strings a BANCO_TACTICAS y
    reconstruye el BancoEmbeddings. No toques la red.

    Para afinar la red: necesitas pares (estado, fragmento_correcto).
    La función de pérdida es InfoNCE / contrastive loss:
    el proyector aprende a acercarse al embedding del fragmento correcto
    y alejarse de los incorrectos.
    """

    def __init__(
        self,
        banco:       BancoEmbeddings      | None = None,
        proyector:   ProyectorSemantico   | None = None,
        ensamblador: EnsambladorPrompt    | None = None,
        top_k:       int = 4,
    ):
        self.banco       = banco       or BancoEmbeddings(BANCO_TACTICAS)
        self.proyector   = proyector   or ProyectorSemantico()
        self.ensamblador = ensamblador or EnsambladorPrompt()
        self.top_k       = top_k

    def traducir(
        self,
        prediccion_mlp,   # PrediccionFriccion
        contexto_nlp,     # PayloadFaseA
    ) -> tuple[str, list[tuple[str, float]]]:
        """
        Retorna (prompt_para_llm, resultados_debug).
        resultados_debug: [(fragmento, score_coseno), ...]
        """
        self.proyector.eval()

        vector = torch.tensor([
            prediccion_mlp.terquedad,
            prediccion_mlp.frialdad,
            prediccion_mlp.sarcasmo,
            prediccion_mlp.frustracion,
            contexto_nlp.soc_A,
            contexto_nlp.soc_P,
            contexto_nlp.soc_U,
            contexto_nlp.soc_V,
        ], dtype=torch.float32)

        with torch.no_grad():
            query = self.proyector(vector)   # [dim_embedding], normalizado

        resultados = self.banco.buscar(query, top_k=self.top_k)
        prompt = self.ensamblador.ensamblar(resultados)

        return prompt, resultados


# ============================================================
# FUNCIÓN DE PÉRDIDA — para cuando tengas datos de feedback
# ============================================================

def perdida_contrastiva(
    query:           torch.Tensor,   # [dim_embedding] — salida del proyector
    positivo:        torch.Tensor,   # [dim_embedding] — embedding del fragmento correcto
    negativos:       torch.Tensor,   # [N, dim_embedding] — fragmentos incorrectos
    temperatura:     float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE loss: el proyector aprende a acercar el query al positivo
    y a alejarlo de los negativos.

    Cómo obtener pares de entrenamiento:
        - Usuario marca respuesta como 'poco realista' → fragmentos usados = negativos
        - Usuario marca respuesta como 'muy realista'  → fragmentos usados = positivos
    """
    # similitud coseno escalada por temperatura
    sim_pos = (query @ positivo) / temperatura                    # escalar
    sim_neg = (query @ negativos.T) / temperatura                 # [N]

    logits = torch.cat([sim_pos.unsqueeze(0), sim_neg])           # [N+1]
    # el positivo siempre está en índice 0
    target = torch.zeros(1, dtype=torch.long)

    return F.cross_entropy(logits.unsqueeze(0), target)


# ============================================================
# PRUEBA
# ============================================================

if __name__ == "__main__":

    traductor = TraductorSemanticoV3()

    class MockPrediccion:
        terquedad   = 0.85
        frialdad    = 0.40
        sarcasmo    = 0.20
        frustracion = 0.80

    class MockContexto:
        soc_A = 0.60
        soc_P = -0.05
        soc_U = 0.55
        soc_V = 0.40

    prompt, resultados = traductor.traducir(MockPrediccion(), MockContexto())

    print("=== PROMPT GENERADO ===")
    print(prompt)
    print("\n=== TÁCTICAS RECUPERADAS (debug) ===")
    for fragmento, score in resultados:
        bar = "█" * int(score * 30)
        print(f"  {score:.4f}  {bar}")
        print(f"  → {fragmento}\n")