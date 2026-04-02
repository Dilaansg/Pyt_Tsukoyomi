from transformers import pipeline

print("Cargando modelo NLP mDeBERTa...")
clasificador = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

# 1. Las "etiquetas cebo" en lenguaje natural que el modelo sí entiende
ETIQUETAS_NLP = [
    "miedo al rechazo o ansiedad social",       # Para soc_A
    "la persona tiene autoridad o es el jefe",  # Para soc_P (Superior)
    "la persona es un subordinado o novato",    # Para soc_P (Subordinado)
    "situación urgente o crisis de tiempo",     # Para soc_U
    "son amigos cercanos o hay mucha confianza" # Para soc_V
]

def extraer_vector_mediator(texto_usuario):
    # Ejecutamos el modelo
    resultado = clasificador(texto_usuario, ETIQUETAS_NLP, multi_label=True)
    
    # Mapeamos los resultados a un diccionario para extraerlos fácil
    # (Convierte las listas de labels y scores en un formato {'etiqueta': puntaje})
    scores = dict(zip(resultado['labels'], resultado['scores']))
    
    # 2. LA TRADUCCIÓN: De NLP a tus parámetros estrictos
    
    # soc_A (0.0 - 1.0): Social Anxiety Index
    soc_a = scores["miedo al rechazo o ansiedad social"]
    
    # soc_P (-1.0 a 1.0): Power Differential
    # Si manda él, se acerca a 1. Si es subordinado, baja a -1. Si son pares, se anulan (cerca a 0).
    soc_p = scores["la persona tiene autoridad o es el jefe"] - scores["la persona es un subordinado o novato"]
    
    # soc_U (0.0 - 1.0): Temporal Urgency
    soc_u = scores["situación urgente o crisis de tiempo"]
    
    # soc_V (0.0 - 1.0): Affinity/Valence
    soc_v = scores["son amigos cercanos o hay mucha confianza"]
    
    # Retornamos tu vector estructurado, redondeado a 3 decimales
    return {
        "soc_A": round(soc_a, 3),
        "soc_P": round(soc_p, 3),
        "soc_U": round(soc_u, 3),
        "soc_V": round(soc_v, 3)
    }

# --- PRUEBAS CON TU NUEVO VECTOR ---
situaciones = [
    # --- MITAD CON ERRORES REALISTAS DE ESTRÉS/RAPIDEZ ---
    "Mi compañero de proyecto no ha escrito una sola linea de codigo en Python y la entrega es en 2 horas. No quiero pelear, pero si no le digo nada perdemos la nota...",
    "El profesor de cálculo me puso un 2.0 en el parcial, pero estoy seguro de q mi procedimiento esta bien. tg q ir a pedirle revisión ya",
    "borré por accidente la base de datos de prueba del proyecto.. el director se va a dar cuenta mañana a primera hora q hago",
    "el entrenador de futsal me dejo en la banca todo el partido, a pesar de que juego mejor q el titular. Quiero reclamarle pero me da miedo que me saque",
    "la oficina de becas me dice que me falta un papel para legalizar el semestre y cerraban hace 10 minutos. el guardia de la puerta no me deja pasar",
    "Mi mamá revisó mi cuarto sin permiso y me botó unos apuntes importantes. Estoy furioso pero no quiero gritarle",
    "le envié un meme sarcástico a un compañero por chat y me respondio super cortante. Creo q se ofendió y no sé como arreglarlo",
    "El conductor del bus me cobró el pasaje doble por error. Le reclamé educadamente pero me ignoro, arrancó y me dejó hablando solo",
    "Mi portátil se dañó de la nada, tengo q entregar un ensayo a medianoche y necesito rogarle a mi vecino q apenas conozco que me preste el suyo",
    "Me rechazaron en el semillero de investigación al q quería entrar.. el correo de la coordinadora fue extremadamente frío y cortante",

    # --- MITAD CON ORTOGRAFÍA Y GRAMÁTICA PERFECTAS ---
    "Soy el líder del grupo y tengo que expulsar a un integrante porque nunca asiste a las reuniones ni responde los mensajes.",
    "Un compañero me pide que le pase mi tarea ya resuelta. Si se la paso, nos pueden anular a los dos por copia, pero si no, se va a enojar.",
    "Todo mi grupo de amigos decidió irse a una fiesta, pero yo no quiero ir y me insisten demasiado. Me siento presionado para no quedar como el aburrido del grupo.",
    "Le presté dinero a mi mejor amigo hace un mes y no me lo ha devuelto. Me da mucha pena cobrarle porque sé que su familia está mal económicamente.",
    "Llevo semanas queriendo invitar a salir a una chica de mi clase, pero cada vez que me acerco me paralizo, sudo y me voy.",
    "Quiero pedirle a mis papás que me dejen llegar más tarde de una reunión, pero sé que me van a decir un no rotundo de inmediato.",
    "Una persona se coló descaradamente en la fila de la cafetería justo cuando yo iba a pedir. Todos miran pero nadie dice nada.",
    "Pedí una hamburguesa sin cebolla porque soy alérgico, y me la trajeron llena de cebolla. El mesero me está mirando con mala actitud.",
    "Tengo que hacer una presentación oral frente a 50 personas en 5 minutos y siento que me voy a desmayar del pánico.",
    "Vi a un compañero copiándose descaradamente en pleno examen final. Si le digo al profesor, me gano un enemigo, pero si no, es injusto para los demás."
]

print("\n=== VECTOR DE ESTRÉS EXTRAÍDO (ARQUITECTURA MEDIATOR-AI) ===")
for i, texto in enumerate(situaciones, 1):
    vector = extraer_vector_mediator(texto)
    print(f"\nSituación {i}:")
    print(f"Texto: '{texto}'")
    for var, val in vector.items():
        print(f"  {var}: {val}")