/**
 * Lógica de cliente: Flujo Escenario -> Modo -> Chat (WhatsApp Style) -> Feedback
 */

let modoActual = "simulador";
let escenarioActual = "";
let historialChat = [];
let tiempoInicioEscritura = null; 
let contadorBorrados = 0; 
let pulsacionesTotales = 0; 
let hizoCopyPaste = false;
let tacticasCapturadas = []; // Guardaremos los IDs de las tácticas usadas

const API_URL = "/simular-friccion";
const FEEDBACK_URL = "/feedback";
const VISION_URL = "/detectar-contexto-visual";

// Estado de visión: base64 de la imagen seleccionada o null
let imagenBase64Pendiente = null;

// Mapping de elementos del DOM
const dom = {
  screens:          document.querySelectorAll('.screen, .modal-overlay'),
  btnNextToModes:   document.getElementById('btn-next-to-modes'),
  btnEnviar:        document.getElementById('btn-enviar'),
  btnRestart:       document.getElementById('btn-restart'),
  inputEscenario:   document.getElementById('input-escenario'),
  previewEscenario: document.getElementById('scenario-preview-text'),
  inputChat:        document.getElementById('chat-input'),
  chatBox:          document.getElementById('chat-box'),
  chatContainer:    document.getElementById('chat-container'),
  chatModeTitle:    document.getElementById('chat-mode-title'),
  outputJson:       document.getElementById('output-json'),
  // Feedback elements
  inputRating:      document.getElementById('input-rating'),
  inputComentario:  document.getElementById('input-comentario'),
  btnEnviarFeedback: document.getElementById('btn-enviar-feedback'),
  stars:            document.querySelectorAll('.star'),
  inputAge:         document.getElementById('user-age'),
  // [VISION] Nuevos elementos del Ojo de Tsukuyomi
  btnAdjuntar:      document.getElementById('btn-adjuntar'),
  inputImagen:      document.getElementById('input-imagen'),
  visionPreview:    document.getElementById('vision-preview'),
  visionThumb:      document.getElementById('vision-thumb'),
  btnQuitarImagen:  document.getElementById('btn-quitar-imagen'),
};

// PERSISTENCIA DE EDAD (LocalStorage)
if (localStorage.getItem('tsukoyomi_age')) {
    dom.inputAge.value = localStorage.getItem('tsukoyomi_age');
}
dom.inputAge.addEventListener('change', () => {
    localStorage.setItem('tsukoyomi_age', dom.inputAge.value);
});

// NAVEGACIÓN Y FLUJO

function showScreen(screenId) {
    dom.screens.forEach(s => s.classList.remove('active'));
    document.getElementById(screenId).classList.add('active');
}

function setModo(modo) {
    modoActual = modo;
    dom.chatModeTitle.textContent = (modo === 'simulador' ? 'Simulador de IA' : 'Estratega (Consejo)');
    showScreen('pantalla-chat');
    inicializarConversacion();
}

// CAPTURA BIOMÉTRICA

function resetSensores() {
  tiempoInicioEscritura = null; 
  contadorBorrados = 0; 
  pulsacionesTotales = 0; 
  hizoCopyPaste = false;
}

function agregarBurbuja(texto, tipo) {
    const div = document.createElement('div');
    div.className = `mensaje msj-${tipo}`;
    if (tipo === 'bot') div.classList.add(`modo-${modoActual}`);

    if (tipo === 'bot') {
        // Parseo mínimo: convertir markdown básico a HTML seguro
        const html = texto
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/^[-•]\s+/gm, '')
            .replace(/\n\n+/g, '</p><p>')
            .replace(/\n/g, '<br>');
        div.innerHTML = `<p style="margin:0">${html}</p>`;
    } else {
        div.textContent = texto;
    }

    dom.chatBox.appendChild(div);
    dom.chatContainer.scrollTop = dom.chatContainer.scrollHeight;
}

// HANDLERS DE INTERFAZ

dom.btnNextToModes.addEventListener('click', async () => {
    // [VISION] Si hay imagen, la procesamos antes de pasar.
    if (imagenBase64Pendiente) {
        dom.btnNextToModes.disabled = true;
        const textoPreLoader = dom.btnNextToModes.innerHTML;
        dom.btnNextToModes.innerHTML = "⏳";
        
        await procesarImagenComoEscenario();
        
        dom.btnNextToModes.innerHTML = textoPreLoader;
        dom.btnNextToModes.disabled = false;
        
        // Si procesarImagen falla completamente y no extrae escenario, retorna.
        if (!escenarioActual) {
            return;
        }
    } else {
        escenarioActual = dom.inputEscenario.value.trim();
    }

    if (!escenarioActual) {
        dom.inputEscenario.style.borderColor = "#D32F2F";
        return;
    }
    
    dom.previewEscenario.textContent = escenarioActual.substring(0, 150) + (escenarioActual.length > 150 ? "..." : "");
    showScreen('pantalla-modos');
});

async function inicializarConversacion() {
    historialChat = [];
    tacticasCapturadas = [];
    dom.chatBox.innerHTML = '';

    const loader = document.createElement('div');
    loader.className = 'escribiendo';
    loader.textContent = 'IA está analizando el escenario...';
    dom.chatBox.appendChild(loader);

    const payload = {
        modo: modoActual,
        escenario: escenarioActual,
        texto_usuario: "Inicia la conversación.",
        metadatos: {
            tiempo_escritura_segundos: 0,
            teclas_borrado: 0,
            pulsaciones_totales: 0,
            ratio_duda: 0,
            copy_paste_detectado: false,
            longitud_caracteres: 1,
            edad_usuario: parseInt(dom.inputAge.value) || 25
        },
        historial: []
    };

    const timeoutId = setTimeout(() => {
        loader.textContent = 'El servidor tardó demasiado. Recarga la página e intenta de nuevo.';
        loader.style.color = '#D32F2F';
    }, 25000);

    try {
        const res = await callBackend(payload);
        clearTimeout(timeoutId);
        loader.remove();
        if (res.respuesta_bot) {
            agregarBurbuja(res.respuesta_bot, 'bot');
            historialChat.push({role: 'model', content: res.respuesta_bot});
            if (res.id_tacticas_usadas) {
                tacticasCapturadas = [...new Set([...tacticasCapturadas, ...res.id_tacticas_usadas])];
            }
            dom.outputJson.textContent = JSON.stringify(res, null, 2);
        }
    } catch (e) {
        clearTimeout(timeoutId);
        loader.textContent = 'Error al conectar con el servidor. Recarga la página.';
        loader.style.color = '#D32F2F';
    }
}

dom.btnEnviar.addEventListener('click', async () => {
    const msg = dom.inputChat.value.trim();
    if (!msg) return;

    agregarBurbuja(msg, 'usuario');
    dom.inputChat.value = '';
    dom.btnEnviar.disabled = true;

    const metas = {
        tiempo_escritura_segundos: (Date.now() - (tiempoInicioEscritura || Date.now())) / 1000,
        teclas_borrado: contadorBorrados,
        pulsaciones_totales: pulsacionesTotales,
        ratio_duda: parseFloat((pulsacionesTotales / Math.max(1, msg.length)).toFixed(2)),
        copy_paste_detectado: hizoCopyPaste,
        longitud_caracteres: msg.length,
        edad_usuario: parseInt(dom.inputAge.value) || 25
    };

    const loader = document.createElement('div');
    loader.className = 'escribiendo';
    loader.textContent = 'Pensando...';
    dom.chatBox.appendChild(loader);

    try {
        const res = await callBackend({
            modo: modoActual,
            escenario: escenarioActual,
            texto_usuario: msg,
            metadatos: metas,
            historial: historialChat
        });

        loader.remove();
        if (res.respuesta_bot) {
            agregarBurbuja(res.respuesta_bot, 'bot');
            historialChat.push({role: 'user', content: msg});
            historialChat.push({role: 'model', content: res.respuesta_bot});
            if (res.id_tacticas_usadas) {
                tacticasCapturadas = [...new Set([...tacticasCapturadas, ...res.id_tacticas_usadas])];
            }
            dom.outputJson.textContent = JSON.stringify(res, null, 2);
        }
    } catch (e) {
        loader.textContent = "Error de conexión.";
    } finally {
        resetSensores();
        dom.btnEnviar.disabled = false;
        dom.inputChat.style.height = "auto";
    }
});

dom.btnRestart.addEventListener('click', () => {
    // Reset completo del estado del modal
    dom.inputRating.value = '0';
    dom.stars.forEach(s => s.classList.remove('selected'));
    dom.inputComentario.value = '';
    dom.btnEnviarFeedback.disabled = false;
    dom.btnEnviarFeedback.textContent = 'Enviar y Finalizar';
    document.getElementById('pantalla-feedback').classList.add('active');
});

dom.btnEnviarFeedback.addEventListener('click', async () => {
    const puntuacion = parseInt(dom.inputRating.value);
    if (puntuacion === 0) {
        alert("Por favor, selecciona una puntuación.");
        return;
    }

    const feedbackData = {
        escenario: escenarioActual,
        modo: modoActual,
        historial: historialChat.map(m => ({
            role: m.role === 'model' ? 'bot' : 'user',
            content: m.content
        })),
        puntuacion: puntuacion,
        comentario: dom.inputComentario.value.trim(),
        tacticas_feedback: tacticasCapturadas.map(id => ({
            id_tactica: id,
            efectiva: puntuacion >= 3
        }))
    };

    dom.btnEnviarFeedback.disabled = true;
    dom.btnEnviarFeedback.textContent = "Guardando evolución...";

    try {
        const response = await fetch(FEEDBACK_URL, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(feedbackData)
        });
        
        if (response.ok) {
            alert("¡Evolución guardada! La IA ha aprendido de esta sesión.");
            location.reload(); 
        } else {
            throw new Error("Error en el servidor");
        }
    } catch (e) {
        location.reload();
    }
});

function setRating(rating) {
    dom.inputRating.value = rating;
    dom.stars.forEach((star, index) => {
        if (index < rating) {
            star.classList.add('selected');
        } else {
            star.classList.remove('selected');
        }
    });
}

// CAPTURA BIOMÉTRICA SOSTENIDA
dom.inputChat.addEventListener('keydown', (e) => {
    if (!tiempoInicioEscritura) tiempoInicioEscritura = Date.now();
    pulsacionesTotales++;
    if (e.key === 'Backspace' || e.key === 'Delete') contadorBorrados++;
    // AÑADIR: Enter sin Shift envía
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!dom.btnEnviar.disabled) dom.btnEnviar.click();
    }
});

dom.inputChat.addEventListener('paste', () => {
    hizoCopyPaste = true;
    if (!tiempoInicioEscritura) tiempoInicioEscritura = Date.now();
});

// Ajuste automático de altura del input
dom.inputChat.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

dom.inputEscenario.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

async function callBackend(data) {
  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  });
  return response.json();
}

function closeFeedback() {
    document.getElementById('pantalla-feedback').classList.remove('active');
}

// ============================================================
// [VISION] OJO DE TSUKUYOMI — Compresión y Análisis Visual
// ============================================================

/**
 * Comprime una imagen a JPEG 1280px max con calidad 0.72 usando Canvas.
 * Retorna una string Base64 pura (sin cabecera data:image/...)
 */
async function compressImageToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onerror = reject;
        reader.onload = (evt) => {
            const img = new Image();
            img.onerror = reject;
            img.onload = () => {
                const MAX_WIDTH = 1280;
                const scale = img.width > MAX_WIDTH ? MAX_WIDTH / img.width : 1;
                const canvas = document.createElement('canvas');
                canvas.width  = Math.round(img.width  * scale);
                canvas.height = Math.round(img.height * scale);
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                // Solo la parte Base64 pura, sin "data:image/jpeg;base64,"
                const dataURL = canvas.toDataURL('image/jpeg', 0.72);
                resolve(dataURL.split(',')[1]);
            };
            img.src = evt.target.result;
        };
        reader.readAsDataURL(file);
    });
}

function mostrarPreviewImagen(src) {
    dom.visionThumb.src = src;
    dom.visionPreview.style.display = 'flex';
}

function limpiarImagen() {
    imagenBase64Pendiente = null;
    dom.visionThumb.src = '';
    dom.visionPreview.style.display = 'none';
    dom.inputImagen.value = '';
}

// Botón clip → abre selector de archivos
dom.btnAdjuntar.addEventListener('click', () => { dom.inputImagen.click(); });

// Archivo seleccionado → thumbnail + compresión
dom.inputImagen.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (!file.type.startsWith('image/')) { alert('Por favor selecciona una imagen.'); return; }
    mostrarPreviewImagen(URL.createObjectURL(file));
    imagenBase64Pendiente = await compressImageToBase64(file);
});

// Quitar imagen
dom.btnQuitarImagen.addEventListener('click', limpiarImagen);

/**
 * Envía la imagen al endpoint de visión al adjuntar en pantalla inicial.
 * Extrae texto y contexto, asimilándolo como escenarioActual.
 */
async function procesarImagenComoEscenario() {
    try {
        const res = await fetch(VISION_URL, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ imagen_base64: imagenBase64Pendiente })
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        // CASO A: Fallback por Rate Limit
        if (data.error || data.fallback_sugerido) {
            alert('El servidor visual está saturado. Por favor, escribe lo que dice la captura en texto.');
            limpiarImagen();
            return;
        }

        // CASO B: Éxito — Inyectar contexto al escenario
        const resumen = data.resumen_escenario || 'Una dinámica analizada por visión.';
        
        // Armamos el escenario inyectando JSON estructurado para que lo atrape RAG_Translator luego
        const extraTexto = dom.inputEscenario.value.trim();
        const baseContext = extraTexto ? `[Nota del usuario: ${extraTexto}]\n\n` : '';
        
        escenarioActual = `${baseContext}He analizado la captura: ${resumen}`;

        // Aquí inyectamos además la transcripción, pero oculta o sumada al escenario
        const transcripcion = Array.isArray(data.transcripcion_cronologica)
            ? data.transcripcion_cronologica.map(m => `${m.emisor || 'Desconocido'}: ${m.mensaje || m}`).join('\n')
            : JSON.stringify(data.transcripcion_cronologica);
            
        // Engrosar el escenario a espaldas del usuario:
        escenarioActual += `\n\n=== HISTORIAL CHAT ===\n${transcripcion}`;
        
        // Log para QA
        dom.outputJson.textContent = JSON.stringify(data, null, 2);

    } catch (err) {
        alert('No pude analizar la imagen. Verifica la conexión o usa texto.');
        console.error('[VISION ERROR]', err);
    }
}
