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
  inputAge:         document.getElementById('user-age')
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
  div.textContent = texto;
  dom.chatBox.appendChild(div);
  dom.chatContainer.scrollTop = dom.chatContainer.scrollHeight;
}

// HANDLERS DE INTERFAZ

dom.btnNextToModes.addEventListener('click', () => {
    escenarioActual = dom.inputEscenario.value.trim();
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

    try {
        const res = await callBackend(payload);
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
        loader.textContent = "Error al conectar con el servidor.";
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
