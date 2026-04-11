import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Send } from "lucide-react";
import { useBiometrics } from "../hooks/useBiometrics";

export function ScreenChat({ data, onEndChat }) {
  const [messages, setMessages] = useState([]);
  const [rawHistory, setRawHistory] = useState([]); // Historial rol/content para backend
  const [sessionTactics, setSessionTactics] = useState([]);
  const [inputVal, setInputVal] = useState("");
  const [isTyping, setIsTyping] = useState(true);
  
  // Track if we should auto-scroll (only if user sends message or is already at bottom)
  const endRef = useRef(null);
  
  const { handlers, getMetricsAndReset } = useBiometrics();
  const hasInitializedRef = useRef(false);

  const parseMarkdown = (texto) => {
    return texto
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/^[-•]\s+/gm, '')
        .replace(/\n\n+/g, '</p><p class="mt-2">')
        .replace(/\n/g, '<br>');
  };

  useEffect(() => {
    if (hasInitializedRef.current) return;
    hasInitializedRef.current = true;

    // Initial setup request to backend
    const initializeChat = async () => {
        try {
            const payload = {
                modo: data.mode,
                escenario: data.contextText,
                texto_usuario: "Inicia la conversación.",
                metadatos: {
                    tiempo_escritura_segundos: 0,
                    teclas_borrado: 0,
                    pulsaciones_totales: 0,
                    ratio_duda: 0,
                    copy_paste_detectado: false,
                    longitud_caracteres: 1,
                    edad_usuario: data.age
                },
                historial: []
            };

            const res = await fetch("/simular-friccion", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const backData = await res.json();
            
            setIsTyping(false);
            if (backData.respuesta_bot) {
                setMessages([{ id: Date.now().toString(), role: "assistant", content: backData.respuesta_bot }]);
                setRawHistory([{ role: "model", content: backData.respuesta_bot }]);
            }
        } catch (e) {
            setIsTyping(false);
            setMessages([{ id: "error", role: "assistant", content: "Error de conexión." }]);
        }
    };
    
    initializeChat();
  }, [data]);

  useEffect(() => {
    // Solo auto-escrolleamos agresivamente si está escribiendo el usuario
    if (!isTyping) {
        // Opcional: Solo hacer scroll si el usuario ya estaba viendo el fondo
    } else {
        endRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isTyping]);

  const handleSend = async () => {
    if (!inputVal.trim()) return;
    
    const textToSend = inputVal.trim();
    const newUserMsg = { id: Date.now().toString(), role: "user", content: textToSend };
    
    setMessages(prev => [...prev, newUserMsg]);
    setInputVal("");
    setIsTyping(true);

    const metrics = getMetricsAndReset(textToSend, data.age);

    try {
        const payload = {
            modo: data.mode,
            escenario: data.contextText,
            texto_usuario: textToSend,
            metadatos: metrics,
            historial: rawHistory
        };

        const res = await fetch("/simular-friccion", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const backData = await res.json();
        
        setIsTyping(false);
        if (backData.respuesta_bot) {
            setMessages(prev => [...prev, { id: Date.now().toString(), role: "assistant", content: backData.respuesta_bot }]);
            setRawHistory(prev => [
                ...prev, 
                { role: "user", content: textToSend }, 
                { role: "model", content: backData.respuesta_bot }
            ]);
            
            if (backData.id_tacticas_usadas && backData.id_tacticas_usadas.length > 0) {
                setSessionTactics(prev => {
                    const dict = {};
                    prev.forEach(t => dict[t.id] = t);
                    backData.id_tacticas_usadas.forEach((id, index) => {
                        dict[id] = { id: id, name: backData.tacticas_usadas[index] || id };
                    });
                    return Object.values(dict);
                });
            }
        }
    } catch (e) {
        setIsTyping(false);
        setMessages(prev => [...prev, { id: Date.now().toString(), role: "assistant", content: "Error enviando el mensaje." }]);
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass rounded-3xl w-full max-w-4xl mx-auto h-[75vh] flex flex-col relative overflow-hidden ring-1 ring-white/20 shadow-2xl"
    >
      <div className="flex items-center justify-between p-4 border-b border-white/10 bg-white/5 dark:bg-black/10 backdrop-blur-md z-10 shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-tr from-[var(--color-primary-base)] to-[var(--color-primary-light)] p-0.5">
            <div className="w-full h-full bg-slate-900 rounded-full flex items-center justify-center">
              <span className="text-white font-bold text-xs">AI</span>
            </div>
          </div>
          <div>
            <h3 className="font-bold text-sm leading-tight">Tsukuyomi</h3>
            <div className="flex items-center gap-1.5 opacity-80">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
              <span className="text-[10px] uppercase tracking-wider font-semibold">{data.mode} Activo</span>
            </div>
          </div>
        </div>
        <button onClick={() => onEndChat(
          rawHistory.map(m => ({ role: m.role === 'model' ? 'bot' : 'user', content: m.content })),
          sessionTactics
        )} className="p-2 hover:bg-red-500/20 text-red-500 rounded-full transition-colors opacity-70 hover:opacity-100" title="Finalizar Sesión">
          <X size={20} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4 no-scrollbar">
        <AnimatePresence initial={false}>
          {messages.map((msg) => {
            const isUser = msg.role === "user";
            return (
              <motion.div 
                key={msg.id}
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}
              >
                <div className={`max-w-[85%] md:max-w-[75%] rounded-2xl p-4 ${
                  isUser 
                    ? "bg-gradient-to-br from-[var(--color-primary-base)] to-[var(--color-primary-light)] text-white rounded-tr-sm shadow-md" 
                    : "glass bg-white/60 dark:bg-slate-800/60 rounded-tl-sm text-slate-800 dark:text-slate-100"
                }`}>
                  {isUser ? (
                    <p className="text-sm md:text-base leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                  ) : (
                    <div 
                      className="text-sm md:text-base leading-relaxed" 
                      dangerouslySetInnerHTML={{ __html: parseMarkdown(msg.content) }} 
                    />
                  )}
                </div>
              </motion.div>
            );
          })}
        </AnimatePresence>

        {isTyping && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex justify-start w-full"
          >
            <div className="glass bg-white/60 dark:bg-slate-800/60 rounded-2xl p-4 px-5">
              <div className="flex gap-1.5 text-[var(--color-primary-base)]">
                <motion.div animate={{ y: [0, -5, 0] }} transition={{ repeat: Infinity, duration: 1, delay: 0 }} className="w-1.5 h-1.5 rounded-full bg-current" />
                <motion.div animate={{ y: [0, -5, 0] }} transition={{ repeat: Infinity, duration: 1, delay: 0.2 }} className="w-1.5 h-1.5 rounded-full bg-current" />
                <motion.div animate={{ y: [0, -5, 0] }} transition={{ repeat: Infinity, duration: 1, delay: 0.4 }} className="w-1.5 h-1.5 rounded-full bg-current" />
              </div>
            </div>
          </motion.div>
        )}
        <div ref={endRef} className="h-4 shrink-0" />
      </div>

      <div className="p-4 bg-white/5 dark:bg-slate-900/40 backdrop-blur-xl border-t border-white/10 shrink-0">
        <div className="relative flex items-end gap-2 bg-white/80 dark:bg-slate-800/80 p-1.5 rounded-2xl shadow-inner focus-within:ring-2 ring-[var(--color-primary-base)] transition-all">
          <textarea 
            className="flex-1 bg-transparent border-none outline-none resize-none px-3 py-3 text-sm md:text-base no-scrollbar min-h-[48px] max-h-[120px] text-left"
            rows={1}
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
            onKeyDown={(e) => {
              handlers.onKeyDown(e);
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            onPaste={handlers.onPaste}
            placeholder="Escribe tu mensaje o táctica..."
          />
          <motion.button 
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={handleSend}
            disabled={!inputVal.trim() || isTyping}
            className={`p-3 lg:p-3 rounded-xl transition-colors mb-0.5 mr-0.5 ${
              inputVal.trim() && !isTyping
                ? "bg-[var(--color-primary-base)] text-white shadow-md shadow-teal-500/20" 
                : "bg-slate-200 dark:bg-slate-700 text-slate-400 cursor-not-allowed"
            }`}
          >
            <Send size={18} />
          </motion.button>
        </div>
      </div>
    </motion.div>
  );
}
