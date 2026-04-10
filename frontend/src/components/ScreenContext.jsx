import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { Paperclip, Send, X, Loader2 } from "lucide-react";

export function ScreenContext({ data, onNext }) {
  const [text, setText] = useState(data.contextText);
  const [preview, setPreview] = useState(data.imagePreview);
  const [base64Img, setBase64Img] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) { alert('Selecciona una imagen.'); return; }
      
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result);
        const b64 = reader.result.split(',')[1];
        setBase64Img(b64);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async () => {
    if (!text.trim() && !preview) return;

    if (base64Img) {
        setIsLoading(true);
        try {
            const res = await fetch("/detectar-contexto-visual", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ imagen_base64: base64Img })
            });
            const dataRes = await res.json();
            
            if (dataRes.error || dataRes.fallback_sugerido) {
                alert('El servidor visual no pudo procesarlo. Escribe el contexto.');
                setPreview(null);
                setBase64Img(null);
                setIsLoading(false);
                return;
            }

            const resumen = dataRes.resumen_escenario || 'Contexto analizado visualmente.';
            const extra = text.trim() ? `[Nota del usuario: ${text}]\n\n` : '';
            let newText = `${extra}He analizado la captura: ${resumen}`;
            
            if (dataRes.transcripcion_cronologica) {
                const tr = Array.isArray(dataRes.transcripcion_cronologica) 
                    ? dataRes.transcripcion_cronologica.map(m => `${m.emisor || 'Desconocido'}: ${m.mensaje || m}`).join('\n')
                    : JSON.stringify(dataRes.transcripcion_cronologica);
                newText += `\n\n=== HISTORIAL CHAT ===\n${tr}`;
            }
            
            onNext({ contextText: newText, imagePreview: preview });
        } catch (e) {
            console.error(e);
            alert("No se pudo conectar con el motor de visión.");
        } finally {
            setIsLoading(false);
        }
    } else {
        onNext({ contextText: text, imagePreview: preview });
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.4 }}
      className="glass glass-card max-w-2xl mx-auto w-full flex flex-col gap-6"
    >
      <h2 className="text-xl font-semibold opacity-90">¿Qué está pasando?</h2>
      
      {preview && (
        <motion.div 
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          className="flex items-center gap-3 p-2 pr-4 bg-white/20 dark:bg-slate-800/40 border border-white/20 rounded-xl w-fit"
        >
          <img src={preview} alt="upload preview" className="h-12 w-12 object-cover rounded-lg border border-white/10 shadow-sm" />
          <span className="text-sm opacity-80 font-medium">Captura adjuntada</span>
          <button 
            onClick={() => { setPreview(null); setBase64Img(null); }}
            className="ml-2 hover:bg-red-500/20 p-1.5 rounded-md transition-colors"
          >
            <X size={16} />
          </button>
        </motion.div>
      )}

      <div className="relative flex items-end gap-2 bg-white/70 dark:bg-slate-900/60 p-2 rounded-2xl border border-white/50 dark:border-white/10 shadow-sm focus-within:ring-2 ring-[var(--color-primary-base)] transition-all">
        
        <button 
          onClick={() => fileInputRef.current?.click()}
          className="p-3 text-slate-500 hover:text-[var(--color-primary-base)] dark:text-slate-400 hover:bg-white/50 dark:hover:bg-slate-800/50 rounded-xl transition-all"
        >
          <Paperclip size={20} />
        </button>
        <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" className="hidden" />

        <textarea 
          className="w-full bg-transparent border-none outline-none resize-none px-2 py-3 text-sm md:text-base no-scrollbar min-h-[50px] max-h-[200px] text-left"
          rows={3}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Describe el contexto o adjunta una captura..."
        />

        <motion.button 
          whileHover={!isLoading ? { scale: 1.05 } : {}}
          whileTap={!isLoading ? { scale: 0.95 } : {}}
          onClick={handleSubmit}
          disabled={(!text.trim() && !preview) || isLoading}
          className={`p-3 rounded-xl transition-colors ${
            (text.trim() || preview) && !isLoading
              ? "bg-[var(--color-primary-base)] hover:bg-[var(--color-primary-light)] text-white shadow-lg" 
              : "bg-slate-200 dark:bg-slate-800 text-slate-400 cursor-not-allowed"
          }`}
        >
          {isLoading ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
        </motion.button>
      </div>
    </motion.div>
  );
}
