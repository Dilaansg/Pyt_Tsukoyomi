import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Star } from "lucide-react";

export function FeedbackModal({ isOpen, onClose, onSubmitFeedback }) {
  const [rating, setRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [comment, setComment] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async () => {
    if (rating === 0) {
      alert("Por favor, selecciona una puntuación.");
      return;
    }
    
    setIsSubmitting(true);
    await onSubmitFeedback({ rating, comment });
    setIsSubmitting(false);
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm"
        >
          <motion.div 
            initial={{ scale: 0.9, y: 20 }}
            animate={{ scale: 1, y: 0 }}
            exit={{ scale: 0.9, y: 20 }}
            className="glass glass-card max-w-md w-full flex flex-col gap-4 text-center ring-1 ring-white/10 shadow-2xl"
          >
            <h2 className="text-2xl font-bold">Sesión Finalizada</h2>
            <p className="text-sm opacity-80 mb-2">¿Qué tan efectiva fue la respuesta de Tsukuyomi IA en este escenario?</p>
            
            {/* Stars */}
            <div className="flex justify-center gap-2 my-2">
              {[1, 2, 3, 4, 5].map((star) => (
                <button 
                  key={star}
                  onClick={() => setRating(star)}
                  onMouseEnter={() => setHoverRating(star)}
                  onMouseLeave={() => setHoverRating(0)}
                  className="p-1 transition-transform hover:scale-110 active:scale-95"
                >
                  <Star 
                    size={32} 
                    className={`transition-colors ${(hoverRating || rating) >= star ? "fill-yellow-400 text-yellow-400" : "text-slate-400 dark:text-slate-600"}`} 
                  />
                </button>
              ))}
            </div>

            <textarea 
              className="w-full bg-white/50 dark:bg-slate-900/50 border border-white/30 dark:border-white/10 rounded-xl px-4 py-3 text-sm focus:outline-none focus:ring-2 ring-[var(--color-primary-base)]"
              rows={3}
              placeholder="Observaciones de la fricción (opcional)..."
              value={comment}
              onChange={(e) => setComment(e.target.value)}
            />

            <div className="flex flex-col gap-2 mt-4">
              <button 
                onClick={handleSubmit}
                disabled={isSubmitting}
                className="w-full py-3 bg-[var(--color-primary-base)] hover:bg-[var(--color-primary-light)] text-white font-semibold rounded-xl transition-colors shadow-lg"
              >
                {isSubmitting ? "Guardando evolución..." : "Enviar y Finalizar"}
              </button>
              <button 
                onClick={onClose}
                className="w-full py-2 text-sm opacity-60 hover:opacity-100 transition-opacity"
              >
                Tal vez luego
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
