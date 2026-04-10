import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Settings, X } from "lucide-react";
import { SketchPicker } from "react-color";

export function ThemeCustomizer() {
  const [isOpen, setIsOpen] = useState(false);
  const [primary, setPrimary] = useState("#DC2626"); // Red 600

  // Apply CSS Variable globally when primary changes
  useEffect(() => {
    document.documentElement.style.setProperty("--color-primary-base", primary);
  }, [primary]);

  return (
    <>
      <motion.button
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        onClick={() => setIsOpen(true)}
        className="fixed top-6 right-20 p-2.5 rounded-full glass hover:scale-105 active:scale-95 transition-all z-50 text-[var(--color-primary-base)] cursor-pointer"
        aria-label="Customize Colors"
      >
        <Settings size={22} className="animate-spin-slow" style={{ animationDuration: '4s' }} />
      </motion.button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="fixed top-20 right-6 z-50 glass glass-card shadow-2xl min-w-[200px]"
          >
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-bold text-sm">Color Primario</h3>
              <button onClick={() => setIsOpen(false)} className="opacity-50 hover:opacity-100">
                <X size={16} />
              </button>
            </div>
            {/* react-color component */}
            <div className="custom-picker">
                <SketchPicker 
                    color={primary} 
                    onChange={(c) => setPrimary(c.hex)}
                    disableAlpha={true}
                    presetColors={["#DC2626", "#EF4444", "#991B1B", "#2563EB", "#7C3AED"]}
                />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
