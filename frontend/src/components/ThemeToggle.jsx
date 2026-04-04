import { Moon, Sun } from "lucide-react";
import { motion } from "framer-motion";

export function ThemeToggle({ isDark, toggleDark }) {
  return (
    <motion.button 
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      onClick={toggleDark}
      className="fixed top-6 right-6 p-2.5 rounded-full glass hover:scale-105 active:scale-95 transition-all z-50 text-[var(--color-text-light)] dark:text-[var(--color-text-dark)] cursor-pointer"
      aria-label="Toggle Dark Mode"
    >
      <motion.div
        initial={false}
        animate={{ rotate: isDark ? 180 : 0 }}
        transition={{ duration: 0.3 }}
      >
        {isDark ? <Sun size={22} className="text-yellow-400" /> : <Moon size={22} className="text-slate-600" />}
      </motion.div>
    </motion.button>
  );
}
