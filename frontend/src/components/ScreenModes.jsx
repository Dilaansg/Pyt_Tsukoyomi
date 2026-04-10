import { useState } from "react";
import { motion } from "framer-motion";
import { Edit3, Swords, BrainCircuit, User } from "lucide-react";

export function ScreenModes({ data, onBack, onSelectMode }) {
  const [age, setAge] = useState(data.age || 25);

  const handleAgeChange = (e) => {
    const val = parseInt(e.target.value);
    setAge(val);
    localStorage.setItem("tsukoyomi_age", val.toString());
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { type: "spring", stiffness: 300, damping: 24 } }
  };

  return (
    <motion.div 
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className="w-full max-w-4xl mx-auto flex flex-col gap-6"
    >
      {/* Context Summary Pill */}
      <motion.div variants={itemVariants} className="glass rounded-2xl p-4 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div className="flex-1">
          <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1">Contexto Analizado</p>
          <p className="text-sm line-clamp-2 opacity-90">{data.contextText || "Imagen adjunta sin texto."}</p>
        </div>
        <button onClick={onBack} className="flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg bg-black/5 dark:bg-white/10 hover:bg-black/10 dark:hover:bg-white/20 transition-colors">
          <Edit3 size={14} /> Editar
        </button>
      </motion.div>

      {/* Age Configuration Slider */}
      <motion.div variants={itemVariants} className="glass rounded-2xl p-5 flex flex-col sm:flex-row justify-between items-center gap-4 border-l-4 border-l-[var(--color-primary-base)]">
        <div>
          <h3 className="font-semibold flex items-center gap-2">
            <User size={18} className="text-[var(--color-primary-base)]" /> 
            Perfil del Usuario
          </h3>
          <p className="text-xs opacity-70">Ajusta tu edad para adaptar el tono.</p>
        </div>
        
        <div className="flex items-center gap-4 w-full sm:w-1/2">
          <span className="text-sm font-medium w-8">{age}</span>
          <input 
            type="range" 
            min="10" max="99" 
            value={age} 
            onChange={handleAgeChange}
            className="w-full h-2 bg-slate-200 dark:bg-slate-700/50 rounded-lg appearance-none cursor-pointer accent-[var(--color-primary-base)]"
          />
        </div>
      </motion.div>

      {/* Modes Grid (Bento Style) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
        {/* Mode 1: Simulador */}
        <motion.button 
          variants={itemVariants}
          whileHover={{ y: -5, scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onSelectMode({ age, mode: 'simulador' })}
          className="glass glass-card group text-left relative overflow-hidden focus:outline-none focus:ring-2 ring-teal-500"
        >
          <div className="absolute -right-8 -top-8 bg-teal-500/10 w-32 h-32 rounded-full blur-2xl group-hover:bg-teal-500/20 transition-colors"></div>
          <Swords className="text-teal-600 dark:text-teal-400 mb-4" size={32} />
          <h3 className="text-xl font-bold mb-2">Simulador de Arena</h3>
          <p className="text-sm opacity-80 leading-relaxed">
            Prueba diferentes escenarios de conversación y observa resultados potenciales en un entorno seguro antes de actuar.
          </p>
        </motion.button>

        {/* Mode 2: Estratega */}
        <motion.button 
          variants={itemVariants}
          whileHover={{ y: -5, scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={() => onSelectMode({ age, mode: 'consejo' })}
          className="glass glass-card group text-left relative overflow-hidden focus:outline-none focus:ring-2 ring-indigo-500"
        >
          <div className="absolute -right-8 -bottom-8 bg-indigo-500/10 w-32 h-32 rounded-full blur-2xl group-hover:bg-indigo-500/20 transition-colors"></div>
          <BrainCircuit className="text-indigo-600 dark:text-indigo-400 mb-4" size={32} />
          <h3 className="text-xl font-bold mb-2">Estratega (Consejo)</h3>
          <p className="text-sm opacity-80 leading-relaxed">
            Recibe un plan de acción personalizado y consejos tácticos basados en un análisis profundo de la situación.
          </p>
        </motion.button>
      </div>

    </motion.div>
  );
}
