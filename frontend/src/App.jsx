import { useState, useEffect } from "react";
import { ThemeToggle } from "./components/ThemeToggle";
import { ThemeCustomizer } from "./components/ThemeCustomizer";

import { ScreenContext } from "./components/ScreenContext";
import { ScreenModes } from "./components/ScreenModes";
import { ScreenChat } from "./components/ScreenChat";
import { FeedbackModal } from "./components/FeedbackModal";

export default function App() {
  const [isDark, setIsDark] = useState(() => {
    const savedTheme = localStorage.getItem("tsukoyomi_theme");
    return savedTheme ? savedTheme === "dark" : (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
  });
  const [currentScreen, setCurrentScreen] = useState("context"); // context | modes | chat
  const [showFeedback, setShowFeedback] = useState(false);
  
  const [sessionData, setSessionData] = useState({
    contextText: "",
    imagePreview: null,
    age: parseInt(localStorage.getItem("tsukoyomi_age")) || 25,
    mode: null // 'simulador' | 'consejo'
  });

  // Historial global para el feedback
  const [globalChatHistory, setGlobalChatHistory] = useState([]);

  useEffect(() => {
    localStorage.setItem("tsukoyomi_theme", isDark ? "dark" : "light");
    if (isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [isDark]);

  const toggleDark = () => setIsDark(!isDark);

  const handleEndChat = (historyObj) => {
    setGlobalChatHistory(historyObj);
    setShowFeedback(true);
  };

  const submitFeedback = async ({ rating, comment }) => {
    const feedbackData = {
        escenario: sessionData.contextText,
        modo: sessionData.mode,
        historial: globalChatHistory, // Debe venir formateado del ScreenChat
        puntuacion: rating,
        comentario: comment,
        tacticas_feedback: [] // En React no las extrajimos aún, pero se envían vacías
    };

    try {
        await fetch('/feedback', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(feedbackData)
        });
    } catch(e) {
        console.error("Error al enviar feedback", e);
    }

    // Reset everything
    setShowFeedback(false);
    setSessionData({ contextText: "", imagePreview: null, age: sessionData.age, mode: null });
    setCurrentScreen("context");
  };

  return (
    <div className="min-h-screen w-full relative transition-colors duration-300 overflow-x-hidden font-sans">
      
      <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-[var(--color-primary-base)] opacity-20 blur-[100px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full bg-[var(--color-accent-orange)] opacity-20 blur-[120px]" />
      </div>

      <ThemeToggle isDark={isDark} toggleDark={toggleDark} />
      <ThemeCustomizer />

      <main className="relative z-10 container mx-auto px-4 py-12 flex flex-col items-center min-h-screen text-left">
        <div className="mb-12 text-center">
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
            TSUKUYOMI <span className="text-[var(--color-primary-base)]">IA</span>
          </h1>
        </div>

        <div className="w-full max-w-4xl flex-grow flex flex-col">
          {currentScreen === "context" && (
            <ScreenContext 
              data={sessionData} 
              onNext={(data) => {
                setSessionData({ ...sessionData, ...data });
                setCurrentScreen("modes");
              }} 
            />
          )}

          {currentScreen === "modes" && (
            <ScreenModes 
              data={sessionData}
              onBack={() => setCurrentScreen("context")}
              onSelectMode={(modeData) => {
                setSessionData({ ...sessionData, ...modeData });
                setCurrentScreen("chat");
              }} 
            />
          )}

          {currentScreen === "chat" && (
            <ScreenChat 
              data={sessionData} 
              onEndChat={handleEndChat}
            />
          )}
        </div>
      </main>

      <FeedbackModal 
        isOpen={showFeedback} 
        onClose={() => {
            setShowFeedback(false);
            setSessionData({ ...sessionData, contextText: "", imagePreview: null, mode: null });
            setCurrentScreen("context");
        }}
        onSubmitFeedback={submitFeedback}
      />
    </div>
  );
}
