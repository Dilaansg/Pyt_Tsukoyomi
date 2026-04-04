import { useState, useRef, useCallback } from "react";

export function useBiometrics() {
  const [sessionStarted, setSessionStarted] = useState(false);
  const startTimeRef = useRef(null);
  const totalKeystrokesRef = useRef(0);
  const deletionCountRef = useRef(0);
  const pastedRef = useRef(false);

  const startTracking = useCallback(() => {
    if (!sessionStarted) {
      startTimeRef.current = Date.now();
      setSessionStarted(true);
    }
  }, [sessionStarted]);

  const recordKeystroke = useCallback((e) => {
    startTracking();
    totalKeystrokesRef.current += 1;

    if (e.key === 'Backspace' || e.key === 'Delete') {
      deletionCountRef.current += 1;
    }
  }, [startTracking]);

  const recordPaste = useCallback(() => {
    startTracking();
    pastedRef.current = true;
  }, [startTracking]);

  const resetBiometrics = useCallback(() => {
    setSessionStarted(false);
    startTimeRef.current = null;
    totalKeystrokesRef.current = 0;
    deletionCountRef.current = 0;
    pastedRef.current = false;
  }, []);

  const getMetricsAndReset = (messageContent, age) => {
    const elapsedSeconds = startTimeRef.current 
      ? (Date.now() - startTimeRef.current) / 1000 
      : 0;
      
    const totalChars = Math.max(1, messageContent.length);
    const doubtRatio = parseFloat((totalKeystrokesRef.current / totalChars).toFixed(2));

    const metrics = {
      tiempo_escritura_segundos: elapsedSeconds,
      teclas_borrado: deletionCountRef.current,
      pulsaciones_totales: totalKeystrokesRef.current,
      ratio_duda: doubtRatio,
      copy_paste_detectado: pastedRef.current,
      longitud_caracteres: messageContent.length,
      edad_usuario: parseInt(age) || 25
    };

    resetBiometrics();
    return metrics;
  };

  return {
    handlers: {
      onKeyDown: recordKeystroke,
      onPaste: recordPaste,
    },
    getMetricsAndReset
  };
}
