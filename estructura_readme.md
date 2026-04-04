# Arquitectura y Flujo de Tsukoyomi-IA (v3.5 - Cloud Integrated)

Este documento detalla el estado actual del flujo del sistema, integrando la nueva capa de persistencia en la nube y las optimizaciones de latencia.

## 1. Flujo del Backend (Asíncrono & Cloud)

El sistema ha sido migrado a un modelo **asíncrono (`asyncio`)** para optimizar las llamadas a la base de datos y al LLM sin bloquear el ciclo de eventos.

### Integración MongoDB Atlas:
- **Doble Escritura**: Todas las interacciones críticas se guardan simultáneamente en MongoDB Atlas (producción) y en ficheros JSONL locales (backup).
- **Colecciones**:
  - `tacticas`: El banco de ADN de 100 comportamientos.
  - `sesiones`: Historial de feedback y puntuaciones de usuarios.
  - `latencias`: Logs técnicos de rendimiento por request.

### Mejoras en el Pipeline:
1. **Fase A (NLP Optimized)**: 
   - Implementación de caché de sistema de archivos y memoria para el modelo DeBERTa. No se recalcula el contexto social si el escenario no ha cambiado.
2. **Fase B y C (Bifurcadas)**:
   - Se ejecutan **únicamente** en modo `Simulador`. 
   - El modo `Consejo` utiliza un canal directo (Fast-track) hacia el LLM para reducir el tiempo de respuesta en un 40% y evitar sesgos de frialdad.
3. **Health Check**: Endpoint `/health` habilitado para monitoreo de estado de conexión con Atlas y disponibilidad de motores de IA.

---

## 2. Paradigma del Frontend (Biometría Pasiva)

El frontend (`biometria.js`) actúa como un sensor biométrico de comportamiento.

### Métricas Capturadas:
- **Velocidad de Escritura**: Tiempo total para formular la respuesta.
- **Teclas de Borrado**: El conteo de Backspaces se traduce en el MLP como "autocensura" o "duda emocional".
- **Ratio de Duda**: Pausas prolongadas durante la redacción.
- **Detección de No-Emoción**: El sistema detecta si el texto fue pegado (Copy/Paste), lo que anula la influencia emocional de la biometría en el motor de fricción.

### Control de Sesión:
- El cliente es el dueño de la **memoria de corto plazo**. Envía el historial de chat completo en cada petición, permitiendo que el backend sea totalmente Stateless (escalable horizontalmente).

---

## 3. Guía de Operación (QuickStart)

### Migración de Datos:
Si necesitas sincronizar los datos locales con un nuevo Cluster de Atlas, usa:
```bash
python -m app.db.migrar
```

### Pre-entrenamiento del MLP:
Para calibrar la red neuronal con los 30 escenarios base antes de usarla:
```bash
python -m app.core.seed_trainer
```

### Ejecución:
```bash
uvicorn app.main:app --reload
```
---
*Nota: Este sistema prioriza la naturalidad del diálogo. Si el bot se siente "robótico", se debe revisar el archivo `tacticas.json` para asegurar que las directivas sean conductuales y no frases literales.*
