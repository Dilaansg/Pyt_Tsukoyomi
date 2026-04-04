from google import genai
from google.genai import types
import groq
from app.config.settings import config

class LLMRouter:
    def __init__(self):
        self.gemini_client = None
        if config.gemini_api_key:
            self.gemini_client = genai.Client(api_key=config.gemini_api_key)
            
        # Setup Groq
        self.groq_client = None
        if config.groq_api_key:
            self.groq_client = groq.AsyncGroq(api_key=config.groq_api_key)

    async def llamar_llm(self, sys_prompt: str, user_text: str, historial: list, imagen_base64: str = None):
        # 1. INTENTO PRIMARIO: GEMINI (Rotación Cíclica)
        if self.gemini_client:
            async with self.gemini_client.aio as async_client:
                for modelo_id in config.gemini_model_list:
                    try:
                        # Config local de generación
                        generation_config = types.GenerateContentConfig(
                            temperature=1,
                            top_p=0.95,
                            top_k=40,
                            max_output_tokens=8192,
                            system_instruction=sys_prompt
                        )
                        
                        historial_formateado = []
                        for h in historial:
                            role = "user" if getattr(h, "rol", "user").lower() == "user" else "model"
                            contenido = getattr(h, "contenido", "")
                            historial_formateado.append(types.Content(role=role, parts=[types.Part.from_text(text=contenido)]))
    
                        chat_session = async_client.chats.create(
                            model=modelo_id, 
                            config=generation_config,
                            history=historial_formateado
                        )
                        if imagen_base64:
                            import base64
                            img_data = base64.b64decode(imagen_base64)
                            mensaje_parts = [
                                types.Part.from_bytes(data=img_data, mime_type="image/jpeg"),
                                types.Part.from_text(text=user_text)
                            ]
                            respuesta = await chat_session.send_message(mensaje_parts)
                        else:
                            respuesta = await chat_session.send_message(user_text)
                        
                        return respuesta.text, f"gemini ({modelo_id})"
                    
                    except Exception as e:
                        err_str = str(e)
                        if "429" in err_str or "quota" in err_str.lower() or "exhausted" in err_str.lower() or "payload too large" in err_str.lower():
                            if imagen_base64:
                                import json
                                return json.dumps({"error": "rate_limit", "fallback_sugerido": True}), "gemini (vision fallback)"
                            continue # Prueba el siguiente modelo de Gemini
                        
                        # Si el error no es de quota, rompe y salta a Groq directamente (si no es vision)
                        print(f"[Gemini] Error no-429 con {modelo_id}: {err_str}")
                        if imagen_base64:
                            import json
                            return json.dumps({"error": "rate_limit", "fallback_sugerido": True}), "gemini (vision fallback)"
                        break
        
        # 2. INTENTO SECUNDARIO (FALLBACK): GROQ (Solo si NO es una imagen, porque la visión depende de Gemini en Fase 1)
        if self.groq_client and not imagen_base64:
            for modelo_id in config.groq_model_list:
                try:
                    # Convertir historial al formato OpenAI/Groq
                    mensajes_groq = [{"role": "system", "content": sys_prompt}]
                    for h in historial:
                        role = "user" if h.rol.lower() == "user" else "assistant"
                        mensajes_groq.append({"role": role, "content": h.contenido})
                        
                    mensajes_groq.append({"role": "user", "content": user_text})
                    
                    response = await self.groq_client.chat.completions.create(
                        model=modelo_id,
                        messages=mensajes_groq,
                        temperature=1,
                        max_completion_tokens=4096,
                        top_p=0.95,
                    )
                    
                    return response.choices[0].message.content, f"groq ({modelo_id})"

                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str:
                        continue # Siguiente de Groq
                    if "connection" in err_str.lower():
                        continue
                    break # Falla crítica

        # 3. FALLA TOTAL
        if imagen_base64:
            import json
            return json.dumps({"error": "rate_limit", "fallback_sugerido": True}), "vision fallback"
        raise RuntimeError("Todos los proveedores LLM fallaron o no están configurados.")
