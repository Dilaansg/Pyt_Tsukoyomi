"""
app/core/llm_router.py — Router LLM unificado con SDK v2 de Google (google-genai)
"""
from google import genai
from google.genai import types
from app.config.settings import config


class LLMRouter:
    def __init__(self):
        self.gemini_client = None
        if config.gemini_api_key:
            self.gemini_client = genai.Client(api_key=config.gemini_api_key)

        self.groq_client = None
        if config.groq_api_key:
            try:
                from groq import AsyncGroq
                self.groq_client = AsyncGroq(api_key=config.groq_api_key)
            except ImportError:
                print("[LLMRouter] groq no instalado. Solo Gemini disponible.")

    async def llamar_llm(
        self,
        sys_prompt: str,
        user_text: str,
        historial: list,
        imagen_base64: str = None,
    ) -> tuple[str, str]:
        """
        Intenta Gemini primero, luego Groq como fallback.
        historial: lista de objetos con atributos .role y .content,
                   o dicts con claves "role"/"content".
        Retorna: (texto_respuesta, nombre_modelo)
        """
        # Normalizar historial a lista de dicts
        historial_norm = []
        for h in historial:
            if hasattr(h, "role"):
                role = h.role
                content = h.content
            else:
                role = h.get("role", "user")
                content = h.get("content", "")
            historial_norm.append({"role": role, "content": content})

        # Intentar Gemini
        if self.gemini_client:
            result = await self._llamar_gemini(
                sys_prompt, user_text, historial_norm, imagen_base64
            )
            if result:
                return result

        # Fallback a Groq (solo texto)
        if self.groq_client and not imagen_base64:
            result = await self._llamar_groq(sys_prompt, user_text, historial_norm)
            if result:
                return result

        # Fallback especial para vision si todo falla
        if imagen_base64:
            import json
            return json.dumps({"error": "rate_limit", "fallback_sugerido": True}), "vision_fallback"

        raise RuntimeError(
            "Todos los proveedores LLM fallaron o no están configurados. "
            "Configura GEMINI_API_KEY y/o GROQ_API_KEY."
        )

    async def _llamar_gemini(
        self, sys_prompt: str, user_text: str, historial: list, imagen_base64: str = None
    ) -> tuple[str, str] | None:
        """Intenta con cada modelo Gemini hasta que uno funcione."""
        gen_config = types.GenerateContentConfig(
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=2048,
            system_instruction=sys_prompt,
        )

        # Convertir historial al formato de Gemini v2
        historial_gemini = []
        for h in historial:
            role = "user" if h["role"].lower() == "user" else "model"
            historial_gemini.append(
                types.Content(role=role, parts=[types.Part.from_text(text=h["content"])])
            )

        for modelo_id in config.gemini_model_list:
            try:
                chat = self.gemini_client.aio.chats.create(
                    model=modelo_id,
                    config=gen_config,
                    history=historial_gemini,
                )

                if imagen_base64:
                    import base64
                    img_bytes = base64.b64decode(imagen_base64)
                    partes = [
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=user_text),
                    ]
                    resp = await chat.send_message(partes)
                else:
                    resp = await chat.send_message(user_text)

                return resp.text, f"gemini ({modelo_id})"

            except Exception as e:
                err = str(e).lower()
                if any(k in err for k in ["429", "quota", "exhausted", "resource_exhausted", "unavailable", "payload too large"]):
                    if imagen_base64:
                        import json
                        return json.dumps({"error": "rate_limit", "fallback_sugerido": True}), "gemini (vision fallback)"
                    print(f"[LLMRouter] Gemini {modelo_id} sin cuota. Siguiente...")
                    continue
                print(f"[LLMRouter] Error Gemini {modelo_id}: {e}")
                if imagen_base64:
                    import json
                    return json.dumps({"error": "rate_limit", "fallback_sugerido": True}), "gemini (vision fallback)"
                break  # Error no-quota → saltar a Groq

        return None

    async def _llamar_groq(
        self, sys_prompt: str, user_text: str, historial: list
    ) -> tuple[str, str] | None:
        """Intenta con cada modelo Groq hasta que uno funcione."""
        mensajes = [{"role": "system", "content": sys_prompt}]
        for h in historial:
            # Obtener atributos usando getattr con fallback a dict.get para soportar objetos Pydantic o dicts
            raw_role = getattr(h, "role", h.get("role", "user") if isinstance(h, dict) else "user")
            raw_content = getattr(h, "content", h.get("content", "") if isinstance(h, dict) else "")
            
            # Groq requiere que el rol sea 'assistant', no 'model' (que es de Gemini) ni 'bot'
            mapped_role = "assistant" if raw_role in ("model", "bot", "assistant") else "user"
            mensajes.append({"role": mapped_role, "content": raw_content})
            
        mensajes.append({"role": "user", "content": user_text})

        for modelo_id in config.groq_model_list:
            try:
                resp = await self.groq_client.chat.completions.create(
                    model=modelo_id,
                    messages=mensajes,
                    max_tokens=1024,
                    temperature=0.9,
                )
                return resp.choices[0].message.content, f"groq ({modelo_id})"
            except Exception as e:
                err = str(e).lower()
                if any(k in err for k in ["429", "rate", "limit", "quota", "connection"]):
                    print(f"[LLMRouter] Groq {modelo_id} con rate limit. Siguiente...")
                    continue
                print(f"[LLMRouter] Error Groq {modelo_id}: {e}")
                break

        return None
