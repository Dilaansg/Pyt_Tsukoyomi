from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Configuración global de la aplicación.
    Lee variables de entorno desde el archivo .env.
    """
    gemini_api_key: str = ""
    mongodb_uri: str = ""

    # RANKING DESCENDENTE: Intentará con el primero, si falla va al siguiente.
    gemini_model_list: list[str] = [
        "models/gemini-3.1-flash-lite-preview",
        "models/gemini-3-flash-preview",
        "models/gemini-2.5-flash",
        "models/gemini-flash-latest",
        "models/gemma-3-27b-it"
    ]
    
    # CONFIGURACIÓN GROQ (TRABAJO 2)
    groq_api_key: str = ""
    groq_model_list: list[str] = [
        "llama-3.1-8b-instant",       # 14.4k RPD, 500k TPD, 30 RPM
        "llama-3.3-70b-versatile"     # Fallback, 1K RPD, 100k TPD, 30 RPM
    ]
    
    max_historial_chat: int = 20

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8', 
        extra='ignore'
    )

config = Settings()
