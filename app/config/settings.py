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
    max_historial_chat: int = 20

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding='utf-8', 
        extra='ignore'
    )

config = Settings()
