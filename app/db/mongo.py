from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config.settings import config

_client: AsyncIOMotorClient | None = None

def get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(
            config.mongodb_uri,
            serverSelectionTimeoutMS=5000,
        )
    return _client

def get_db() -> AsyncIOMotorDatabase:
    return get_client()["tsukoyomi"]

def col_tacticas():  return get_db()["tacticas"]
def col_sesiones():  return get_db()["sesiones"]
def col_latencias(): return get_db()["latencias"]

async def ping() -> bool:
    try:
        await get_client().admin.command("ping")
        return True
    except Exception:
        return False
