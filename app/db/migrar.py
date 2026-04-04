"""
Migra los datos locales a MongoDB Atlas.
Ejecutar UNA SOLA VEZ: python -m app.db.migrar
"""
import asyncio
import json
from pathlib import Path
from app.db.mongo import col_tacticas, col_sesiones, col_latencias

async def main():
    print("=== MIGRACIÓN TSUKOYOMI → MONGODB ATLAS ===\n")

    # Tácticas
    tacticas_path = Path("app/config/tacticas.json")
    if tacticas_path.exists():
        with open(tacticas_path, encoding="utf-8") as f:
            tacticas = json.load(f)
        col = col_tacticas()
        await col.drop()
        if tacticas:
            await col.insert_many(tacticas)
        await col.create_index("id", unique=True)
        print(f"Tácticas: {len(tacticas)} documentos insertados.")
    else:
        print("SKIP: tacticas.json no encontrado.")

    # Sesiones de feedback
    sesiones_path = Path("app/data/dataset_interacciones.jsonl")
    sesiones = _leer_jsonl(sesiones_path)
    if sesiones:
        await col_sesiones().insert_many(sesiones)
        await col_sesiones().create_index("timestamp")
        print(f"Sesiones: {len(sesiones)} documentos insertados.")
    else:
        print("SKIP: dataset_interacciones.jsonl vacío o no encontrado.")

    # Logs de latencia
    latencias_path = Path("app/data/latency_logs.jsonl")
    logs = _leer_jsonl(latencias_path)
    if logs:
        await col_latencias().insert_many(logs)
        print(f"Latencias: {len(logs)} documentos insertados.")
    else:
        print("SKIP: latency_logs.jsonl vacío o no encontrado.")

    print("\nMigración completa.")

def _leer_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    docs = []
    with open(path, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                try:
                    docs.append(json.loads(linea))
                except json.JSONDecodeError:
                    continue
    return docs

if __name__ == "__main__":
    asyncio.run(main())
