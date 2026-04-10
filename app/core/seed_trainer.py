"""
seed_trainer.py — Fine-tuning de arranque para RedMediacionMLP

Ejecutar una sola vez antes del primer despliegue:
    python -m app.core.seed_trainer

Guarda los pesos en app/models/mlp_seed.pt
El lifespan de main.py los cargará automáticamente si existen.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path

from app.core.friction_model import RedMediacionMLP

# ─────────────────────────────────────────────────────────────
#  SEED DATA (30 ejemplos etiquetados manualmente)
#  Formato: (features_11d, targets_4d)
#  Features: [soc_A, soc_P, soc_U, soc_V,
#             tiempo_escritura, teclas_borrado, pulsaciones,
#             ratio_duda, longitud, edad, copy_paste]
#  Targets:  [terquedad, frialdad, sarcasmo, frustracion]
# ─────────────────────────────────────────────────────────────
SEED_DATA = [
    # Jefe autoritario, directo, sin tiempo
    ([0.3, 0.8, 0.2, 0.1, 3.0, 1, 20, 0.3, 80, 28, 0],    [0.9, 0.8, 0.1, 0.2]),
    # Compañero que evade responsabilidades
    ([0.5, -0.3, 0.4, 0.3, 8.0, 5, 45, 0.7, 120, 22, 0],   [0.7, 0.4, 0.3, 0.6]),
    # Ex pareja emocionalmente manipuladora
    ([0.7, 0.0, 0.2, 0.6, 12.0, 8, 60, 0.9, 200, 19, 0],   [0.5, 0.2, 0.6, 0.3]),
    # Entrenador/autoridad fría
    ([0.4, 0.6, 0.1, 0.1, 2.0, 0, 15, 0.1, 50, 25, 0],     [0.8, 0.9, 0.2, 0.1]),
    # Amigo con presión social (alcohol, fiestas)
    ([0.6, -0.1, 0.3, 0.7, 10.0, 4, 40, 0.6, 90, 16, 0],   [0.4, 0.2, 0.5, 0.7]),
    # Colega que interrumpe constantemente
    ([0.3, 0.4, 0.6, 0.4, 5.0, 3, 35, 0.5, 70, 30, 0],     [0.6, 0.5, 0.4, 0.6]),
    # Padre/madre controlador/a
    ([0.5, 0.7, 0.1, 0.2, 6.0, 2, 25, 0.3, 100, 16, 0],    [0.8, 0.6, 0.2, 0.3]),
    # Chica/chico que da señales mixtas
    ([0.8, 0.0, 0.1, 0.5, 15.0, 10, 80, 0.95, 180, 18, 0], [0.3, 0.3, 0.4, 0.5]),
    # Profesor/a que ignora al alumno
    ([0.6, 0.5, 0.3, 0.1, 4.0, 1, 20, 0.2, 60, 20, 0],     [0.7, 0.8, 0.3, 0.4]),
    # Compañero sarcástico en reunión
    ([0.4, 0.3, 0.4, 0.3, 7.0, 4, 50, 0.6, 130, 27, 0],    [0.5, 0.5, 0.8, 0.5]),
    # Vendedor/a insistente
    ([0.2, 0.5, 0.7, 0.2, 2.0, 0, 10, 0.1, 40, 35, 0],     [0.6, 0.4, 0.3, 0.7]),
    # Amigo/a pasivo-agresivo/a
    ([0.5, 0.1, 0.2, 0.5, 9.0, 6, 55, 0.8, 160, 21, 0],    [0.4, 0.4, 0.9, 0.3]),
    # Jefe micromanager
    ([0.3, 0.9, 0.5, 0.0, 3.0, 1, 15, 0.2, 45, 32, 0],     [0.95, 0.7, 0.2, 0.5]),
    # Familiar que no escucha
    ([0.6, 0.4, 0.1, 0.3, 8.0, 5, 40, 0.7, 100, 17, 0],    [0.8, 0.5, 0.3, 0.4]),
    # Situación de alta urgencia y confrontación directa
    ([0.5, 0.5, 0.9, 0.1, 1.0, 0, 8, 0.1, 30, 28, 0],      [0.7, 0.8, 0.1, 0.9]),
    # Usuario ansioso, escribe mucho y borra mucho
    ([0.9, 0.0, 0.3, 0.4, 20.0, 15, 120, 0.95, 250, 19, 0],[0.2, 0.1, 0.2, 0.9]),
    # Interacción casual y cómoda
    ([0.1, 0.0, 0.1, 0.9, 5.0, 1, 30, 0.2, 80, 22, 0],     [0.1, 0.1, 0.1, 0.1]),
    # Jefe que da feedback negativo fríamente
    ([0.4, 0.8, 0.3, 0.0, 4.0, 2, 22, 0.3, 70, 29, 0],     [0.8, 0.9, 0.3, 0.3]),
    # Compañero de trabajo que toma crédito ajeno
    ([0.5, 0.2, 0.3, 0.3, 9.0, 6, 60, 0.8, 140, 25, 0],    [0.6, 0.5, 0.5, 0.5]),
    # Amigo que cancela planes a última hora
    ([0.4, -0.1, 0.5, 0.6, 11.0, 7, 70, 0.8, 170, 20, 0],  [0.4, 0.3, 0.6, 0.6]),
    # Persona que minimiza tus logros
    ([0.7, 0.3, 0.2, 0.3, 10.0, 6, 55, 0.8, 120, 23, 0],   [0.5, 0.6, 0.7, 0.4]),
    # Negociador/a duro/a en contexto laboral
    ([0.3, 0.6, 0.6, 0.2, 3.0, 1, 18, 0.2, 55, 35, 0],     [0.85, 0.7, 0.2, 0.6]),
    # Usuario que usa copy-paste (bajo esfuerzo)
    ([0.2, 0.1, 0.1, 0.7, 1.0, 0, 5, 0.05, 300, 24, 1],    [0.2, 0.3, 0.3, 0.2]),
    # Persona que rompe confianza (traición)
    ([0.8, 0.0, 0.1, 0.7, 14.0, 9, 75, 0.9, 210, 18, 0],   [0.6, 0.2, 0.5, 0.7]),
    # Debate acalorado entre iguales
    ([0.5, 0.1, 0.7, 0.4, 8.0, 5, 60, 0.7, 150, 26, 0],    [0.7, 0.3, 0.5, 0.8]),
    # Persona distante pero no hostil
    ([0.3, 0.3, 0.1, 0.2, 6.0, 2, 30, 0.3, 80, 30, 0],     [0.4, 0.7, 0.2, 0.2]),
    # Adulto mayor que no acepta cambios
    ([0.4, 0.5, 0.1, 0.2, 5.0, 3, 28, 0.4, 70, 50, 0],     [0.9, 0.6, 0.1, 0.2]),
    # Líder carismático que manipula emocionalmente
    ([0.6, 0.6, 0.2, 0.5, 6.0, 2, 35, 0.4, 100, 28, 0],    [0.7, 0.3, 0.6, 0.3]),
    # Situación de alta ansiedad social (primer día)
    ([0.95, -0.1, 0.4, 0.1, 18.0, 12, 100, 0.9, 200, 17, 0], [0.2, 0.2, 0.1, 0.95]),
    # Resolución empática (mentor natural)
    ([0.3, 0.2, 0.1, 0.8, 7.0, 2, 35, 0.3, 90, 28, 0],     [0.3, 0.1, 0.0, 0.1]),
]


def entrenar_seed(
    epocas: int = 50,
    lr: float = 1e-3,
    output_path: str = "app/models/mlp_seed.pt",
):
    print(f"[SEED TRAINER] Iniciando fine-tuning con {len(SEED_DATA)} ejemplos por {epocas} épocas...")

    modelo = RedMediacionMLP()
    modelo.train()

    X = torch.tensor([d[0] for d in SEED_DATA], dtype=torch.float32)
    Y = torch.tensor([d[1] for d in SEED_DATA], dtype=torch.float32)

    # Calcular mean y std del seed dataset y guardarlos como buffers
    modelo.normalizador.mean = X.mean(dim=0)
    modelo.normalizador.std = X.std(dim=0).clamp(min=1e-6)

    optim = torch.optim.Adam(modelo.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoca in range(epocas):
        optim.zero_grad()
        salida = modelo(X)
        loss = loss_fn(salida, Y)
        loss.backward()
        optim.step()
        if (epoca + 1) % 10 == 0:
            print(f"  Época {epoca + 1:03d} — Loss: {loss.item():.6f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), output_path)
    print(f"[SEED TRAINER] Pesos guardados en: {output_path}")


if __name__ == "__main__":
    entrenar_seed()
