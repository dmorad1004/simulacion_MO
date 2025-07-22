"""
Modulo: bb84.py
--------------------
Contiene la lógica del protocolo BB84 y utilidades relacionadas con la Ley de Malus.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Utilidades de generación y codificación para Alice


def generate_alice_data(num_bits: int, rng: np.random.Generator | None = None):
    """Genera bits y bases ("+" rectilínea, "x" diagonal) para Alice."""
    if rng is None:
        rng = np.random.default_rng()
    bits = rng.integers(0, 2, num_bits, dtype=int)
    bases = rng.choice(["+", "x"], num_bits)
    return bits, bases


def encode_photons(bits: np.ndarray, bases: np.ndarray) -> np.ndarray:
    """Convierte (bit, base) → ángulo de polarización en grados."""
    # Mapeo directo usando lógica vectorizada
    polarizations = np.empty_like(bits, dtype=int)
    # Base rectilínea “+” → 0° (bit 0) o 90° (bit 1)
    plus_mask = bases == "+"
    polarizations[plus_mask] = np.where(bits[plus_mask] == 1, 90, 0)
    # Base diagonal “x” → 45° (bit 0) o 135° (bit 1)
    x_mask = ~plus_mask
    polarizations[x_mask] = np.where(bits[x_mask] == 1, 135, 45)
    return polarizations


# Bob y mediciones (incluyendo Ley de Malus)


def generate_bob_bases(num_bits: int, rng: np.random.Generator | None = None):
    """Genera bases aleatorias de medición para Bob."""
    if rng is None:
        rng = np.random.default_rng()
    return rng.choice(["+", "x"], num_bits)


def _angle_for_base(base: str) -> int:
    """Devuelve el ángulo de referencia de la base para representar el estado |0⟩."""
    return 0 if base == "+" else 45


def measure_photons(
    polarizations: np.ndarray,
    bob_bases: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Realiza la medición de cada fotón usando la Ley de Malus."""
    if rng is None:
        rng = np.random.default_rng()
    #  ángulo de referencia para |0⟩ en cada base de Bob
    ref_angles = np.vectorize(_angle_for_base)(bob_bases)
    # Probabilidad de colapsar a |0⟩
    prob_0 = np.cos(np.deg2rad(polarizations - ref_angles)) ** 2
    # Genera bits medidos
    random_vals = rng.random(len(polarizations))
    measured_bits = (random_vals >= prob_0).astype(
        int
    )  # 0 si r < prob_0, 1 si r ≥ prob_0
    return measured_bits


# Intervención de Eve(espia :v)


def eve_intervention(polarizations: np.ndarray, rng: np.random.Generator | None = None):
    """Eve mide con bases aleatorias y reenvía fotones modificados."""
    num_bits = len(polarizations)
    if rng is None:
        rng = np.random.default_rng()
    eve_bases = generate_bob_bases(num_bits, rng)
    eve_bits = measure_photons(polarizations, eve_bases, rng)
    resent_photons = encode_photons(eve_bits, eve_bases)
    return resent_photons, eve_bases, eve_bits


# Comparación y clave


def compare_bases_and_get_key(
    alice_bases: np.ndarray,
    bob_bases: np.ndarray,
    alice_bits: np.ndarray,
    bob_bits: np.ndarray,
):
    """Devuelve las claves filtradas y un array booleano con coincidencias."""
    matches = alice_bases == bob_bases
    alice_key = alice_bits[matches]
    bob_key = bob_bits[matches]
    return alice_key, bob_key, matches


# Utilidades de presentación
# Ola ahijada, si lees esto reclama un dulce.


def polarization_to_symbol(angle: int) -> str:
    """Representación unicode sencilla para ilustrar la polarización."""
    return {0: "→ (0°)", 90: "↑ (90°)", 45: "↗ (45°)", 135: "↖ (135°)"}.get(angle, "?")


def create_results_dataframe(
    num_bits: int,
    alice_bits: np.ndarray,
    alice_bases: np.ndarray,
    alice_photons: np.ndarray,
    bob_bases: np.ndarray,
    bob_bits: np.ndarray,
    matches: np.ndarray,
    is_eve_present: bool = False,
    eve_bases: np.ndarray | None = None,
    eve_bits: np.ndarray | None = None,
    eve_photons: np.ndarray | None = None,
) -> pd.DataFrame:
    """Regresa un pd.DataFrame con los datos de cada fotón."""
    data: dict[str, list] = {
        "Bit de Gatalice": list(alice_bits),
        "Base de Gatalice": list(alice_bases),
        "Fotón Enviado (Polarización)": [
            polarization_to_symbol(p) for p in alice_photons
        ],
    }

    if is_eve_present and eve_bases is not None:
        data["Base de MeowEve"] = list(eve_bases)
        data["Bit Medido por MeowEve"] = list(eve_bits)
        data["Fotón Reenviado por MeowEve"] = [
            polarization_to_symbol(p) for p in eve_photons
        ]

    data.update(
        {
            "Base de MichiBob": list(bob_bases),
            "Bit Medido por MichiBob": list(bob_bits),
            "Bases Coinciden": ["Sí ✅" if m else "No ❌" for m in matches],
        }
    )

    df = pd.DataFrame(data)
    df.index = [f"Fotón {i+1}" for i in range(num_bits)]
    return df


__all__ = [
    "generate_alice_data",
    "encode_photons",
    "generate_bob_bases",
    "measure_photons",
    "eve_intervention",
    "compare_bases_and_get_key",
    "polarization_to_symbol",
    "create_results_dataframe",
]
