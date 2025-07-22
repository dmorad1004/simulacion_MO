"""
Streamlit UI – bb84_streamlit_app.py
-----------------------------------
Interfaz interactiva para demostrar la Ley de Malus y el protocolo BB84.
La lógica cuántica/estadística vive en `bb84.py` y se importa como `bb`.

Ejecución:
    streamlit run bb84_streamlit_app.py
Dependencias mínimas:
    pip install streamlit numpy pandas
"""

from __future__ import annotations

import streamlit as st
import numpy as np

# import pandas as pd
import bb84 as bb  # Módulo de lógica

# Configuración de la página
st.set_page_config(
    page_title="Simulador BB84 y Ley de Malus",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚛️ Simulación del Protocolo BB84 y la Ley de Malus")
st.markdown(
    """
    Esta aplicación de manera interactiva, demuestra el protocolo de criptografía cuántica **BB84** y su
    relación con la **Ley de Malus**. Utiliza la polarización de fotones para establecer una
    clave secreta segura entre dos partes (Gatalice y MichiBob).
    """
)

# Ley de Malus
with st.expander("Demostración Interactiva de la Ley de Malus", expanded=True):
    st.markdown(
        r"""
    La Ley de Malus describe cómo la intensidad (o probabilidad de transmisión de un fotón)
    cambia según el ángulo entre la polarización incidente y el eje del polarizador:
    $$ P(\theta) = \cos^2(\theta) $$
    """
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        polarizer_angle = st.slider("Ángulo del polarizador (θ)", 0, 180, 45, 1)
        probability = np.cos(np.deg2rad(polarizer_angle)) ** 2
        st.metric(
            label=f"Probabilidad de transmisión para θ = {polarizer_angle}°",
            value=f"{probability:.2%}",
        )
        if polarizer_angle % 90 == 0:
            if polarizer_angle == 0:
                st.success("Bases coinciden ⇒ transmisión 100 %.")
            else:
                st.error("Bases ortogonales ⇒ transmisión 0 %.")
        elif polarizer_angle == 45:
            st.info("Bases diagonales ⇒ probabilidad 50 %.")

# Sidebar – Controles de la simulación BB84
with st.sidebar:
    st.header("️Controles de la Simulación")
    num_bits = st.slider(
        "Número de fotones / bits a enviar:",
        min_value=4,
        max_value=40,
        value=12,
        step=2,
    )
    is_eve_present = st.checkbox("Incluir un espía (MeowEve) en el canal")
    seed = st.number_input(
        "Semilla aleatoria (0 = aleatoria)", min_value=0, value=0, step=1
    )

    start_simulation = st.button(" 🏃🏻 Correr la Simulación", type="primary")


# Ejecución de la simulación
if start_simulation:
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    alice_bits, alice_bases = bb.generate_alice_data(num_bits, rng)
    photons_sent = bb.encode_photons(alice_bits, alice_bases)

    photons_for_bob = photons_sent
    eve_bases = eve_bits = photons_resent = None
    if is_eve_present:
        st.subheader("Intervención de MeowEve")
        st.warning("¡MeowEve intercepta el canal cuántico!")
        st.image("images/meoweve.png", caption="MeowEve está espiando")

        photons_for_bob, eve_bases, eve_bits = bb.eve_intervention(photons_sent, rng)
        photons_resent = bb.encode_photons(eve_bits, eve_bases)

    bob_bases = bb.generate_bob_bases(num_bits, rng)
    bob_measured_bits = bb.measure_photons(photons_for_bob, bob_bases, rng)

    # 4. Comparación de bases y clave
    alice_key, bob_key, matches = bb.compare_bases_and_get_key(
        alice_bases, bob_bases, alice_bits, bob_measured_bits
    )

    # 5. Tabla de resultados
    st.subheader("📊 Resultados de cada fotón")
    df = bb.create_results_dataframe(
        num_bits,
        alice_bits,
        alice_bases,
        photons_sent,
        bob_bases,
        bob_measured_bits,
        matches,
        is_eve_present,
        eve_bases,
        eve_bits,
        photons_resent,
    )
    st.dataframe(df)

    # 6. Clave final y QBER
    st.subheader("🔑 Claves finales")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Clave de Gatalice:**")
        st.code("".join(map(str, alice_key)) if len(alice_key) else "—")
    with col2:
        st.markdown("**Clave de MichiBob:**")
        st.code("".join(map(str, bob_key)) if len(bob_key) else "—")

    if len(alice_key):
        qber = (alice_key != bob_key).mean() * 100
        st.metric("QBER (Tasa de error)", f"{qber:.2f}%")
        if is_eve_present and qber > 0:
            st.error("Eve detectada: QBER > 0%")
        elif is_eve_present:
            st.info(
                "MeowEve no introdujo errores detectables en esta muestra (poco probable a gran escala)."
            )
        else:
            st.success("Canal seguro: QBER ≈ 0%")
    else:
        st.warning("No hubo coincidencias de base — aumenta el número de fotones.")

# Conclusión
st.markdown("---")
st.header("Conclusiones")
st.markdown(
    """
    * **Ley de Malus:** explica la probabilidad de detección cuando las bases no coinciden.
    * **Principio de no clonación:** Eve no puede copiar un qubit sin colapsar su estado; al
      medir en la base errónea, introduce errores detectables.
    * **BB84:** garantiza la detección de espionaje mediante la comparación pública de bases y
      el cálculo del QBER.
    """
)
