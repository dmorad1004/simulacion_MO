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

st.title("⚛️ Simulación del Protocolo BB84")
st.markdown(
    """
    Esta simulación interactiva demuestra el protocolo de criptografía cuántica **BB84** y su
    relación con la **probabilidad de detección cuántica**. Utiliza la polarización de fotones 
    para establecer una clave secreta segura entre dos partes: **Gatalice** y **MichiBob**.
    """
)

# Probabilidad de detección
with st.expander(
    "Probabilidad de Detección según el Ángulo de Medición", expanded=True
):
    st.markdown(
        r"""
    En el contexto del protocolo BB84, la **probabilidad de detección** de un fotón por un polarizador
    depende del ángulo entre la dirección de polarización del fotón y la orientación del polarizador.
    
    Esta probabilidad está dada por la expresión:
    $$
    P(\theta) = \cos^2(\theta)
    $$
    donde $\theta$ es la diferencia angular entre la polarización del fotón y el eje del polarizador.

    Esta ley probabilística es fundamental para comprender por qué, cuando las bases no coinciden, 
    el resultado de la medición es aleatorio.
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
        if polarizer_angle == 0 or polarizer_angle == 180:
            st.success("Bases alineadas ⇒ detección 100 %.")
        elif polarizer_angle == 90:
            st.error("Bases ortogonales ⇒ detección 0 %.")
        elif polarizer_angle == 45 or polarizer_angle == 135:
            st.info("Bases diagonales ⇒ detección 50 %.")
        else:
            st.info("Probabilidad de detección intermedia.")


# Sidebar – Controles de la simulación BB84
with st.sidebar:
    st.header("️Controles de la Simulación")
    num_bits = st.slider(
        "Número de fotones / bits a enviar:",
        min_value=4,
        max_value=100,
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
            st.error("MeowEve detectada: QBER > 0%")
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
st.header("Consideraciones")
st.markdown(
    """
    * La **probabilidad de detección** de un fotón depende del ángulo entre la base en que fue preparado y la base en que se mide. Cuando las bases no coinciden, el resultado es impredecible.
    * El **principio de no clonación cuántica** impide que un espía (MeowEve) copie el estado de un fotón sin alterar su comportamiento. Esto garantiza que cualquier intento de interceptación deje huella.
    * El protocolo **BB84** permite a dos partes (como Gatalice y MichiBob) detectar la presencia de espionaje comparando públicamente sus bases y calculando la tasa de error (**QBER**).
    """
)
