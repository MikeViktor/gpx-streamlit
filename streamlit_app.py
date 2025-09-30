import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import gpxpy
import io

# =====================
# Costanti
# =====================
SEVERITY_GAIN = 1.52  # severità aumentata del 10%
APP_TITLE = "Analisi Tracce GPX"
APP_ICON = "⛰️"

# =====================
# Funzioni di supporto
# =====================
def parse_gpx(file) -> pd.DataFrame:
    gpx = gpxpy.parse(file)
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append([point.latitude, point.longitude, point.elevation])
    df = pd.DataFrame(points, columns=["lat", "lon", "ele"])
    df["dist"] = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(df.lat)**2 + np.diff(df.lon)**2) * 111000)])
    return df

def compute_stats(df: pd.DataFrame) -> dict:
    ascent = np.sum(np.diff(df.ele).clip(min=0))
    descent = np.sum(-np.diff(df.ele).clip(max=0))
    distance_km = df.dist.iloc[-1] / 1000
    stats = {
        "distance": distance_km,
        "ascent": ascent,
        "descent": descent,
        "time_h": distance_km / 4.0 + ascent / 400.0  # stima T = km/4 + 400m/h
    }
    return stats

def compute_difficulty(stats: dict) -> tuple:
    score = (stats["distance"] * 2 + stats["ascent"] * 0.01) * SEVERITY_GAIN
    score = min(score, 100)  # cap a 100
    if score < 20:
        level, color = "Facile", "green"
    elif score < 40:
        level, color = "Medio", "yellowgreen"
    elif score < 60:
        level, color = "Impegnativo", "orange"
    elif score < 80:
        level, color = "Difficile", "red"
    elif score < 90:
        level, color = "Molto difficile", "purple"
    else:
        level, color = "Estremo", "black"
    return score, level, color

def gauge_chart(score: float, level: str, color: str):
    base = alt.Chart(pd.DataFrame({"value":[score]})).mark_arc(
        innerRadius=50, outerRadius=100
    ).encode(
        theta=alt.Theta("value", stack=True),
        color=alt.value(color)
    )
    text = alt.Chart(pd.DataFrame({"score":[f"{score:.1f} ({level})"]})).mark_text(
        align="center", baseline="middle", fontSize=18, fontWeight="bold", dy=-20
    ).encode(text="score:N")
    return base + text

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
st.title(APP_TITLE)

# File upload
uploaded_file = st.file_uploader("📂 Carica GPX", type=["gpx"])

# Layout top KPIs
col1, col2, col3 = st.columns(3)

if uploaded_file:
    df = parse_gpx(uploaded_file)
    stats = compute_stats(df)
    score, level, color = compute_difficulty(stats)

    with col1:
        st.metric("Distanza (km)", f"{stats['distance']:.2f}")
    with col2:
        st.metric("Dislivello + (m)", f"{stats['ascent']:.0f}")
    with col3:
        st.metric("Tempo Totale", f"{stats['time_h']:.1f} h")

    st.subheader("Indice di Difficoltà")
    st.altair_chart(gauge_chart(score, level, color), use_container_width=True)

    st.subheader("Profilo altimetrico")
    chart = alt.Chart(df).mark_line(color="blue").encode(
        x=alt.X("dist", title="Distanza (m)"),
        y=alt.Y("ele", title="Quota (m)")
    )
    st.altair_chart(chart, use_container_width=True)

else:
    # Mostra gauge vuoto ma con scala pronta
    st.subheader("Indice di Difficoltà")
    st.write("0 (Facile)")
    st.altair_chart(gauge_chart(0, "Facile", "green"), use_container_width=True)
