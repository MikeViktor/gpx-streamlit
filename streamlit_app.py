# streamlit_app.py
import io
import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

APP_TITLE = "Analisi Tracce GPX"
APP_ICON = "⛰️"
SEVERITY_GAIN = 1.52  # severità +10% rispetto alla versione precedente

# ====== tentativo di import gpxpy; se non c'è useremo un parser XML di fallback ======
try:
    import gpxpy  # type: ignore
    HAS_GPXPY = True
except Exception:
    import xml.etree.ElementTree as ET
    HAS_GPXPY = False

# ----------------- utilità -----------------
def _dist_km(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.hypot(dx, dy)

def _difficulty_from_stats(distance_km, ascent_m):
    # scorings semplice (come prima) + severità globale
    score = (distance_km * 2.0 + ascent_m * 0.01) * SEVERITY_GAIN
    score = max(0.0, min(100.0, score))
    if score < 20:   return score, "Facile", "green"
    if score < 40:   return score, "Medio", "yellowgreen"
    if score < 60:   return score, "Impegnativo", "orange"
    if score < 80:   return score, "Difficile", "red"
    if score < 90:   return score, "Molto difficile", "purple"
    return score, "Estremo", "black"

def _format_time_h(distance_km, ascent_m):
    # stima veloce: T(h) = km/4 + D+/400 (come prima)
    t_h = (distance_km / 4.0) + (ascent_m / 400.0)
    return f"{t_h:.1f} h"

# ----------------- parsing GPX -----------------
def parse_gpx_gpxpy(uploaded_file) -> pd.DataFrame:
    # Streamlit dà un UploadedFile (bytes). gpxpy vuole testo.
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    gpx = gpxpy.parse(io.StringIO(text))
    pts = []
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if p.elevation is None:  # saltiamo i punti senza quota
                    continue
                pts.append((p.latitude, p.longitude, float(p.elevation)))
    if not pts:
        # rotta/waypoint di fallback
        for rte in gpx.routes:
            for p in rte.points:
                if p.elevation is None:
                    continue
                pts.append((p.latitude, p.longitude, float(p.elevation)))
    df = pd.DataFrame(pts, columns=["lat", "lon", "ele"])
    if df.empty:
        return df
    # distanza cumulata
    dist = [0.0]
    for i in range(1, len(df)):
        dist.append(dist[-1] + _dist_km(df.lat.iloc[i-1], df.lon.iloc[i-1], df.lat.iloc[i], df.lon.iloc[i]) * 1000.0)
    df["dist_m"] = dist
    return df

def parse_gpx_xml(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    root = ET.fromstring(raw)
    pts = []
    def is_tag(el, name):
        t = el.tag
        return t.endswith('}' + name) or t == name
    # cerca trkpt, poi rtept, poi wpt
    for wanted in ("trkpt", "rtept", "wpt"):
        pts.clear()
        for el in root.iter():
            if is_tag(el, wanted):
                la = el.attrib.get("lat"); lo = el.attrib.get("lon")
                if la is None or lo is None: continue
                ele = None
                for ch in el:
                    if is_tag(ch, "ele"):
                        ele = ch.text; break
                if ele is None: continue
                try:
                    pts.append((float(la), float(lo), float(ele)))
                except:
                    pass
        if pts: break
    df = pd.DataFrame(pts, columns=["lat","lon","ele"])
    if df.empty:
        return df
    dist = [0.0]
    for i in range(1, len(df)):
        dist.append(dist[-1] + _dist_km(df.lat.iloc[i-1], df.lon.iloc[i-1], df.lat.iloc[i], df.lon.iloc[i]) * 1000.0)
    df["dist_m"] = dist
    return df

def parse_gpx(uploaded_file) -> pd.DataFrame:
    if HAS_GPXPY:
        return parse_gpx_gpxpy(uploaded_file)
    return parse_gpx_xml(uploaded_file)

# ----------------- grafici -----------------
def difficulty_gauge(score: float, level: str, color: str):
    # gauge semplice: arc + testo
    # base: mostriamo sempre qualcosa, anche a 0
    base = alt.Chart(pd.DataFrame({"x":[0, score], "y":[1,1]})).mark_arc(innerRadius=60, outerRadius=100).encode(
        theta=alt.Theta("x:Q", stack=False), color=alt.value(color)
    )
    text = alt.Chart(pd.DataFrame({"t":[f"{score:.1f} ({level})"]})).mark_text(
        font="Segoe UI", fontSize=20, fontWeight="bold", dy=-20
    ).encode(text="t:N")
    return (base + text).properties(height=220)

def profile_chart(df: pd.DataFrame):
    return alt.Chart(df).mark_line().encode(
        x=alt.X("dist_m:Q", title="Distanza (m)"),
        y=alt.Y("ele:Q", title="Quota (m)")
    ).properties(height=260).interactive()

# ===================== UI =====================
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
st.title(APP_TITLE)

uploaded = st.file_uploader("📂 Carica GPX", type=["gpx"])

# KPI sopra
col1, col2, col3 = st.columns(3)

if uploaded:
    df = parse_gpx(uploaded)
    if df.empty or df["ele"].isna().all():
        st.warning("Il GPX non contiene punti utilizzabili con quota.")
    else:
        distance_km = df["dist_m"].iloc[-1] / 1000.0
        # D+ / D-
        diff = np.diff(df["ele"].to_numpy())
        ascent  = float(np.sum(np.clip(diff,  a_min=0, a_max=None)))
        descent = float(np.sum(np.clip(-diff, a_min=0, a_max=None)))

        # KPI
        col1.metric("Distanza (km)", f"{distance_km:.2f}")
        col2.metric("Dislivello + (m)", f"{ascent:.0f}")
        col3.metric("Tempo totale", _format_time_h(distance_km, ascent))

        # Indice di difficoltà
        score, level, color = _difficulty_from_stats(distance_km, ascent)
        st.subheader("Indice di Difficoltà")
        st.altair_chart(difficulty_gauge(score, level, color), use_container_width=True)

        # Profilo
        st.subheader("Profilo altimetrico")
        st.altair_chart(profile_chart(df), use_container_width=True)

else:
    # Stato iniziale: gauge a 0 già visibile
    col1.metric("Distanza (km)", "-")
    col2.metric("Dislivello + (m)", "-")
    col3.metric("Tempo totale", "-")
    st.subheader("Indice di Difficoltà")
    st.altair_chart(difficulty_gauge(0.0, "Facile", "green"), use_container_width=True)
