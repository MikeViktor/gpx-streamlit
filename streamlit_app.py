# streamlit_app.py
# -*- coding: utf-8 -*-

import math
import io
import datetime as dt
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ---------------------------------------------
# Parametri filtro/ricampionamento (come desktop)
# ---------------------------------------------
RS_STEP_M     = 3.0
RS_MIN_DELEV  = 0.25
RS_MED_K      = 3
RS_AVG_K      = 3

# velocità “base” (min/km) e contributi salita/discesa
BASE_MIN_PER_KM   = 15.0
UP_MIN_PER_100M   = 15.0
DOWN_MIN_PER_200M = 15.0


# ------------------------------
# Utility formato tempo e distanze
# ------------------------------
def fmt_hm(minutes: float) -> str:
    if minutes is None:
        return "-"
    m = max(0.0, float(minutes))
    h = int(m // 60)
    mm = int(round(m - h * 60))
    if mm == 60:
        h += 1
        mm = 0
    return f"{h}:{mm:02d}" if h else f"{mm:02d}"

def hav_km(lat1, lon1, lat2, lon2):
    # approccio locale (veloce) come nelle app precedenti
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.hypot(dx, dy)


# ------------------------------
# Parsing GPX senza gpxpy (Soluz. B)
# ------------------------------
def read_gpx(gpx_bytes: bytes):
    """
    Parser GPX senza gpxpy: estrae liste lat[], lon[], ele[].
    Cerca 'trkpt', poi 'rtept', poi 'wpt'.
    """
    data = gpx_bytes.decode("utf-8", errors="ignore") if isinstance(gpx_bytes, (bytes, bytearray)) else gpx_bytes
    root = ET.fromstring(data)

    def is_tag(el, name: str):
        t = el.tag
        return t.endswith('}' + name) or t == name

    lat, lon, ele = [], [], []
    for wanted in ("trkpt", "rtept", "wpt"):
        lat.clear(); lon.clear(); ele.clear()
        for el in root.iter():
            if is_tag(el, wanted):
                la = el.attrib.get("lat"); lo = el.attrib.get("lon")
                if la is None or lo is None:
                    continue
                z = None
                for ch in el:
                    if is_tag(ch, "ele"):
                        z = ch.text
                        break
                if z is None:
                    continue
                try:
                    lat.append(float(la)); lon.append(float(lo)); ele.append(float(z))
                except Exception:
                    pass
        if lat:
            return lat, lon, ele
    return [], [], []


# ------------------------------
# Filtri / ricampionamento elevazione
# ------------------------------
def median_k(seq, k=3):
    if k < 1: k = 1
    if k % 2 == 0: k += 1
    half = k // 2
    out = []
    n = len(seq)
    for i in range(n):
        a = max(0, i - half); b = min(n, i + half + 1)
        window = sorted(seq[a:b])
        out.append(window[len(window)//2])
    return out

def moving_avg(seq, k=3):
    if k < 1: k = 1
    half = k // 2
    out = []
    n = len(seq)
    for i in range(n):
        a = max(0, i - half); b = min(n, i + half + 1)
        out.append(sum(seq[a:b]) / max(1, b - a))
    return out

def resample_profile(lat, lon, ele, step_m=RS_STEP_M):
    if len(lat) < 2:
        return [], [], [], []

    # cumulata distanze (m)
    cum_m = [0.0]
    for i in range(1, len(lat)):
        cum_m.append(cum_m[-1] + hav_km(lat[i-1], lon[i-1], lat[i], lon[i]) * 1000.0)
    total_m = cum_m[-1]

    # ricampionamento elevazione a passo costante
    n = int(total_m // step_m) + 1
    ele_rs = []
    j = 0
    t = 0.0
    for _ in range(n):
        while j < len(cum_m) - 1 and cum_m[j+1] < t:
            j += 1
        if t <= cum_m[0]:
            ele_rs.append(ele[0])
        elif t >= cum_m[-1]:
            ele_rs.append(ele[-1])
        else:
            u = (t - cum_m[j]) / (cum_m[j+1] - cum_m[j])
            ele_rs.append(ele[j] + u * (ele[j+1] - ele[j]))
        t += step_m

    # filtri
    ele_med = median_k(ele_rs, RS_MED_K)
    ele_sm  = moving_avg(ele_med, RS_AVG_K)

    x_km = [i * step_m / 1000.0 for i in range(len(ele_sm))]
    return x_km, ele_sm, cum_m, total_m


# ------------------------------
# Calcolo tempi per km (split)
# ------------------------------
def compute_time_per_step(e_prev, e_curr, seg_m,
                          base_min_per_km=BASE_MIN_PER_KM,
                          up_min_per_100m=UP_MIN_PER_100M,
                          down_min_per_200m=DOWN_MIN_PER_200M):
    """
    tempo (min) per un segmento di lunghezza seg_m e dislivello (e_curr - e_prev)
    """
    d = max(0.0, seg_m) / 1000.0
    up  = max(0.0, e_curr - e_prev)
    down= max(0.0, e_prev - e_curr)
    return d * base_min_per_km + (up / 100.0) * up_min_per_100m + (down / 200.0) * down_min_per_200m


def splits_from_profile(x_km, ele_sm, step_m=RS_STEP_M,
                        base_min_per_km=BASE_MIN_PER_KM,
                        up_min_per_100m=UP_MIN_PER_100M,
                        down_min_per_200m=DOWN_MIN_PER_200M):
    if not x_km:
        return [], [], []

    tot_km = x_km[-1]
    N = len(x_km)
    step = step_m

    # tempo cumulato ad ogni campione
    t_cum = [0.0]
    for i in range(1, N):
        seg_m = step
        dtm = compute_time_per_step(ele_sm[i-1], ele_sm[i], seg_m,
                                    base_min_per_km, up_min_per_100m, down_min_per_200m)
        t_cum.append(t_cum[-1] + dtm)

    # tempo per ogni km intero
    km_marks = list(range(1, int(math.floor(tot_km)) + 1))
    split_min = []
    split_cum = []
    last_t = 0.0

    for km in km_marks:
        # trova indice del primo campione con x_km >= km
        idx = next((i for i, x in enumerate(x_km) if x >= km), N-1)
        t_here = t_cum[idx]
        split_min.append(t_here - last_t)
        split_cum.append(t_here)
        last_t = t_here

    return km_marks, split_min, split_cum


# ------------------------------
# Etichette tempo al km (opzione orario del giorno)
# ------------------------------
def km_labels(km_marks, split_cum, start_time: dt.time | None):
    labels = []
    if start_time:
        # trasforma in datetime di oggi
        base_dt = dt.datetime.combine(dt.date.today(), start_time)
        for tmin in split_cum:
            t = base_dt + dt.timedelta(minutes=float(tmin))
            labels.append(t.strftime("%H:%M"))
    else:
        for tmin in split_cum:
            labels.append(fmt_hm(tmin))
    return labels


# ------------------------------
# Export GPX con waypoint per km
# ------------------------------
def gpx_with_waypoints(lat, lon, ele, km_marks, labels):
    """
    Crea un GPX (trk originale) + wpt per ogni km con nome=labels[k-1]
    """
    gpx = ET.Element("gpx", attrib={
        "version": "1.1",
        "creator": "gpx-streamlit",
        "xmlns": "http://www.topografix.com/GPX/1/1"
    })
    trk = ET.SubElement(gpx, "trk")
    trkseg = ET.SubElement(trk, "trkseg")
    for la, lo, el in zip(lat, lon, ele):
        trkpt = ET.SubElement(trkseg, "trkpt", attrib={"lat": f"{la:.7f}", "lon": f"{lo:.7f}"})
        ET.SubElement(trkpt, "ele").text = f"{el:.2f}"

    # cumulata su originali per posizionare wpt ogni km in modo approssimato
    cum_km = [0.0]
    for i in range(1, len(lat)):
        cum_km.append(cum_km[-1] + hav_km(lat[i-1], lon[i-1], lat[i], lon[i]))
    total_km = cum_km[-1]

    for k, name in zip(km_marks, labels):
        target = float(k)
        # trova punto più vicino alla distanza target
        best_i = min(range(len(cum_km)), key=lambda i: abs(cum_km[i] - target))
        wpt = ET.SubElement(gpx, "wpt", attrib={
            "lat": f"{lat[best_i]:.7f}",
            "lon": f"{lon[best_i]:.7f}",
        })
        ET.SubElement(wpt, "name").text = f"Km {k} – {name}"

    # serializza
    data = ET.tostring(gpx, encoding="utf-8", method="xml")
    return data


def gpx_waypoints_only(lat, lon, km_marks, labels):
    """
    Crea un GPX contenente solo i waypoint (niente traccia).
    """
    gpx = ET.Element("gpx", attrib={
        "version": "1.1",
        "creator": "gpx-splits",
        "xmlns": "http://www.topografix.com/GPX/1/1"
    })
    # cumulata su originali
    cum_km = [0.0]
    for i in range(1, len(lat)):
        cum_km.append(cum_km[-1] + hav_km(lat[i-1], lon[i-1], lat[i], lon[i]))

    for k, name in zip(km_marks, labels):
        target = float(k)
        best_i = min(range(len(cum_km)), key=lambda i: abs(cum_km[i] - target))
        wpt = ET.SubElement(gpx, "wpt", attrib={
            "lat": f"{lat[best_i]:.7f}",
            "lon": f"{lon[best_i]:.7f}",
        })
        ET.SubElement(wpt, "name").text = f"Km {k} – {name}"

    return ET.tostring(gpx, encoding="utf-8", method="xml")


# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="Analisi Tracce GPX", layout="wide")

st.title("Analisi Tracce GPX")

uploaded = st.file_uploader("Carica un file GPX", type=["gpx"])

with st.sidebar:
    st.header("Impostazioni")
    inverti = st.checkbox("Inverti traccia", value=False)

    st.markdown("---")
    st.subheader("Tempi per km")
    show_clock = st.checkbox("Mostra orario del giorno", value=True)
    show_labels = st.checkbox("Mostra etichette sul grafico", value=True)
    start_time = st.time_input("Orario di partenza", value=dt.time(8, 0))

    st.markdown("---")
    st.subheader("Parametri di passo (min)")
    base = st.number_input("Min/km (piano)", value=BASE_MIN_PER_KM, step=0.5, min_value=1.0, max_value=60.0)
    up   = st.number_input("Min/100 m (salita)", value=UP_MIN_PER_100M, step=0.5, min_value=1.0, max_value=60.0)
    down = st.number_input("Min/200 m (discesa)", value=DOWN_MIN_PER_200M, step=0.5, min_value=1.0, max_value=60.0)

if not uploaded:
    st.info("Carica un file GPX per iniziare.")
    st.stop()

# Parse GPX
lat, lon, ele = read_gpx(uploaded.read())
if len(lat) < 2:
    st.error("GPX non valido o privo di elevazione.")
    st.stop()

# Inversione traccia
if inverti:
    lat = list(reversed(lat))
    lon = list(reversed(lon))
    ele = list(reversed(ele))

# Ricampionamento + profilo
x_km, ele_sm, cum_m, total_m = resample_profile(lat, lon, ele, step_m=RS_STEP_M)
if not x_km:
    st.error("Impossibile creare il profilo.")
    st.stop()

tot_km = x_km[-1]

# Splits
km_marks, split_min, split_cum = splits_from_profile(
    x_km, ele_sm, step_m=RS_STEP_M, base_min_per_km=base, up_min_per_100m=up, down_min_per_200m=down
)

# Etichette per km
labels = km_labels(km_marks, split_cum, start_time if show_clock else None)

# ------------------------------
# Grafico profilo + etichette km
# ------------------------------
df_profile = pd.DataFrame({"km": x_km, "ele": ele_sm})

line = alt.Chart(df_profile).mark_line().encode(
    x=alt.X("km:Q", title="Distanza (km)"),
    y=alt.Y("ele:Q", title="Quota (m)")
)

# DataFrame per etichette sui km
df_km = pd.DataFrame({
    "km": km_marks,
    "label": labels
})
# quota approssimata al km (interpolazione semplice)
def elev_at_km(k):
    # trova indice più vicino
    idx = min(range(len(x_km)), key=lambda i: abs(x_km[i] - k))
    return ele_sm[idx]

df_km["ele"] = [elev_at_km(k) for k in km_marks]

points = alt.Chart(df_km).mark_point(size=50, color="#333").encode(
    x="km:Q", y="ele:Q"
)

if show_labels and not df_km.empty:
    text = alt.Chart(df_km).mark_text(
        align="left", dx=6, dy=-6, font="Segoe UI", fontSize=12, fontWeight="bold", color="#111"
    ).encode(x="km:Q", y="ele:Q", text="label:N")
    chart = (line + points + text).properties(height=340)
else:
    chart = (line + points).properties(height=340)

st.subheader("Profilo altimetrico")
st.altair_chart(chart, use_container_width=True)

# ------------------------------
# Tabella split (righe compatte, km sopra / tempo sotto)
# ------------------------------
if km_marks:
    # km sopra, tempo sotto: usiamo due colonne formattate
    rows = []
    for i, k in enumerate(km_marks, start=1):
        rows.append({
            "Km": f"{k}",
            "Split": fmt_hm(split_min[i-1]),
            "Cumulato": fmt_hm(split_cum[i-1]),
        })
    df_split = pd.DataFrame(rows)

    st.subheader("Split per km")

    # altezza dinamica per rendere visibile la scrollbar
    table_height = min(480, max(220, 32 * (len(df_split) + 1)))
    st.dataframe(
        df_split.style.set_properties(**{"font-size": "14px"}),
        use_container_width=True,
        height=table_height
    )
else:
    st.info("Nessun km intero nella traccia (troppo corta).")

# ------------------------------
# Download GPX (con waypoint) e solo waypoint
# ------------------------------
if km_marks:
    st.markdown("---")
    st.subheader("Esporta")

    # nome etichetta: se orologio attivo -> HH:MM, altrimenti mm:ss
    name_labels = labels

    data_gpx = gpx_with_waypoints(lat, lon, ele, km_marks, name_labels)
    st.download_button(
        label="⬇️ Scarica GPX (traccia + waypoint al km)",
        data=data_gpx,
        file_name="traccia_con_waypoint_km.gpx",
        mime="application/gpx+xml"
    )

    data_wpt = gpx_waypoints_only(lat, lon, km_marks, name_labels)
    st.download_button(
        label="⬇️ Scarica GPX (solo waypoint al km)",
        data=data_wpt,
        file_name="waypoint_km.gpx",
        mime="application/gpx+xml"
    )
