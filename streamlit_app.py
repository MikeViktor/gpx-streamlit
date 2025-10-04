# streamlit_app.py
# -*- coding: utf-8 -*-

import math
import io
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import xml.etree.ElementTree as ET
from datetime import time as _time

# ------------------------------------------------------------
# GPX parsing leggero (senza gpxpy obbligatorio)
# ------------------------------------------------------------
def _is_tag(e, name: str) -> bool:
    t = e.tag
    return t.endswith('}' + name) or t == name

def parse_gpx_bytes(data: bytes):
    """
    Ritorna liste lat, lon, ele (float). Cerca trkpt -> rtept -> wpt.
    """
    root = ET.fromstring(data)
    for wanted in ("trkpt", "rtept", "wpt"):
        lat, lon, ele = [], [], []
        for el in root.iter():
            if _is_tag(el, wanted):
                la = el.attrib.get("lat"); lo = el.attrib.get("lon")
                if la is None or lo is None:
                    continue
                z = None
                for ch in el:
                    if _is_tag(ch, "ele"):
                        z = ch.text
                        break
                if z is None:
                    continue
                try:
                    lat.append(float(la))
                    lon.append(float(lo))
                    ele.append(float(z))
                except:
                    pass
        if lat:
            return lat, lon, ele
    return [], [], []

# ------------------------------------------------------------
# Geodesia semplice + ricampionamento
# ------------------------------------------------------------
def dist_km(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.hypot(dx, dy)

def cumulative_dist_m(lat, lon):
    n = len(lat)
    cum = np.zeros(n, dtype=float)
    for i in range(1, n):
        cum[i] = cum[i-1] + dist_km(lat[i-1], lon[i-1], lat[i], lon[i]) * 1000.0
    return cum

def resample_along(cum_m, values, step_m=3.0):
    """
    Interpola 'values' ai punti [0, step_m, ..., cum_m[-1]]
    """
    total = float(cum_m[-1])
    if total <= 0:
        return np.array([0.0]), np.array([values[0]])
    t = np.arange(0.0, total + step_m, step_m, dtype=float)
    idx = np.searchsorted(cum_m, t, side="right") - 1
    idx = np.clip(idx, 0, len(cum_m)-2)
    w = (t - cum_m[idx]) / np.maximum(1e-9, (cum_m[idx+1] - cum_m[idx]))
    out = values[idx] + w * (values[idx+1] - values[idx])
    return t, out

def median_k(x, k=3):
    x = np.asarray(x, dtype=float)
    if k < 1: return x.copy()
    if k % 2 == 0: k += 1
    r = np.zeros_like(x); half = k // 2
    for i in range(len(x)):
        a = max(0, i-half); b = min(len(x), i+half+1)
        r[i] = np.median(x[a:b])
    return r

def moving_avg(x, k=3):
    x = np.asarray(x, dtype=float)
    if k < 1: return x.copy()
    r = np.zeros_like(x); half = k // 2
    for i in range(len(x)):
        a = max(0, i-half); b = min(len(x), i+half+1)
        r[i] = np.mean(x[a:b])
    return r

def is_loop(lat, lon, tol_m=200.0) -> bool:
    if len(lat) < 2: return False
    return dist_km(lat[0], lon[0], lat[-1], lon[-1]) * 1000.0 <= tol_m

def correct_linear_drift(elev_m, lat, lon, min_abs=2.0):
    if not is_loop(lat, lon) or len(elev_m) < 2:
        return elev_m, False, 0.0
    drift = float(elev_m[-1] - elev_m[0])
    if abs(drift) < min_abs:
        return elev_m, False, drift
    n = len(elev_m) - 1
    idx = np.arange(len(elev_m), dtype=float)
    corr = elev_m - drift * (idx / n)
    return corr, True, drift

# ------------------------------------------------------------
# Profilo + tempi
# ------------------------------------------------------------
def compute_profile_and_times(lat, lon, ele,
                              base_min_per_km=15.0,
                              up_min_per_100m=15.0,
                              down_min_per_200m=15.0,
                              reverse=False,
                              rs_step_m=3.0,
                              med_k=3, avg_k=3,
                              min_delev=0.25):
    lat = list(lat); lon = list(lon); ele = list(ele)
    if reverse:
        lat.reverse(); lon.reverse(); ele.reverse()
    if len(ele) < 2:
        raise ValueError("GPX privo di elevazioni sufficienti.")

    cum_raw = cumulative_dist_m(lat, lon)
    grid_m, ele_rs = resample_along(cum_raw, np.asarray(ele, dtype=float), step_m=rs_step_m)

    ele_med = median_k(ele_rs, med_k)
    ele_sm  = moving_avg(ele_med, avg_k)
    ele_fix, loop_fixed, drift = correct_linear_drift(ele_sm, lat, lon, min_abs=2.0)

    _, lat_rs = resample_along(cum_raw, np.asarray(lat, dtype=float), step_m=rs_step_m)
    _, lon_rs = resample_along(cum_raw, np.asarray(lon, dtype=float), step_m=rs_step_m)

    dplus = 0.0; dneg = 0.0
    dt_min = np.zeros_like(grid_m)
    for i in range(1, len(grid_m)):
        ds = grid_m[i] - grid_m[i-1]
        dh = ele_fix[i] - ele_fix[i-1]
        t_dist = (ds/1000.0) * base_min_per_km
        t_up   = (max(dh, 0.0)/100.0) * up_min_per_100m
        t_down = (max(-dh,0.0)/200.0) * down_min_per_200m
        dt_min[i] = t_dist + t_up + t_down
        if dh > min_delev: dplus += dh
        elif dh < -min_delev: dneg += -dh

    cum_min = np.cumsum(dt_min)
    tot_km  = float(grid_m[-1] / 1000.0)
    t_total = float(cum_min[-1])

    return {
        "dist_km": grid_m/1000.0,
        "elev_m": ele_fix,
        "lat_rs": lat_rs,
        "lon_rs": lon_rs,
        "cum_min": cum_min,
        "tot_km": round(tot_km,2),
        "dplus": int(round(dplus)),
        "dneg": int(round(dneg)),
        "t_total": t_total,
        "loop_fixed": loop_fixed,
        "drift_abs_m": abs(float(drift))
    }

# ------------------------------------------------------------
# Etichette e tabella split
# ------------------------------------------------------------
def _build_km_table(dist_km, elev_m, lat, lon, cum_min, start_time=None, show_clock=False):
    D = np.asarray(dist_km); E = np.asarray(elev_m)
    LAT = np.asarray(lat);   LON = np.asarray(lon)
    T = np.asarray(cum_min)

    if len(D) == 0:
        return pd.DataFrame(columns=["Km","x_km","y_m","lat","lon","Split","Cumulato","Orario","label"])

    max_km = int(math.floor(D[-1]))
    rows = []
    prev_cum = 0.0
    for km in range(1, max_km + 1):
        idx = int(np.searchsorted(D, km, side="left"))
        if idx >= len(D): idx = len(D)-1

        total_minutes = float(T[idx])
        split_minutes = total_minutes - prev_cum
        prev_cum = total_minutes

        # format hh:mm
        def fmt(mins):
            h = int(mins // 60); m = int(round(mins - h*60))
            if m == 60: h, m = h+1, 0
            return f"{h}:{m:02d}"

        cumul_str = fmt(total_minutes)
        split_str = fmt(split_minutes)

        if show_clock and start_time is not None:
            hh = start_time.hour; mm = start_time.minute
            add_h = int(total_minutes // 60); add_m = int(round(total_minutes - add_h*60))
            hh = (hh + add_h + (mm + add_m)//60) % 24
            mm = (mm + add_m) % 60
            clock_str = f"{hh:02d}:{mm:02d}"
            label = f"Km {km} — {clock_str}"
        else:
            clock_str = ""
            label = f"Km {km} — {cumul_str}"

        rows.append({
            "Km": km,
            "x_km": float(D[idx]),
            "y_m": float(E[idx]),
            "lat": float(LAT[idx]),
            "lon": float(LON[idx]),
            "Split": split_str,
            "Cumulato": cumul_str,
            "Orario": clock_str,
            "label": label
        })
    return pd.DataFrame(rows)

def draw_altitude_profile_altair(dist_km, elev_m, labels_df, show_labels=True):
    df = pd.DataFrame({"km": dist_km, "quota": elev_m})

    base = alt.Chart(df).mark_line().encode(
        x=alt.X("km:Q", title="Distanza (km)",
                axis=alt.Axis(values=list(range(0, int(max(dist_km))+1)))),
        y=alt.Y("quota:Q", title="Quota (m)")
    ).properties(height=380)

    if show_labels and labels_df is not None and not labels_df.empty:
        pts = alt.Chart(labels_df).mark_point(size=55, color="#666", filled=True).encode(
            x="x_km:Q", y="y_m:Q"
        )
        txt = alt.Chart(labels_df).mark_text(
            align="left", baseline="middle", dx=6, dy=-8,
            fontSize=12, color="#222", stroke="white", strokeWidth=3  # contorno per leggibilità
        ).encode(
            x="x_km:Q", y="y_m:Q", text="label:N"
        )
        chart = base + pts + txt
    else:
        chart = base

    return chart.interactive()

# ------------------------------------------------------------
# Export GPX con waypoint
# ------------------------------------------------------------
def make_gpx_with_waypoints(original_gpx_text: str, labels_df: pd.DataFrame) -> str:
    if labels_df is None or labels_df.empty:
        return original_gpx_text
    wpts = []
    for _, r in labels_df.iterrows():
        wpts.append(
f'''  <wpt lat="{r.lat:.6f}" lon="{r.lon:.6f}">
    <name>{r.label}</name>
  </wpt>'''
        )
    blob = "\n".join(wpts)
    if "</gpx>" in original_gpx_text:
        return original_gpx_text.replace("</gpx>", blob + "\n</gpx>")
    else:
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="streamlit-app" xmlns="http://www.topografix.com/GPX/1/1">
{blob}
</gpx>'''

def make_gpx_waypoints_only(labels_df: pd.DataFrame) -> str:
    if labels_df is None or labels_df.empty:
        return '''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="streamlit-app" xmlns="http://www.topografix.com/GPX/1/1">
</gpx>'''
    wpts = []
    for _, r in labels_df.iterrows():
        wpts.append(
f'''  <wpt lat="{r.lat:.6f}" lon="{r.lon:.6f}">
    <name>{r.label}</name>
  </wpt>'''
        )
    blob = "\n".join(wpts)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="streamlit-app" xmlns="http://www.topografix.com/GPX/1/1">
{blob}
</gpx>'''

# ------------------------------------------------------------
# Streamlit
# ------------------------------------------------------------
st.set_page_config(page_title="Analisi Tracce GPX", layout="wide")
st.title("Analisi Tracce GPX")

with st.sidebar:
    st.subheader("Parametri calcolo")
    base = st.number_input("Min/Km (piano)", 1.0, 60.0, 15.0, 0.5)
    up   = st.number_input("Min/100 m (salita)", 1.0, 60.0, 15.0, 0.5)
    down = st.number_input("Min/200 m (discesa)", 1.0, 60.0, 15.0, 0.5)
    reverse = st.checkbox("Inverti traccia", value=False)

    st.subheader("Tempi per km")
    show_clock = st.checkbox("Mostra orario del giorno", value=False)
    show_labels = st.checkbox("Mostra etichette sul grafico", value=True)
    if show_clock:
        start_time = st.time_input("Orario di partenza", value=_time(8, 0))
    else:
        start_time = None

uploaded = st.file_uploader("Carica GPX", type=["gpx"])
if uploaded is None:
    st.info("Carica un file **.gpx** per iniziare.")
    st.stop()

data_bytes = uploaded.read()
try:
    lat, lon, ele = parse_gpx_bytes(data_bytes)
except Exception as e:
    st.error(f"Errore nel parsing GPX: {e}")
    st.stop()

if len(ele) < 2:
    st.warning("GPX troppo breve o senza elevazioni.")
    st.stop()

# Calcolo profilo e tempi
try:
    res = compute_profile_and_times(
        lat, lon, ele,
        base_min_per_km=base,
        up_min_per_100m=up,
        down_min_per_200m=down,
        reverse=reverse
    )
except Exception as e:
    st.error(f"Errore nel calcolo: {e}")
    st.stop()

# Tabella per km (split + cumulato + orario)
km_df = _build_km_table(
    res["dist_km"], res["elev_m"], res["lat_rs"], res["lon_rs"], res["cum_min"],
    start_time=start_time, show_clock=show_clock
)

# Metriche principali
c1, c2, c3 = st.columns(3)
c1.metric("Distanza (km)", f"{res['tot_km']:.2f}")
c2.metric("Dislivello + (m)", f"{res['dplus']}")
tot_min = res["t_total"]; hh = int(tot_min // 60); mm = int(round(tot_min - hh*60))
if mm == 60: hh, mm = hh+1, 0
c3.metric("Tempo totale", f"{hh}:{mm:02d}")

# Grafico Altair
st.subheader("Profilo altimetrico")
chart = draw_altitude_profile_altair(
    res["dist_km"], res["elev_m"],
    km_df.rename(columns={"Km":"km"}),  # riuso per etichette
    show_labels=show_labels
)
st.altair_chart(chart, use_container_width=True)

# Tabella split/km + download
st.subheader("Split per km")
show_cols = ["Km", "Split", "Cumulato"] + (["Orario"] if show_clock else [])
st.dataframe(km_df[show_cols], use_container_width=True)
csv_buf = io.StringIO()
km_df[show_cols].to_csv(csv_buf, index=False)
st.download_button(
    "Scarica split (CSV)",
    data=csv_buf.getvalue(),
    file_name=f"{uploaded.name.rsplit('.',1)[0]}_splits.csv",
    mime="text/csv"
)

# Export GPX (traccia + waypoint) e solo waypoint
st.subheader("Esporta waypoint km/tempo")
orig_text = data_bytes.decode("utf-8", errors="ignore")
gpx_with = make_gpx_with_waypoints(orig_text, km_df.rename(columns={"Km":"km"}))
gpx_only = make_gpx_waypoints_only(km_df.rename(columns={"Km":"km"}))

col_a, col_b = st.columns(2)
col_a.download_button(
    "Scarica GPX (traccia + waypoint)",
    data=gpx_with.encode("utf-8"),
    file_name=f"{uploaded.name.rsplit('.',1)[0]}_with_km_times.gpx",
    mime="application/gpx+xml"
)
col_b.download_button(
    "Scarica GPX (solo waypoint)",
    data=gpx_only.encode("utf-8"),
    file_name=f"{uploaded.name.rsplit('.',1)[0]}_km_times_waypoints.gpx",
    mime="application/gpx+xml"
)

# Info correzioni
info = []
if res.get("loop_fixed"):
    info.append(f"Correzione deriva altimetrica applicata (~{res['drift_abs_m']:.1f} m).")
if info:
    st.caption(" ".join(info))
