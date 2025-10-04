# -*- coding: utf-8 -*-
import math, datetime as dt
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ------------------ util ------------------
def equirect_km(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return (dx*dx + dy*dy) ** 0.5

def parse_gpx_bytes(b: bytes):
    root = ET.fromstring(b)
    for wanted in ("trkpt", "rtept", "wpt"):
        lat, lon, ele = [], [], []
        for el in root.iter():
            tag = el.tag.split('}')[-1]
            if tag == wanted:
                la = el.attrib.get("lat"); lo = el.attrib.get("lon")
                if la is None or lo is None: continue
                z = None
                for ch in el:
                    if ch.tag.split('}')[-1] == "ele":
                        z = ch.text; break
                if z is None: continue
                try:
                    lat.append(float(la)); lon.append(float(lo)); ele.append(float(z))
                except:
                    pass
        if lat: return lat, lon, ele
    return [], [], []

def median_k(seq, k=3):
    if k < 1: k = 1
    if k % 2 == 0: k += 1
    half = k // 2; out = []; n = len(seq)
    for i in range(n):
        a = max(0, i-half); b = min(n, i+half+1)
        w = sorted(seq[a:b]); out.append(w[len(w)//2])
    return out

def moving_avg(seq, k=3):
    if k < 1: k = 1
    half = k // 2; out = []; n = len(seq)
    for i in range(n):
        a = max(0, i-half); b = min(n, i+half+1)
        out.append(sum(seq[a:b]) / max(1, b-a))
    return out

def resample_profile(lat, lon, ele, step_m=3.0):
    # distanza cumulata grezza
    cum = [0.0]
    for i in range(1, len(lat)):
        cum.append(cum[-1] + equirect_km(lat[i-1],lon[i-1],lat[i],lon[i])*1000.0)

    total_m = cum[-1]
    # ricampionamento per distanza equispaziata
    n = int(total_m // step_m) + 1
    new_ele = []
    t = 0.0; j = 0
    for _ in range(n):
        while j < len(cum)-1 and cum[j+1] < t: j += 1
        if t <= cum[0]:
            new_ele.append(ele[0])
        elif t >= cum[-1]:
            new_ele.append(ele[-1])
        else:
            u = (t - cum[j]) / (cum[j+1] - cum[j])
            new_ele.append(ele[j] + u*(ele[j+1]-ele[j]))
        t += step_m
    x_km = [min(i*step_m, total_m)/1000.0 for i in range(n)]
    return x_km, new_ele, total_m

# ------------------ gauge (SVG, spicchi perfetti) ------------------
def gauge_svg_html(value: float) -> str:
    v = max(0.0, min(100.0, float(value)))
    bins = [
        (0,30,"#2ecc71","Facile"),
        (30,50,"#f1c40f","Medio"),
        (50,70,"#e67e22","Impeg."),
        (70,80,"#e74c3c","Diffic."),
        (80,90,"#8e44ad","Molto diff."),
        (90,100,"#111111","Estremo"),
    ]
    cx, cy = 150, 150
    r_outer, r_inner = 140, 105

    def pol(r, ang):
        a = math.radians(ang)
        return cx + r*math.cos(a), cy - r*math.sin(a)

    def arc_path(r, start_deg, end_deg):
        # sweep in senso orario (0° a destra, 180° a sinistra)
        large = 1 if (start_deg - end_deg) > 180 else 0
        x1,y1 = pol(r, start_deg)
        x2,y2 = pol(r, end_deg)
        return f"M{x1:.1f},{y1:.1f} A{r:.1f},{r:.1f} 0 {large} 0 {x2:.1f},{y2:.1f}"

    def ring(a,b,col):
        a_ang = 180.0 - (a/100.0)*180.0
        b_ang = 180.0 - (b/100.0)*180.0
        outer = arc_path(r_outer, a_ang, b_ang)
        inner = arc_path(r_inner, b_ang, a_ang)
        # vertice di collegamento sul bordo interno all'angolo b
        xi,yi = pol(r_inner, b_ang)
        return f'<path d="{outer} L{xi:.1f},{yi:.1f} {inner} Z" fill="{col}" stroke="{col}" />'

    val_ang = 180.0 - (v/100.0)*180.0
    px = cx + (r_inner-5)*math.cos(math.radians(val_ang))
    py = cy - (r_inner-5)*math.sin(math.radians(val_ang))

    svg = [f'<svg viewBox="0 0 300 170" width="100%" height="170" xmlns="http://www.w3.org/2000/svg">']
    for a,b,c,_ in bins:
        svg.append(ring(a,b,c))
    # disco centrale
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r_inner-2}" fill="white"/>')
    # ago
    svg.append(f'<line x1="{cx}" y1="{cy}" x2="{px:.1f}" y2="{py:.1f}" stroke="#333" stroke-width="3"/>')
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="#333"/>')
    # valore
    svg.append(f'<text x="{cx}" y="{cy-20}" text-anchor="middle" font-size="18" font-weight="700">{v:.1f}</text>')
    svg.append('</svg>')
    return ''.join(svg)

# ------------------ tempo per segmento + split ------------------
def per_step_times(x_km, y_m, base_min_per_km=15.0, up_min_per_100m=15.0, down_min_per_200m=15.0, step_m=3.0):
    """Distribuisce il tempo sul profilo: piano + salita + discesa per ogni step."""
    dt = []
    step_km = step_m / 1000.0
    for i in range(1, len(y_m)):
        dz = y_m[i] - y_m[i-1]
        t_flat = base_min_per_km * step_km
        t_up   = up_min_per_100m * max(0.0, dz)/100.0
        t_down = down_min_per_200m * max(0.0, -dz)/200.0
        dt.append(t_flat + t_up + t_down)
    # stesso numero di punti di x/y (primo step=0)
    return [0.0] + dt

def cumulative_minutes(arr):
    s = 0.0; out=[0.0]
    for i in range(1,len(arr)):
        s += arr[i]
        out.append(s)
    return out

def format_hm(m):
    h = int(m // 60); mm = int(round(m - h*60))
    if mm == 60: h += 1; mm = 0
    return f"{h}:{mm:02d}"

# ------------------ IF semplice (stesso schema visuale) ------------------
def if_category(v):
    if v < 30: return "Facile"
    if v < 50: return "Medio"
    if v < 70: return "Impegnativo"
    if v < 80: return "Difficile"
    if v <= 90: return "Molto difficile"
    return "Estremamente difficile"

def difficulty_index(x_km, y_m):
    # indicatore semplice che tiene conto di lunghezza + D+
    tot_km = x_km[-1] if x_km else 0.0
    dplus = sum(max(0.0, y_m[i]-y_m[i-1]) for i in range(1,len(y_m)))
    # compressione logistica come nella tua app
    W_D, W_PLUS = 0.5, 1.0
    S0 = 80.0
    S = W_D*tot_km + W_PLUS*(dplus/100.0)
    IF = 100.0*(1.0 - math.exp(-S/max(1e-6,S0)))
    return round(min(100.0, IF),1)

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="GPX – indice + profilo + split", layout="wide")

st.title("Indice di Difficoltà")

# Sidebar
st.sidebar.header("Impostazioni")
invert = st.sidebar.checkbox("Inverti traccia", value=False)

st.sidebar.subheader("Tempi per km")
show_daytime = st.sidebar.checkbox("Mostra orario del giorno", value=True)
show_labels  = st.sidebar.checkbox("Mostra etichette sul grafico", value=True)
start_time   = st.sidebar.time_input("Orario di partenza", value=dt.time(8,0))

st.sidebar.subheader("Parametri di passo (min)")
base_p  = st.sidebar.number_input("Min/km (piano)",  5.0, 60.0, 15.0, 0.5)
up_p    = st.sidebar.number_input("Min/100 m (salita)", 5.0, 60.0, 15.0, 0.5)
down_p  = st.sidebar.number_input("Min/200 m (discesa)",5.0, 60.0, 15.0, 0.5)

uploaded = st.file_uploader("Trascina qui il file GPX", type=["gpx"])

if not uploaded:
    st.info("Carica un GPX per iniziare.")
    st.stop()

content = uploaded.read()
lat,lon,ele = parse_gpx_bytes(content)
if not ele:
    st.error("GPX senza quote utili.")
    st.stop()

if invert:
    lat = list(reversed(lat)); lon = list(reversed(lon)); ele = list(reversed(ele))

# profilo ricampionato + filtri leggeri
RS_STEP_M = 3.0
x_km, y_raw, total_m = resample_profile(lat,lon,ele, RS_STEP_M)
y_med = median_k(y_raw, 3)
y_sm  = moving_avg(y_med, 3)

# tempi per step + cumulato
dt_steps = per_step_times(x_km, y_sm, base_p, up_p, down_p, RS_STEP_M)
cum_min  = cumulative_minutes(dt_steps)
tot_time_min = cum_min[-1]
tot_km = x_km[-1]
dplus = round(sum(max(0.0, y_sm[i]-y_sm[i-1]) for i in range(1,len(y_sm))), 0)

# IF
IF = difficulty_index(x_km, y_sm)
st.write(f"**{IF}** ({if_category(IF)})")
st.markdown(
    f'<div style="max-width:600px;margin:auto;">{gauge_svg_html(IF)}</div>',
    unsafe_allow_html=True
)

# ---- Profilo con etichette km/tempo ----
st.subheader("Profilo altimetrico")

df = pd.DataFrame({"km": x_km, "ele": y_sm, "tmin": cum_min})

# punti interi per etichette
km_int = list(range(0, int(math.ceil(tot_km))+1))
ann = []
for k in km_int:
    # indice del punto più vicino a quel km
    idx = int(np.argmin(np.abs(df["km"].values - k)))
    y = float(df.loc[idx, "ele"])
    t = float(df.loc[idx, "tmin"])
    if show_daytime:
        base_dt = dt.datetime.combine(dt.date.today(), start_time)
        lab_time = (base_dt + dt.timedelta(minutes=t)).strftime("%H:%M")
    else:
        lab_time = format_hm(t)
    ann.append({"km": float(k), "ele": y, "km_txt": f"{k} km", "tm_txt": lab_time})

ann_df = pd.DataFrame(ann)

# chart linea
base_chart = alt.Chart(df).mark_line().encode(
    x=alt.X("km:Q",
            axis=alt.Axis(title="Distanza (km)", values=km_int, tickCount=len(km_int))),
    y=alt.Y("ele:Q", axis=alt.Axis(title="Quota (m)"))
).properties(height=360)

if show_labels:
    km_text = alt.Chart(ann_df).mark_text(fontSize=12, dy=-14, fontWeight="bold").encode(
        x="km:Q", y="ele:Q", text="km_txt:N"
    )
    tm_text = alt.Chart(ann_df).mark_text(fontSize=12, dy=12).encode(
        x="km:Q", y="ele:Q", text="tm_txt:N"
    )
    chart = alt.layer(base_chart, km_text, tm_text).resolve_scale(y='shared')
else:
    chart = base_chart

st.altair_chart(chart, use_container_width=True)

# ---- Split / tabella ----
st.subheader("Tempi / Orario ai diversi Km")
rows = []
for i in range(1, len(km_int)):
    k0, k1 = km_int[i-1], km_int[i]
    # tempo al km esimo = differenza fra cumulati
    t0 = float(ann_df.loc[ann_df["km"]==k0, "tm_txt"].index[0])
    t1 = float(ann_df.loc[ann_df["km"]==k1, "tm_txt"].index[0])

# ricava gli split corretti (parziali + cumulato)
splits = []
for i in range(1, len(km_int)):
    k = km_int[i]
    # indice km k
    idx_k   = int(np.argmin(np.abs(df["km"].values - k)))
    idx_km1 = int(np.argmin(np.abs(df["km"].values - (k-1))))
    t_cum = df.loc[idx_k, "tmin"]               # cumulato minuti
    t_prev = df.loc[idx_km1, "tmin"]
    t_split = t_cum - t_prev
    if show_daytime:
        base_dt = dt.datetime.combine(dt.date.today(), start_time)
        h_cum = (base_dt + dt.timedelta(minutes=float(t_cum))).strftime("%H:%M")
    else:
        h_cum = format_hm(float(t_cum))
    splits.append({"Km": k, "Tempo parziale": format_hm(float(t_split)), "Cumulativo": h_cum})

st.dataframe(pd.DataFrame(splits), use_container_width=True, height=360)
