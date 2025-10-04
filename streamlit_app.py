# streamlit_app.py
# -*- coding: utf-8 -*-

import math
import io
from datetime import time as _time, datetime, timedelta

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import gpxpy

# =========================
# Config base pagina
# =========================
st.set_page_config(page_title="Analisi Tracce GPX", layout="wide")

APP_TITLE = "Analisi Tracce GPX — tempi/km + indice di difficoltà"
RS_STEP_M     = 3.0
RS_MIN_DELEV  = 0.25
RS_MED_K      = 3
RS_AVG_K      = 3
ABS_JUMP_RAW  = 100.0

LOOP_TOL_M        = 200.0
DRIFT_MIN_ABS_M   = 2.0
BALANCE_MIN_DIFFM = 10.0
BALANCE_REL_FRAC  = 0.05

# Pesi IF (allineati alla tua app desktop)
W_D=0.5; W_PLUS=1.0; W_COMP=0.5; W_STEEP=0.4; W_STEEP_D=0.3
W_LCS=0.25; W_BLOCKS=0.15; W_SURGE=0.25
IF_S0=80.0; ALPHA_METEO=0.6; SEVERITY_GAIN=1.52


# =========================
# Utilità
# =========================
def fmt_hm(minutes: float) -> str:
    h = int(minutes // 60)
    m = int(round(minutes - h*60))
    if m == 60:
        h += 1; m = 0
    return f"{h}:{m:02d}"

def _is_loop(lat, lon, tol_m=LOOP_TOL_M) -> bool:
    if len(lat) < 2: return False
    def dist_km(a1, o1, a2, o2):
        dy = (a2 - a1) * 111.32
        dx = (o2 - o1) * 111.32 * math.cos(math.radians((a1 + a2)/2))
        return math.hypot(dx, dy)
    return dist_km(lat[0], lon[0], lat[-1], lon[-1]) * 1000.0 <= tol_m

def median_k(seq, k=3):
    if k < 1: k = 1
    if k % 2 == 0: k += 1
    half = k // 2
    out = []
    n = len(seq)
    for i in range(n):
        a = max(0, i-half); b = min(n, i+half+1)
        w = sorted(seq[a:b])
        out.append(w[len(w)//2])
    return out

def moving_avg(seq, k=3):
    if k < 1: k = 1
    half = k // 2
    out = []
    n = len(seq)
    for i in range(n):
        a = max(0, i-half); b = min(n, i+half+1)
        out.append(sum(seq[a:b]) / max(1, b-a))
    return out

# =========================
# GPX parsing
# =========================
def read_gpx(gpx_bytes: bytes):
    gpx = gpxpy.parse(io.StringIO(gpx_bytes.decode("utf-8")))
    pts = []
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if p.elevation is not None:
                    pts.append((p.latitude, p.longitude, p.elevation))
    # fallback anche per route
    if not pts:
        for rte in gpx.routes:
            for p in rte.points:
                if p.elevation is not None:
                    pts.append((p.latitude, p.longitude, p.elevation))
    lat = [p[0] for p in pts]
    lon = [p[1] for p in pts]
    ele = [p[2] for p in pts]
    return lat, lon, ele

def dist_km(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2)/2.0))
    return math.hypot(dx, dy)

def resample_profile(lat, lon, ele):
    # distanza cumulata (m)
    cum=[0.0]
    for i in range(1, len(lat)):
        cum.append(cum[-1] + dist_km(lat[i-1], lon[i-1], lat[i], lon[i])*1000.0)
    if not cum or cum[-1] == 0:
        return [], [], 0.0
    total_m = cum[-1]

    # ricampionamento altimetria
    n = int(total_m // RS_STEP_M) + 1
    e_res = []
    j = 0
    t = 0.0
    for _ in range(n):
        while j < len(cum)-1 and cum[j+1] < t:
            j += 1
        if t <= cum[0]:
            e_res.append(ele[0])
        elif t >= cum[-1]:
            e_res.append(ele[-1])
        else:
            u = (t - cum[j]) / (cum[j+1] - cum[j])
            e_res.append(ele[j] + u*(ele[j+1]-ele[j]))
        t += RS_STEP_M

    # correzione deriva per anello
    if _is_loop(lat, lon) and len(e_res) > 1:
        drift = e_res[-1] - e_res[0]
        if abs(drift) >= DRIFT_MIN_ABS_M:
            n1 = len(e_res)-1
            e_res = [e_res[i] - (drift*(i/n1)) for i in range(len(e_res))]

    # filtri
    e_med = median_k(e_res, RS_MED_K)
    e_sm  = moving_avg(e_med, RS_AVG_K)

    x_km = [min(i*RS_STEP_M, total_m)/1000.0 for i in range(len(e_sm))]
    return x_km, e_sm, total_m/1000.0

# =========================
# Calcoli tempo & metriche
# =========================
def compute_metrics(lat, lon, ele,
                    base_min_km=15.0, up_min_100m=15.0, down_min_200m=15.0,
                    reverse=False, weight_kg=70.0):
    if reverse:
        lat = list(reversed(lat)); lon = list(reversed(lon)); ele = list(reversed(ele))
    if len(ele) < 2:
        raise ValueError("GPX privo di quota utile.")

    x_km, e_sm, total_km = resample_profile(lat, lon, ele)
    if not x_km:
        raise ValueError("Profilo non ricampionabile.")

    # accumuli
    dplus=dneg=0.0
    asc_len=desc_len=flat_len=0.0
    asc_bins=[0,0,0,0,0]; desc_bins=[0,0,0,0,0]
    longest_steep_run=0.0; current_run=0.0; last_state=0; surge_trans=0
    time_profile_min = 0.0

    for i in range(1, len(e_sm)):
        seg = RS_STEP_M if i*RS_STEP_M <= total_km*1000 else max(0.0,(total_km*1000)-(i-1)*RS_STEP_M)
        if seg <= 0: continue

        d = e_sm[i] - e_sm[i-1]
        # tempo segmentato (coerente con la desktop)
        up = max(d,0.0); down = max(-d,0.0)
        time_profile_min += base_min_km * (seg/1000.0) + up_min_100m*(up/100.0) + down_min_200m*(down/200.0)

        if d > RS_MIN_DELEV:
            dplus += d; asc_len += seg; g = (d/seg)*100.0
            if   g<10: asc_bins[0]+=seg
            elif g<20: asc_bins[1]+=seg
            elif g<30: asc_bins[2]+=seg
            elif g<40: asc_bins[3]+=seg
            else:      asc_bins[4]+=seg
            if g >= 25:
                current_run += seg; longest_steep_run = max(longest_steep_run, current_run); state=2
            else:
                if current_run>=100: pass
                current_run=0.0; state=1 if g<15 else 0
            if (last_state in (1,2)) and (state in (1,2)) and state!=last_state:
                surge_trans += 1
            if state!=0: last_state=state
        elif d < -RS_MIN_DELEV:
            drop = -d; dneg += drop; desc_len += seg; g = (drop/seg)*100.0
            if   g<10: desc_bins[0]+=seg
            elif g<20: desc_bins[1]+=seg
            elif g<30: desc_bins[2]+=seg
            elif g<40: desc_bins[3]+=seg
            else:      desc_bins[4]+=seg
            if current_run>=100: pass
            current_run=0.0; last_state=0
        else:
            flat_len += seg
            if current_run>=100: pass
            current_run=0.0; last_state=0

    # tempi (totali coerenti con profilo segmentato)
    t_total = time_profile_min
    t_dist  = base_min_km * total_km
    t_up    = (dplus/100.0)*up_min_100m
    t_down  = (dneg/200.0)*down_min_200m

    holes = sum(1 for i in range(1, len(ele)) if abs(ele[i]-ele[i-1]) >= ABS_JUMP_RAW)

    # per-km split (da profilo segmentato)
    km_split = []  # minuti del km corrente
    km_cum   = []  # cumulati
    labels   = []  # x_km label position
    km_target = 1.0
    last_dist_m = 0.0
    acc_time = 0.0
    for i in range(1, len(e_sm)):
        d_m = min(i*RS_STEP_M, total_km*1000.0)
        seg = d_m - last_dist_m
        if seg <= 0: continue
        d = e_sm[i] - e_sm[i-1]
        up = max(d,0.0); down = max(-d,0.0)
        seg_min = base_min_km*(seg/1000.0) + up_min_100m*(up/100.0) + down_min_200m*(down/200.0)
        acc_time += seg_min
        last_dist_m = d_m

        # ogni volta che superiamo 1,2,3… km
        while (d_m/1000.0) >= km_target - 1e-9:
            km_split.append(acc_time)
            km_cum.append(sum(km_split))
            labels.append(km_target)
            acc_time = 0.0
            km_target += 1.0
            if km_target > total_km + 1e-6:
                break

    # metrics DF
    df_splits = pd.DataFrame({
        "Km": list(range(1, len(km_split)+1)),
        "Split_min": km_split,
        "Cumulato_min": km_cum
    })

    res = dict(
        x_km=x_km, y_m=e_sm, tot_km=round(total_km,2),
        dplus=round(dplus,0), dneg=round(dneg,0),
        t_total=t_total, t_dist=t_dist, t_up=t_up, t_down=t_down,
        holes=int(holes),
        asc_bins_m=[round(v,0) for v in asc_bins],
        desc_bins_m=[round(v,0) for v in desc_bins],
        len_up_km=round(asc_len/1000.0,2), len_down_km=round(desc_len/1000.0,2), len_flat_km=round(flat_len/1000.0,2),
        grade_up_pct=round((dplus/max(1.0,asc_len))*100.0,1) if asc_len>0 else 0.0,
        grade_down_pct=round((dneg/max(1.0,desc_len))*100.0,1) if desc_len>0 else 0.0,
        df_splits=df_splits,
        label_km_positions=labels,
        avg_alt_m=float(np.mean(ele)) if ele else None
    )
    return res


# =========================
# Sidebar (parametri)
# =========================
with st.sidebar:
    st.header("Impostazioni")

    uploaded = st.file_uploader("Carica GPX", type=["gpx"])

    st.subheader("Parametri base")
    base = st.number_input("Min/Km (piano)", 1.0, 60.0, 15.0, 0.5)
    up   = st.number_input("Min/100 m (salita)", 1.0, 60.0, 15.0, 0.5)
    down = st.number_input("Min/200 m (discesa)", 1.0, 60.0, 15.0, 0.5)
    weight = st.number_input("Peso (kg)", 30.0, 150.0, 70.0, 1.0)
    reverse = st.checkbox("Inverti traccia", value=False)

    # 🔵🔵🔵 Tempi per km (spostato qui, subito sotto "Inverti traccia")
    st.subheader("Tempi per km")
    show_clock  = st.checkbox("Mostra orario del giorno", value=False)
    show_labels = st.checkbox("Mostra etichette sul grafico", value=True)
    if show_clock:
        start_time = st.time_input("Orario di partenza", value=_time(8, 0))
    else:
        start_time = None

    st.subheader("Condizioni")
    temp = st.number_input("Temperatura (°C)", -30.0, 50.0, 15.0, 1.0)
    hum  = st.number_input("Umidità (%)", 0.0, 100.0, 50.0, 1.0)
    wind = st.number_input("Vento (km/h)", 0.0, 150.0, 5.0, 1.0)
    precip = st.selectbox("Precipitazioni", ["assenza pioggia","pioviggine","pioggia","pioggia forte","neve fresca","neve profonda"])
    surface = st.selectbox("Fondo", ["asciutto","fango","roccia bagnata","neve dura","ghiaccio"])
    expo = st.selectbox("Esposizione", ["ombra","misto","pieno sole"])
    tech = st.selectbox("Tecnica", ["facile","normale","roccioso","passaggi di roccia (scrambling)","neve/ghiaccio"])
    loadkg = st.number_input("Zaino extra (kg)", 0.0, 40.0, 6.0, 1.0)

# =========================
# Corpo pagina
# =========================
st.title(APP_TITLE)

if not uploaded:
    st.info("Carica un file GPX per iniziare.")
    st.stop()

try:
    lat, lon, ele = read_gpx(uploaded.read())
    res = compute_metrics(lat, lon, ele,
                          base_min_km=base, up_min_100m=up, down_min_200m=down,
                          reverse=reverse, weight_kg=weight)
except Exception as e:
    st.error(str(e))
    st.stop()

# =========================
# Box riepilogo
# =========================
c1, c2, c3 = st.columns([1,1,1])
with c1:
    st.metric("Distanza (km)", f"{res['tot_km']}")
with c2:
    st.metric("Dislivello + (m)", f"{int(res['dplus'])}")
with c3:
    st.metric("Tempo totale", fmt_hm(res["t_total"]))

# =========================
# Profilo altimetrico (Altair)
# =========================
df_profile = pd.DataFrame({"km": res["x_km"], "quota": res["y_m"]})

base_chart = alt.Chart(df_profile).mark_line().encode(
    x=alt.X("km:Q",
            title="Distanza (km)",
            scale=alt.Scale(nice=False),  # no auto nice
            axis=alt.Axis(tickMinStep=1, values=list(range(0, int(math.ceil(res["tot_km"]))+1)))
           ),
    y=alt.Y("quota:Q", title="Quota (m)"),
    tooltip=[
        alt.Tooltip("km:Q", format=".2f", title="Km"),
        alt.Tooltip("quota:Q", format=".0f", title="Quota (m)")
    ]
).properties(height=320)

layers = [base_chart]

# Etichette dei km (due righe)
if show_labels and res["label_km_positions"]:
    lab_km = []
    lab_q  = []
    lab_txt = []
    # prendo quota in prossimità di quel km
    km_to_y = dict(zip(np.round(df_profile["km"],3), df_profile["quota"]))
    for kmv in res["label_km_positions"]:
        # quota dal profilo (vicinanza)
        idx = (np.abs(df_profile["km"] - kmv)).argmin()
        yv = float(df_profile.iloc[idx]["quota"])
        # testo: Km N + sotto orario/cumulo
        row_idx = kmv-1
        if 0 <= row_idx < len(res["df_splits"]):
            cumul_min = float(res["df_splits"].iloc[int(row_idx)]["Cumulato_min"])
            if show_clock and start_time is not None:
                start_dt = datetime.combine(datetime.today(), start_time)
                tt = start_dt + timedelta(minutes=cumul_min)
                sub = tt.strftime("%H:%M")
            else:
                sub = fmt_hm(cumul_min)
        else:
            sub = ""

        lab_km.append(kmv)
        lab_q.append(yv)
        lab_txt.append(f"Km {int(kmv)}\n{sub}")

    df_labels = pd.DataFrame({"km": lab_km, "quota": lab_q, "label": lab_txt})
    label_layer = alt.Chart(df_labels).mark_text(
        align="left", baseline="bottom", dx=4, dy=-4, fontSize=11, fontWeight="bold"
    ).encode(
        x="km:Q", y="quota:Q", text="label:N"
    )
    layers.append(label_layer)

chart = alt.layer(*layers).resolve_scale(y='shared').interactive()
st.subheader("Profilo altimetrico")
st.altair_chart(chart, use_container_width=True)

# =========================
# Split per km (tabella compatta con scrollbar visibile)
# =========================
# Mini CSS per righe compatte + scrollbar sempre visibile
st.markdown("""
<style>
[data-testid="stDataFrame"] div[role="gridcell"] { padding-top: 3px; padding-bottom: 3px; font-size: 0.95rem; }
[data-testid="stDataFrame"] div[role="columnheader"] { font-size: 0.95rem; }
[data-testid="stDataFrame"] div[role="grid"] { overflow-y: auto !important; }
</style>
""", unsafe_allow_html=True)

df_km = res["df_splits"].copy()
# Stringhe formattate per tabella
df_km["Split"]    = df_km["Split_min"].apply(fmt_hm)
df_km["Cumulato"] = df_km["Cumulato_min"].apply(fmt_hm)
if show_clock and start_time is not None:
    start_dt = datetime.combine(datetime.today(), start_time)
    df_km["Orario"] = [(start_dt + timedelta(minutes=m)).strftime("%H:%M") for m in df_km["Cumulato_min"]]
else:
    df_km["Orario"] = ""

# Mostra colonne principali
st.subheader("Split per km")
st.caption("⬇️ Scorri per vedere tutti i km")
st.dataframe(df_km[["Km","Split","Cumulato"] + (["Orario"] if show_clock else [])],
             use_container_width=True, hide_index=True, height=360)
