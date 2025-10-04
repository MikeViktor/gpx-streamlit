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

# ------------------- Costanti/Parametri -------------------
RS_STEP_M     = 3.0
RS_MIN_DELEV  = 0.25
RS_MED_K      = 3
RS_AVG_K      = 3
ABS_JUMP_RAW  = 100.0

BASE_MIN_PER_KM   = 15.0
UP_MIN_PER_100M   = 15.0
DOWN_MIN_PER_200M = 15.0

# opzioni IT
PRECIP_OPTIONS = ["assenza pioggia","pioviggine","pioggia","pioggia forte","neve fresca","neve profonda"]
SURF_OPTIONS   = ["asciutto","fango","roccia bagnata","neve dura","ghiaccio"]
EXPO_OPTIONS   = ["ombra","misto","pieno sole"]
TECH_OPTIONS   = ["facile","normale","roccioso","passaggi di roccia (scrambling)","neve/ghiaccio"]

# mappe interne per i moltiplicatori (se vorrai usare anche IF più avanti)
precip_map = {"assenza pioggia":"dry","pioviggine":"drizzle","pioggia":"rain","pioggia forte":"heavy_rain","neve fresca":"snow_shallow","neve profonda":"snow_deep"}
surface_map= {"asciutto":"dry","fango":"mud","roccia bagnata":"wet_rock","neve dura":"hard_snow","ghiaccio":"ice"}
expo_map   = {"ombra":"shade","misto":"mixed","pieno sole":"sun"}

# ------------------- Utility -------------------
def fmt_hm(minutes: float) -> str:
    if minutes is None: return "-"
    m = max(0.0, float(minutes))
    h = int(m // 60); mm = int(round(m - h*60))
    if mm == 60: h += 1; mm = 0
    return f"{h}:{mm:02d}" if h else f"{mm:02d}"

def hav_km(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.hypot(dx, dy)

def read_gpx(gpx_bytes: bytes):
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
                if la is None or lo is None: continue
                z = None
                for ch in el:
                    if is_tag(ch, "ele"): z = ch.text; break
                if z is None: continue
                try:
                    lat.append(float(la)); lon.append(float(lo)); ele.append(float(z))
                except: pass
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

def resample_profile(lat, lon, ele, step_m=RS_STEP_M):
    if len(lat) < 2: return [], [], [], []
    cum_m=[0.0]
    for i in range(1,len(lat)):
        cum_m.append(cum_m[-1] + hav_km(lat[i-1],lon[i-1],lat[i],lon[i])*1000.0)
    total_m=cum_m[-1]
    n = int(total_m // step_m) + 1
    e_rs=[]; j=0; t=0.0
    for _ in range(n):
        while j < len(cum_m)-1 and cum_m[j+1] < t: j+=1
        if t <= cum_m[0]: e_rs.append(ele[0])
        elif t >= cum_m[-1]: e_rs.append(ele[-1])
        else:
            u=(t-cum_m[j])/(cum_m[j+1]-cum_m[j])
            e_rs.append(ele[j]+u*(ele[j+1]-ele[j]))
        t += step_m
    e_med=median_k(e_rs, RS_MED_K)
    e_sm =moving_avg(e_med, RS_AVG_K)
    x_km=[i*step_m/1000.0 for i in range(len(e_sm))]
    return x_km, e_sm, cum_m, total_m

def compute_time_seg(e_prev, e_curr, seg_m,
                     base_min_per_km, up_min_per_100m, down_min_per_200m):
    d = max(0.0, seg_m) / 1000.0
    up   = max(0.0, e_curr - e_prev)
    down = max(0.0, e_prev - e_curr)
    return d*base_min_per_km + (up/100.0)*up_min_per_100m + (down/200.0)*down_min_per_200m

def splits_from_profile(x_km, ele_sm, step_m,
                        base_min_per_km, up_min_per_100m, down_min_per_200m):
    if not x_km: return [], [], []
    N=len(x_km); t_cum=[0.0]
    for i in range(1,N):
        t_cum.append(t_cum[-1] + compute_time_seg(ele_sm[i-1], ele_sm[i], step_m,
                                                  base_min_per_km, up_min_per_100m, down_min_per_200m))
    km_marks=list(range(1, int(math.floor(x_km[-1]))+1))
    split_min=[]; split_cum=[]; last=0.0
    for km in km_marks:
        idx = next((i for i,x in enumerate(x_km) if x>=km), N-1)
        here = t_cum[idx]
        split_min.append(here-last)
        split_cum.append(here)
        last = here
    return km_marks, split_min, split_cum

def km_labels(km_marks, split_cum, start_time: dt.time|None):
    labels=[]
    if start_time:
        base_dt = dt.datetime.combine(dt.date.today(), start_time)
        for m in split_cum:
            labels.append((base_dt + dt.timedelta(minutes=float(m))).strftime("%H:%M"))
    else:
        for m in split_cum:
            labels.append(fmt_hm(m))
    return labels

# ---- Metriche dettagliate (come desktop) ----
def detailed_metrics(lat, lon, ele_raw,
                     base_min_per_km, up_min_per_100m, down_min_per_200m):
    if len(ele_raw) < 2: raise ValueError("Dati elevazione insufficienti.")
    # cumulata originale
    cum=[0.0]
    for i in range(1,len(lat)):
        cum.append(cum[-1]+hav_km(lat[i-1],lon[i-1],lat[i],lon[i])*1000.0)
    total_m=cum[-1]; tot_km=total_m/1000.0

    # profilo ricampionato+filtri
    x_km, e_sm, _, _ = resample_profile(lat, lon, ele_raw, RS_STEP_M)

    dplus=dneg=0.0; asc_len=desc_len=flat_len=0.0
    asc_bins=[0,0,0,0,0]; desc_bins=[0,0,0,0,0]
    longest=0.0; cur=0.0; blocks=0; last_state=0; surge=0
    for i in range(1,len(e_sm)):
        seg=RS_STEP_M
        d=e_sm[i]-e_sm[i-1]
        if d>RS_MIN_DELEV:
            dplus+=d; asc_len+=seg; g=(d/seg)*100.0
            if   g<10: asc_bins[0]+=seg
            elif g<20: asc_bins[1]+=seg
            elif g<30: asc_bins[2]+=seg
            elif g<40: asc_bins[3]+=seg
            else:      asc_bins[4]+=seg
            if g>=25: cur+=seg; longest=max(longest,cur); state=2
            else:
                if cur>=100: blocks+=1
                cur=0.0; state=1 if g<15 else 0
            if (last_state in (1,2)) and (state in (1,2)) and (state!=last_state): surge+=1
            if state!=0: last_state=state
        elif d<-RS_MIN_DELEV:
            drop=-d; dneg+=drop; desc_len+=seg; g=(drop/seg)*100.0
            if   g<10: desc_bins[0]+=seg
            elif g<20: desc_bins[1]+=seg
            elif g<30: desc_bins[2]+=seg
            elif g<40: desc_bins[3]+=seg
            else:      desc_bins[4]+=seg
            if cur>=100: blocks+=1
            cur=0.0; last_state=0
        else:
            flat_len+=seg
            if cur>=100: blocks+=1
            cur=0.0; last_state=0
    if cur>=100: blocks+=1

    asc_gain=dplus; desc_drop=dneg
    grade_up   = (asc_gain/asc_len*100.0)  if asc_len>0 else 0.0
    grade_down = (desc_drop/desc_len*100.0) if desc_len>0 else 0.0

    t_dist  = tot_km*base_min_per_km
    t_up    = (dplus/100.0)*up_min_per_100m
    t_down  = (dneg/200.0)*down_min_per_200m
    t_total = t_dist+t_up+t_down

    holes = sum(1 for i in range(1,len(ele_raw)) if abs(ele_raw[i]-ele_raw[i-1])>=ABS_JUMP_RAW)
    weight_kg = 70.0
    cal_flat = weight_kg*0.6*max(0.0,tot_km)
    cal_up   = weight_kg*0.006*max(0.0,dplus)
    cal_down = weight_kg*0.003*max(0.0,dneg)
    cal_tot  = int(round(cal_flat+cal_up+cal_down))

    return {
        "tot_km": round(tot_km,2),
        "dplus": round(dplus,0),
        "dneg": round(dneg,0),
        "t_dist": t_dist, "t_up": t_up, "t_down": t_down, "t_total": t_total,
        "cal_total": cal_tot,
        "len_flat_km": round(flat_len/1000.0,2),
        "len_up_km":   round(asc_len/1000.0,2),
        "len_down_km": round(desc_len/1000.0,2),
        "grade_up_pct": round(grade_up,1),
        "grade_down_pct": round(grade_down,1),
        "lcs25_m": round(longest,0),
        "blocks25_count": int(blocks),
        "surge_idx_per_km": round(surge / max(0.1, tot_km), 2),
        "holes": int(holes),
        "profile_x_km": x_km, "profile_y_m": e_sm,
    }

# ---- Export GPX ----
def gpx_with_waypoints(lat, lon, ele, km_marks, labels):
    gpx = ET.Element("gpx", attrib={
        "version": "1.1", "creator": "gpx-streamlit",
        "xmlns":"http://www.topografix.com/GPX/1/1"
    })
    trk = ET.SubElement(gpx,"trk"); trkseg=ET.SubElement(trk,"trkseg")
    for la,lo,el in zip(lat,lon,ele):
        trkpt=ET.SubElement(trkseg,"trkpt",attrib={"lat":f"{la:.7f}","lon":f"{lo:.7f}"})
        ET.SubElement(trkpt,"ele").text=f"{el:.2f}"
    cum_km=[0.0]
    for i in range(1,len(lat)):
        cum_km.append(cum_km[-1]+hav_km(lat[i-1],lon[i-1],lat[i],lon[i]))
    for k,name in zip(km_marks,labels):
        target=float(k)
        best_i=min(range(len(cum_km)),key=lambda i:abs(cum_km[i]-target))
        wpt=ET.SubElement(gpx,"wpt",attrib={"lat":f"{lat[best_i]:.7f}","lon":f"{lon[best_i]:.7f}"})
        ET.SubElement(wpt,"name").text=f"Km {k} – {name}"
    return ET.tostring(gpx,encoding="utf-8",method="xml")

def gpx_waypoints_only(lat, lon, km_marks, labels):
    gpx = ET.Element("gpx", attrib={
        "version": "1.1", "creator": "gpx-splits",
        "xmlns":"http://www.topografix.com/GPX/1/1"
    })
    cum_km=[0.0]
    for i in range(1,len(lat)):
        cum_km.append(cum_km[-1]+hav_km(lat[i-1],lon[i-1],lat[i],lon[i]))
    for k,name in zip(km_marks,labels):
        target=float(k)
        best_i=min(range(len(cum_km)),key=lambda i:abs(cum_km[i]-target))
        wpt=ET.SubElement(gpx,"wpt",attrib={"lat":f"{lat[best_i]:.7f}","lon":f"{lon[best_i]:.7f}"})
        ET.SubElement(wpt,"name").text=f"Km {k} – {name}"
    return ET.tostring(gpx,encoding="utf-8",method="xml")

# ------------------- UI -------------------
st.set_page_config(page_title="Analisi Tracce GPX", layout="wide")
st.title("Analisi Tracce GPX")

uploaded = st.file_uploader("Carica un file GPX", type=["gpx"])

with st.sidebar:
    st.header("Impostazioni")
    inverti = st.checkbox("Inverti traccia", value=False)

    st.markdown("---")
    st.subheader("Parametri di passo (min)")
    base = st.number_input("Min/km (piano)", value=BASE_MIN_PER_KM, step=0.5, min_value=1.0, max_value=60.0)
    up   = st.number_input("Min/100 m (salita)", value=UP_MIN_PER_100M, step=0.5, min_value=1.0, max_value=60.0)
    down = st.number_input("Min/200 m (discesa)", value=DOWN_MIN_PER_200M, step=0.5, min_value=1.0, max_value=60.0)

    st.markdown("---")
    st.subheader("Condizioni")
    temp = st.number_input("Temperatura (°C)", value=15.0, step=1.0, min_value=-40.0, max_value=60.0)
    hum  = st.number_input("Umidità (%)", value=50.0, step=1.0, min_value=0.0, max_value=100.0)
    wind = st.number_input("Vento (km/h)", value=5.0, step=1.0, min_value=0.0, max_value=200.0)
    precip_it = st.selectbox("Precipitazioni", PRECIP_OPTIONS, index=0)
    surf_it   = st.selectbox("Fondo", SURF_OPTIONS, index=0)
    expo_it   = st.selectbox("Esposizione", EXPO_OPTIONS, index=1)
    tech      = st.selectbox("Tecnica", TECH_OPTIONS, index=1)
    loadkg    = st.number_input("Zaino extra (kg)", value=6.0, step=1.0, min_value=0.0, max_value=40.0)

    st.markdown("---")
    st.subheader("Tempi per km")
    show_clock  = st.checkbox("Mostra orario del giorno", value=True)
    show_labels = st.checkbox("Mostra etichette sul grafico", value=True)
    start_time  = st.time_input("Orario di partenza", value=dt.time(8, 0))

if not uploaded:
    st.info("Carica un file GPX per iniziare.")
    st.stop()

lat, lon, ele = read_gpx(uploaded.read())
if len(lat) < 2:
    st.error("GPX non valido o senza elevazioni.")
    st.stop()

if inverti:
    lat = list(reversed(lat)); lon = list(reversed(lon)); ele = list(reversed(ele))

# Profilo (ricampionato)
x_km, ele_sm, _, _ = resample_profile(lat, lon, ele, RS_STEP_M)

# Metriche dettagliate
res = detailed_metrics(lat, lon, ele, base, up, down)

# Top KPIs
c1,c2,c3 = st.columns(3)
c1.metric("Distanza (km)", f"{res['tot_km']:.2f}")
c2.metric("Dislivello + (m)", f"{int(res['dplus'])}")
c3.metric("Tempo totale", fmt_hm(res["t_total"]))

# Dettaglio
with st.expander("Risultati dettagliati", expanded=True):
    colA,colB = st.columns(2)
    with colA:
        st.write(f"**Dislivello − (m):** {int(res['dneg'])}")
        st.write(f"**Tempo piano:** {fmt_hm(res['t_dist'])}")
        st.write(f"**Tempo salita:** {fmt_hm(res['t_up'])}")
        st.write(f"**Tempo discesa:** {fmt_hm(res['t_down'])}")
        st.write(f"**Calorie stimate:** {res['cal_total']}")
        st.write(f"**Piano (km):** {res['len_flat_km']:.2f}")
    with colB:
        st.write(f"**Salita (km):** {res['len_up_km']:.2f}")
        st.write(f"**Discesa (km):** {res['len_down_km']:.2f}")
        st.write(f"**Pend. media salita (%):** {res['grade_up_pct']:.1f}")
        st.write(f"**Pend. media discesa (%):** {res['grade_down_pct']:.1f}")
        st.write(f"**LCS ≥25% (m):** {int(res['lcs25_m'])}")
        st.write(f"**Blocchi ripidi (≥100 m @ ≥25%):** {int(res['blocks25_count'])}")
        st.write(f"**Surge (cambi ritmo)/km:** {res['surge_idx_per_km']:.2f}")

    # Buchi GPX con avviso
    holes = int(res["holes"])
    if holes > 0:
        st.warning(f"⚠️ **Buchi GPX:** {holes} — Attenzione, calcoli e profilo possono risultare alterati.")
    else:
        st.success("Buchi GPX: 0")

# Splits
km_marks, split_min, split_cum = splits_from_profile(
    x_km, ele_sm, RS_STEP_M, base, up, down
)
labels = km_labels(km_marks, split_cum, start_time if show_clock else None)

# Grafico profilo con etichette
st.subheader("Profilo altimetrico")
df_prof = pd.DataFrame({"km": x_km, "ele": ele_sm})

axis_vals = list(range(0, int(math.ceil(x_km[-1])) + 1))
line = alt.Chart(df_prof).mark_line().encode(
    x=alt.X("km:Q", title="Distanza (km)", axis=alt.Axis(values=axis_vals)),
    y=alt.Y("ele:Q", title="Quota (m)")
)

df_km = pd.DataFrame({
    "km": km_marks,
    "ele": [ele_sm[min(range(len(x_km)), key=lambda i: abs(x_km[i]-k))] for k in km_marks],
    "label": labels
})
points = alt.Chart(df_km).mark_point(size=50, color="#333").encode(x="km:Q", y="ele:Q")
if show_labels and not df_km.empty:
    text = alt.Chart(df_km).mark_text(align="left", dx=6, dy=-6, fontSize=12, font="Segoe UI", fontWeight="bold").encode(
        x="km:Q", y="ele:Q", text="label:N"
    )
    chart = (line + points + text).properties(height=340)
else:
    chart = (line + points).properties(height=340)

st.altair_chart(chart, use_container_width=True)

# Tabella split per km (senza indice, colonne strette)
if km_marks:
    rows=[]
    for i,k in enumerate(km_marks, start=1):
        rows.append({"Km":f"{k}", "Split":fmt_hm(split_min[i-1]), "Cumulato":fmt_hm(split_cum[i-1])})
    df_split=pd.DataFrame(rows)

    st.subheader("Split per km")
    table_height = min(420, max(220, 30*(len(df_split)+1)))
    st.dataframe(
        df_split.style.set_properties(**{"font-size":"14px"}),
        use_container_width=True, height=table_height, hide_index=True,
        column_config={
            "Km": st.column_config.TextColumn(width="small"),
            "Split": st.column_config.TextColumn(width="small"),
            "Cumulato": st.column_config.TextColumn(width="small"),
        }
    )
else:
    st.info("Nessun km intero nella traccia (troppo corta).")

# Export GPX
if km_marks:
    st.markdown("---")
    st.subheader("Esporta")
    data_gpx = gpx_with_waypoints(lat, lon, ele, km_marks, labels)
    st.download_button("⬇️ Scarica GPX (traccia + waypoint al km)",
                       data=data_gpx, file_name="traccia_con_waypoint_km.gpx",
                       mime="application/gpx+xml")
    data_wpt = gpx_waypoints_only(lat, lon, km_marks, labels)
    st.download_button("⬇️ Scarica GPX (solo waypoint al km)",
                       data=data_wpt, file_name="waypoint_km.gpx",
                       mime="application/gpx+xml")
