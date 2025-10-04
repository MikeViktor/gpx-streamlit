# streamlit_app.py
# -*- coding: utf-8 -*-

import math
import io
import datetime as dt
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

APP_TITLE = "Analisi Tracce GPX"
APP_VER   = "v5.1 – IF fix gauge + unico grafico con tempi + testo IF grande"

# ----- Ricampionamento / filtri -----
RS_STEP_M     = 3.0
RS_MIN_DELEV  = 0.25
RS_MED_K      = 3
RS_AVG_K      = 3
ABS_JUMP_RAW  = 100.0

# ----- Correzione anello -----
LOOP_TOL_M        = 200.0
DRIFT_MIN_ABS_M   = 2.0
BALANCE_MIN_DIFFM = 10.0
BALANCE_REL_FRAC  = 0.05

# ----- Pesi IF -----
W_D       = 0.5
W_PLUS    = 1.0
W_COMP    = 0.5
W_STEEP   = 0.4
W_STEEP_D = 0.3
W_LCS     = 0.25
W_BLOCKS  = 0.15
W_SURGE   = 0.25
IF_S0     = 80.0
ALPHA_METEO   = 0.6
SEVERITY_GAIN = 1.52

# -------------------- Utility --------------------
def _is_tag(e, name: str) -> bool:
    t = e.tag
    return t.endswith('}' + name) or t == name

def parse_gpx_file(file_bytes: bytes):
    root = ET.fromstring(file_bytes)
    for wanted in ("trkpt", "rtept", "wpt"):
        lat, lon, ele = [], [], []
        for el in root.iter():
            if _is_tag(el, wanted):
                la = el.attrib.get("lat"); lo = el.attrib.get("lon")
                if la is None or lo is None: continue
                z = None
                for ch in el:
                    if _is_tag(ch, "ele"): z = ch.text; break
                if z is None: continue
                try:
                    lat.append(float(la)); lon.append(float(lo)); ele.append(float(z))
                except: pass
        if lat: return lat, lon, ele
    return [], [], []

def dist_km(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.hypot(dx, dy)

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

def resample_elev(cum_m, ele, step_m=3.0):
    total = cum_m[-1]; n = int(total // step_m) + 1
    out = []; t = 0.0; j = 0
    for _ in range(n):
        while j < len(cum_m)-1 and cum_m[j+1] < t: j += 1
        if t <= cum_m[0]: out.append(ele[0])
        elif t >= cum_m[-1]: out.append(ele[-1])
        else:
            u = (t - cum_m[j]) / (cum_m[j+1] - cum_m[j])
            out.append(ele[j] + u * (ele[j+1] - ele[j]))
        t += step_m
    return out

def _is_loop(lat, lon, tol_m=LOOP_TOL_M) -> bool:
    if len(lat) < 2: return False
    return dist_km(lat[0], lon[0], lat[-1], lon[-1]) * 1000.0 <= tol_m

def apply_loop_drift_correction(elev_series, lat, lon, min_abs=DRIFT_MIN_ABS_M):
    if not _is_loop(lat, lon) or len(elev_series) < 2:
        return elev_series, False, 0.0
    drift = elev_series[-1] - elev_series[0]
    if abs(drift) < min_abs: return elev_series, False, drift
    n = len(elev_series)-1
    fixed = [elev_series[i] - (drift*(i/n)) for i in range(len(elev_series))]
    return fixed, True, drift

def meteo_multiplier(temp_c, humidity_pct, precip, surface, wind_kmh, exposure):
    if   temp_c < -5: M_temp = 1.20
    elif temp_c < 0:  M_temp = 1.10
    elif temp_c < 5:  M_temp = 1.05
    elif temp_c <= 20:M_temp = 1.00
    elif temp_c <= 25:M_temp = 1.05
    elif temp_c <= 30:M_temp = 1.10
    elif temp_c <= 35:M_temp = 1.20
    else:             M_temp = 1.35
    if   humidity_pct > 80: M_temp += 0.10
    elif humidity_pct > 60: M_temp += 0.05
    precip_map  = {"dry":1.00,"drizzle":1.05,"rain":1.15,"heavy_rain":1.30,"snow_shallow":1.25,"snow_deep":1.60}
    surface_map = {"dry":1.00,"mud":1.10,"wet_rock":1.15,"hard_snow":1.30,"ice":1.60}
    exposure_map= {"shade":1.00,"mixed":1.05,"sun":1.10}
    M_precip  = precip_map.get(precip, 1.00)
    M_surface = surface_map.get(surface, 1.00)
    M_sun     = exposure_map.get(exposure, 1.00)
    if   wind_kmh <= 10: M_wind = 1.00
    elif wind_kmh <= 20: M_wind = 1.05
    elif wind_kmh <= 35: M_wind = 1.10
    elif wind_kmh <= 60: M_wind = 1.20
    else:                M_wind = 1.35
    return min(1.4, M_temp * max(M_precip, M_surface) * M_wind * M_sun)

def altitude_multiplier(avg_alt_m):
    if avg_alt_m is None: return 1.0
    excess = max(0.0, (avg_alt_m - 2000.0) / 500.0)
    return 1.0 + 0.03 * excess

def technique_multiplier(level="normale"):
    table = {
        "facile": 0.95, "normale": 1.00, "roccioso": 1.10,
        "passaggi di roccia (scrambling)": 1.20, "neve/ghiaccio": 1.30,
        "scrambling": 1.20,
    }
    return table.get(level, 1.0)

def pack_load_multiplier(extra_load_kg=0.0):
    return 1.0 + 0.02 * max(0.0, extra_load_kg / 5.0)

def cat_from_if(val: float) -> str:
    if val < 30: return "Facile"
    if val < 50: return "Medio"
    if val < 70: return "Impegnativo"
    if val < 80: return "Difficile"
    if val <= 90: return "Molto difficile"
    return "Estremamente difficile"

# -------------------- Calcoli percorso --------------------
def compute_from_arrays(lat, lon, ele_raw,
                        base_min_per_km=15.0, up_min_per_100m=15.0, down_min_per_200m=15.0,
                        reverse=False, weight_kg=70.0):
    if len(ele_raw) < 2:
        raise ValueError("Nessun punto utile con elevazione nel GPX.")
    if reverse:
        lat, lon, ele_raw = list(reversed(lat)), list(reversed(lon)), list(reversed(ele_raw))

    cum = [0.0]
    for i in range(1, len(lat)):
        cum.append(cum[-1] + dist_km(lat[i-1], lon[i-1], lat[i], lon[i]) * 1000.0)
    tot_km = cum[-1] / 1000.0
    total_m = cum[-1]

    e_res = resample_elev(cum, ele_raw, RS_STEP_M)
    e_res, loop_fix, loop_drift = apply_loop_drift_correction(e_res, lat, lon)
    e_med = median_k(e_res, RS_MED_K)
    e_sm  = moving_avg(e_med, RS_AVG_K)

    dplus=dneg=0.0; asc_len=desc_len=flat_len=0.0; asc_gain=desc_drop=0.0
    asc_bins=[0,0,0,0,0]; desc_bins=[0,0,0,0,0]
    longest_steep_run=0.0; current_run=0.0; blocks25=0; last_state=0; surge_trans=0

    for i in range(1, len(e_sm)):
        t_prev=(i-1)*RS_STEP_M; t_curr=min(i*RS_STEP_M, total_m); seg=max(0.0, t_curr-t_prev)
        if seg <= 0: continue
        d = e_sm[i] - e_sm[i-1]
        if d > RS_MIN_DELEV:
            dplus += d; asc_len += seg; asc_gain += d; g=(d/seg)*100.0
            if   g<10: asc_bins[0]+=seg
            elif g<20: asc_bins[1]+=seg
            elif g<30: asc_bins[2]+=seg
            elif g<40: asc_bins[3]+=seg
            else:      asc_bins[4]+=seg
            if g>=25:
                current_run += seg
                longest_steep_run = max(longest_steep_run, current_run)
                state = 2
            else:
                if current_run>=100: blocks25+=1
                current_run=0.0; state=1 if g<15 else 0
            if (last_state in (1,2)) and (state in (1,2)) and (state!=last_state):
                surge_trans += 1
            if state != 0: last_state = state
        elif d < -RS_MIN_DELEV:
            drop = -d; dneg += drop; desc_len += seg; desc_drop += drop; g = (drop/seg)*100.0
            if   g<10: desc_bins[0]+=seg
            elif g<20: desc_bins[1]+=seg
            elif g<30: desc_bins[2]+=seg
            elif g<40: desc_bins[3]+=seg
            else:      desc_bins[4]+=seg
            if current_run>=100: blocks25+=1
            current_run=0.0; last_state=0
        else:
            flat_len += seg
            if current_run>=100: blocks25+=1
            current_run=0.0; last_state=0
    if current_run>=100: blocks25+=1

    loop_like=_is_loop(lat,lon,LOOP_TOL_M)
    diff=abs(dplus-dneg)
    need_balance = loop_like and (diff>max(BALANCE_MIN_DIFFM,BALANCE_REL_FRAC*max(dplus,dneg)))
    balance=False
    if need_balance:
        dplus=dneg; asc_gain=dplus; balance=True

    grade_up   = (asc_gain/asc_len*100.0)  if asc_len>0 else 0.0
    grade_down = (desc_drop/desc_len*100.0) if desc_len>0 else 0.0

    t_dist  = tot_km * base_min_per_km
    t_up    = (dplus/100.0) * up_min_per_100m
    t_down  = (dneg/200.0) * down_min_per_200m
    t_total = t_dist + t_up + t_down

    holes = sum(1 for i in range(1,len(ele_raw)) if abs(ele_raw[i]-ele_raw[i-1])>=ABS_JUMP_RAW)

    weight_kg=max(1.0, float(weight_kg))
    cal_flat = weight_kg*0.6*max(0.0, tot_km)
    cal_up   = weight_kg*0.006*max(0.0, dplus)
    cal_down = weight_kg*0.003*max(0.0, dneg)
    cal_tot  = int(round(cal_flat+cal_up+cal_down))

    x_km=[min(i*RS_STEP_M,total_m)/1000.0 for i in range(len(e_sm))]
    surge_idx = round(surge_trans / max(0.1, tot_km), 2)

    return {
        "tot_km": round(tot_km,2), "dplus": round(dplus,0), "dneg": round(dneg,0),
        "t_dist": t_dist, "t_up": t_up, "t_down": t_down, "t_total": t_total,
        "holes": holes,
        "len_flat_km": round(flat_len/1000.0,2), "len_up_km": round(asc_len/1000.0,2), "len_down_km": round(desc_len/1000.0,2),
        "grade_up_pct": round(grade_up,1), "grade_down_pct": round(grade_down,1),
        "cal_total": cal_tot,
        "asc_bins_m": [round(v,0) for v in asc_bins],
        "desc_bins_m": [round(v,0) for v in desc_bins],
        "lcs25_m": round(longest_steep_run,0),
        "blocks25_count": int(blocks25),
        "surge_idx_per_km": surge_idx,
        "avg_alt_m": sum(ele_raw)/len(ele_raw) if ele_raw else None,
        "profile_x_km": x_km, "profile_y_m": e_sm[:],
        "loop_fix_applied": bool(loop_fix), "loop_drift_abs_m": round(abs(loop_drift),1),
        "loop_balance_applied": bool(balance), "loop_like": bool(loop_like), "balance_diff_m": round(diff,1),
    }

# ----- IF + gauge -----
precip_map = {
    "assenza pioggia":"dry","pioviggine":"drizzle","pioggia":"rain","pioggia forte":"heavy_rain",
    "neve fresca":"snow_shallow","neve profonda":"snow_deep",
}
surface_map = {"asciutto":"dry","fango":"mud","roccia bagnata":"wet_rock","neve dura":"hard_snow","ghiaccio":"ice"}
expo_map    = {"ombra":"shade","misto":"mixed","pieno sole":"sun"}

def compute_if(res, temp_c, humidity_pct, precip_it, surface_it, wind_kmh, expo_it, tech, loadkg):
    D_km = float(res["tot_km"])
    Dp   = float(res["dplus"])
    C    = (Dp / max(0.001, D_km))
    f_up25, f_down25 = 0.0, 0.0  # mantenuti a 0 per coerenza

    lcs    = float(res.get("lcs25_m", 0.0))
    blocks = float(res.get("blocks25_count", 0.0))
    surge  = float(res.get("surge_idx_per_km", 0.0))
    lcs_sc = lcs / 200.0

    S = (W_D*D_km + W_PLUS*(Dp/100.0) + W_COMP*(C/100.0) +
         W_STEEP*(100.0*f_up25) + W_STEEP_D*(100.0*f_down25) +
         W_LCS*lcs_sc + W_BLOCKS*blocks + W_SURGE*surge)

    IF_base = 100.0 * (1.0 - math.exp(-S / max(1e-6, IF_S0)))

    M = meteo_multiplier(
            temp_c, humidity_pct, precip_map[precip_it], surface_map[surface_it],
            wind_kmh, expo_map[expo_it]
        ) * altitude_multiplier(res.get("avg_alt_m")) \
          * technique_multiplier(tech) \
          * pack_load_multiplier(loadkg)

    bump = (100.0 - IF_base) * max(0.0, (M - 1.0)) * ALPHA_METEO
    IF = min(100.0, (IF_base + bump) * SEVERITY_GAIN)
    return round(IF, 1), cat_from_if(IF)

def gauge_svg(value: float) -> str:
    """Gauge semicerchio con settori nell'ordine corretto (sinistra -> destra)."""
    v = max(0.0, min(100.0, float(value)))
    ang = 180.0 - (v/100.0)*180.0

    arcs = [
        (0,30,"#2ecc71"),
        (30,50,"#f1c40f"),
        (50,70,"#e67e22"),
        (70,80,"#e74c3c"),
        (80,90,"#8e44ad"),
        (90,100,"#111111"),
    ]
    def arc_path(a1, a2, r=90, cx=120, cy=120):
        def pol(a_deg):
            a = math.radians(a_deg)
            return cx + r*math.cos(a), cy - r*math.sin(a)
        # 0% -> 180°, 100% -> 0° ; disegniamo in senso orario (sweep-flag = 1)
        x1,y1 = pol(180 - (a1/100)*180)
        x2,y2 = pol(180 - (a2/100)*180)
        large = 1 if (a2 - a1) > 50 else 0
        return f"M {x1:.1f},{y1:.1f} A {r},{r} 0 {large} 1 {x2:.1f},{y2:.1f}"

    ang_r = math.radians(ang)
    cx,cy = 120,120
    x_tip = cx + 70*math.cos(ang_r)
    y_tip = cy - 70*math.sin(ang_r)

    arcs_svg = "\n".join(
        f'<path d="{arc_path(a1,a2)}" stroke="{col}" stroke-width="14" fill="none" />'
        for a1,a2,col in arcs
    )
    return f"""
<svg viewBox="0 0 240 140" width="100%" height="140">
  {arcs_svg}
  <circle cx="120" cy="120" r="52" fill="white" stroke="white"/>
  <line x1="{cx}" y1="{cy}" x2="{x_tip:.1f}" y2="{y_tip:.1f}" stroke="#333" stroke-width="4" />
  <circle cx="{cx}" cy="{cy}" r="5" fill="#333"/>
  <text x="120" y="90" text-anchor="middle" font-family="Segoe UI" font-size="24" font-weight="bold">{v:.1f}</text>
</svg>
"""

# ----- Split / export -----
def format_hm(minutes: float) -> str:
    h = int(minutes // 60)
    m = int(round(minutes - h*60))
    if m == 60: h += 1; m = 0
    return f"{h}:{m:02d}"

def compute_segment_minutes(seg_m, d_ele, base_min_per_km, up_min_per_100m, down_min_per_200m):
    t = (seg_m/1000.0) * base_min_per_km
    if d_ele > 0:
        t += (d_ele/100.0) * up_min_per_100m
    elif d_ele < 0:
        t += ((-d_ele)/200.0) * down_min_per_200m
    return t

def split_per_km(lat, lon, y_m, base_min_per_km, up_min_per_100m, down_min_per_200m):
    if len(lat) < 2: return pd.DataFrame(columns=["Km","Split","Cumulato","Alt (m)"])
    seg_t = [0.0]
    for i in range(1, len(y_m)):
        seg_t.append(compute_segment_minutes(RS_STEP_M, y_m[i]-y_m[i-1],
                                             base_min_per_km, up_min_per_100m, down_min_per_200m))
    t_cum = np.cumsum(seg_t)
    out_rows = []
    max_km = int((len(y_m)*RS_STEP_M)//1000)
    for k in range(1, max_km+1):
        idx = min(max(1, int((k*1000.0)//RS_STEP_M)), len(y_m)-1)
        idx_prev = min(max(0, int(((k-1)*1000.0)//RS_STEP_M)), len(y_m)-1)
        split_min = t_cum[idx] - t_cum[idx_prev]
        cum_min   = t_cum[idx]
        out_rows.append([k, format_hm(split_min), format_hm(cum_min), round(y_m[idx],0)])
    return pd.DataFrame(out_rows, columns=["Km","Split","Cumulato","Alt (m)"])

def add_waypoints_gpx(original_gpx_bytes: bytes, km_df: pd.DataFrame, names: list):
    try:
        root = ET.fromstring(original_gpx_bytes)
    except Exception:
        return None
    first_pt = None
    for el in root.iter():
        if _is_tag(el, "trkpt"):
            first_pt = (el.attrib.get("lat"), el.attrib.get("lon"))
            break
    if first_pt is None:
        return None
    for i, row in km_df.iterrows():
        name = names[i]
        wpt = ET.Element("wpt", attrib={"lat": first_pt[0], "lon": first_pt[1]})
        nm  = ET.Element("name"); nm.text = name
        wpt.append(nm)
        root.append(wpt)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)

def waypoints_only_gpx(km_df: pd.DataFrame, names: list, lat0=0.0, lon0=0.0):
    root = ET.Element("gpx", attrib={"version":"1.1", "creator":"streamlit"})
    for i, row in km_df.iterrows():
        wpt = ET.Element("wpt", attrib={"lat": f"{lat0:.6f}", "lon": f"{lon0:.6f}"})
        nm  = ET.Element("name"); nm.text = names[i]
        wpt.append(nm)
        root.append(wpt)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)

# -------------------- UI --------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(APP_VER)

with st.sidebar:
    st.header("Impostazioni")
    uploaded = st.file_uploader("Carica GPX", type=["gpx"])

    invert_trace = st.checkbox("Inverti traccia", value=False)

    st.subheader("Tempi per km")
    show_clock = st.checkbox("Mostra orario del giorno", value=True)
    show_labels= st.checkbox("Mostra etichette sul grafico", value=True)
    start_time = st.time_input("Orario di partenza", value=dt.time(8,0))

    st.subheader("Parametri di passo (min)")
    base_p = st.number_input("Min/km (piano)", min_value=1.0, max_value=60.0, value=15.0, step=0.5)
    up_p   = st.number_input("Min/100 m (salita)", min_value=1.0, max_value=60.0, value=15.0, step=0.5)
    down_p = st.number_input("Min/200 m (discesa)", min_value=1.0, max_value=60.0, value=15.0, step=0.5)

    st.subheader("Condizioni")
    temp   = st.number_input("Temperatura (°C)", -30.0, 50.0, 15.0, 1.0)
    hum    = st.number_input("Umidità (%)", 0.0, 100.0, 50.0, 1.0)
    wind   = st.number_input("Vento (km/h)", 0.0, 150.0, 5.0, 1.0)
    precip_it = st.selectbox("Precipitazioni", ["assenza pioggia","pioviggine","pioggia","pioggia forte","neve fresca","neve profonda"], index=0)
    surf_it   = st.selectbox("Fondo", ["asciutto","fango","roccia bagnata","neve dura","ghiaccio"], index=0)
    expo_it   = st.selectbox("Esposizione", ["ombra","misto","pieno sole"], index=1)
    tech      = st.selectbox("Tecnica", ["facile","normale","roccioso","passaggi di roccia (scrambling)","neve/ghiaccio"], index=1)
    loadkg    = st.number_input("Zaino extra (kg)", 0.0, 40.0, 6.0, 1.0)

if not uploaded:
    st.info("Carica un file GPX per iniziare.")
    st.stop()

gpx_bytes = uploaded.read()
lat, lon, ele = parse_gpx_file(gpx_bytes)
if len(ele) < 2:
    st.error("GPX privo di quota utilizzabile.")
    st.stop()

# Calcolo core
res = compute_from_arrays(
    lat, lon, ele,
    base_min_per_km=base_p, up_min_per_100m=up_p, down_min_per_200m=down_p,
    reverse=invert_trace, weight_kg=70.0
)

# Metriche top
c1,c2,c3 = st.columns(3)
c1.metric("Distanza (km)", f"{res['tot_km']:.2f}")
c2.metric("Dislivello + (m)", f"{int(res['dplus'])}")
c3.metric("Tempo totale", format_hm(res["t_total"]))

# IF + Dettagli + Grafico (unico, con etichette opzionali)
cL, cR = st.columns([1,2])

with cL:
    st.subheader("Indice di Difficoltà")
    IF_val, IF_cat = compute_if(res, temp, hum, precip_it, surf_it, wind, expo_it, tech, loadkg)
    st.markdown(f"<div style='font-size:26px;font-weight:700'>{IF_val} ({IF_cat})</div>", unsafe_allow_html=True)
    st.markdown(gauge_svg(IF_val), unsafe_allow_html=True)

    st.subheader("Risultati")
    st.write(f"Dislivello − (m): **{int(res['dneg'])}**")
    st.write(f"Tempo piano: **{format_hm(res['t_dist'])}**")
    st.write(f"Tempo salita: **{format_hm(res['t_up'])}**")
    st.write(f"Tempo discesa: **{format_hm(res['t_down'])}**")
    st.write(f"Calorie stimate: **{res['cal_total']}**")
    st.write(f"Piano (km): **{res['len_flat_km']:.2f}** — Salita (km): **{res['len_up_km']:.2f}** — Discesa (km): **{res['len_down_km']:.2f}**")
    st.write(f"Pend. media salita: **{res['grade_up_pct']:.1f}%** — discesa: **{res['grade_down_pct']:.1f}%**")
    st.write(f"LCS ≥25% (m): **{int(res['lcs25_m'])}**")
    st.write(f"Blocchi ripidi (≥100 m @ ≥25%): **{int(res['blocks25_count'])}**")
    st.write(f"Surge (cambi ritmo)/km: **{res['surge_idx_per_km']:.2f}**")
    holes = int(res["holes"])
    st.write(f"Buchi GPX: **{holes}**" + (" ⚠️ *Attenzione, calcoli potenzialmente alterati*" if holes>0 else " ✅"))

# Preparazione split e grafico unico con etichette
km_df = split_per_km(lat, lon, np.array(res["profile_y_m"]), base_p, up_p, down_p)

with cR:
    st.subheader("Profilo altimetrico")
    x = np.array(res["profile_x_km"])
    y = np.array(res["profile_y_m"])
    dfp = pd.DataFrame({"Distanza (km)": x, "Quota (m)": y})
    chart = alt.Chart(dfp).mark_line().encode(
        x=alt.X("Distanza (km):Q"),
        y=alt.Y("Quota (m):Q")
    ).properties(height=340)

    if show_labels and not km_df.empty:
        lab_df = km_df.copy()
        lab_df["Distanza (km)"] = lab_df["Km"].astype(float)
        if show_clock:
            t0 = dt.datetime.combine(dt.date.today(), start_time)
            def add_clock(s):
                h, m = s.split(":")
                return (t0 + dt.timedelta(hours=int(h), minutes=int(m))).strftime("%H:%M")
            lab_df["label"] = lab_df["Km"].astype(str) + " - " + lab_df["Cumulato"].apply(add_clock)
        else:
            lab_df["label"] = lab_df["Km"].astype(str) + " - " + lab_df["Cumulato"]

        points = alt.Chart(lab_df).mark_point(size=40).encode(
            x="Distanza (km):Q", y=alt.Y("Alt (m):Q", scale=alt.Scale(zero=False))
        )
        texts = alt.Chart(lab_df).mark_text(dy=-10, fontSize=11).encode(
            x="Distanza (km):Q", y="Alt (m):Q", text="label"
        )
        st.altair_chart(chart + points + texts, use_container_width=True)
    else:
        st.altair_chart(chart, use_container_width=True)

# Tabella split (niente grafico duplicato)
st.subheader("Split per km")
if not km_df.empty:
    if show_clock:
        t0 = dt.datetime.combine(dt.date.today(), start_time)
        def to_clock(tstr):
            h, m = tstr.split(":")
            return (t0 + dt.timedelta(hours=int(h), minutes=int(m))).strftime("%H:%M")
        km_df_show = km_df.copy()
        km_df_show["Cumulato"] = km_df_show["Cumulato"].apply(to_clock)
    else:
        km_df_show = km_df
    st.dataframe(
        km_df_show[["Km","Split","Cumulato"]],
        use_container_width=True,
        height=min(360, 42*(len(km_df_show)+1))
    )

# Export waypoint
if not km_df.empty:
    if show_clock:
        base_dt = dt.datetime.combine(dt.date.today(), start_time)
        def name_for_row(row):
            h, m = row["Cumulato"].split(":")
            clock = (base_dt + dt.timedelta(hours=int(h), minutes=int(m))).strftime("%H:%M")
            return f"Km {int(row['Km'])} - {clock}"
    else:
        def name_for_row(row):
            return f"Km {int(row['Km'])} - {row['Cumulato']}"
    wp_names = [name_for_row(r) for _, r in km_df.iterrows()]

    st.subheader("Esportazione waypoint")
    colA, colB = st.columns(2)
    with colA:
        gpx_aug = add_waypoints_gpx(gpx_bytes, km_df, wp_names)
        if gpx_aug is not None:
            st.download_button("⬇️ Scarica GPX (traccia + waypoint)",
                               data=gpx_aug, file_name="traccia_con_waypoint.gpx",
                               mime="application/gpx+xml", use_container_width=True)
        else:
            st.info("Impossibile creare GPX con waypoints (file non compatibile).")
    with colB:
        gpx_wp = waypoints_only_gpx(km_df, wp_names)
        st.download_button("⬇️ Scarica GPX (solo waypoint)",
                           data=gpx_wp, file_name="waypoint_km.gpx",
                           mime="application/gpx+xml", use_container_width=True)

# Note di stato
note = []
if res.get("loop_fix_applied"):
    note.append(f"corretta deriva ~{res['loop_drift_abs_m']} m")
if res.get("loop_balance_applied"):
    note.append(f"chiusura anello: D+ allineato a D− (diff {res['balance_diff_m']} m)")
if note:
    st.info(" / ".join(note))
