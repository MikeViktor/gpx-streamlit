# -*- coding: utf-8 -*-
import math, datetime as dt
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# ================== Costanti e default (come gpx_gui.py) ==================
APP_TITLE = "Tempo percorrenza sentiero — web"
APP_VER   = "v5 (allineata ai calcoli desktop)"

# Ricampionamento / filtri
RS_STEP_M     = 3.0
RS_MIN_DELEV  = 0.25
RS_MED_K      = 3
RS_AVG_K      = 3
ABS_JUMP_RAW  = 100.0

# Correzione anello
LOOP_TOL_M        = 200.0
DRIFT_MIN_ABS_M   = 2.0
BALANCE_MIN_DIFFM = 10.0
BALANCE_REL_FRAC  = 0.05

# Pesi Indice di Difficoltà
W_D       = 0.5
W_PLUS    = 1.0
W_COMP    = 0.5
W_STEEP   = 0.4
W_STEEP_D = 0.3
W_LCS     = 0.25
W_BLOCKS  = 0.15
W_SURGE   = 0.25
IF_S0     = 80.0
ALPHA_METEO = 0.6
SEVERITY_GAIN = 1.52

DEFAULTS = {
    "base": 15.0, "up": 15.0, "down": 15.0,
    "weight": 70.0, "reverse": False,
    "temp": 15.0, "hum": 50.0, "wind": 5.0,
    "precip": "assenza pioggia",
    "surface": "asciutto",
    "expo": "misto",
    "tech": "normale",
    "loadkg": 6.0,
}

PRECIP_OPTIONS = ["assenza pioggia","pioviggine","pioggia","pioggia forte","neve fresca","neve profonda"]
SURF_OPTIONS   = ["asciutto","fango","roccia bagnata","neve dura","ghiaccio"]
EXPO_OPTIONS   = ["ombra","misto","pieno sole"]
TECH_OPTIONS   = ["facile","normale","roccioso","passaggi di roccia (scrambling)","neve/ghiaccio"]

# ================== Util GPX/geom ==================
def _is_tag(e, name: str) -> bool:
    t = e.tag
    return t.endswith('}' + name) or t == name

def parse_gpx_bytes(data: bytes):
    root = ET.fromstring(data)
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

# ================== Fattori e IF (identici al desktop) ==================
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

    precip_map = {"dry":1.00,"drizzle":1.05,"rain":1.15,"heavy_rain":1.30,"snow_shallow":1.25,"snow_deep":1.60}
    surface_map = {"dry":1.00,"mud":1.10,"wet_rock":1.15,"hard_snow":1.30,"ice":1.60}
    exposure_map = {"shade":1.00,"mixed":1.05,"sun":1.10}

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

def compute_from_arrays(lat, lon, ele_raw,
                        base_min_per_km=15.0, up_min_per_100m=15.0, down_min_per_200m=15.0,
                        weight_kg=70.0, reverse=False):

    if len(ele_raw) < 2: raise ValueError("Nessun punto utile con elevazione nel GPX.")
    if reverse: lat=list(reversed(lat)); lon=list(reversed(lon)); ele_raw=list(reversed(ele_raw))

    cum=[0.0]
    for i in range(1,len(lat)):
        cum.append(cum[-1]+dist_km(lat[i-1],lon[i-1],lat[i],lon[i])*1000.0)
    tot_km=cum[-1]/1000.0; total_m=cum[-1]

    e_res=resample_elev(cum,ele_raw,RS_STEP_M)
    e_res, loop_fix, loop_drift = apply_loop_drift_correction(e_res,lat,lon)
    e_med=median_k(e_res,RS_MED_K); e_sm=moving_avg(e_med,RS_AVG_K)

    dplus=dneg=0.0; asc_len=desc_len=flat_len=0.0; asc_gain=desc_drop=0.0
    asc_bins=[0,0,0,0,0]; desc_bins=[0,0,0,0,0]
    longest_steep_run=0.0; current_run=0.0; blocks25=0; last_state=0; surge_trans=0

    for i in range(1,len(e_sm)):
        t_prev=(i-1)*RS_STEP_M; t_curr=min(i*RS_STEP_M,total_m); seg=max(0.0,t_curr-t_prev)
        if seg<=0: continue
        d=e_sm[i]-e_sm[i-1]
        if d>RS_MIN_DELEV:
            dplus+=d; asc_len+=seg; asc_gain+=d; g=(d/seg)*100.0
            if   g<10: asc_bins[0]+=seg
            elif g<20: asc_bins[1]+=seg
            elif g<30: asc_bins[2]+=seg
            elif g<40: asc_bins[3]+=seg
            else:      asc_bins[4]+=seg
            if g>=25: current_run+=seg; longest_steep_run=max(longest_steep_run,current_run); state=2
            else:
                if current_run>=100: blocks25+=1
                current_run=0.0; state=1 if g<15 else 0
            if (last_state in (1,2)) and (state in (1,2)) and (state!=last_state): surge_trans+=1
            if state!=0: last_state=state
        elif d<-RS_MIN_DELEV:
            drop=-d; dneg+=drop; desc_len+=seg; desc_drop+=drop; g=(drop/seg)*100.0
            if   g<10: desc_bins[0]+=seg
            elif g<20: desc_bins[1]+=seg
            elif g<30: desc_bins[2]+=seg
            elif g<40: desc_bins[3]+=seg
            else:      desc_bins[4]+=seg
            if current_run>=100: blocks25+=1
            current_run=0.0; last_state=0
        else:
            flat_len+=seg
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

    t_dist  = tot_km*base_min_per_km
    t_up    = (dplus/100.0)*up_min_per_100m
    t_down  = (dneg/200.0)*down_min_per_200m
    t_total = t_dist+t_up+t_down

    holes = sum(1 for i in range(1,len(ele_raw)) if abs(ele_raw[i]-ele_raw[i-1])>=ABS_JUMP_RAW)

    weight_kg=max(1.0,float(weight_kg))
    cal_flat=weight_kg*0.6*max(0.0,tot_km)
    cal_up  =weight_kg*0.006*max(0.0,dplus)
    cal_down=weight_kg*0.003*max(0.0,dneg)
    cal_tot=int(round(cal_flat+cal_up+cal_down))

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
        "loop_fix_applied": False, "loop_drift_abs_m": 0.0,
        "loop_balance_applied": balance, "loop_like": loop_like, "balance_diff_m": float(diff),
    }

def compute_if_from_res(res, temp_c, humidity_pct, precip_it, surface_it, wind_kmh, expo_it, technique_level, extra_load_kg):
    D_km=float(res["tot_km"]); Dp=float(res["dplus"])
    C=(Dp/max(0.001,D_km))

    ascL=1000.0*float(res["len_up_km"]); descL=1000.0*float(res["len_down_km"])
    ab = res.get("asc_bins_m",[0,0,0,0,0]); db = res.get("desc_bins_m",[0,0,0,0,0])
    up25_m   = (ab[3] if len(ab)>3 else 0) + (ab[4] if len(ab)>4 else 0)
    down25_m = (db[3] if len(db)>3 else 0) + (db[4] if len(db)>4 else 0)
    f_up25   = up25_m   / max(1.0, ascL)
    f_down25 = down25_m / max(1.0, descL)

    lcs   = float(res.get("lcs25_m",0.0))
    blocks= float(res.get("blocks25_count",0.0))
    surge = float(res.get("surge_idx_per_km",0.0))
    lcs_sc = lcs/200.0

    S=(W_D*D_km + W_PLUS*(Dp/100.0) + W_COMP*(C/100.0) +
       W_STEEP*(100.0*f_up25) + W_STEEP_D*(100.0*f_down25) +
        W_LCS*lcs_sc + W_BLOCKS*blocks + W_SURGE*surge)

    IF_base=100.0*(1.0 - math.exp(-S/max(1e-6,IF_S0)))

    precip_map={"assenza pioggia":"dry","pioviggine":"drizzle","pioggia":"rain","pioggia forte":"heavy_rain","neve fresca":"snow_shallow","neve profonda":"snow_deep"}
    surf_map={"asciutto":"dry","fango":"mud","roccia bagnata":"wet_rock","neve dura":"hard_snow","ghiaccio":"ice"}
    expo_map={"ombra":"shade","misto":"mixed","pieno sole":"sun"}

    M = meteo_multiplier(temp_c, humidity_pct, precip_map[precip_it], surf_map[surface_it], wind_kmh, expo_map[expo_it]) \
        * altitude_multiplier(res.get("avg_alt_m")) \
        * technique_multiplier(technique_level) \
        * pack_load_multiplier(extra_load_kg)

    bump = (100.0 - IF_base) * max(0.0, (M - 1.0)) * ALPHA_METEO
    IF = min(100.0, (IF_base + bump) * SEVERITY_GAIN)
    IF = round(IF,1)
    return {"IF": IF, "cat": cat_from_if(IF)}

# ================== Gauge SVG (spicchi pieni + ago dal centro) ==================
def gauge_svg_html(value: float, width: int = 620, height: int = 210, show_labels: bool = True) -> str:
    v = max(0.0, min(100.0, float(value)))
    cx, cy = width / 2.0, height - 12.0
    R_outer = min(width * 0.42, height * 0.95)
    R_inner = R_outer - 24.0

    bands = [
        (0, 30,  "#2ecc71", "Facile"),
        (30, 50, "#f1c40f", "Medio"),
        (50, 70, "#e67e22", "Impeg."),
        (70, 80, "#e74c3c", "Diffic."),
        (80, 90, "#8e44ad", "Molto diff."),
        (90, 100,"#111111", "Estremo"),
    ]

    def val2ang(pct: float) -> float:
        return 180.0 - (pct / 100.0) * 180.0

    def polar(r: float, deg: float):
        rad = math.radians(deg)
        return (cx + r * math.cos(rad), cy - r * math.sin(rad))

    def ring_segment(a0: float, a1: float, color: str) -> str:
        xo0, yo0 = polar(R_outer, a0)
        xo1, yo1 = polar(R_outer, a1)
        xi1, yi1 = polar(R_inner, a1)
        xi0, yi0 = polar(R_inner, a0)
        large = 1 if abs(a0 - a1) > 180 else 0
        d = (
            f"M {xo0:.1f},{yo0:.1f} "
            f"A {R_outer:.1f},{R_outer:.1f} 0 {large} 1 {xo1:.1f},{yo1:.1f} "
            f"L {xi1:.1f},{yi1:.1f} "
            f"A {R_inner:.1f},{R_inner:.1f} 0 {large} 0 {xi0:.1f},{yi0:.1f} Z"
        )
        return f'<path d="{d}" fill="{color}" stroke="{color}" stroke-width="1"/>'

    segs = []
    for a, b, col, _lab in bands:
        a0 = val2ang(a); a1 = val2ang(b)
        if a0 < a1:  # garantiamo ordine
            a0, a1 = a1, a0
        segs.append(ring_segment(a0, a1, col))

    # Ago lungo dal centro
    ang = val2ang(v)
    x_tip, y_tip = polar(R_outer - 8.0, ang)
    needle = (
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x_tip:.1f}" y2="{y_tip:.1f}" stroke="#333" stroke-width="5" />'
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="#333"/>'
    )

    labels_svg = ""
    if show_labels:
        r_lab = (R_inner + R_outer) / 2.0 - 8.0
        for a, b, _col, lab in bands:
            mid = (a + b) / 2.0
            ax = val2ang(mid)
            lx, ly = polar(r_lab, ax)
            labels_svg += (
                f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
                f'font-family="Segoe UI, Roboto, Arial" font-size="12" fill="#111">{lab}</text>'
            )

    svg = (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">{"".join(segs)}{needle}{labels_svg}</svg>'
    )
    return svg

# ================== UI ==================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE + " — " + APP_VER)

# Sidebar
st.sidebar.header("Impostazioni")
rev = st.sidebar.checkbox("Inverti traccia", value=DEFAULTS["reverse"])

st.sidebar.subheader("Tempi per km")
show_daytime = st.sidebar.checkbox("Mostra orario del giorno", value=True)
show_labels_graph  = st.sidebar.checkbox("Mostra etichette sul grafico", value=True)
start_time   = st.sidebar.time_input("Orario di partenza", value=dt.time(8,0))

st.sidebar.subheader("Parametri di passo (min)")
base = st.sidebar.number_input("Min/km (piano)",  5.0, 60.0, DEFAULTS["base"], 0.5)
up   = st.sidebar.number_input("Min/100 m (salita)", 5.0, 60.0, DEFAULTS["up"], 0.5)
down = st.sidebar.number_input("Min/200 m (discesa)",5.0, 60.0, DEFAULTS["down"], 0.5)

st.sidebar.subheader("Condizioni")
temp = st.sidebar.number_input("Temperatura (°C)", -30.0, 50.0, DEFAULTS["temp"], 1.0)
hum  = st.sidebar.number_input("Umidità (%)", 0.0, 100.0, DEFAULTS["hum"], 1.0)
wind = st.sidebar.number_input("Vento (km/h)", 0.0, 150.0, DEFAULTS["wind"], 1.0)
precip = st.sidebar.selectbox("Precipitazioni", PRECIP_OPTIONS, index=PRECIP_OPTIONS.index(DEFAULTS["precip"]))
surface= st.sidebar.selectbox("Fondo", SURF_OPTIONS, index=SURF_OPTIONS.index(DEFAULTS["surface"]))
expo   = st.sidebar.selectbox("Esposizione", EXPO_OPTIONS, index=EXPO_OPTIONS.index(DEFAULTS["expo"]))
tech   = st.sidebar.selectbox("Tecnica", TECH_OPTIONS, index=TECH_OPTIONS.index(DEFAULTS["tech"]))
loadkg = st.sidebar.number_input("Zaino extra (kg)", 0.0, 40.0, DEFAULTS["loadkg"], 1.0)

st.sidebar.subheader("Aspetto profilo")
graph_width_ratio = st.sidebar.slider("Larghezza grafico", 2, 5, 4, help="Allarga o restringi il riquadro del profilo per rendere più 'ripide' o 'dolci' le pendenze (impatta l'asse X).")

# --- Uploader + calcolo con modalità placeholder ---
uploaded   = st.file_uploader("Trascina qui il file GPX", type=["gpx"])
have_data  = uploaded is not None
res, fi    = None, None

if have_data:
    try:
        data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        lat, lon, ele = parse_gpx_bytes(data)
        if len(ele) < 2:
            have_data = False
            st.warning("Il GPX non contiene quote utili. Mostro solo il layout.")
        else:
            res = compute_from_arrays(
                lat, lon, ele,
                base_min_per_km=base,
                up_min_per_100m=up,
                down_min_per_200m=down,
                weight_kg=DEFAULTS["weight"],
                reverse=rev
            )
            fi  = compute_if_from_res(
                res,
                temp_c=temp, humidity_pct=hum,
                precip_it=precip, surface_it=surface,
                wind_kmh=wind, expo_it=expo,
                technique_level=tech, extra_load_kg=loadkg
            )
    except Exception as e:
        have_data = False
        st.warning(f"Impossibile calcolare: {e}. Mostro solo il layout.")

if not have_data:
    # Placeholder: valori a zero e profilo piatto 0..10 km
    res = {
        "tot_km": 0.0, "dplus": 0.0, "dneg": 0.0,
        "t_dist": 0.0, "t_up": 0.0, "t_down": 0.0, "t_total": 0.0,
        "holes": 0,
        "len_flat_km": 0.0, "len_up_km": 0.0, "len_down_km": 0.0,
        "grade_up_pct": 0.0, "grade_down_pct": 0.0,
        "cal_total": 0,
        "asc_bins_m": [0,0,0,0,0], "desc_bins_m": [0,0,0,0,0],
        "lcs25_m": 0, "blocks25_count": 0, "surge_idx_per_km": 0.0,
        "avg_alt_m": None,
        "profile_x_km": list(np.linspace(0, 10, 41)),
        "profile_y_m": [0.0]*41,
    }
    fi = {"IF": 0.0, "cat": "—"}

# === Testate: Distanza, D+ e Tempo totale ===
c1,c2,c3 = st.columns(3)
c1.metric("Distanza (km)", f"{res['tot_km']:.2f}")
c2.metric("Dislivello + (m)", f"{int(res['dplus'])}")
c3.metric("Tempo totale", f"{int(res['t_total']//60)}:{int(round(res['t_total']%60)):02d}")

# === INDICE DI DIFFICOLTÀ (titolo/numero a sinistra, gauge a destra) ===
gc1, gc2 = st.columns([1, 2])

with gc1:
    st.subheader("Indice di Difficoltà")
    st.markdown(
        f"""
        <div style="font-size:44px;font-weight:400;line-height:1;margin:2px 0 4px 0;">
            {fi['IF']:.1f}
        </div>
        <div style="font-size:16px;color:#666;margin-top:-2px;">
            {fi['cat']}
        </div>
        """,
        unsafe_allow_html=True
    )

with gc2:
    st.markdown(
        f"""
        <div style="margin-top:-8px; max-width:640px;">
            {gauge_svg_html(fi['IF'], show_labels=True)}
        </div>
        """,
        unsafe_allow_html=True
    )

# === Risultati dettagliati ===
st.subheader("Risultati")
cols = st.columns(2)
with cols[0]:
    st.write(f"- **Dislivello − (m):** {int(res['dneg'])}")
    st.write(f"- **Tempo piano:** {int(res['t_dist']//60)}:{int(round(res['t_dist']%60)):02d}")
    st.write(f"- **Tempo salita:** {int(res['t_up']//60)}:{int(round(res['t_up']%60)):02d}")
    st.write(f"- **Tempo discesa:** {int(res['t_down']//60)}:{int(round(res['t_down']%60)):02d}")
    st.write(f"- **Calorie stimate:** {res['cal_total']}")
    st.write(f"- **Piano (km):** {res['len_flat_km']:.2f} — **Salita (km):** {res['len_up_km']:.2f} — **Discesa (km):** {res['len_down_km']:.2f}")
    st.write(f"- **Pend. media salita (%):** {res['grade_up_pct']:.1f} — **discesa (%):** {res['grade_down_pct']:.1f}")
with cols[1]:
    st.write(f"- **LCS ≥25% (m):** {int(res['lcs25_m'])}")
    st.write(f"- **Blocchi ripidi (≥100 m @ ≥25%):** {int(res['blocks25_count'])}")
    st.write(f"- **Surge (cambi ritmo)/km:** {res['surge_idx_per_km']:.2f}")
    if have_data:
        holes = int(res["holes"])
        st.write(f"- **Buchi GPX:** {'OK (0)' if holes==0 else f'ATTENZIONE ({holes})'}")
    else:
        st.write(f"- **Buchi GPX:** —")

# === Profilo altimetrico con etichette Km/Tempo ===
st.subheader("Profilo altimetrico")

# Regolo la larghezza del riquadro tramite colonne a rapporto variabile
gcol, spacer = st.columns([graph_width_ratio, max(1, 6-graph_width_ratio)])
with gcol:
    x = res["profile_x_km"]; y = res["profile_y_m"]
    df = pd.DataFrame({"km": x, "ele": y})

    # etichette
    km_ticks = list(range(0, int(math.ceil(res["tot_km"]))+1)) if have_data else list(range(0, 11))
    ann = []
    step_km = RS_STEP_M/1000.0
    dt_steps=[0.0]
    for i in range(1,len(y)):
        dz = y[i]-y[i-1]
        t_flat = base * step_km
        t_up   = up   * max(0.0, dz)/100.0
        t_down = down * max(0.0,-dz)/200.0
        dt_steps.append(t_flat+t_up+t_down)
    cum = np.cumsum(dt_steps)

    if have_data:
        for k in km_ticks:
            idx = int(np.argmin(np.abs(df["km"].values - k)))
            yk = float(df.loc[idx,"ele"])
            t  = float(cum[idx])
            if show_daytime:
                base_dt = dt.datetime.combine(dt.date.today(), start_time)
                txt = (base_dt + dt.timedelta(minutes=t)).strftime("%H:%M")
            else:
                hh = int(t//60); mm = int(round(t - hh*60))
                if mm==60: hh+=1; mm=0
                txt = f"{hh}:{mm:02d}"
            ann.append({"km": float(k), "ele": yk, "top": f"{k} km", "bot": txt})
    ann_df = pd.DataFrame(ann) if ann else pd.DataFrame(columns=["km","ele","top","bot"])

    line = alt.Chart(df).mark_line().encode(
        x=alt.X("km:Q", axis=alt.Axis(title="Distanza (km)", values=km_ticks)),
        y=alt.Y("ele:Q", axis=alt.Axis(title="Quota (m)")),
    ).properties(height=360)

    if have_data and show_labels_graph and not ann_df.empty:
        text1 = alt.Chart(ann_df).mark_text(fontSize=12, dy=-14, fontWeight="bold").encode(x="km:Q", y="ele:Q", text="top:N")
        text2 = alt.Chart(ann_df).mark_text(fontSize=12, dy=12).encode(x="km:Q", y="ele:Q", text="bot:N")
        st.altair_chart(alt.layer(line, text1, text2).resolve_scale(y='shared'), use_container_width=True)
    else:
        st.altair_chart(line, use_container_width=True)

# === Tempi / Orario ai diversi Km (tabella) ===
st.subheader("Tempi / Orario ai diversi Km")
if have_data:
    rows=[]
    for k in range(1, len(km_ticks)):
        idx_k   = int(np.argmin(np.abs(df["km"].values - k)))
        idx_km1 = int(np.argmin(np.abs(df["km"].values - (k-1))))
        t_cum = float(cum[idx_k]); t_prev = float(cum[idx_km1]); t_split = t_cum - t_prev
        hh_s=int(t_split//60); mm_s=int(round(t_split-hh_s*60))
        if mm_s==60: hh_s+=1; mm_s=0
        split_txt=f"{hh_s}:{mm_s:02d}"
        if show_daytime:
            base_dt = dt.datetime.combine(dt.date.today(), start_time)
            cum_txt = (base_dt + dt.timedelta(minutes=t_cum)).strftime("%H:%M")
        else:
            hh_c=int(t_cum//60); mm_c=int(round(t_cum-hh_c*60))
            if mm_c==60: hh_c+=1; mm_c=0
            cum_txt=f"{hh_c}:{mm_c:02d}"
        rows.append({"Km": k, "Tempo parziale": split_txt, "Cumulativo": cum_txt})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=360)
else:
    # Tabella “vuota” con km 1..10 e celle senza valore
    rows = [{"Km": k, "Tempo parziale": "—", "Cumulativo": "—"} for k in range(1,11)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=360)
