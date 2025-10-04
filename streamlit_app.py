# -*- coding: utf-8 -*-
# streamlit_app.py

import math
import io
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import xml.etree.ElementTree as ET
from datetime import time as _time

# ===================== Costanti/Default (allineate alla GUI) =====================
RS_STEP_M     = 3.0
RS_MIN_DELEV  = 0.25
RS_MED_K      = 3
RS_AVG_K      = 3
ABS_JUMP_RAW  = 100.0

LOOP_TOL_M        = 200.0
DRIFT_MIN_ABS_M   = 2.0
BALANCE_MIN_DIFFM = 10.0
BALANCE_REL_FRAC  = 0.05

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

# ===================== Utility GPX =====================
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

def cumulative_dist_m(lat, lon):
    n = len(lat)
    cum = np.zeros(n, dtype=float)
    for i in range(1, n):
        cum[i] = cum[i-1] + dist_km(lat[i-1], lon[i-1], lat[i], lon[i]) * 1000.0
    return cum

def resample_along(cum_m, values, step_m=3.0):
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

def is_loop(lat, lon, tol_m=LOOP_TOL_M) -> bool:
    if len(lat) < 2: return False
    return dist_km(lat[0], lon[0], lat[-1], lon[-1]) * 1000.0 <= tol_m

def correct_linear_drift(elev_m, lat, lon, min_abs=DRIFT_MIN_ABS_M):
    if not is_loop(lat, lon) or len(elev_m) < 2:
        return elev_m, False, 0.0
    drift = float(elev_m[-1] - elev_m[0])
    if abs(drift) < min_abs: return elev_m, False, drift
    n = len(elev_m) - 1
    idx = np.arange(len(elev_m), dtype=float)
    corr = elev_m - drift * (idx / n)
    return corr, True, drift

# ===================== Moltiplicatori (meteo, quota, tecnica, zaino) =====================
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
    M_precip  = precip_map.get(precip,1.00)
    M_surface = surface_map.get(surface,1.00)
    M_sun     = exposure_map.get(exposure,1.00)
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
    return table.get(level,1.0)

def pack_load_multiplier(extra_load_kg=0.0):
    return 1.0 + 0.02 * max(0.0, extra_load_kg / 5.0)

def cat_from_if(val: float) -> str:
    if val < 30: return "Facile"
    if val < 50: return "Medio"
    if val < 70: return "Impegnativo"
    if val < 80: return "Difficile"
    if val <= 90: return "Molto difficile"
    return "Estremamente difficile"

# ===================== Calcoli principali (allineati alla GUI) =====================
def compute_all(lat, lon, ele,
                base_min_per_km=15.0, up_min_per_100m=15.0, down_min_per_200m=15.0,
                weight_kg=70.0, reverse=False):

    lat = list(lat); lon = list(lon); ele = list(ele)
    if reverse:
        lat.reverse(); lon.reverse(); ele.reverse()
    if len(ele) < 2:
        raise ValueError("Nessun punto utile con elevazione nel GPX.")

    cum_raw = cumulative_dist_m(lat, lon)
    grid_m, e_res = resample_along(cum_raw, np.asarray(ele, dtype=float), step_m=RS_STEP_M)
    e_med = median_k(e_res, RS_MED_K); e_sm = moving_avg(e_med, RS_AVG_K)
    e_fix, loop_fix, loop_drift = correct_linear_drift(e_sm, lat, lon, min_abs=DRIFT_MIN_ABS_M)

    # ricampiono anche lat/lon
    _, lat_rs = resample_along(cum_raw, np.asarray(lat, dtype=float), step_m=RS_STEP_M)
    _, lon_rs = resample_along(cum_raw, np.asarray(lon, dtype=float), step_m=RS_STEP_M)

    total_m = grid_m[-1]
    tot_km = total_m / 1000.0

    # metriche salita/discesa + tempi per ogni step
    dplus=dneg=0.0; asc_len=desc_len=flat_len=0.0; asc_gain=desc_drop=0.0
    asc_bins=[0,0,0,0,0]; desc_bins=[0,0,0,0,0]
    longest_steep_run=0.0; current_run=0.0; blocks25=0; last_state=0; surge_trans=0

    dt_min = np.zeros_like(grid_m)

    for i in range(1, len(grid_m)):
        seg = max(0.0, grid_m[i]-grid_m[i-1])
        dh  = e_fix[i]-e_fix[i-1]

        # tempi segmentali
        t_dist = (seg/1000.0) * base_min_per_km
        t_up   = (max(dh, 0.0)/100.0) * up_min_per_100m
        t_down = (max(-dh,0.0)/200.0) * down_min_per_200m
        dt_min[i] = t_dist + t_up + t_down

        if dh>RS_MIN_DELEV:
            dplus+=dh; asc_len+=seg; asc_gain+=dh; g=(dh/seg)*100.0
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
        elif dh<-RS_MIN_DELEV:
            drop=-dh; dneg+=drop; desc_len+=seg; desc_drop+=drop; g=(drop/seg)*100.0
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

    # chiusura anello
    loop_like = is_loop(lat,lon,LOOP_TOL_M)
    diff = abs(dplus-dneg)
    need_balance = loop_like and (diff>max(BALANCE_MIN_DIFFM,BALANCE_REL_FRAC*max(dplus,dneg)))
    balance=False
    if need_balance:
        dplus=dneg; asc_gain=dplus; balance=True

    grade_up   = (asc_gain/asc_len*100.0)  if asc_len>0 else 0.0
    grade_down = (desc_drop/desc_len*100.0) if desc_len>0 else 0.0

    t_dist  = (tot_km)*base_min_per_km
    t_up    = (dplus/100.0)*up_min_per_100m
    t_down  = (dneg/200.0)*down_min_per_200m
    t_total = t_dist+t_up+t_down

    holes = sum(1 for i in range(1,len(ele)) if abs(ele[i]-ele[i-1])>=ABS_JUMP_RAW)

    weight_kg=max(1.0,float(weight_kg))
    cal_flat=weight_kg*0.6*max(0.0,tot_km)
    cal_up  =weight_kg*0.006*max(0.0,dplus)
    cal_down=weight_kg*0.003*max(0.0,dneg)
    cal_tot=int(round(cal_flat+cal_up+cal_down))

    cum_min = np.cumsum(dt_min)

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
        "avg_alt_m": float(np.mean(ele)) if len(ele)>0 else None,
        "profile_x_km": list(grid_m/1000.0), "profile_y_m": list(e_fix),
        "cum_min": list(cum_min), "lat_rs": list(lat_rs), "lon_rs": list(lon_rs),
        "loop_fix_applied": bool(loop_fix), "loop_drift_abs_m": round(abs(loop_drift),1),
        "loop_balance_applied": bool(balance), "loop_like": bool(loop_like), "balance_diff_m": round(diff,1),
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

# ===================== Split/etichette e grafico =====================
def build_km_table(res, start_time=None, show_clock=False):
    D = np.asarray(res["profile_x_km"], dtype=float)
    E = np.asarray(res["profile_y_m"], dtype=float)
    LAT = np.asarray(res["lat_rs"], dtype=float)
    LON = np.asarray(res["lon_rs"], dtype=float)
    T = np.asarray(res["cum_min"], dtype=float)

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

def draw_altitude_profile(dist_km, elev_m, labels_df, show_labels=True):
    df = pd.DataFrame({"km": dist_km, "quota": elev_m})

    axis_vals = list(range(0, int(max(dist_km))+1))
    base = alt.Chart(df).mark_line().encode(
        x=alt.X("km:Q", title="Distanza (km)", axis=alt.Axis(values=axis_vals)),
        y=alt.Y("quota:Q", title="Quota (m)")
    ).properties(height=360)

    if show_labels and labels_df is not None and not labels_df.empty:
        pts = alt.Chart(labels_df).mark_point(size=50, color="#666", filled=True).encode(
            x="x_km:Q", y="y_m:Q"
        )
        txt = alt.Chart(labels_df).mark_text(align="left", baseline="middle", dx=6, dy=-8, fontSize=12).encode(
            x="x_km:Q", y="y_m:Q", text="label:N"
        )
        return (base + pts + txt).interactive()
    else:
        return base.interactive()

# ===================== Export GPX =====================
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

# ===================== Streamlit UI =====================
st.set_page_config(page_title="Analisi Tracce GPX", layout="wide")
st.title("Analisi Tracce GPX")

with st.sidebar:
    st.subheader("Parametri base")
    base = st.number_input("Min/Km (piano)", 1.0, 60.0, 15.0, 0.5)
    up   = st.number_input("Min/100 m (salita)", 1.0, 60.0, 15.0, 0.5)
    down = st.number_input("Min/200 m (discesa)", 1.0, 60.0, 15.0, 0.5)
    weight = st.number_input("Peso (kg)", 30.0, 150.0, 70.0, 1.0)
    reverse = st.checkbox("Inverti traccia", value=False)

    st.subheader("Condizioni")
    temp = st.number_input("Temperatura (°C)", -30.0, 50.0, 15.0, 1.0)
    hum  = st.number_input("Umidità (%)", 0.0, 100.0, 50.0, 1.0)
    wind = st.number_input("Vento (km/h)", 0.0, 150.0, 5.0, 1.0)
    precip = st.selectbox("Precipitazioni", ["assenza pioggia","pioviggine","pioggia","pioggia forte","neve fresca","neve profonda"])
    surface = st.selectbox("Fondo", ["asciutto","fango","roccia bagnata","neve dura","ghiaccio"])
    expo = st.selectbox("Esposizione", ["ombra","misto","pieno sole"])
    tech = st.selectbox("Tecnica", ["facile","normale","roccioso","passaggi di roccia (scrambling)","neve/ghiaccio"])
    loadkg = st.number_input("Zaino extra (kg)", 0.0, 40.0, 6.0, 1.0)

    st.subheader("Tempi per km")
    show_clock = st.checkbox("Mostra orario del giorno", value=False)
    show_labels = st.checkbox("Mostra etichette sul grafico", value=True)
    if show_clock:
        start_time = st.time_input("Orario di partenza", value=_time(8, 0))
    else:
        start_time = None

uploaded = st.file_uploader("Carica GPX", type=["gpx"])
if not uploaded:
    st.info("Carica un file **.gpx** per iniziare.")
    st.stop()

data_bytes = uploaded.read()
try:
    lat, lon, ele = parse_gpx_bytes(data_bytes)
except Exception as e:
    st.error(f"Errore parsing GPX: {e}")
    st.stop()

if len(ele) < 2:
    st.warning("GPX troppo breve o senza elevazioni.")
    st.stop()

# --- Calcolo principale (identico alla GUI) ---
res = compute_all(
    lat, lon, ele,
    base_min_per_km=base, up_min_per_100m=up, down_min_per_200m=down,
    weight_kg=weight, reverse=reverse
)

# --- Indice difficoltà (identico alla GUI) ---
fi = compute_if_from_res(
    res, temp, hum, precip, surface, wind, expo, tech, loadkg
)

# ----------------- Pannello risultati -----------------
st.subheader("Risultati")

cA, cB, cC = st.columns(3)
cA.metric("Distanza (km)", f"{res['tot_km']:.2f}")
cB.metric("Dislivello + (m)", f"{int(res['dplus'])}")
tt = res["t_total"]; hh=int(tt//60); mm=int(round(tt-hh*60)); 
if mm==60: hh,mm=hh+1,0
cC.metric("Tempo totale", f"{hh}:{mm:02d}")

c1, c2, c3 = st.columns(3)
c1.write(f"**Tempo piano:** {int(res['t_dist']//60)}:{int(round(res['t_dist']%60)):02d}")
c2.write(f"**Tempo salita:** {int(res['t_up']//60)}:{int(round(res['t_up']%60)):02d}")
c3.write(f"**Tempo discesa:** {int(res['t_down']//60)}:{int(round(res['t_down']%60)):02d}")

c4, c5, c6 = st.columns(3)
c4.write(f"**Piano (km):** {res['len_flat_km']:.2f}")
c5.write(f"**Salita (km):** {res['len_up_km']:.2f}")
c6.write(f"**Discesa (km):** {res['len_down_km']:.2f}")

c7, c8, c9 = st.columns(3)
c7.write(f"**Pend. media salita (%):** {res['grade_up_pct']:.1f}")
c8.write(f"**Pend. media discesa (%):** {res['grade_down_pct']:.1f}")
c9.write(f"**Calorie stimate:** {res['cal_total']}")

c10, c11, c12 = st.columns(3)
c10.write(f"**LCS ≥25% (m):** {int(res['lcs25_m'])}")
c11.write(f"**Blocchi ripidi (≥100 m @ ≥25%):** {int(res['blocks25_count'])}")
c12.write(f"**Surge (cambi ritmo)/km:** {res['surge_idx_per_km']:.2f}")

holes_col, if_col = st.columns([1,2])
holes = int(res["holes"])
holes_col.write(f"**Buchi GPX:** {'❗ ' if holes>0 else ''}{holes}")
if holes>0:
    holes_col.caption("⚠️ Attenzione: possibili salti altimetrici, alcuni calcoli possono essere alterati.")

if_col.markdown(f"**Indice di Difficoltà:** **{fi['IF']}**  ({fi['cat']})")

if res.get("loop_fix_applied"):
    st.caption(f"Applicata correzione deriva altimetrica (~{res['loop_drift_abs_m']:.1f} m).")
if res.get("loop_balance_applied"):
    st.caption(f"Chiusura anello: D+ allineato a D− (diff {res['balance_diff_m']} m).")

# ----------------- Profilo + etichette -----------------
st.subheader("Profilo altimetrico")
km_df = build_km_table(res, start_time=start_time, show_clock=show_clock)
chart = draw_altitude_profile(res["profile_x_km"], res["profile_y_m"], km_df, show_labels=show_labels)
st.altair_chart(chart, use_container_width=True)

# ----------------- Tabella split -----------------
st.subheader("Split per km")
show_cols = ["Km", "Split", "Cumulato"] + (["Orario"] if show_clock else [])
st.dataframe(km_df[show_cols], use_container_width=True, hide_index=True)

# CSV split
csv_buf = io.StringIO()
km_df[show_cols].to_csv(csv_buf, index=False)
st.download_button(
    "Scarica split (CSV)",
    data=csv_buf.getvalue(),
    file_name=f"{uploaded.name.rsplit('.',1)[0]}_splits.csv",
    mime="text/csv"
)

# ----------------- Export GPX con waypoint -----------------
st.subheader("Esporta waypoint km/tempo")
orig_text = data_bytes.decode("utf-8", errors="ignore")
gpx_with = make_gpx_with_waypoints(orig_text, km_df)
gpx_only = make_gpx_waypoints_only(km_df)

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
