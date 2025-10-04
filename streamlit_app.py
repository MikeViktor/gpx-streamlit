# -*- coding: utf-8 -*-
# streamlit_app.py — v6.1 (robusta: niente layer Altair sul profilo)

import io
import math
import datetime as dt
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from typing import List, Tuple, Dict, Any

st.set_page_config(page_title="GPX – tempi e difficoltà", layout="wide")

APP_TITLE = "GPX → tempi, profilo e indice di difficoltà"
APP_VER = "v6.1"

# ===== Ricampionamento / filtri =====
RS_STEP_M     = 3.0
RS_MIN_DELEV  = 0.25
RS_MED_K      = 3
RS_AVG_K      = 3
ABS_JUMP_RAW  = 100.0

# ===== Correzione anello =====
LOOP_TOL_M        = 200.0
DRIFT_MIN_ABS_M   = 2.0
BALANCE_MIN_DIFFM = 10.0
BALANCE_REL_FRAC  = 0.05

# ===== Pesi IF =====
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

# ===== Default =====
DEFAULTS = dict(
    base=15.0, up=15.0, down=15.0, weight=70.0, reverse=False,
    temp=15.0, hum=50.0, wind=5.0,
    precip="assenza pioggia",
    surface="asciutto",
    expo="misto",
    tech="normale",
    loadkg=6.0,
)

PRECIP_OPTIONS = ["assenza pioggia","pioviggine","pioggia","pioggia forte","neve fresca","neve profonda"]
SURF_OPTIONS   = ["asciutto","fango","roccia bagnata","neve dura","ghiaccio"]
EXPO_OPTIONS   = ["ombra","misto","pieno sole"]
TECH_OPTIONS   = ["facile","normale","roccioso","passaggi di roccia (scrambling)","neve/ghiaccio"]

# ---------- util ----------
def fmt_hm(minutes: float) -> str:
    h = int(minutes // 60)
    m = int(round(minutes - h*60))
    if m == 60: h += 1; m = 0
    return f"{h}:{m:02d}"

def fmt_mmss(minutes: float) -> str:
    total_sec = int(round(minutes*60))
    mm = total_sec // 60
    ss = total_sec % 60
    return f"{mm}:{ss:02d}"

def cat_from_if(val: float) -> str:
    if val < 30: return "Facile"
    if val < 50: return "Medio"
    if val < 70: return "Impegnativo"
    if val < 80: return "Difficile"
    if val <= 90: return "Molto difficile"
    return "Estremamente difficile"

# ---------- GPX parsing ----------
def _is_tag(e, name: str) -> bool:
    t = e.tag
    return t.endswith('}' + name) or t == name

def parse_gpx_bytes(data: bytes) -> Tuple[List[float], List[float], List[float]]:
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

# ---------- modulatori ----------
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

# ---------- core ----------
def compute_from_arrays(lat, lon, ele_raw,
                        base_min_per_km=15.0, up_min_per_100m=15.0, down_min_per_200m=15.0,
                        weight_kg=70.0, reverse=False):
    if len(ele_raw) < 2: raise ValueError("Nessun punto utile con elevazione nel GPX.")
    if reverse:
        lat=list(reversed(lat)); lon=list(reversed(lon)); ele_raw=list(reversed(ele_raw))

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
            if g>=25:
                current_run+=seg; longest_steep_run=max(longest_steep_run,current_run); state=2
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

# ---------- splits ----------
def elevation_at_dist_m(elev: List[float], dist_m: float) -> float:
    if not elev: return 0.0
    idx = dist_m / RS_STEP_M
    i = int(idx)
    if i <= 0: return elev[0]
    if i >= len(elev)-1: return elev[-1]
    u = idx - i
    return elev[i] + u*(elev[i+1]-elev[i])

def compute_km_splits(res: Dict[str,Any], base, up100, down200,
                      start_time: dt.time=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_m = (len(res["profile_y_m"])-1)*RS_STEP_M
    km_count = int(res["tot_km"])
    e = res["profile_y_m"]

    cum_min = 0.0
    splits = []
    labels = []

    curr_k_mark = 1
    dist_pass = 0.0

    for i in range(1, len(e)):
        seg = RS_STEP_M
        if dist_pass + seg > total_m:
            seg = total_m - dist_pass
            if seg <= 0: break

        d = e[i] - e[i-1]
        t = base*(seg/1000.0)
        if d > RS_MIN_DELEV: t += (d/100.0)*up100
        elif d < -RS_MIN_DELEV: t += ((-d)/200.0)*down200
        cum_min += t

        while curr_k_mark*1000.0 <= dist_pass + seg + 1e-9 and curr_k_mark <= km_count:
            overshoot = dist_pass + seg - curr_k_mark*1000.0
            frac = 1.0 - overshoot/seg if seg>0 else 1.0
            add_t = t*max(0.0,min(1.0,frac))
            exact_cum = cum_min - t + add_t

            if len(splits)==0:
                split_min = exact_cum
            else:
                split_min = exact_cum - splits[-1]["cum_min"]

            y_at_k = elevation_at_dist_m(e, curr_k_mark*1000.0)
            lab = {"km": float(curr_k_mark), "quota": float(y_at_k)}

            if start_time:
                base_dt = dt.datetime.combine(dt.date.today(), start_time)
                label_time = (base_dt + dt.timedelta(minutes=exact_cum)).strftime("%H:%M")
                display_cum = label_time
            else:
                display_cum = fmt_hm(exact_cum)

            splits.append({"Km": curr_k_mark,
                           "split_min": float(split_min),
                           "cum_min": float(exact_cum),
                           "Parziale": fmt_mmss(split_min),
                           "Cumulativo": display_cum})

            lab["label"] = f"Km {curr_k_mark}\n{display_cum}"
            labels.append(lab)
            curr_k_mark += 1

        dist_pass += seg
        if curr_k_mark > km_count: break

    df = pd.DataFrame(splits)
    labels_df = pd.DataFrame(labels)
    return df, labels_df

# ---------- Gauge SVG (robusto) ----------
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
    def arc(cx,cy,r,start,end):
        def pol(angle_deg):
            rad = math.radians(angle_deg)
            return cx + r*math.cos(rad), cy - r*math.sin(rad)
        large = 1 if (start-end) > 180 else 0
        x1,y1 = pol(start); x2,y2 = pol(end)
        return f"M{x1:.1f},{y1:.1f} A{r:.1f},{r:.1f} 0 {large} 0 {x2:.1f},{y2:.1f}"

    cx, cy = 170, 170
    r_outer, r_inner = 150, 115
    def ring_path(s,e,color):
        p1 = arc(cx,cy,r_outer,s,e)
        p2 = arc(cx,cy,r_inner,e,s)
        return f'<path d="{p1} L{arc(cx,cy,r_inner,e,e)[1:]} {p2} Z" fill="{color}" stroke="{color}" />'

    val_ang = 180.0 - (v/100.0)*180.0

    parts = []
    parts.append(f'<svg viewBox="0 0 340 190" width="100%" height="190" xmlns="http://www.w3.org/2000/svg">')
    for a,b,col,_ in bins:
        a_ang = 180.0 - (a/100.0)*180.0
        b_ang = 180.0 - (b/100.0)*180.0
        parts.append(ring_path(a_ang, b_ang, col))
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r_inner-2}" fill="white"/>')
    px = cx + (r_inner-5)*math.cos(math.radians(val_ang))
    py = cy - (r_inner-5)*math.sin(math.radians(val_ang))
    parts.append(f'<line x1="{cx}" y1="{cy}" x2="{px:.1f}" y2="{py:.1f}" stroke="#333" stroke-width="3"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="#333"/>')
    parts.append(f'<text x="{cx}" y="{cy-20}" text-anchor="middle" font-size="18" font-weight="700" fill="#000">{v:.1f}</text>')
    parts.append('</svg>')
    return "".join(parts)

# ---------- Export GPX con waypoint ----------
def make_gpx_with_waypoints(original_bytes: bytes,
                            km_df: pd.DataFrame,
                            wpt_only: bool=False) -> bytes:
    lat, lon, ele = parse_gpx_bytes(original_bytes)
    if not lat: return b""
    cum=[0.0]
    for i in range(1,len(lat)):
        cum.append(cum[-1]+dist_km(lat[i-1],lon[i-1],lat[i],lon[i])*1000.0)
    total = cum[-1]

    def interp_xy_at_m(target_m):
        if target_m <= 0: return lat[0], lon[0]
        if target_m >= total: return lat[-1], lon[-1]
        j = 0
        while j < len(cum)-1 and cum[j+1] < target_m:
            j += 1
        t = (target_m - cum[j]) / (cum[j+1]-cum[j]) if cum[j+1]>cum[j] else 0.0
        La = lat[j] + t*(lat[j+1]-lat[j])
        Lo = lon[j] + t*(lon[j+1]-lon[j])
        return La, Lo

    root = ET.Element("gpx", version="1.1", creator="streamlit")
    if not wpt_only:
        gtrk = ET.SubElement(root, "trk")
        gtrkseg = ET.SubElement(gtrk, "trkseg")
        for La,Lo,El in zip(lat,lon,ele):
            p = ET.SubElement(gtrkseg, "trkpt", lat=str(La), lon=str(Lo))
            ET.SubElement(p, "ele").text = f"{El:.2f}"

    for _,row in km_df.iterrows():
        km = int(row["Km"])
        name = f"Km {km} — {row['Cumulativo']}"
        mpos = km*1000.0
        La,Lo = interp_xy_at_m(mpos)
        w = ET.SubElement(root, "wpt", lat=str(La), lon=str(Lo))
        ET.SubElement(w, "name").text = name

    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml

# ================= UI Sidebar =================
st.title(APP_TITLE)
st.caption(APP_VER)

with st.sidebar:
    st.header("Impostazioni")
    uploaded = st.file_uploader("Trascina qui il file GPX", type=["gpx"])
    reverse = st.checkbox("Inverti traccia", value=False)

    st.subheader("Tempi per km")
    show_day_time = st.checkbox("Mostra orario del giorno", value=True)
    # (le etichette nel grafico sono state disattivate per robustezza)
    start_time = st.time_input("Orario di partenza", value=dt.time(8,0))

    st.subheader("Parametri di passo (min)")
    base = st.number_input("Min/km (piano)", min_value=1.0, max_value=60.0, value=DEFAULTS["base"], step=0.5)
    up100 = st.number_input("Min/100 m (salita)", min_value=1.0, max_value=60.0, value=DEFAULTS["up"], step=0.5)
    down200 = st.number_input("Min/200 m (discesa)", min_value=1.0, max_value=60.0, value=DEFAULTS["down"], step=0.5)

    st.subheader("Condizioni")
    temp = st.number_input("Temperatura (°C)", -30.0, 50.0, DEFAULTS["temp"], 1.0)
    hum  = st.number_input("Umidità (%)", 0.0, 100.0, DEFAULTS["hum"], 1.0)
    wind = st.number_input("Vento (km/h)", 0.0, 150.0, DEFAULTS["wind"], 1.0)
    precip = st.selectbox("Precipitazioni", PRECIP_OPTIONS, index=0)
    surface = st.selectbox("Fondo", SURF_OPTIONS, index=0)
    expo = st.selectbox("Esposizione", EXPO_OPTIONS, index=1)
    tech = st.selectbox("Tecnica", TECH_OPTIONS, index=1)
    loadkg = st.number_input("Zaino extra (kg)", 0.0, 40.0, DEFAULTS["loadkg"], 1.0)

    st.subheader("Aspetto grafico")
    prof_h = st.slider("Altezza profilo (px)", min_value=220, max_value=600, value=320, step=10)

if not uploaded:
    st.info("Carica un file GPX per iniziare.")
    st.stop()

gpx_bytes = uploaded.read()
lat,lon,ele = parse_gpx_bytes(gpx_bytes)
if len(ele) < 2:
    st.error("Il GPX non contiene elevazioni.")
    st.stop()

res = compute_from_arrays(lat,lon,ele,
                          base_min_per_km=base, up_min_per_100m=up100,
                          down_min_per_200m=down200,
                          weight_kg=DEFAULTS["weight"],
                          reverse=reverse)

fi = compute_if_from_res(
    res, temp_c=temp, humidity_pct=hum,
    precip_it=precip, surface_it=surface, wind_kmh=wind, expo_it=expo,
    technique_level=tech, extra_load_kg=loadkg
)

df_splits, labels_df = compute_km_splits(res, base, up100, down200,
                                         start_time if show_day_time else None)

# ===== summary =====
c1,c2,c3 = st.columns(3)
with c1:
    st.metric("Distanza (km)", f"{res['tot_km']:.2f}")
with c2:
    st.metric("Dislivello + (m)", f"{int(res['dplus'])}")
with c3:
    st.metric("Tempo totale", fmt_hm(res["t_total"]))

# ===== IF gauge =====
st.subheader("Indice di Difficoltà")
st.write(f"**{fi['IF']:.1f}** ({fi['cat']})")
import streamlit.components.v1 as components
components.html(gauge_svg_html(fi["IF"]), height=190)

# ===== profilo (SOLO linea: nessun layer) =====
st.subheader("Profilo altimetrico")
dfp = pd.DataFrame({"km": res["profile_x_km"], "quota": res["profile_y_m"]})

profile_chart = (
    alt.Chart(dfp)
    .mark_line()
    .encode(
        x=alt.X(
            "km:Q",
            axis=alt.Axis(title="Distanza (km)", tickMinStep=1, labelPadding=8, titlePadding=28),
        ),
        y=alt.Y("quota:Q", axis=alt.Axis(title="Quota (m)")),
        tooltip=[alt.Tooltip("km:Q", title="Km", format=".2f"),
                 alt.Tooltip("quota:Q", title="Quota (m)", format=".0f")]
    )
    .properties(height=prof_h)
    .configure_view(strokeWidth=0)
)
st.altair_chart(profile_chart, use_container_width=True)

# ===== risultati =====
st.subheader("Risultati")
colA, colB = st.columns([1,2])

with colA:
    st.write(f"**Dislivello – (m):** {int(res['dneg'])}")
    st.write(f"**Tempo piano:** {fmt_hm(res['t_dist'])}")
    st.write(f"**Tempo salita:** {fmt_hm(res['t_up'])}")
    st.write(f"**Tempo discesa:** {fmt_hm(res['t_down'])}")
    st.write(f"**Calorie stimate:** {res['cal_total']}")
    st.write(f"**Piano (km):** {res['len_flat_km']:.2f} — **Salita (km):** {res['len_up_km']:.2f} — **Discesa (km):** {res['len_down_km']:.2f}")
    st.write(f"**Pend. media salita:** {res['grade_up_pct']:.1f}% — **discesa:** {res['grade_down_pct']:.1f}%")
    st.write(f"**LCS ≥25% (m):** {int(res['lcs25_m'])}")
    st.write(f"**Blocchi ripidi (≥100 m @ ≥25%):** {int(res['blocks25_count'])}")
    st.write(f"**Surge (cambi ritmo)/km:** {res['surge_idx_per_km']:.2f}")
    if res["holes"] == 0:
        st.success("Buchi GPX: 0")
    else:
        st.warning(f"Buchi GPX: {res['holes']} — Attenzione, i calcoli potrebbero essere leggermente alterati.")

with colB:
    st.subheader("Tempi/Orario ai diversi Km")
    right_label = "Orario del giorno" if show_day_time else "Cumulativo"
    df_show = df_splits[["Km","Parziale","Cumulativo"]].rename(columns={"Cumulativo": right_label})
    st.dataframe(df_show, use_container_width=True, height=min(380, 42*(len(df_show)+1)))

# ===== export =====
st.subheader("Export GPX con waypoint (ogni km)")
colx, coly = st.columns(2)
with colx:
    gpx_wpt = make_gpx_with_waypoints(gpx_bytes, df_splits, wpt_only=False)
    st.download_button("Scarica traccia + waypoint", data=gpx_wpt,
                       file_name="traccia_con_km.gpx", mime="application/gpx+xml")
with coly:
    gpx_wpt_only = make_gpx_with_waypoints(gpx_bytes, df_splits, wpt_only=True)
    st.download_button("Scarica SOLO waypoint", data=gpx_wpt_only,
                       file_name="waypoint_km.gpx", mime="application/gpx+xml")
