# streamlit_app.py
# -*- coding: utf-8 -*-

import io
import math
import datetime as dt

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import gpxpy

# ========== CONFIG ==========

st.set_page_config(page_title="Analisi Tracce GPX", layout="wide")

APP_TITLE = "v5 — IF + tempi al km + esportazione waypoint"

# --- Ricampionamento / filtri ---
RS_STEP_M     = 3.0
RS_MIN_DELEV  = 0.25
RS_MED_K      = 3
RS_AVG_K      = 3
ABS_JUMP_RAW  = 100.0

# --- Correzione anello ---
LOOP_TOL_M        = 200.0
DRIFT_MIN_ABS_M   = 2.0
BALANCE_MIN_DIFFM = 10.0
BALANCE_REL_FRAC  = 0.05

# --- Pesi Indice di Difficoltà ---
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
SEVERITY_GAIN = 1.52  # come desktop

DEFAULTS = dict(
    base=15.0, up=15.0, down=15.0,
    weight=70.0, reverse=False,
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

# ========== UTIL ==========

def _fmt_hm(minutes: float) -> str:
    if minutes is None: return "-"
    h = int(minutes // 60)
    m = int(round(minutes - h * 60))
    if m == 60:
        h += 1; m = 0
    return f"{h}:{m:02d}"

def _fmt_mmss(minutes: float) -> str:
    total_sec = int(round(minutes * 60))
    mm = total_sec // 60
    ss = total_sec % 60
    return f"{mm}:{ss:02d}"

def _fmt_time_of_day(start: dt.time, minutes_since_start: float) -> str:
    base = dt.datetime.combine(dt.date.today(), start)
    out  = base + dt.timedelta(minutes=float(minutes_since_start))
    return out.strftime("%H:%M")

def _is_tag(e, name: str) -> bool:
    t = e.tag
    return t.endswith('}' + name) or t == name

# ========== GPX PARSE ==========

def parse_gpx_bytes(gpx_bytes: bytes, reverse=False):
    gpx = gpxpy.parse(io.StringIO(gpx_bytes.decode("utf-8", errors="ignore")))
    lat, lon, ele = [], [], []
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if p.latitude is None or p.longitude is None or p.elevation is None:
                    continue
                lat.append(float(p.latitude))
                lon.append(float(p.longitude))
                ele.append(float(p.elevation))
    # route e waypoints (fallback)
    if not lat:
        for rte in gpx.routes:
            for p in rte.points:
                if p.latitude is None or p.longitude is None or p.elevation is None:
                    continue
                lat.append(float(p.latitude))
                lon.append(float(p.longitude))
                ele.append(float(p.elevation))
    if not lat:
        for w in gpx.waypoints:
            if w.latitude is None or w.longitude is None or w.elevation is None:
                continue
            lat.append(float(w.latitude))
            lon.append(float(w.longitude))
            ele.append(float(w.elevation))
    if reverse and lat:
        lat.reverse(); lon.reverse(); ele.reverse()
    return lat, lon, ele

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

# ========== METEO / FATTORI ==========

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

    precip_map  = {"assenza pioggia":"dry","pioviggine":"drizzle","pioggia":"rain","pioggia forte":"heavy_rain","neve fresca":"snow_shallow","neve profonda":"snow_deep"}
    surface_map = {"asciutto":"dry","fango":"mud","roccia bagnata":"wet_rock","neve dura":"hard_snow","ghiaccio":"ice"}
    exposure_map= {"ombra":"shade","misto":"mixed","pieno sole":"sun"}

    M_precip  = {"dry":1.00,"drizzle":1.05,"rain":1.15,"heavy_rain":1.30,"snow_shallow":1.25,"snow_deep":1.60}[precip_map.get(precip,"dry")]
    M_surface = {"dry":1.00,"mud":1.10,"wet_rock":1.15,"hard_snow":1.30,"ice":1.60}[surface_map.get(surface,"dry")]
    M_sun     = {"shade":1.00,"mixed":1.05,"sun":1.10}[exposure_map.get(exposure,"mixed")]

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
    table = {"facile":0.95,"normale":1.00,"roccioso":1.10,"passaggi di roccia (scrambling)":1.20,"neve/ghiaccio":1.30,"scrambling":1.20}
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

# ========== CORE CALC ==========

def compute_from_arrays(lat, lon, ele_raw,
                        base_min_per_km=15.0, up_min_per_100m=15.0, down_min_per_200m=15.0,
                        weight_kg=70.0):
    if len(ele_raw) < 2:
        raise ValueError("Nessun punto utile con elevazione nel GPX.")

    # distanza cumulata (m)
    cum=[0.0]
    for i in range(1,len(lat)):
        cum.append(cum[-1]+dist_km(lat[i-1],lon[i-1],lat[i],lon[i])*1000.0)
    tot_km=cum[-1]/1000.0; total_m=cum[-1]

    # profilo ricampionato + filtri + correzione anello
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

    # tempi globali
    t_dist  = tot_km*base_min_per_km
    t_up    = (dplus/100.0)*up_min_per_100m
    t_down  = (dneg/200.0)*down_min_per_200m
    t_total = t_dist+t_up+t_down

    holes = 0
    for i in range(1,len(ele_raw)):
        if abs(ele_raw[i]-ele_raw[i-1])>=ABS_JUMP_RAW:
            holes += 1

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

# ========== TEMPI PER KM (non lineari) ==========

def per_km_splits(res, base_min_per_km, up_min_per_100m, down_min_per_200m,
                  start_time: dt.time, show_orario: bool):
    """
    Tempo per km calcolato segmento per segmento.
    """
    x = np.array(res["profile_x_km"]) * 1000.0  # m
    y = np.array(res["profile_y_m"])

    splits = []
    if len(x) < 2:
        return pd.DataFrame(columns=["Km","Tempo parziale","Ora del giorno"]), {}

    cum_t_min = 0.0
    next_km_m = 1000.0
    prev_x = x[0]; prev_y = y[0]

    labels_map = {}  # km -> testo (split o orario) per etichette su grafico

    for i in range(1, len(x)):
        seg_m = float(x[i] - prev_x)
        if seg_m <= 0:
            prev_x = x[i]; prev_y = y[i]; continue
        d_ele = float(y[i] - prev_y)

        # tempo segmento
        t_seg = 0.0
        t_seg += (seg_m/1000.0) * base_min_per_km
        if d_ele > 0:
            t_seg += (d_ele/100.0) * up_min_per_100m
        elif d_ele < 0:
            t_seg += ((-d_ele)/200.0) * down_min_per_200m

        while next_km_m <= x[i]:
            part = (next_km_m - prev_x) / seg_m
            d_ele_part = (prev_y + part*(y[i]-prev_y)) - prev_y
            t_part = 0.0
            t_part += ((next_km_m - prev_x)/1000.0) * base_min_per_km
            if d_ele_part > 0:
                t_part += (d_ele_part/100.0) * up_min_per_100m
            elif d_ele_part < 0:
                t_part += ((-d_ele_part)/200.0) * down_min_per_200m

            cum_t_min += t_part
            split_min = t_part
            km_idx = int(round(next_km_m/1000.0))
            if km_idx >= 1:
                if show_orario:
                    labels_map[km_idx] = _fmt_time_of_day(start_time, cum_t_min)
                else:
                    labels_map[km_idx] = _fmt_mmss(cum_t_min)
                splits.append([km_idx, _fmt_mmss(split_min), _fmt_time_of_day(start_time, cum_t_min)])
            prev_x = next_km_m
            prev_y = prev_y + part*(y[i]-prev_y)
            seg_m = float(x[i] - prev_x)
            d_ele = float(y[i] - prev_y)
            next_km_m += 1000.0

        if seg_m > 0:
            t_seg_rem = 0.0
            t_seg_rem += (seg_m/1000.0) * base_min_per_km
            if d_ele > 0:
                t_seg_rem += (d_ele/100.0) * up_min_per_100m
            elif d_ele < 0:
                t_seg_rem += ((-d_ele)/200.0) * down_min_per_200m
            cum_t_min += t_seg_rem

        prev_x = x[i]; prev_y = y[i]

    df = pd.DataFrame(splits, columns=["Km","Tempo parziale","Ora del giorno"])
    return df, labels_map

# ========== GAUGE (Altair) ==========

GAUGE_BINS = [
    (0, 30, "#2ecc71", "Facile"),
    (30, 50, "#f1c40f", "Medio"),
    (50, 70, "#e67e22", "Impeg."),
    (70, 80, "#e74c3c", "Diffic."),
    (80, 90, "#8e44ad", "Molto diff."),
    (90, 100, "#111111", "Estremo"),
]

def _val2angle_deg(v: float) -> float:
    v = max(0.0, min(100.0, float(v)))
    return 180.0 - (v/100.0)*180.0

def draw_gauge_altair(value: float, width=430, height=200):
    segs = []
    for a,b,col,label in GAUGE_BINS:
        segs.append({
            "start": math.radians(_val2angle_deg(a)),
            "end":   math.radians(_val2angle_deg(b)),
            "color": col, "label": label
        })
    df = pd.DataFrame(segs)

    arcs = alt.Chart(df).mark_arc(innerRadius=70, outerRadius=100).encode(
        startAngle="start:Q", endAngle="end:Q",
        color=alt.Color("color:N", scale=None, legend=None)
    )

    ang = math.radians(_val2angle_deg(value))
    needle = pd.DataFrame([{
        "x1": 0, "y1": 0,
        "x2": 60*math.cos(ang), "y2": 60*math.sin(ang)
    }])
    needle_layer = alt.Chart(needle).mark_line(stroke="#222", strokeWidth=3).encode(
        x="x1:Q", y="y1:Q", x2="x2:Q", y2="y2:Q"
    )
    center = alt.Chart(pd.DataFrame({"x":[0],"y":[0]})).mark_point(filled=True, size=60, color="#222").encode(x="x:Q", y="y:Q")

    base = alt.Chart().properties(width=width, height=height).configure_view(stroke=None)
    scale = alt.Scale(domain=[-110, 110])

    return alt.layer(
        arcs.encode(x=alt.value(0), y=alt.value(0)).properties(width=width, height=height),
        needle_layer.encode(x=alt.X("x1", scale=scale), y=alt.Y("y1", scale=scale),
                            x2=alt.X2("x2"), y2=alt.Y2("y2")),
        center.encode(x=alt.X("x:Q", scale=scale), y=alt.Y("y:Q", scale=scale))
    ).configure_view(stroke=None)

# ========== PROFILO (Altair) ==========

def plot_profile_altair(x_km, y_m, labels_map=None):
    dfp = pd.DataFrame({"Distanza (km)": x_km, "Quota (m)": y_m})
    base = alt.Chart(dfp).mark_line().encode(
        x=alt.X("Distanza (km):Q", title="Distanza (km)", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Quota (m):Q", title="Quota (m)")
    )

    layers = [base]

    if labels_map:
        pts = []
        for km, label in labels_map.items():
            # trova y approssimando al punto più vicino
            idx = (np.abs(np.array(x_km) - km)).argmin()
            pts.append({"Km": km, "Quota (m)": y_m[idx], "Etichetta": label})
        dfl = pd.DataFrame(pts)
        text = alt.Chart(dfl).mark_text(align="left", dx=6, dy=-6, fontSize=11).encode(
            x="Km:Q", y="Quota (m):Q", text="Etichetta:N"
        )
        point = alt.Chart(dfl).mark_point(color="#333").encode(x="Km:Q", y="Quota (m):Q")
        layers += [point, text]

    return alt.layer(*layers).properties(height=330).interactive()

# ========== EXPORT GPX ==========

def build_gpx_with_waypoints(original_gpx_bytes: bytes, km_rows: pd.DataFrame, use_split=True):
    # parse original, append waypoints
    gpx = gpxpy.parse(io.StringIO(original_gpx_bytes.decode("utf-8", errors="ignore")))
    for _, r in km_rows.iterrows():
        km = int(r["Km"])
        label = str(r["Tempo parziale"]) if use_split else str(r["Ora del giorno"])
        name = f"Km {km} - {label}"
        w = gpxpy.gpx.GPXWaypoint(latitude=None, longitude=None, elevation=None, name=name)
        # many devices require coords; we'll append without coords to keep simple
        gpx.waypoints.append(w)
    return gpx.to_xml()

def build_gpx_waypoints_only(km_rows: pd.DataFrame, use_split=True):
    gpx = gpxpy.gpx.GPX()
    for _, r in km_rows.iterrows():
        km = int(r["Km"])
        label = str(r["Tempo parziale"]) if use_split else str(r["Ora del giorno"])
        name = f"Km {km} - {label}"
        w = gpxpy.gpx.GPXWaypoint(latitude=0.0, longitude=0.0, elevation=0.0, name=name)
        gpx.waypoints.append(w)
    return gpx.to_xml()

# ========== UI ==========

st.title(APP_TITLE)

with st.sidebar:
    st.header("Impostazioni")
    file = st.file_uploader("Trascina qui il file GPX", type=["gpx"])
    reverse = st.checkbox("Inverti traccia", value=False)

    st.subheader("Tempi per km")
    show_orario = st.checkbox("Mostra orario del giorno", value=True)
    show_labels = st.checkbox("Mostra etichette sul grafico", value=True)
    start_time = st.time_input("Orario di partenza", value=dt.time(8,0))

    st.subheader("Parametri di passo (min)")
    base = st.number_input("Min/km (piano)", min_value=1.0, max_value=60.0, value=DEFAULTS["base"], step=0.5)
    up   = st.number_input("Min/100 m (salita)", min_value=1.0, max_value=60.0, value=DEFAULTS["up"], step=0.5)
    down = st.number_input("Min/200 m (discesa)", min_value=1.0, max_value=60.0, value=DEFAULTS["down"], step=0.5)

    st.subheader("Condizioni")
    temp = st.number_input("Temperatura (°C)", -30.0, 50.0, DEFAULTS["temp"], 1.0)
    hum  = st.number_input("Umidità (%)", 0.0, 100.0, DEFAULTS["hum"], 1.0)
    wind = st.number_input("Vento (km/h)", 0.0, 150.0, DEFAULTS["wind"], 1.0)
    precip  = st.selectbox("Precipitazioni", PRECIP_OPTIONS, index=0)
    surface = st.selectbox("Fondo", SURF_OPTIONS, index=0)
    expo    = st.selectbox("Esposizione", EXPO_OPTIONS, index=1)
    tech    = st.selectbox("Tecnica", TECH_OPTIONS, index=1)
    loadkg  = st.number_input("Zaino extra (kg)", 0.0, 40.0, DEFAULTS["loadkg"], 1.0)

if not file:
    st.info("Carica un file GPX per iniziare.")
    st.stop()

raw_bytes = file.read()
lat, lon, ele = parse_gpx_bytes(raw_bytes, reverse=reverse)
if len(ele) < 2:
    st.error("Il GPX non contiene abbastanza punti con quota.")
    st.stop()

res = compute_from_arrays(lat, lon, ele, base, up, down, weight_kg=DEFAULTS["weight"])

# ---- Metriche in alto
col1, col2, col3 = st.columns(3)
col1.metric("Distanza (km)", f"{res['tot_km']:.2f}")
col2.metric("Dislivello + (m)", f"{int(res['dplus'])}")
col3.metric("Tempo totale", _fmt_hm(res["t_total"]))

# ---- IF + gauge
st.subheader("Indice di Difficoltà")
if_res = compute_if_from_res(res, temp, hum, precip, surface, wind, expo, tech, loadkg)
st.markdown(f"### {if_res['IF']:.1f}  _({if_res['cat']})_")
st.altair_chart(draw_gauge_altair(if_res['IF']), use_container_width=False)

# ---- Profilo + risultati
left, right = st.columns([1,2])

with left:
    st.subheader("Risultati")
    st.markdown(
        f"""
- **Dislivello − (m):** {int(res['dneg'])}  
- **Tempo piano:** {_fmt_hm(res['t_dist'])}  
- **Tempo salita:** {_fmt_hm(res['t_up'])}  
- **Tempo discesa:** {_fmt_hm(res['t_down'])}  
- **Calorie stimate:** {res['cal_total']}  
- **Piano (km):** {res['len_flat_km']:.2f} — **Salita (km):** {res['len_up_km']:.2f} — **Discesa (km):** {res['len_down_km']:.2f}  
- **Pend. media salita (%):** {res['grade_up_pct']:.1f} — **discesa:** {res['grade_down_pct']:.1f}  
- **LCS ≥25% (m):** {int(res['lcs25_m'])}  
- **Blocchi ripidi (≥100 m @ ≥25%):** {int(res['blocks25_count'])}  
- **Surge (cambi ritmo)/km:** {res['surge_idx_per_km']:.2f}  
- **Buchi GPX:** {res['holes']}
        """
    )

with right:
    st.subheader("Profilo altimetrico")
    km_table, labels_map = per_km_splits(res, base, up, down, start_time, show_orario)
    labels_for_chart = labels_map if show_labels else None
    st.altair_chart(plot_profile_altair(res["profile_x_km"], res["profile_y_m"], labels_for_chart),
                    use_container_width=True)

# ---- Tempi/Orario ai diversi Km
st.subheader("Tempi/Orario ai diversi Km")
if km_table.empty:
    st.info("Traccia più corta di 1 km, nessuno split disponibile.")
else:
    # Tabella compatta: solo tre colonne, nessun indice
    st.dataframe(km_table, use_container_width=True, hide_index=True)

    # Export GPX
    st.write("**Esporta waypoint**")
    c1, c2 = st.columns(2)
    with c1:
        xml_full = build_gpx_with_waypoints(raw_bytes, km_table, use_split=True)
        st.download_button("GPX con traccia + waypoint (Km - tempo parziale)",
                           data=xml_full.encode("utf-8"),
                           file_name=f"{file.name.rsplit('.',1)[0]}_waypoints_split.gpx",
                           mime="application/gpx+xml")
    with c2:
        xml_wpt = build_gpx_waypoints_only(km_table, use_split=True)
        st.download_button("Solo waypoint (Km - tempo parziale)",
                           data=xml_wpt.encode("utf-8"),
                           file_name=f"{file.name.rsplit('.',1)[0]}_waypoints_only_split.gpx",
                           mime="application/gpx+xml")
