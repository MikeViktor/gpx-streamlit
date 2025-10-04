# -*- coding: utf-8 -*-
# Streamlit GPX analyser — IF + splits/km + waypoint export (with gpxpy fallback)
import io
import math
import json
import datetime as dt
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ─────────────────────────────────────────────────────────────
# gpxpy (opzionale): se manca useremo il fallback con ElementTree
try:
    import gpxpy
    GPXPY_OK = True
except Exception:
    GPXPY_OK = False

# ─────────────────────────────────────────────────────────────
APP_TITLE = "Analisi Tracce GPX"
APP_VER   = "v5 — IF + tempi/km + esportazione waypoint"

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

# ===== Pesi Indice di Difficoltà =====
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

# ─────────────────────────────────────────────────────────────
# Utility numeriche
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

# ─────────────────────────────────────────────────────────────
# Parser GPX con fallback
def _parse_gpx_bytes_et(gpx_bytes: bytes, reverse=False):
    txt = gpx_bytes.decode("utf-8", errors="ignore")
    root = ET.fromstring(txt)
    def is_tag(e, name):
        t = e.tag
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
                        z = ch.text; break
                if z is None:
                    continue
                try:
                    lat.append(float(la)); lon.append(float(lo)); ele.append(float(z))
                except:
                    pass
        if lat:
            break
    if reverse and lat:
        lat.reverse(); lon.reverse(); ele.reverse()
    return lat, lon, ele

def parse_gpx_bytes(gpx_bytes: bytes, reverse=False):
    if GPXPY_OK:
        gpx = gpxpy.parse(io.StringIO(gpx_bytes.decode("utf-8", errors="ignore")))
        lat, lon, ele = [], [], []
        # track
        for trk in gpx.tracks:
            for seg in trk.segments:
                for p in seg.points:
                    if p.latitude is None or p.longitude is None or p.elevation is None:
                        continue
                    lat.append(float(p.latitude))
                    lon.append(float(p.longitude))
                    ele.append(float(p.elevation))
        # route
        if not lat:
            for rte in gpx.routes:
                for p in rte.points:
                    if p.latitude is None or p.longitude is None or p.elevation is None:
                        continue
                    lat.append(float(p.latitude))
                    lon.append(float(p.longitude))
                    ele.append(float(p.elevation))
        # waypoints
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
    # fallback
    return _parse_gpx_bytes_et(gpx_bytes, reverse=reverse)

# ─────────────────────────────────────────────────────────────
# Fattori meteo/altitudine/tecnica/zaino
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

# ─────────────────────────────────────────────────────────────
# Calcolo principale (replica desktop)
def compute_from_arrays(lat, lon, ele_raw,
                        base_min_per_km=15.0, up_min_per_100m=15.0, down_min_per_200m=15.0,
                        weight_kg=70.0, reverse=False):
    if len(ele_raw) < 2: raise ValueError("Nessun punto utile con elevazione nel GPX.")
    if reverse: lat=list(reversed(lat)); lon=list(reversed(lon)); ele_raw=list(reversed(ele_raw))

    # distanza cumulata (m)
    cum=[0.0]
    for i in range(1,len(lat)):
        cum.append(cum[-1]+dist_km(lat[i-1],lon[i-1],lat[i],lon[i])*1000.0)
    tot_km=cum[-1]/1000.0; total_m=cum[-1]

    # profilo ricampionato + filtri + correzione anello
    e_res=resample_elev(cum,ele_raw,RS_STEP_M)
    e_res, _loop_fix, loop_drift = apply_loop_drift_correction(e_res,lat,lon)
    e_med=median_k(e_res,RS_MED_K); e_sm=moving_avg(e_med,RS_AVG_K)

    # metriche
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
        "loop_like": bool(loop_like),
        "balance_diff_m": round(diff,1),
        "loop_drift_abs_m": round(abs(loop_drift),1),
        "loop_balance_applied": bool(need_balance),
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

# ─────────────────────────────────────────────────────────────
# Distribuzione tempo per segmento + split per km
def per_segment_minutes(x_m, y_m, base_min_per_km, up_min_per_100m, down_min_per_200m):
    """Ritorna liste cumulative: distance_m, time_min; tempo calcolato per ogni segmento RS_STEP_M."""
    time_cum = [0.0]
    dist_cum = [0.0]
    for i in range(1, len(y_m)):
        seg = RS_STEP_M
        d = y_m[i] - y_m[i-1]
        # contributi
        dt = base_min_per_km * (seg/1000.0)
        if d >  RS_MIN_DELEV:  dt += (d/100.0)*up_min_per_100m
        if d < -RS_MIN_DELEV:  dt += ((-d)/200.0)*down_min_per_200m
        time_cum.append(time_cum[-1] + dt)
        dist_cum.append(dist_cum[-1] + seg)
    return dist_cum, time_cum

def km_splits(dist_cum_m, time_cum_min, tot_km, start_time: dt.time|None):
    """Intervalli per ogni km intero (1..floor(tot_km))."""
    rows = []
    def interp(m):
        # tempo/min a distanza m
        if m <= 0: return 0.0
        if m >= dist_cum_m[-1]: return time_cum_min[-1]
        # binario semplice
        lo, hi = 0, len(dist_cum_m)-1
        while lo+1 < hi:
            mid = (lo+hi)//2
            if dist_cum_m[mid] <= m: lo = mid
            else: hi = mid
        m0, m1 = dist_cum_m[lo], dist_cum_m[hi]
        t0, t1 = time_cum_min[lo], time_cum_min[hi]
        u = (m - m0) / max(1e-9, (m1-m0))
        return t0 + u*(t1-t0)
    km_int = int(math.floor(tot_km))
    for k in range(1, km_int+1):
        t_curr = interp(k*1000.0)
        t_prev = interp((k-1)*1000.0)
        split = t_curr - t_prev
        split_str = hm_str(split)
        if start_time is not None:
            base_dt = dt.datetime.combine(dt.date.today(), start_time)
            tm = (base_dt + dt.timedelta(minutes=t_curr)).time()
            day_str = tm.strftime("%H:%M")
        else:
            day_str = ""
        rows.append(dict(Km=k, **{"Tempo parziale":split_str}, **({"Ora del giorno":day_str} if day_str else {})))
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────
# Stringhe temporali
def hm_str(minutes: float):
    h = int(minutes//60)
    m = int(round(minutes - h*60))
    if m == 60:
        h += 1; m = 0
    return f"{h}:{m:02d}"

# ─────────────────────────────────────────────────────────────
# Gauge (Altair) semicircolare
def gauge_chart(value: float):
    # bins in ordine sinistra->destra
    bins = [
        (0,30,"#2ecc71","Facile"),
        (30,50,"#f1c40f","Medio"),
        (50,70,"#e67e22","Impeg."),
        (70,80,"#e74c3c","Diffic."),
        (80,90,"#8e44ad","Molto diff."),
        (90,100,"#111111","Estremo"),
    ]
    # mapping: 0..100 -> angolo 180..0 (radianti)
    def v2rad(v): return math.radians(180.0 - (v/100.0)*180.0)

    segs = []
    for a,b,col,label in bins:
        segs.append(dict(
            start=v2rad(a), end=v2rad(b),
            color=col, label=label
        ))
    base = alt.Chart(pd.DataFrame(segs)).mark_arc(innerRadius=55, outerRadius=90).encode(
        theta=alt.Theta("start:Q", stack=None, title=None),
        theta2="end:Q",
        color=alt.Color("color:N", scale=None, legend=None)
    ).properties(width=260, height=140)

    # puntatore
    val = float(max(0,min(100,value)))
    ang = v2rad(val)
    r0, r1 = 40, 95
    x0, y0 = 0, 0
    x1, y1 = r1*math.cos(ang), r1*math.sin(ang)
    pointer = alt.Chart(pd.DataFrame([dict(x0=x0,y0=y0,x1=x1,y1=y1)])).mark_line(strokeWidth=3,color="#222").encode(
        x="x0:Q", y="y0:Q", x2="x1:Q", y2="y1:Q"
    )
    dot = alt.Chart(pd.DataFrame([dict(x=0,y=0)])).mark_circle(size=80,color="#222").encode(x="x:Q",y="y:Q")

    label = alt.Chart(pd.DataFrame([dict(v=f"{val:.1f}")])).mark_text(y=-10, fontSize=16, fontWeight="bold").encode(text="v:N")

    return (base + pointer + dot + label).configure_view(strokeWidth=0)

# ─────────────────────────────────────────────────────────────
# Export GPX con waypoint (gpxpy se presente, altrimenti ET)
def build_gpx_with_waypoints(original_gpx_bytes: bytes, km_rows: pd.DataFrame, use_split=True):
    title = "Km {km} - {label}"
    if GPXPY_OK:
        gpx = gpxpy.parse(io.StringIO(original_gpx_bytes.decode("utf-8", errors="ignore")))
        for _, r in km_rows.iterrows():
            km = int(r["Km"])
            label = str(r["Tempo parziale"]) if use_split else str(r["Ora del giorno"])
            name = title.format(km=km, label=label)
            w = gpxpy.gpx.GPXWaypoint(latitude=0.0, longitude=0.0, elevation=0.0, name=name)
            gpx.waypoints.append(w)
        return gpx.to_xml()

    # Fallback ET: aggiunge <wpt> alla radice
    txt = original_gpx_bytes.decode("utf-8", errors="ignore")
    root = ET.fromstring(txt)

    def tag(name):
        if "}" in root.tag:
            uri = root.tag.split("}")[0].strip("{")
            return f"{{{uri}}}{name}"
        return name

    for _, r in km_rows.iterrows():
        km = int(r["Km"])
        label = str(r["Tempo parziale"]) if use_split else str(r["Ora del giorno"])
        name = title.format(km=km, label=label)
        w = ET.Element(tag("wpt"), attrib={"lat":"0.0","lon":"0.0"})
        nm = ET.SubElement(w, tag("name")); nm.text = name
        root.append(w)
    return ET.tostring(root, encoding="unicode")

def build_gpx_waypoints_only(km_rows: pd.DataFrame, use_split=True):
    title = "Km {km} - {label}"
    if GPXPY_OK:
        gpx = gpxpy.gpx.GPX()
        for _, r in km_rows.iterrows():
            km = int(r["Km"])
            label = str(r["Tempo parziale"]) if use_split else str(r["Ora del giorno"])
            name = title.format(km=km, label=label)
            w = gpxpy.gpx.GPXWaypoint(latitude=0.0, longitude=0.0, elevation=0.0, name=name)
            gpx.waypoints.append(w)
        return gpx.to_xml()

    # Fallback ET: GPX minimale
    gpx = ET.Element("gpx", attrib={
        "version":"1.1",
        "creator":"streamlit",
        "xmlns":"http://www.topografix.com/GPX/1/1"
    })
    for _, r in km_rows.iterrows():
        km = int(r["Km"])
        label = str(r["Tempo parziale"]) if use_split else str(r["Ora del giorno"])
        name = title.format(km=km, label=label)
        w = ET.SubElement(gpx, "wpt", attrib={"lat":"0.0","lon":"0.0"})
        nm = ET.SubElement(w, "name"); nm.text = name
    return ET.tostring(gpx, encoding="unicode")

# ─────────────────────────────────────────────────────────────
# UI
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(f"{APP_TITLE}")
st.caption(APP_VER)

with st.sidebar:
    st.subheader("Impostazioni")
    inverti = st.checkbox("Inverti traccia", value=False)

    st.markdown("---")
    st.subheader("Tempi per km")
    show_daytime = st.checkbox("Mostra orario del giorno", value=True)
    show_labels  = st.checkbox("Mostra etichette sul grafico", value=True)
    start_time = st.time_input("Orario di partenza", value=dt.time(8,0))

    st.markdown("---")
    st.subheader("Parametri di passo (min)")
    base = st.number_input("Min/km (piano)", min_value=1.0, max_value=60.0, step=0.5, value=DEFAULTS["base"])
    up   = st.number_input("Min/100 m (salita)", min_value=1.0, max_value=60.0, step=0.5, value=DEFAULTS["up"])
    down = st.number_input("Min/200 m (discesa)", min_value=1.0, max_value=60.0, step=0.5, value=DEFAULTS["down"])
    weight = st.number_input("Peso (kg)", min_value=30.0, max_value=150.0, step=1.0, value=DEFAULTS["weight"])

    st.markdown("---")
    st.subheader("Condizioni")
    temp = st.number_input("Temperatura (°C)", -30.0, 50.0, DEFAULTS["temp"])
    hum  = st.number_input("Umidità (%)", 0.0, 100.0, DEFAULTS["hum"])
    wind = st.number_input("Vento (km/h)", 0.0, 150.0, DEFAULTS["wind"])
    precip = st.selectbox("Precipitazioni", PRECIP_OPTIONS, index=PRECIP_OPTIONS.index(DEFAULTS["precip"]))
    surface = st.selectbox("Fondo", SURF_OPTIONS, index=SURF_OPTIONS.index(DEFAULTS["surface"]))
    expo = st.selectbox("Esposizione", EXPO_OPTIONS, index=EXPO_OPTIONS.index(DEFAULTS["expo"]))
    tech = st.selectbox("Tecnica", TECH_OPTIONS, index=TECH_OPTIONS.index(DEFAULTS["tech"]))
    loadkg = st.number_input("Zaino extra (kg)", 0.0, 40.0, DEFAULTS["loadkg"])

# Uploader
uploaded = st.file_uploader("Trascina qui il file GPX", type=["gpx"])

if not uploaded:
    st.info("Carica un file GPX per iniziare.")
    st.stop()

gpx_bytes = uploaded.read()

# Parse GPX
lat, lon, ele = parse_gpx_bytes(gpx_bytes, reverse=inverti)
if len(ele) < 2:
    st.error("Nessun dato di quota valido nel GPX.")
    st.stop()

# Calcolo
res = compute_from_arrays(
    lat, lon, ele,
    base_min_per_km=base, up_min_per_100m=up, down_min_per_200m=down,
    weight_kg=weight, reverse=False # già fatto sopra
)

# IF
fi = compute_if_from_res(
    res,
    temp_c=temp, humidity_pct=hum,
    precip_it=precip, surface_it=surface,
    wind_kmh=wind, expo_it=expo,
    technique_level=tech, extra_load_kg=loadkg
)

# Per-km split: distribuzione temporale per segmento e poi ai km interi
dist_cum_m, time_cum_min = per_segment_minutes(res["profile_x_km"], res["profile_y_m"], base, up, down)
splits_df = km_splits(dist_cum_m, time_cum_min, res["tot_km"], start_time if show_daytime else None)

# ─────────────────────────────────────────────────────────────
# Intestazione valori principali
c1, c2, c3 = st.columns(3)
with c1:
    st.caption("Distanza (km)")
    st.subheader(f"{res['tot_km']:.2f}")
with c2:
    st.caption("Dislivello + (m)")
    st.subheader(f"{int(res['dplus'])}")
with c3:
    st.caption("Tempo totale")
    st.subheader(hm_str(res["t_total"]))

# ─────────────────────────────────────────────────────────────
# Sezione IF con gauge + profilo
cL, cR = st.columns([1.1, 2.2])

with cL:
    st.subheader("Indice di Difficoltà")
    st.write(f"**{fi['IF']:.1f}** ({fi['cat']})")
    st.altair_chart(gauge_chart(fi["IF"]), use_container_width=True)

    st.subheader("Risultati")
    st.markdown(
        f"- **Dislivello − (m):** {int(res['dneg'])}\n"
        f"- **Tempo piano:** {hm_str(res['t_dist'])}\n"
        f"- **Tempo salita:** {hm_str(res['t_up'])}\n"
        f"- **Tempo discesa:** {hm_str(res['t_down'])}\n"
        f"- **Calorie stimate:** {res['cal_total']}\n"
        f"- **Piano (km):** {res['len_flat_km']:.2f} — **Salita (km):** {res['len_up_km']:.2f} — **Discesa (km):** {res['len_down_km']:.2f}\n"
        f"- **Pend. media salita (%):** {res['grade_up_pct']:.1f} — **discesa (%):** {res['grade_down_pct']:.1f}\n"
        f"- **LCS ≥25% (m):** {int(res['lcs25_m'])}\n"
        f"- **Blocchi ripidi (≥100 m @ ≥25%):** {int(res['blocks25_count'])}\n"
        f"- **Surge (cambi ritmo)/km:** {res['surge_idx_per_km']:.2f}\n"
        f"- **Buchi GPX:** {'0' if res['holes']==0 else str(res['holes'])}"
    )

with cR:
    st.subheader("Profilo altimetrico")
    dfp = pd.DataFrame(dict(km=res["profile_x_km"], quota=res["profile_y_m"]))
    chart = alt.Chart(dfp).mark_line().encode(
        x=alt.X("km:Q", axis=alt.Axis(title="Distanza (km)", tickMinStep=1)),
        y=alt.Y("quota:Q", axis=alt.Axis(title="Quota (m)"))
    ).properties(height=280)
    # etichette su km interi
    if show_labels and not splits_df.empty:
        lab = pd.DataFrame({
            "km": splits_df["Km"].astype(float),
            "quota": np.interp(splits_df["Km"].values, dfp["km"].values, dfp["quota"].values),
            "txt": (splits_df["Tempo parziale"] if "Tempo parziale" in splits_df else "")
        })
        chart = chart + alt.Chart(lab).mark_text(dy=-8, fontSize=10).encode(x="km:Q", y="quota:Q", text="txt:N")
    st.altair_chart(chart, use_container_width=True)

# ─────────────────────────────────────────────────────────────
st.subheader("Tempi/Orario ai diversi Km")
# Tabella compatta (altezza fissa)
cols = ["Km","Tempo parziale"]
if show_daytime and "Ora del giorno" in splits_df:
    cols.append("Ora del giorno")
st.dataframe(splits_df[cols], height=320, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# Download GPX con waypoint
st.markdown("### Esporta waypoint")
colA, colB = st.columns(2)
with colA:
    g1 = build_gpx_with_waypoints(gpx_bytes, splits_df, use_split=True)
    st.download_button(
        "Scarica GPX (waypoint con tempo split)",
        data=g1.encode("utf-8"),
        file_name="gpx_con_waypoint_split.gpx",
        mime="application/gpx+xml"
    )
with colB:
    g2 = build_gpx_waypoints_only(splits_df, use_split=not show_daytime)  # scegli cosa scrivere nel nome
    st.download_button(
        "Scarica GPX SOLO waypoint",
        data=g2.encode("utf-8"),
        file_name="gpx_waypoint_solo.gpx",
        mime="application/gpx+xml"
    )
