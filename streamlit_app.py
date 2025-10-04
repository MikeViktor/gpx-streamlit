# -*- coding: utf-8 -*-
import math, io, xml.etree.ElementTree as ET
from datetime import timedelta
# --- guardia facoltativa per librerie mancanti ---
import streamlit as st

def _ensure_deps():
    try:
        import numpy as _np          # noqa: F401
        import matplotlib.pyplot as _plt   # noqa: F401
    except ModuleNotFoundError as e:
        pkg = str(e).split("'")[1]
        st.error(
            f"Manca la libreria **{pkg}**. "
            f"Installa i pacchetti richiesti (es. `pip install {pkg}`) "
            f"o aggiungili al `requirements.txt` del repo."
        )
        st.stop()

_ensure_deps()
# --- fine guardia facoltativa ---

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ===== Meta/app =====
APP_TITLE = "Tempo percorrenza sentiero"
APP_VER   = "v3.5.0-web (allineata a gpx_gui v3.5.0)"

# ===== Ricampionamento / filtri (identici a gpx_gui) =====
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
ALPHA_METEO   = 0.6
SEVERITY_GAIN = 1.52

# ===== Default =====
DEFAULTS = {
    "base": 15.0, "up": 15.0, "down": 15.0,
    "weight": 70.0, "reverse": False,
    "temp": 15.0, "hum": 50.0, "wind": 5.0,
    "precip": "assenza pioggia",
    "surface": "asciutto",
    "expo": "misto",
    "tech": "normale",
    "loadkg": 6.0,
    # split/orario
    "show_splits": True,
    "use_start": False,
    "start_h": 8,
    "start_m": 0,
}

PRECIP_OPTIONS = ["assenza pioggia","pioviggine","pioggia","pioggia forte","neve fresca","neve profonda"]
SURF_OPTIONS   = ["asciutto","fango","roccia bagnata","neve dura","ghiaccio"]
EXPO_OPTIONS   = ["ombra","misto","pieno sole"]
TECH_OPTIONS   = ["facile","normale","roccioso","passaggi di roccia (scrambling)","neve/ghiaccio"]

# ---------- Utils identici a desktop ----------
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
    if reverse:
        lat=list(reversed(lat)); lon=list(reversed(lon)); ele_raw=list(reversed(ele_raw))

    # distanza cumulata (m)
    cum=[0.0]
    for i in range(1,len(lat)):
        cum.append(cum[-1]+dist_km(lat[i-1],lon[i-1],lat[i],lon[i])*1000.0)
    tot_km=cum[-1]/1000.0; total_m=cum[-1]

    # profilo ricampionato + filtri + correzione anello
    e_res=resample_elev(cum,ele_raw,RS_STEP_M)
    e_res, loop_fix, loop_drift = apply_loop_drift_correction(e_res,lat,lon)
    e_med=median_k(e_res,RS_MED_K); e_sm=moving_avg(e_med,RS_AVG_K)

    # metriche + tempo cumulativo
    dplus=dneg=0.0; asc_len=desc_len=flat_len=0.0; asc_gain=desc_drop=0.0
    asc_bins=[0,0,0,0,0]; desc_bins=[0,0,0,0,0]
    longest_steep_run=0.0; current_run=0.0; blocks25=0; last_state=0; surge_trans=0

    cum_time_min=[0.0]; cum_dist_m=[0.0]

    for i in range(1,len(e_sm)):
        t_prev=(i-1)*RS_STEP_M
        t_curr=min(i*RS_STEP_M,total_m)
        seg=max(0.0,t_curr-t_prev)
        if seg<=0:
            cum_time_min.append(cum_time_min[-1]); cum_dist_m.append(cum_dist_m[-1])
            continue

        d = e_sm[i]-e_sm[i-1]

        ds_km = seg/1000.0
        up_m  = d if d>RS_MIN_DELEV else 0.0
        down_m= (-d) if d < -RS_MIN_DELEV else 0.0
        dt = ds_km*base_min_per_km + (up_m/100.0)*up_min_per_100m + (down_m/200.0)*down_min_per_200m

        cum_time_min.append(cum_time_min[-1] + dt)
        cum_dist_m.append(min(t_curr, total_m))

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

    # tempi totali
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

    # SPLIT per KM
    km_splits=[]
    if tot_km>=1.0:
        target=1000.0; j=1; last=len(cum_dist_m)-1
        while target<=total_m and j<=last:
            while j<=last and cum_dist_m[j] < target:
                j+=1
            if j>last: break
            d1,d2 = cum_dist_m[j-1], cum_dist_m[j]
            t1,t2 = cum_time_min[j-1], cum_time_min[j]
            if d2==d1:
                t_at=t2; y_at=e_sm[j]
            else:
                u=(target-d1)/(d2-d1)
                t_at=t1+u*(t2-t1)
                y_at=e_sm[j-1]+u*(e_sm[j]-e_sm[j-1])
            km_idx=int(round(target/1000.0))
            km_splits.append((km_idx, target/1000.0, t_at, y_at))
            target+=1000.0

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
        "cum_dist_m": cum_dist_m, "cum_time_min": cum_time_min,
        "km_splits": km_splits,
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

# ---------- Gauge (matplotlib) ----------
def draw_if_gauge(IF_value: float, width=5.2, height=2.2, dpi=150):
    bins = [
        (0,30,"#2ecc71","Facile"),
        (30,50,"#f1c40f","Medio"),
        (50,70,"#e67e22","Impeg."),
        (70,80,"#e74c3c","Diffic."),
        (80,90,"#8e44ad","Molto diff."),
        (90,100,"#111111","Estremo"),
    ]
    def v2ang(v): return math.radians(180.0 - (max(0,min(100,v))/100.0)*180.0)

    fig, ax = plt.subplots(figsize=(width,height), dpi=dpi)
    ax.axis("equal"); ax.axis("off")
    R = 1.0; r = 0.72
    for a,b,col,label in bins:
        th1=v2ang(a); th2=v2ang(b)
        tt = plt.Polygon([[0,0]], visible=False)  # placeholder

        # draw wedge by many points
        n=80
        thetas=[th1 + t*(th2-th1)/n for t in range(n+1)]
        outer=[[R*math.cos(t), R*math.sin(t)] for t in thetas]
        inner=[[r*math.cos(t), r*math.sin(t)] for t in reversed(thetas)]
        poly = plt.Polygon(outer+inner, color=col, ec=col)
        ax.add_patch(poly)

    # ticks 0/50/100
    for v in (0,50,100):
        th=v2ang(v)
        x1,y1 = 0.68*math.cos(th), 0.68*math.sin(th)
        x2,y2 = 1.08*math.cos(th), 1.08*math.sin(th)
        ax.plot([x1,x2],[y1,y2], lw=1, color="#888")
        tx,ty = 1.20*math.cos(th), 1.20*math.sin(th)
        ax.text(tx,ty, str(v), ha="center", va="center", fontsize=8, color="#555")

    # labels
    for a,b,col,label in bins:
        m=(a+b)/2.0; th=v2ang(m)
        tx,ty = 0.88*math.cos(th), 0.88*math.sin(th)
        ax.text(tx,ty, label, ha="center", va="center", fontsize=9, fontweight="bold", color="#111")

    # hub
    circ = plt.Circle((0,0), 0.62, color="white")
    ax.add_patch(circ)

    # needle
    v = 0.0 if IF_value is None else float(IF_value)
    th=v2ang(v)
    xt,yt = 0.74*math.cos(th), 0.74*math.sin(th)
    ax.plot([0,xt],[0,yt], lw=3, color="#333")
    ax.add_patch(plt.Circle((0,0), 0.04, color="#333"))
    ax.text(0, -0.32, f"{v:.1f}", ha="center", va="center", fontsize=14, fontweight="bold", color="#000")

    ax.set_xlim(-1.35,1.35); ax.set_ylim(-0.25,1.25)
    return fig

# ---------- Export GPX ----------
def build_waypoints_from_splits(lat, lon, ele, km_splits, use_start, start_h, start_m):
    if not km_splits or len(lat)<2: return []
    # distanza cumulata sugli originali
    cum=[0.0]
    for i in range(1,len(lat)):
        cum.append(cum[-1]+dist_km(lat[i-1],lon[i-1],lat[i],lon[i])*1000.0)

    start_min = int(start_h)*60 + int(start_m)

    out=[]
    for (kidx, km_x, tmin, _y) in km_splits:
        target = km_x*1000.0
        j=1
        while j<len(cum) and cum[j] < target:
            j+=1
        if j>=len(cum): break
        d1,d2 = cum[j-1], cum[j]
        u = 0.0 if d2==d1 else (target-d1)/(d2-d1)
        la = lat[j-1] + u*(lat[j]-lat[j-1])
        lo = lon[j-1] + u*(lon[j]-lon[j-1])
        el = ele[j-1] + u*(ele[j]-ele[j-1])
        # label tempo
        if use_start:
            tot = start_min + int(round(tmin))
            tot %= (24*60)
            hh,mm = tot//60, tot%60
            label = f"{hh:02d}:{mm:02d}"
        else:
            h = int(tmin//60); m = int(round(tmin - h*60))
            if m==60: h+=1; m=0
            label = f"{h}:{m:02d}"
        name = f"Km {int(round(km_x))} - {label}"
        out.append((la,lo,el,name))
    return out

def gpx_bytes_with_track_and_wpt(lat, lon, ele, waypoints):
    ns = "http://www.topografix.com/GPX/1/1"
    gpx = ET.Element("gpx", version="1.1", creator=f"{APP_TITLE} {APP_VER}", xmlns=ns)
    for (la,lo,el,name) in waypoints:
        wpt = ET.SubElement(gpx, "wpt", lat=f"{la:.7f}", lon=f"{lo:.7f}")
        ET.SubElement(wpt, "ele").text = f"{el:.2f}"
        ET.SubElement(wpt, "name").text = name
    trk = ET.SubElement(gpx, "trk")
    ET.SubElement(trk, "name").text = "Traccia originale"
    seg = ET.SubElement(trk, "trkseg")
    for la,lo,el in zip(lat,lon,ele):
        pt = ET.SubElement(seg, "trkpt", lat=f"{la:.7f}", lon=f"{lo:.7f}")
        ET.SubElement(pt, "ele").text = f"{el:.2f}"
    tree = ET.ElementTree(gpx)
    bio = io.BytesIO()
    ET.indent(tree, space="  ", level=0)
    tree.write(bio, encoding="utf-8", xml_declaration=True)
    return bio.getvalue()

def gpx_bytes_waypoints_only(waypoints):
    ns = "http://www.topografix.com/GPX/1/1"
    gpx = ET.Element("gpx", version="1.1", creator=f"{APP_TITLE} {APP_VER}", xmlns=ns)
    for (la,lo,el,name) in waypoints:
        wpt = ET.SubElement(gpx, "wpt", lat=f"{la:.7f}", lon=f"{lo:.7f}")
        ET.SubElement(wpt, "ele").text = f"{el:.2f}"
        ET.SubElement(wpt, "name").text = name
    tree = ET.ElementTree(gpx)
    bio = io.BytesIO()
    ET.indent(tree, space="  ", level=0)
    tree.write(bio, encoding="utf-8", xml_declaration=True)
    return bio.getvalue()

# ---------- UI ----------
st.set_page_config(APP_TITLE, layout="wide")
st.title(f"{APP_TITLE} · {APP_VER}")

# File upload
uploaded = st.file_uploader("Carica GPX", type=["gpx"])

# Reset parametri ai default se carichi un nuovo file
if "last_filename" not in st.session_state:
    st.session_state["last_filename"] = None
if uploaded and uploaded.name != st.session_state["last_filename"]:
    # reset a default
    for k,v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state["last_filename"] = uploaded.name

colL, colR = st.columns([0.9,1.1])

with colL:
    st.subheader("Impostazioni")
    c1,c2,c3 = st.columns(3)
    base  = c1.number_input("Min/km (piano)", min_value=1.0, max_value=60.0, step=0.5, value=st.session_state.get("base", DEFAULTS["base"]), key="base")
    up    = c2.number_input("Min/100 m (salita)", min_value=1.0, max_value=60.0, step=0.5, value=st.session_state.get("up", DEFAULTS["up"]), key="up")
    down  = c3.number_input("Min/200 m (discesa)", min_value=1.0, max_value=60.0, step=0.5, value=st.session_state.get("down", DEFAULTS["down"]), key="down")

    c4,c5 = st.columns(2)
    weight = c4.number_input("Peso (kg)", min_value=30.0, max_value=150.0, step=1.0, value=st.session_state.get("weight", DEFAULTS["weight"]), key="weight")
    reverse = c5.checkbox("Inverti traccia", value=st.session_state.get("reverse", DEFAULTS["reverse"]), key="reverse")

    st.divider()
    st.markdown("**Condizioni**")
    cA,cB,cC = st.columns(3)
    temp = cA.number_input("Temperatura (°C)", min_value=-30.0, max_value=50.0, step=1.0, value=st.session_state.get("temp", DEFAULTS["temp"]), key="temp")
    hum  = cB.number_input("Umidità (%)", min_value=0.0, max_value=100.0, step=1.0, value=st.session_state.get("hum", DEFAULTS["hum"]), key="hum")
    wind = cC.number_input("Vento (km/h)", min_value=0.0, max_value=150.0, step=1.0, value=st.session_state.get("wind", DEFAULTS["wind"]), key="wind")

    cD,cE = st.columns(2)
    precip = cD.selectbox("Precipitazioni", options=PRECIP_OPTIONS, index=PRECIP_OPTIONS.index(st.session_state.get("precip", DEFAULTS["precip"])), key="precip")
    surface = cE.selectbox("Fondo", options=SURF_OPTIONS, index=SURF_OPTIONS.index(st.session_state.get("surface", DEFAULTS["surface"])), key="surface")

    cF,cG = st.columns(2)
    expo = cF.selectbox("Esposizione", options=EXPO_OPTIONS, index=EXPO_OPTIONS.index(st.session_state.get("expo", DEFAULTS["expo"])), key="expo")
    tech = cG.selectbox("Tecnica", options=TECH_OPTIONS, index=TECH_OPTIONS.index(st.session_state.get("tech", DEFAULTS["tech"])), key="tech")

    cH,cI = st.columns(2)
    loadkg = cH.number_input("Zaino extra (kg)", min_value=0.0, max_value=40.0, step=1.0, value=st.session_state.get("loadkg", DEFAULTS["loadkg"]), key="loadkg")

    st.divider()
    show_splits = st.checkbox("Mostra tempi per km sul grafico", value=st.session_state.get("show_splits", DEFAULTS["show_splits"]), key="show_splits")
    use_start   = st.checkbox("Usa orario di partenza", value=st.session_state.get("use_start", DEFAULTS["use_start"]), key="use_start")
    cJ,cK = st.columns(2)
    start_h = cJ.number_input("Partenza ora (0–23)", min_value=0, max_value=23, value=st.session_state.get("start_h", DEFAULTS["start_h"]), key="start_h")
    start_m = cK.number_input("Partenza minuti (0–59)", min_value=0, max_value=59, value=st.session_state.get("start_m", DEFAULTS["start_m"]), key="start_m")

    cZ1, cZ2 = st.columns(2)
    do_calc = cZ1.button("Calcola")
    if cZ2.button("Reset parametri"):
        for k,v in DEFAULTS.items():
            st.session_state[k] = v
        st.experimental_rerun()

# Computo: eseguiamo se c’è file (il bottone Calcola è “decorativo” perché Streamlit ricalcola comunque)
if uploaded is not None:
    try:
        lat,lon,ele = parse_gpx_bytes(uploaded.read())
        if len(ele) < 2:
            st.error("Mancano dati elevazione nel GPX."); st.stop()
        res = compute_from_arrays(
            lat,lon,ele,
            base_min_per_km=base, up_min_per_100m=up, down_min_per_200m=down,
            weight_kg=weight, reverse=reverse
        )
    except Exception as e:
        st.error(f"Errore: {e}")
        st.stop()

    fi = compute_if_from_res(
        res,
        temp_c=temp, humidity_pct=hum,
        precip_it=precip, surface_it=surface,
        wind_kmh=wind, expo_it=expo,
        technique_level=tech, extra_load_kg=loadkg
    )

    with colR:
        # top 3
        c1,c2,c3 = st.columns(3)
        c1.metric("Distanza (km)", f"{res['tot_km']:.2f}")
        c2.metric("Dislivello + (m)", f"{int(res['dplus'])}")
        c3.markdown("**Tempo totale**")
        c3.markdown(f"<div style='font-size:1.6rem;font-weight:800'>{int(res['t_total']//60)}:{int(round(res['t_total']-60*int(res['t_total']//60))):02d}</div>", unsafe_allow_html=True)

        c4,c5,c6 = st.columns(3)
        c4.metric("Dislivello − (m)", f"{int(res['dneg'])}")
        c5.metric("Tempo salita", f"{int(res['t_up']//60)}:{int(round(res['t_up']-60*int(res['t_up']//60))):02d}")
        c6.metric("Tempo discesa", f"{int(res['t_down']//60)}:{int(round(res['t_down']-60*int(res['t_down']//60))):02d}")

        c7,c8,c9 = st.columns(3)
        c7.metric("Tempo piano", f"{int(res['t_dist']//60)}:{int(round(res['t_dist']-60*int(res['t_dist']//60))):02d}")
        c8.metric("Calorie stimate", f"{res['cal_total']}")
        holes = int(res["holes"])
        if holes>0:
            c9.markdown(f"<div style='padding:6px 10px;border-radius:6px;background:#ffe6e6;color:#b00000;font-weight:700;'>⚠️ Buchi GPX: {holes}</div>", unsafe_allow_html=True)
        else:
            c9.markdown(f"<div style='padding:6px 10px;border-radius:6px;background:#e8f7e8;color:#0b6b0b;font-weight:700;'>Buchi GPX: {holes}</div>", unsafe_allow_html=True)

        c10,c11,c12 = st.columns(3)
        c10.metric("Piano (km)", f"{res['len_flat_km']:.2f}")
        c11.metric("Salita (km)", f"{res['len_up_km']:.2f}")
        c12.metric("Discesa (km)", f"{res['len_down_km']:.2f}")

        c13,c14 = st.columns(2)
        c13.metric("Pend. media salita (%)", f"{res['grade_up_pct']:.1f}")
        c14.metric("Pend. media discesa (%)", f"{res['grade_down_pct']:.1f}")

        c15,c16,c17 = st.columns(3)
        c15.metric("LCS ≥25% (m)", f"{int(res['lcs25_m'])}")
        c16.metric("Blocchi ripidi (≥100 m @ ≥25%)", f"{int(res['blocks25_count'])}")
        c17.metric("Surge (cambi ritmo)/km", f"{res['surge_idx_per_km']:.2f}")

        # Gauge
        st.subheader(f"Indice di Difficoltà: **{fi['IF']}**  ({fi['cat']})")
        fig_g = draw_if_gauge(fi["IF"])
        st.pyplot(fig_g, use_container_width=False)

        # Profilo altimetrico
        st.subheader("Profilo altimetrico")
        fig, ax = plt.subplots(figsize=(9,4), dpi=120)
        ax.plot(res["profile_x_km"], res["profile_y_m"], lw=2.8, color="#1f77b4")
        ax.set_xlabel("Distanza (km)"); ax.set_ylabel("Quota (m)")
        ax.grid(True, which="both", linestyle=":")
        # ticks km
        max_km = max(res["profile_x_km"]) if res["profile_x_km"] else 0.0
        ax.xaxis.set_major_locator(MultipleLocator(1.0))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        ax.set_xlim(0, math.ceil(max_km + 1e-9))

        # etichette split
        if show_splits and res["km_splits"]:
            start_min = int(start_h)*60 + int(start_m)
            for (kidx, xkm, tmin, y_here) in res["km_splits"]:
                if use_start:
                    tot = (start_min + int(round(tmin))) % (24*60)
                    hh,mm = tot//60, tot%60
                    label = f"{hh:02d}:{mm:02d}"
                else:
                    h = int(tmin//60); m = int(round(tmin - h*60))
                    if m==60: h+=1; m=0
                    label = f"{h}:{m:02d}"
                ax.annotate(label, xy=(xkm, y_here), xytext=(0, 10),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999", alpha=0.9))
        st.pyplot(fig, use_container_width=True)

        # Note su correzione anello
        msg=[]
        if res.get("loop_fix_applied"): msg.append(f"corretta deriva ~{res['loop_drift_abs_m']} m")
        if res.get("loop_balance_applied"): msg.append(f"chiusura anello: D+ allineato a D− (diff {res['balance_diff_m']} m)")
        if msg:
            st.info(" · ".join(msg))

        # Export GPX (traccia + wpt) e solo waypoint
        st.subheader("Esporta")
        wpts = build_waypoints_from_splits(lat,lon,ele,res["km_splits"], use_start, start_h, start_m)
        if wpts:
            fname = (uploaded.name.rsplit(".",1)[0] if uploaded.name else "percorso")
            colA, colB = st.columns(2)
            with colA:
                b1 = gpx_bytes_with_track_and_wpt(lat,lon,ele,wpts)
                st.download_button(
                    "Scarica GPX (traccia + waypoint/km)",
                    data=b1, file_name=f"{fname}_kmwpt.gpx", mime="application/gpx+xml"
                )
            with colB:
                b2 = gpx_bytes_waypoints_only(wpts)
                st.download_button(
                    "Scarica SOLO waypoint/km (GPX)",
                    data=b2, file_name=f"{fname}_kmwpt_only.gpx", mime="application/gpx+xml"
                )
        else:
            st.caption("Percorso più corto di 1 km: nessun waypoint/km generato.")
else:
    st.info("Carica un file GPX per iniziare.")
