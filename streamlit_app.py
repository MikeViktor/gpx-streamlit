# -*- coding: utf-8 -*-
import io, math
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---- gpxpy se disponibile ----
try:
    import gpxpy  # type: ignore
    HAS_GPXPY = True
except Exception:
    import xml.etree.ElementTree as ET
    HAS_GPXPY = False

APP_TITLE = "Analisi Tracce GPX"
APP_ICON  = "⛰️"

# ====== Parametri ======
RS_STEP_M     = 3.0
RS_MIN_DELEV  = 0.25
RS_MED_K      = 3
RS_AVG_K      = 3
ABS_JUMP_RAW  = 100.0

LOOP_TOL_M        = 200.0
DRIFT_MIN_ABS_M   = 2.0
BALANCE_MIN_DIFFM = 10.0
BALANCE_REL_FRAC  = 0.05

W_D, W_PLUS, W_COMP = 0.5, 1.0, 0.5
W_STEEP, W_STEEP_D  = 0.4, 0.3
W_LCS, W_BLOCKS, W_SURGE = 0.25, 0.15, 0.25
IF_S0 = 80.0
ALPHA_METEO = 0.6
SEVERITY_GAIN = 1.52

PRECIP_OPTIONS = ["assenza pioggia","pioviggine","pioggia","pioggia forte","neve fresca","neve profonda"]
SURF_OPTIONS   = ["asciutto","fango","roccia bagnata","neve dura","ghiaccio"]
EXPO_OPTIONS   = ["ombra","misto","pieno sole"]
TECH_OPTIONS   = ["facile","normale","roccioso","passaggi di roccia (scrambling)","neve/ghiaccio"]

DEFAULTS = dict(
    base=15.0, up=15.0, down=15.0, weight=70.0, reverse=False,
    temp=15.0, hum=50.0, wind=5.0,
    precip="assenza pioggia", surface="asciutto",
    expo="misto", tech="normale", loadkg=6.0,
    prof_scale=1.0
)

# ===== Utility =====
def _dist_km(la1, lo1, la2, lo2):
    dy = (la2 - la1) * 111.32
    dx = (lo2 - lo1) * 111.32 * math.cos(math.radians((la1 + la2) / 2.0))
    return math.hypot(dx, dy)

def median_k(seq, k=3):
    if k < 1: k = 1
    if k % 2 == 0: k += 1
    h = k // 2; out = []; n = len(seq)
    for i in range(n):
        a = max(0, i-h); b = min(n, i+h+1)
        w = sorted(seq[a:b]); out.append(w[len(w)//2])
    return out

def moving_avg(seq, k=3):
    if k < 1: k = 1
    h = k // 2; out = []; n = len(seq)
    for i in range(n):
        a = max(0, i-h); b = min(n, i+h+1)
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

def _is_loop(lat, lon, tol_m=LOOP_TOL_M):
    if len(lat) < 2: return False
    return _dist_km(lat[0], lon[0], lat[-1], lon[-1])*1000.0 <= tol_m

def apply_loop_drift_correction(elev_series, lat, lon, min_abs=DRIFT_MIN_ABS_M):
    if not _is_loop(lat, lon) or len(elev_series) < 2: return elev_series, False, 0.0
    drift = elev_series[-1] - elev_series[0]
    if abs(drift) < min_abs: return elev_series, False, drift
    n = len(elev_series)-1
    fixed = [elev_series[i] - (drift * (i/n)) for i in range(len(elev_series))]
    return fixed, True, drift

def meteo_multiplier(temp_c, humidity_pct, precip, surface, wind_kmh, exposure):
    if   temp_c < -5: M_temp = 1.20
    elif temp_c < 0:  M_temp = 1.10
    elif temp_c < 5:  M_temp = 1.05
    elif temp_c <= 20: M_temp = 1.00
    elif temp_c <= 25: M_temp = 1.05
    elif temp_c <= 30: M_temp = 1.10
    elif temp_c <= 35: M_temp = 1.20
    else: M_temp = 1.35
    if   humidity_pct > 80: M_temp += 0.10
    elif humidity_pct > 60: M_temp += 0.05
    precip_map  = {"dry":1.00,"drizzle":1.05,"rain":1.15,"heavy_rain":1.30,"snow_shallow":1.25,"snow_deep":1.60}
    surface_map = {"dry":1.00,"mud":1.10,"wet_rock":1.15,"hard_snow":1.30,"ice":1.60}
    exposure_map= {"shade":1.00,"mixed":1.05,"sun":1.10}
    if   wind_kmh <= 10: M_wind = 1.00
    elif wind_kmh <= 20: M_wind = 1.05
    elif wind_kmh <= 35: M_wind = 1.10
    elif wind_kmh <= 60: M_wind = 1.20
    else: M_wind = 1.35
    M_precip  = precip_map.get(precip,1.0)
    M_surface = surface_map.get(surface,1.0)
    M_sun     = exposure_map.get(exposure,1.0)
    return min(1.4, M_temp * max(M_precip, M_surface) * M_wind * M_sun)

def altitude_multiplier(avg_alt_m):
    if avg_alt_m is None: return 1.0
    excess = max(0.0, (avg_alt_m - 2000.0)/500.0)
    return 1.0 + 0.03 * excess

def technique_multiplier(level="normale"):
    table = {"facile":0.95,"normale":1.00,"roccioso":1.10,
             "passaggi di roccia (scrambling)":1.20,"neve/ghiaccio":1.30,
             "scrambling":1.20}
    return table.get(level,1.0)

def pack_load_multiplier(extra_load_kg=0.0):
    return 1.0 + 0.02 * max(0.0, extra_load_kg/5.0)

def cat_from_if(v):
    if v < 30: return "Facile"
    if v < 50: return "Medio"
    if v < 70: return "Impegnativo"
    if v < 80: return "Difficile"
    if v <= 90: return "Molto diff."
    return "Estremo"

# ==== Parsing GPX ====
def parse_gpx(uploaded_file) -> pd.DataFrame:
    if HAS_GPXPY:
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        gpx = gpxpy.parse(io.StringIO(text))
        pts = []
        for trk in gpx.tracks:
            for seg in trk.segments:
                for p in seg.points:
                    if p.elevation is None: continue
                    pts.append((p.latitude, p.longitude, float(p.elevation)))
        if not pts:
            for rte in gpx.routes:
                for p in rte.points:
                    if p.elevation is None: continue
                    pts.append((p.latitude, p.longitude, float(p.elevation)))
        df = pd.DataFrame(pts, columns=["lat","lon","ele"])
    else:
        raw = uploaded_file.getvalue()
        root = ET.fromstring(raw)
        def is_tag(el, name):
            t = el.tag; return t.endswith('}' + name) or t == name
        pts = []
        for wanted in ("trkpt","rtept","wpt"):
            pts.clear()
            for el in root.iter():
                if is_tag(el, wanted):
                    la = el.attrib.get("lat"); lo = el.attrib.get("lon")
                    if la is None or lo is None: continue
                    ele = None
                    for ch in el:
                        if is_tag(ch, "ele"): ele = ch.text; break
                    if ele is None: continue
                    try: pts.append((float(la), float(lo), float(ele)))
                    except: pass
            if pts: break
        df = pd.DataFrame(pts, columns=["lat","lon","ele"])
    if df.empty: return df
    dist = [0.0]
    for i in range(1, len(df)):
        dist.append(dist[-1] + _dist_km(df.lat[i-1], df.lon[i-1], df.lat[i], df.lon[i])*1000.0)
    df["dist_m"] = dist
    return df

# ==== Motore calcolo (identico alla versione precedente) ====
def compute_from_arrays(lat, lon, ele_raw,
                        base_min_per_km=15.0, up_min_per_100m=15.0, down_min_per_200m=15.0,
                        weight_kg=70.0, reverse=False):
    if len(ele_raw) < 2: raise ValueError("Nessun punto utile con quota.")
    if reverse:
        lat=list(reversed(lat)); lon=list(reversed(lon)); ele_raw=list(reversed(ele_raw))

    cum=[0.0]
    for i in range(1,len(lat)):
        cum.append(cum[-1]+_dist_km(lat[i-1],lon[i-1],lat[i],lon[i])*1000.0)
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
        "asc_bins_m": [round(v,0) for v in asc_bins], "desc_bins_m": [round(v,0) for v in desc_bins],
        "lcs25_m": round(longest_steep_run,0), "blocks25_count": int(blocks25),
        "surge_idx_per_km": surge_idx,
        "avg_alt_m": float(np.mean(ele_raw)) if ele_raw else None,
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

# ===== Gauge SVG con etichette =====
def _pt(cx, cy, r, deg):
    rad = math.radians(180 - deg)
    return cx + r*math.cos(rad), cy - r*math.sin(rad)

def _arc_path(cx, cy, r, a0, a1, sweep=1):
    x0,y0 = _pt(cx,cy,r,a0); x1,y1 = _pt(cx,cy,r,a1)
    large = 1 if abs(a0-a1) > 180 else 0
    return f"M {x0:.2f},{y0:.2f} A {r:.2f},{r:.2f} 0 {large} {sweep} {x1:.2f},{y1:.2f}"

def _ring_segment_path(cx, cy, r_out, r_in, a0, a1):
    x0,y0 = _pt(cx,cy,r_out,a0); x1,y1 = _pt(cx,cy,r_out,a1)
    xi,yi = _pt(cx,cy,r_in ,a1); xo,yo = _pt(cx,cy,r_in ,a0)
    large = 1 if abs(a0-a1) > 180 else 0
    return " ".join([
        f"M {x0:.2f},{y0:.2f}",
        f"A {r_out:.2f},{r_out:.2f} 0 {large} 1 {x1:.2f},{y1:.2f}",
        f"L {xi:.2f},{yi:.2f}",
        f"A {r_in:.2f},{r_in:.2f} 0 {large} 0 {xo:.2f},{yo:.2f}",
        "Z"
    ])

def draw_gauge_svg(score: float):
    score = float(np.clip(score,0,100))
    cx, cy = 300, 230
    r_out, r_in = 200, 140
    r_lab = r_out - 18  # raggio per etichette lungo arco
    r_rad = r_out + 24  # raggio per etichette radiali

    # (label, colore, p0, p1, tipo)
    segs = [
        ("Facile"      , "#2ecc71", 0 , 30, "arc"),
        ("Medio"       , "#f1c40f", 30, 50, "arc"),
        ("Impegnativo" , "#e67e22", 50, 70, "arc"),
        ("Difficile"   , "#e74c3c", 70, 80, "rad"),
        ("Molto diff." , "#8e44ad", 80, 90, "rad"),
        ("Estremo"     , "#111111", 90, 100,"rad"),
    ]

    paths = []
    defs  = []
    texts = []

    for idx,(lab,col,p0,p1,kind) in enumerate(segs):
        a0 = 180*(p0/100); a1 = 180*(p1/100)
        # settore colorato
        d = _ring_segment_path(cx,cy,r_out,r_in,a0,a1)
        paths.append(f'<path d="{d}" fill="{col}" stroke="#fff" stroke-width="2"/>')

        if kind=="arc":
            # etichetta “parallela” all’arco
            d_lab = _arc_path(cx,cy,r_lab,a0,a1, sweep=1)
            defs.append(f'<path id="lab{idx}" d="{d_lab}" fill="none" stroke="none"/>')
            texts.append(
                f'<text font-size="14" font-weight="600" fill="#222">'
                f'<textPath href="#lab{idx}" startOffset="50%" text-anchor="middle">{lab}</textPath>'
                f'</text>'
            )
        else:
            # etichetta radiale
            amid = (a0 + a1)/2.0
            x,y = _pt(cx,cy,r_rad,amid)
            rot = amid - 180  # radialmente verso l’esterno
            fill = "#222" if col != "#111111" else "#222"
            texts.append(
                f'<text x="{x:.1f}" y="{y:.1f}" font-size="14" font-weight="700" fill="{fill}" '
                f'text-anchor="middle" dominant-baseline="middle" '
                f'transform="rotate({rot:.1f},{x:.1f},{y:.1f})">{lab}</text>'
            )

    # lancetta
    aN = 180*(score/100.0)
    xN, yN = _pt(cx, cy, r_out, aN)
    needle = f'<line x1="{cx}" y1="{cy}" x2="{xN:.1f}" y2="{yN:.1f}" stroke="#000" stroke-width="4"/>'
    hub    = f'<circle cx="{cx}" cy="{cy}" r="6" fill="#000"/>'

    svg = f'''
    <svg width="100%" height="260" viewBox="0 0 600 260" xmlns="http://www.w3.org/2000/svg">
      <defs>{''.join(defs)}</defs>
      {''.join(paths)}
      {needle}{hub}
      {''.join(texts)}
    </svg>
    '''
    return svg

# ===== UI =====
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
st.title(APP_TITLE)

uploaded = st.file_uploader("📂 Carica GPX", type=["gpx"])
if "gpx_name" not in st.session_state: st.session_state.gpx_name = None
if uploaded is not None and uploaded.name != st.session_state.gpx_name:
    st.session_state.gpx_name = uploaded.name
    for k,v in DEFAULTS.items(): st.session_state[k] = v

with st.sidebar:
    st.markdown("### Impostazioni")
    base = st.number_input("Min/km (piano)", 1.0, 60.0, value=st.session_state.get("base", 15.0), step=0.5, key="base")
    up   = st.number_input("Min/100 m (salita)", 1.0, 60.0, value=st.session_state.get("up", 15.0), step=0.5, key="up")
    down = st.number_input("Min/200 m (discesa)", 1.0, 60.0, value=st.session_state.get("down", 15.0), step=0.5, key="down")
    weight = st.number_input("Peso (kg)", 30.0, 150.0, value=st.session_state.get("weight", 70.0), step=1.0, key="weight")
    reverse = st.checkbox("Inverti traccia", value=st.session_state.get("reverse", False), key="reverse")

    st.markdown("### Condizioni")
    temp = st.number_input("Temperatura (°C)", -30.0, 50.0, value=st.session_state.get("temp", 15.0), key="temp")
    hum  = st.number_input("Umidità (%)", 0.0, 100.0, value=st.session_state.get("hum", 50.0), key="hum")
    wind = st.number_input("Vento (km/h)", 0.0, 150.0, value=st.session_state.get("wind", 5.0), key="wind")
    precip = st.selectbox("Precipitazioni", PRECIP_OPTIONS, index=PRECIP_OPTIONS.index(st.session_state.get("precip","assenza pioggia")), key="precip")
    surface = st.selectbox("Fondo", SURF_OPTIONS, index=SURF_OPTIONS.index(st.session_state.get("surface","asciutto")), key="surface")
    expo = st.selectbox("Esposizione", EXPO_OPTIONS, index=EXPO_OPTIONS.index(st.session_state.get("expo","misto")), key="expo")
    tech = st.selectbox("Tecnica", TECH_OPTIONS, index=TECH_OPTIONS.index(st.session_state.get("tech","normale")), key="tech")
    loadkg = st.number_input("Zaino extra (kg)", 0.0, 40.0, value=st.session_state.get("loadkg", 6.0), key="loadkg")

    st.markdown("### Profilo")
    prof_scale = st.slider("Scala verticale profilo (Y)", 0.6, 1.6, value=float(st.session_state.get("prof_scale",1.0)), step=0.05, key="prof_scale")

k1, k2, k3 = st.columns(3)

if uploaded is None:
    k1.metric("Distanza (km)", "-")
    k2.metric("Dislivello + (m)", "-")
    k3.metric("Tempo totale", "-")
    st.subheader("Indice di Difficoltà")
    st.markdown(draw_gauge_svg(0.0), unsafe_allow_html=True)
else:
    df = parse_gpx(uploaded)
    if df.empty or df["ele"].isna().all():
        st.warning("Il GPX non contiene punti utilizzabili con quota.")
    else:
        lat = df["lat"].to_list(); lon = df["lon"].to_list(); ele = df["ele"].to_list()
        res = compute_from_arrays(lat, lon, ele, base, up, down, weight, reverse)

        def fmt_hm(minutes):
            h=int(minutes//60); m=int(round(minutes-h*60))
            if m==60: h+=1; m=0
            return f"{h}:{m:02d}"

        k1.metric("Distanza (km)", f"{res['tot_km']:.2f}")
        k2.metric("Dislivello + (m)", f"{int(res['dplus'])}")
        k3.metric("Tempo totale", fmt_hm(res["t_total"]))

        st.subheader("Risultati")
        cA, cB, cC = st.columns(3)
        with cA:
            st.write(f"**Dislivello − (m):** {int(res['dneg'])}")
            st.write(f"**Tempo piano:** {fmt_hm(res['t_dist'])}")
            st.write(f"**Tempo salita:** {fmt_hm(res['t_up'])}")
            st.write(f"**Tempo discesa:** {fmt_hm(res['t_down'])}")
            st.write(f"**Calorie stimate:** {res['cal_total']}")
        with cB:
            st.write(f"**Piano (km):** {res['len_flat_km']:.2f}")
            st.write(f"**Salita (km):** {res['len_up_km']:.2f}")
            st.write(f"**Discesa (km):** {res['len_down_km']:.2f}")
            st.write(f"**Pend. media salita (%):** {res['grade_up_pct']:.1f}")
            st.write(f"**Pend. media discesa (%):** {res['grade_down_pct']:.1f}")
        with cC:
            holes = int(res["holes"])
            st.write(f"**Buchi GPX:** {'🟥 ' if holes>0 else '🟩 '} {holes}")
            st.write(f"**LCS ≥25% (m):** {int(res['lcs25_m'])}")
            st.write(f"**Blocchi ripidi (≥100 m @ ≥25%):** {int(res['blocks25_count'])}")
            st.write(f"**Surge (cambi ritmo)/km:** {res['surge_idx_per_km']:.2f}")

        fi = compute_if_from_res(
            res, temp_c=temp, humidity_pct=hum,
            precip_it=precip, surface_it=surface,
            wind_kmh=wind, expo_it=expo,
            technique_level=tech, extra_load_kg=loadkg
        )
        st.subheader("Indice di Difficoltà")
        st.markdown(f"**{fi['IF']}  ({fi['cat']})**")
        st.markdown(draw_gauge_svg(fi["IF"]), unsafe_allow_html=True)

        st.subheader("Profilo altimetrico")
        prof = pd.DataFrame({"km":res["profile_x_km"], "elev":res["profile_y_m"]})
        y_min = float(np.min(prof["elev"])) - 60
        y_max = float(np.max(prof["elev"])) + 60
        base_height = 520
        height = int(base_height * float(prof_scale))
        chart = alt.Chart(prof).mark_line(strokeWidth=2.8).encode(
            x=alt.X("km:Q", title="Distanza (km)"),
            y=alt.Y("elev:Q", title="Quota (m)", scale=alt.Scale(domain=[y_min, y_max]))
        ).properties(width=900, height=height)
        st.altair_chart(chart, use_container_width=False)

        msgs=[]
        if res.get("loop_fix_applied"): msgs.append(f"Corretta deriva ~{res['loop_drift_abs_m']} m")
        if res.get("loop_balance_applied"): msgs.append(f"Chiusura anello: D+ allineato a D− (diff {res['balance_diff_m']} m)")
        if msgs: st.info(" – ".join(msgs))
