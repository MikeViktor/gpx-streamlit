# streamlit_app.py
# v5 — IF + split/orari + esport waypoint — settori gauge corretti
import io
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import gpxpy
import gpxpy.gpx
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

st.set_page_config(page_title="Analisi Tracce GPX", layout="wide")

# ------------------------------
# Parametri calcolo (come GUI)
# ------------------------------
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

DEFAULTS = dict(
    base=15.0, up=15.0, down=15.0,
    temp=15.0, hum=50.0, wind=5.0,
    precip="assenza pioggia", surface="asciutto", expo="misto",
    tech="normale", loadkg=6.0,
)

# ------------------------------
# Utility/Calcolo
# ------------------------------
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

def parse_gpx(file_bytes):
    gpx = gpxpy.parse(io.BytesIO(file_bytes).read().decode("utf-8", errors="ignore"))
    lat, lon, ele = [], [], []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                if p.elevation is not None:
                    lat.append(p.latitude); lon.append(p.longitude); ele.append(p.elevation)
    # fallback routes/waypoints
    if not lat:
        for rte in gpx.routes:
            for p in rte.points:
                if p.elevation is not None:
                    lat.append(p.latitude); lon.append(p.longitude); ele.append(p.elevation)
    return lat, lon, ele

def compute_from_arrays(lat, lon, ele_raw,
                        base_min_per_km=15.0, up_min_per_100m=15.0, down_min_per_200m=15.0,
                        reverse=False):
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
    x_km=[min(i*RS_STEP_M,total_m)/1000.0 for i in range(len(e_sm))]

    surge_idx = round(surge_trans / max(0.1, tot_km), 2)

    return {
        "tot_km": round(tot_km,2), "dplus": round(dplus,0), "dneg": round(dneg,0),
        "t_dist": t_dist, "t_up": t_up, "t_down": t_down, "t_total": t_total,
        "holes": holes,
        "len_flat_km": round(flat_len/1000.0,2), "len_up_km": round(asc_len/1000.0,2), "len_down_km": round(desc_len/1000.0,2),
        "grade_up_pct": round(grade_up,1), "grade_down_pct": round(grade_down,1),
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

def fmt_hm(minutes):
    h=int(minutes//60); m=int(round(minutes-h*60))
    if m==60: h+=1; m=0
    return f"{h}:{m:02d}"

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("Impostazioni")
    inverti = st.checkbox("Inverti traccia", value=False)

    st.subheader("Tempi per km")
    show_clock = st.checkbox("Mostra orario del giorno", value=True)
    show_labels = st.checkbox("Mostra etichette sul grafico", value=True)
    start_time = st.time_input("Orario di partenza", value=datetime.strptime("08:00","%H:%M").time())

    st.subheader("Parametri di passo (min)")
    base = st.number_input("Min/km (piano)", 1.0, 60.0, DEFAULTS["base"], 0.5)
    up   = st.number_input("Min/100 m (salita)", 1.0, 60.0, DEFAULTS["up"], 0.5)
    down = st.number_input("Min/200 m (discesa)",1.0, 60.0, DEFAULTS["down"],0.5)

    st.subheader("Condizioni")
    temp = st.number_input("Temperatura (°C)", -30.0, 50.0, DEFAULTS["temp"], 1.0)
    hum  = st.number_input("Umidità (%)", 0.0, 100.0, DEFAULTS["hum"], 1.0)
    wind = st.number_input("Vento (km/h)", 0.0, 150.0, DEFAULTS["wind"], 1.0)
    precip = st.selectbox("Precipitazioni", ["assenza pioggia","pioviggine","pioggia","pioggia forte","neve fresca","neve profonda"], index=0)
    surface = st.selectbox("Fondo", ["asciutto","fango","roccia bagnata","neve dura","ghiaccio"], index=0)
    expo = st.selectbox("Esposizione", ["misto","ombra","pieno sole"], index=0)
    tech = st.selectbox("Tecnica", ["facile","normale","roccioso","passaggi di roccia (scrambling)","neve/ghiaccio"], index=1)
    loadkg = st.number_input("Zaino extra (kg)", 0.0, 40.0, DEFAULTS["loadkg"], 1.0)

# ------------------------------
# Body
# ------------------------------
st.title("v5 — IF + split/orari + esportazione waypoint")

uploaded = st.file_uploader("Carica GPX (trascina qui il file)", type=["gpx"])

if not uploaded:
    st.stop()

lat, lon, ele = parse_gpx(uploaded.getvalue())
if len(ele) < 2:
    st.error("Il GPX non contiene punti quota sufficienti.")
    st.stop()

res = compute_from_arrays(lat, lon, ele, base, up, down, reverse=inverti)
fi = compute_if_from_res(
    res, temp, hum, precip, surface, wind, expo, tech, loadkg
)

# metriche in testata
colA, colB, colC = st.columns(3)
colA.metric("Distanza (km)", f"{res['tot_km']:.2f}")
colB.metric("Dislivello + (m)", f"{int(res['dplus'])}")
colC.metric("Tempo totale", fmt_hm(res["t_total"]))

# ---------------- Gauge (matplotlib) ----------------
st.subheader("Indice di Difficoltà")
st.write(f"**{fi['IF']} ({fi['cat']})**")

fig_g, axg = plt.subplots(figsize=(4.8, 1.8))
axg.set_aspect('equal'); axg.axis('off')

# bins left->right
bins = [
    (0,30,"#2ecc71","Facile"),
    (30,50,"#f1c40f","Medio"),
    (50,70,"#e67e22","Impeg."),
    (70,80,"#e74c3c","Diffic."),
    (80,90,"#8e44ad","Molto diff."),
    (90,100,"#111111","Estremo"),
]
# geometry
cx, cy, r_o, r_i = 0, 0, 1.0, 0.68
# draw wedges (start 180° toward 0°)
for a,b,col,_ in bins:
    ang1 = 180 - (a/100)*180
    ang2 = 180 - (b/100)*180
    w = Wedge((cx,cy), r_o, ang2, ang1, width=r_o-r_i, facecolor=col, edgecolor=col)
    axg.add_patch(w)
# needle
v = max(0.0, min(100.0, fi["IF"]))
ang = math.radians(180 - (v/100)*180)
x_tip = cx + r_i*math.cos(ang)
y_tip = cy + r_i*math.sin(ang)
axg.plot([cx, x_tip], [cy, y_tip], color="#222", linewidth=3)
axg.scatter([cx],[cy], s=30, color="#222")
# labels over arcs
for a,b,_,lab in bins:
    m = (a+b)/2
    angm = math.radians(180 - (m/100)*180)
    xr = cx + (r_o-0.18)*math.cos(angm)
    yr = cy + (r_o-0.18)*math.sin(angm)
    axg.text(xr, yr, lab, fontsize=9, ha='center', va='center')

# big number
axg.text(0, -0.35, f"{fi['IF']:.1f}", fontsize=18, fontweight='bold', ha='center')
st.pyplot(fig_g, transparent=True)

# ---------------- Dettagli “Risultati” ----------------
st.subheader("Risultati")
left, right = st.columns([1,2])
with left:
    st.markdown(
        f"""
- **Dislivello − (m):** {int(res['dneg'])}  
- **Tempo piano:** {fmt_hm(res['t_dist'])}  
- **Tempo salita:** {fmt_hm(res['t_up'])}  
- **Tempo discesa:** {fmt_hm(res['t_down'])}  
- **Calorie stimate:** ~{int( (res['t_total']/60)*300 ):,}  
- **Piano (km):** {res['len_flat_km']:.2f} — **Salita (km):** {res['len_up_km']:.2f} — **Discesa (km):** {res['len_down_km']:.2f}  
- **Pend. media salita (%):** {res['grade_up_pct']:.1f} — **discesa:** {res['grade_down_pct']:.1f}  
- **LCS ≥25% (m):** {int(res['lcs25_m'])}  
- **Blocchi ripidi (≥100 m @ ≥25%):** {int(res['blocks25_count'])}  
- **Surge (cambi ritmo)/km:** {res['surge_idx_per_km']:.2f}  
- **Buchi GPX:** {"✅ 0" if res['holes']==0 else f"⚠️ {res['holes']}"}  
        """.replace(",", ".")
    )

# ---------------- Tempi/Orari ai km (grafico & tabella) ----------------
st.subheader("Tempi/Orario ai diversi Km")

# calcolo split km → tempo
tot_km = res["tot_km"]
x = np.array(res["profile_x_km"])
y = np.array(res["profile_y_m"])

# velocità istantanea “equivalente” in funzione di pendenza
# (stesso modello usato per t_dist/t_up/t_down a step)
pace_flat = base
minutes = 0.0
km_marks = np.arange(1, int(math.floor(tot_km))+1)
split_rows = []
curr_km_target = 1.0
last_t = 0.0
last_dist = 0.0
for i in range(1, len(x)):
    seg_m = (x[i]-x[i-1])*1000.0
    if seg_m <= 0: continue
    dz = y[i]-y[i-1]
    g = dz/seg_m  # pendenza frazionale (es 0.10 = 10%)
    if dz > RS_MIN_DELEV:          # salita
        seg_min = (seg_m/1000.0)*pace_flat + (dz/100.0)*up
    elif dz < -RS_MIN_DELEV:       # discesa
        seg_min = (seg_m/1000.0)*pace_flat + ((-dz)/200.0)*down
    else:                          # quasi piano
        seg_min = (seg_m/1000.0)*pace_flat
    minutes += seg_min

    # km mark raggiunta?
    while x[i] >= curr_km_target and curr_km_target <= tot_km + 1e-6:
        # tempo parziale tra (last_dist -> curr_km_target)
        frac = (curr_km_target - x[i-1]) / max(1e-9, (x[i]-x[i-1]))
        t_mark = last_t + frac * (minutes - last_t)
        km_index = int(round(curr_km_target))
        split_rows.append((km_index, t_mark))
        curr_km_target += 1.0
    last_t = minutes
    last_dist = x[i]

# tabella
start_dt = datetime.combine(datetime.today(), start_time) if show_clock else None
km_list = []
for km_i, t_min in split_rows:
    km_list.append(dict(
        Km=km_i,
        **({"Tempo parziale": fmt_hm(t_min)}),
        **({"Ora del giorno": (start_dt + timedelta(minutes=t_min)).strftime("%H:%M") if start_dt else ""})
    ))
df = pd.DataFrame(km_list)

# grafico altimetrico con etichette tempi
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(x, y, linewidth=2.0)
ax.set_xlabel("Distanza (km)")
ax.set_ylabel("Quota (m)")
ax.grid(True, linestyle=":", alpha=0.5)
if show_labels and len(km_list):
    for row in km_list:
        km_i = row["Km"]
        if km_i <= x[-1]:
            # trova quota vicino al km_i
            j = np.searchsorted(x, km_i)
            j = min(max(j,1), len(x)-1)
            y_km = y[j]
            label = row["Ora del giorno"] if show_clock else row["Tempo parziale"]
            ax.text(km_i, y_km, label, ha="center", va="bottom", fontsize=9)
st.pyplot(fig, use_container_width=True, transparent=True)

st.dataframe(
    df[["Km","Tempo parziale","Ora del giorno"]],
    use_container_width=True, hide_index=True, height=min(420, 40*max(1, len(df)))
)
