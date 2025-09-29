# -*- coding: utf-8 -*-
import io, math, xml.etree.ElementTree as ET
import pandas as pd
import streamlit as st
import altair as alt

# ===================== PDF backends =====================
# Prova ReportLab, altrimenti fallback FPDF2. Se nessuno dei due è presente,
# mostreremo un messaggio informativo e non un errore.
PDF_BACKEND = None
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    PDF_BACKEND = "reportlab"
except Exception:
    try:
        from fpdf import FPDF
        PDF_BACKEND = "fpdf2"
    except Exception:
        PDF_BACKEND = None

def _safe(s: str) -> str:
    return (s.replace("−","-").replace("•","-").replace("°"," deg")
              .replace("≥",">=").replace("≤","<="))

APP_TITLE = "Tempo percorrenza sentiero (web)"
APP_VER   = "v2.5"

# ===== Filtri/ricampionamento =====
RS_STEP_M     = 3.0
RS_MIN_DELEV  = 0.25
RS_MED_K      = 3
RS_AVG_K      = 3
ABS_JUMP_RAW  = 100.0

# ===== Pesi Indice di Fatica =====
W_D      = 0.5
W_PLUS   = 1.0
W_COMP   = 0.5
W_STEEP  = 0.4
W_STEEP_D= 0.3
W_LCS    = 0.25
W_BLOCKS = 0.15
W_SURGE  = 0.25
IF_S0    = 80.0
ALPHA_METEO = 0.6

# ---------------- Utility GPX / calcolo base ----------------
def _is_tag(e, name: str) -> bool:
    t = e.tag
    return t.endswith('}' + name) or t == name

def parse_gpx_bytes(file_bytes: bytes):
    """Parsa GPX da bytes, supportando trkpt/rtept/wpt con ele."""
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
        if lat: return lat, lon, ele, wanted
    return [], [], [], None

def dist_km(lat1, lon1, lat2, lon2):
    dy = (lat2 - lat1) * 111.32
    dx = (lon2 - lon1) * 111.32 * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.hypot(dx, dy)

def median_k(seq, k=3):
    if k < 1: k = 1
    if k % 2 == 0: k += 1
    half = k // 2
    out = []
    n = len(seq)
    for i in range(n):
        a = max(0, i - half); b = min(n, i + half + 1)
        window = sorted(seq[a:b])
        out.append(window[len(window)//2])
    return out

def moving_avg(seq, k=3):
    if k < 1: k = 1
    half = k // 2
    out = []
    n = len(seq)
    for i in range(n):
        a = max(0, i - half); b = min(n, i + half + 1)
        out.append(sum(seq[a:b]) / max(1, b - a))
    return out

def resample_elev(cum_m, ele, step_m=3.0):
    if len(cum_m) != len(ele): raise ValueError("cum_m/ele di diversa lunghezza")
    total = cum_m[-1]
    n = int(total // step_m) + 1
    out = []
    t = 0.0; j = 0
    for _ in range(n):
        while j < len(cum_m) - 1 and cum_m[j+1] < t: j += 1
        if t <= cum_m[0]:
            out.append(ele[0])
        elif t >= cum_m[-1]:
            out.append(ele[-1])
        else:
            u = (t - cum_m[j]) / (cum_m[j+1] - cum_m[j])
            out.append(ele[j] + u * (ele[j+1] - ele[j]))
        t += step_m
    return out

def fmt_hm(minutes):
    h = int(minutes // 60); m = int(round(minutes - h*60))
    if m == 60: h += 1; m = 0
    return f"{h}:{m:02d}"

# ---------------- Meteo & fattori ----------------
def meteo_multiplier(temp_c: float, humidity_pct: float, precip: str, surface: str,
                     wind_kmh: float, exposure: str) -> float:
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

    M_meteo = M_temp * max(M_precip, M_surface) * M_wind * M_sun
    return min(1.4, M_meteo)

def altitude_multiplier(avg_alt_m):
    if avg_alt_m is None: return 1.0
    excess = max(0.0, (avg_alt_m - 2000.0) / 500.0)
    return 1.0 + 0.03 * excess

def technique_multiplier(level: str = "normale") -> float:
    table = {"facile":0.95,"normale":1.00,"roccioso":1.10,"scrambling":1.20,"neve/ghiaccio":1.30}
    return table.get(level, 1.0)

def pack_load_multiplier(extra_load_kg: float = 0.0) -> float:
    return 1.0 + 0.02 * max(0.0, extra_load_kg / 5.0)

def cat_from_if(val: float) -> str:
    if val < 30: return "Facile"
    if val < 50: return "Medio"
    if val < 70: return "Impegnativo"
    return "Molto impegnativo"

# ---------------- DEFAULT e reset su nuovo file ----------------
DEFAULTS = {
    "base": 15.0, "up": 15.0, "down": 15.0,
    "weight": 70.0, "reverse": False,
    "temp": 15.0, "hum": 50.0, "wind": 5.0,
    "precip_sel": "assenza pioggia",
    "surface_sel": "asciutto",
    "expo_sel": "misto",
    "tech_sel": "normale",
    "loadkg": 6.0,
}
def ensure_defaults():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ---------------- Core calcoli ----------------
def compute_from_gpx_bytes(file_bytes: bytes,
                           base_min_per_km=15.0, up_min_per_100m=15.0, down_min_per_200m=15.0,
                           weight_kg=70.0, reverse=False):
    lat, lon, ele_raw, _kind = parse_gpx_bytes(file_bytes)
    if not ele_raw or len(ele_raw) < 2:
        raise ValueError("Nessun punto utile con elevazione nel GPX.")

    if reverse:
        lat = list(reversed(lat)); lon = list(reversed(lon)); ele_raw = list(reversed(ele_raw))

    # distanza cumulata (m)
    cum = [0.0]
    for i in range(1, len(lat)):
        cum.append(cum[-1] + dist_km(lat[i-1], lon[i-1], lat[i], lon[i]) * 1000.0)
    tot_km = cum[-1] / 1000.0
    total_m = cum[-1]

    # ricampionamento + filtri
    e_res = resample_elev(cum, ele_raw, RS_STEP_M)
    e_med = median_k(e_res, RS_MED_K)
    e_sm  = moving_avg(e_med, RS_AVG_K)

    # metriche base
    dplus = dneg = 0.0
    asc_len = desc_len = flat_len = 0.0
    asc_gain = desc_drop = 0.0

    # fasce pendenza (metri): <10, 10–20, 20–30, 30–40, >40
    asc_bins = [0.0, 0.0, 0.0, 0.0, 0.0]
    desc_bins= [0.0, 0.0, 0.0, 0.0, 0.0]

    # IF avanzate su SALITA
    longest_steep_run = 0.0
    current_run = 0.0
    blocks25_count = 0
    last_state = 0   # 0=altro, 1=gentle up (<15%), 2=steep up (>=25%)
    surge_transitions = 0

    for i in range(1, len(e_sm)):
        t_prev = (i-1) * RS_STEP_M
        t_curr = min(i * RS_STEP_M, total_m)
        seg = max(0.0, t_curr - t_prev)
        if seg <= 0: continue

        d = e_sm[i] - e_sm[i-1]
        if d > RS_MIN_DELEV:
            dplus += d; asc_len += seg; asc_gain += d
            g = (d / seg) * 100.0
            # fasce
            if   g < 10: asc_bins[0] += seg
            elif g < 20: asc_bins[1] += seg
            elif g < 30: asc_bins[2] += seg
            elif g < 40: asc_bins[3] += seg
            else:        asc_bins[4] += seg
            # LCS / blocchi / surge su salita
            if g >= 25.0:
                current_run += seg
                if current_run > longest_steep_run: longest_steep_run = current_run
                state = 2
            else:
                if current_run >= 100.0: blocks25_count += 1
                current_run = 0.0
                state = 1 if g < 15.0 else 0
            if (last_state in (1,2)) and (state in (1,2)) and (state != last_state):
                surge_transitions += 1
            if state != 0: last_state = state
        elif d < -RS_MIN_DELEV:
            drop = -d; dneg += drop; desc_len += seg; desc_drop += drop
            g = (drop / seg) * 100.0
            if   g < 10: desc_bins[0] += seg
            elif g < 20: desc_bins[1] += seg
            elif g < 30: desc_bins[2] += seg
            elif g < 40: desc_bins[3] += seg
            else:        desc_bins[4] += seg
            if current_run >= 100.0: blocks25_count += 1
            current_run = 0.0
            last_state = 0
        else:
            flat_len += seg
            if current_run >= 100.0: blocks25_count += 1
            current_run = 0.0
            last_state = 0

    if current_run >= 100.0: blocks25_count += 1

    grade_up_pct   = (asc_gain / asc_len * 100.0)  if asc_len  > 0 else 0.0
    grade_down_pct = (desc_drop / desc_len * 100.0) if desc_len > 0 else 0.0

    t_dist  = tot_km * base_min_per_km
    t_up    = (dplus / 100.0) * up_min_per_100m
    t_down  = (dneg  / 200.0) * down_min_per_200m
    t_total = t_dist + t_up + t_down

    holes = sum(1 for i in range(1, len(ele_raw)) if abs(ele_raw[i] - ele_raw[i-1]) >= ABS_JUMP_RAW)

    # calorie
    weight_kg = max(1.0, float(weight_kg))
    cal_flat = weight_kg * 0.6  * max(0.0, tot_km)
    cal_up   = weight_kg * 0.006* max(0.0, dplus)
    cal_down = weight_kg * 0.003* max(0.0, dneg)
    cal_tot  = int(round(cal_flat + cal_up + cal_down))

    surge_per_km = (surge_transitions / max(0.001, tot_km))

    df = pd.DataFrame({"km":[c/1000.0 for c in cum], "elev_m": ele_raw})

    return {
        "tot_km": round(tot_km, 2),
        "dplus": round(dplus, 0),
        "dneg": round(dneg, 0),
        "t_dist": t_dist, "t_up": t_up, "t_down": t_down, "t_total": t_total,
        "holes": holes,
        "len_flat_km": round(flat_len/1000.0, 2),
        "len_up_km":   round(asc_len /1000.0, 2),
        "len_down_km": round(desc_len/1000.0, 2),
        "grade_up_pct":   round(grade_up_pct, 1),
        "grade_down_pct": round(grade_down_pct, 1),
        "cal_total": cal_tot,
        "cal_flat":  int(round(cal_flat)),
        "cal_up":    int(round(cal_up)),
        "cal_down":  int(round(cal_down)),
        "asc_bins_m":  asc_bins,
        "desc_bins_m": desc_bins,
        "lcs25_m": int(round(longest_steep_run)),
        "blocks25_count": int(blocks25_count),
        "surge_idx_per_km": round(surge_per_km, 2),
        "avg_alt_m": sum(ele_raw)/len(ele_raw) if ele_raw else None,
        "df_profile": df
    }

def compute_if_from_res(res: dict,
                        temp_c: float, humidity_pct: float, precip: str, surface: str,
                        wind_kmh: float, exposure: str,
                        technique_level: str, extra_load_kg: float):
    D_km = float(res["tot_km"]); Dp = float(res["dplus"])
    ascL_m = 1000.0 * float(res["len_up_km"]); descL_m = 1000.0 * float(res["len_down_km"])
    C = (Dp / max(0.001, D_km))
    asc_bins = res["asc_bins_m"]; desc_bins = res["desc_bins_m"]

    def frac_ge25(bins_m, total_len_m):
        if total_len_m <= 0: return 0.0
        m_20_30, m_30_40, m_over40 = bins_m[2], bins_m[3], bins_m[4]
        approx = 0.5*m_20_30 + m_30_40 + m_over40
        return min(1.0, max(0.0, approx / total_len_m))

    f_up25   = frac_ge25(asc_bins,  ascL_m)
    f_down25 = frac_ge25(desc_bins, descL_m)

    lcs = float(res["lcs25_m"]); blocks = float(res["blocks25_count"]); surge = float(res["surge_idx_per_km"])
    lcs_scaled = lcs / 200.0

    S = (W_D*D_km + W_PLUS*(Dp/100.0) + W_COMP*(C/100.0) +
         W_STEEP*(100.0*f_up25) + W_STEEP_D*(100.0*f_down25) +
         W_LCS*lcs_scaled + W_BLOCKS*blocks + W_SURGE*surge)

    IF_base = 100.0 * (1.0 - math.exp(-S / max(1e-6, IF_S0)))

    # Mappature IT -> codici
    precip_map = {"assenza pioggia":"dry","pioviggine":"drizzle","pioggia":"rain","pioggia forte":"heavy_rain","neve fresca":"snow_shallow","neve profonda":"snow_deep"}
    surf_map   = {"asciutto":"dry","fango":"mud","roccia bagnata":"wet_rock","neve dura":"hard_snow","ghiaccio":"ice"}
    expo_map   = {"ombra":"shade","misto":"mixed","pieno sole":"sun"}

    M_meteo = meteo_multiplier(temp_c, humidity_pct,
                               precip_map.get(precip, "dry"),
                               surf_map.get(surface, "dry"),
                               wind_kmh, expo_map.get(exposure, "mixed"))
    M_alt   = altitude_multiplier(res.get("avg_alt_m"))
    M_tech  = technique_multiplier(technique_level)
    M_load  = pack_load_multiplier(extra_load_kg)
    M_tot   = M_meteo * M_alt * M_tech * M_load

    bump = (100.0 - IF_base) * max(0.0, (M_tot - 1.0)) * ALPHA_METEO
    IF = min(100.0, IF_base + bump)

    return {"IF": round(IF,1), "IF_base": round(IF_base,1),
            "M_meteo": round(M_meteo,2), "M_alt": round(M_alt,2),
            "M_tech": round(M_tech,2), "M_load": round(M_load,2),
            "cat": cat_from_if(round(IF,1))}

# ---------------- Build PDF (usa backend disponibile) ----------------
def build_pdf(res: dict, fi: dict, params: dict, conds: dict, gpx_name: str) -> bytes:
    if PDF_BACKEND == "reportlab":
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        W, H = A4; M = 36; x = M; y = H - M

        c.setFont("Helvetica-Bold", 16)
        c.drawString(x, y, _safe("Tempo percorrenza sentiero — Export PDF"))
        c.setFont("Helvetica", 10)
        c.drawString(x, y-14, _safe(f"Versione {APP_VER} — File: {gpx_name or 'GPX'}"))
        y -= 28

        c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Risultati principali")
        y -= 14; c.setFont("Helvetica", 10)
        lines = [
            f"Distanza: {res['tot_km']} km",
            f"Dislivello +: {int(res['dplus'])} m   Dislivello -: {int(res['dneg'])} m",
            f"Tempo totale: {fmt_hm(res['t_total'])}",
            f"  • Piano: {fmt_hm(res['t_dist'])}   • Salita: {fmt_hm(res['t_up'])}   • Discesa: {fmt_hm(res['t_down'])}",
            f"Calorie stimate: {res['cal_total']} kcal",
            f"Buchi GPX: {int(res['holes'])}",
            f"Lunghezze — Piano: {res['len_flat_km']:.2f} km, Salita: {res['len_up_km']:.2f} km, Discesa: {res['len_down_km']:.2f} km",
            f"Pend. media — Salita: {res['grade_up_pct']:.1f} %, Discesa: {res['grade_down_pct']:.1f} %",
        ]
        for L in lines: c.drawString(x, y, _safe(L)); y -= 12

        ab = [int(round(v)) for v in res["asc_bins_m"]]
        db = [int(round(v)) for v in res["desc_bins_m"]]
        y -= 4
        c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Fasce di pendenza (metri)")
        y -= 14; c.setFont("Helvetica", 10)
        c.drawString(x, y, _safe(f"Salita: <10% {ab[0]} m — 10–20% {ab[1]} m — 20–30% {ab[2]} m — 30–40% {ab[3]} m — >40% {ab[4]} m")); y -= 12
        c.drawString(x, y, _safe(f"Discesa: <10% {db[0]} m — 10–20% {db[1]} m — 20–30% {db[2]} m — 30–40% {db[3]} m — >40% {db[4]} m")); y -= 18

        c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Indice di fatica")
        y -= 14; c.setFont("Helvetica", 10)
        c.drawString(x, y, _safe(f"IF: {fi['IF']} ({fi['cat']}) — IF base: {fi['IF_base']}"))
        y -= 12
        c.drawString(x, y, _safe(f"Meteo: {fi['M_meteo']}  • Alt: {fi['M_alt']}  • Tec: {fi['M_tech']}  • Zaino: {fi['M_load']}"))
        y -= 18

        c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Parametri e condizioni")
        y -= 14; c.setFont("Helvetica", 10)
        p1 = f"Min/km (piano): {params['base']} — Min/100m salita: {params['up']} — Min/200m discesa: {params['down']} — Peso: {params['weight']} kg"
        p2 = f"Inverti traccia: {'sì' if params['reverse'] else 'no'} — Zaino: {conds['loadkg']} kg — Tecnica: {conds['tech_it']}"
        p3 = f"Met: T={conds['temp']} °C  U={conds['hum']} %  Vento={conds['wind']} km/h — Prec: {conds['precip_it']} — Fondo: {conds['surface_it']} — Esposizione: {conds['expo_it']}"
        for L in (p1, p2, p3): c.drawString(x, y, _safe(L)); y -= 12

        # Profilo altimetrico semplice
        y -= 12
        c.setFont("Helvetica-Bold", 12); c.drawString(x, y, "Profilo altimetrico")
        y -= 6
        chart_h = 200
        chart_w = A4[0] - 2*M
        box_y = y - chart_h
        c.setStrokeColor(colors.black)
        c.rect(x, box_y, chart_w, chart_h, stroke=1, fill=0)

        df = res["df_profile"]
        if len(df) >= 2:
            xs = df["km"].tolist(); ys = df["elev_m"].tolist()
            y_min, y_max = min(ys), max(ys)
            if y_max - y_min < 1: y_max = y_min + 1
            x0, x1 = xs[0], xs[-1]
            if x1 - x0 < 1e-6: x1 = x0 + 1e-6
            def X(u): return x + ((u - x0)/(x1 - x0))*chart_w
            def Y(v): return box_y + ((v - y_min)/(y_max - y_min))*chart_h
            # griglia km
            c.setStrokeColor(colors.lightgrey)
            for k in range(int(math.floor(x0)), int(math.ceil(x1))+1):
                gx = X(k); c.line(gx, box_y, gx, box_y+chart_h)
            # traccia
            c.setStrokeColor(colors.darkblue); c.setLineWidth(1)
            px, py = X(xs[0]), Y(ys[0])
            for u, v in zip(xs[1:], ys[1:]):
                nx, ny = X(u), Y(v); c.line(px, py, nx, ny); px, py = nx, ny
            c.setStrokeColor(colors.black); c.setLineWidth(0.5)
            c.line(x, box_y, x+chart_w, box_y); c.line(x, box_y, x, box_y+chart_h)

        c.showPage(); c.save()
        return buf.getvalue()

    if PDF_BACKEND == "fpdf2":
        pdf = FPDF(unit="pt", format="A4")
        pdf.add_page()
        W, H = 595.27, 841.89; M = 36; x = M; y = H - M

        pdf.set_font("Helvetica", "B", 16)
        pdf.text(x, y, _safe("Tempo percorrenza sentiero — Export PDF"))
        pdf.set_font("Helvetica", "", 10)
        pdf.text(x, y-14, _safe(f"Versione {APP_VER} — File: {gpx_name or 'GPX'}"))
        y -= 28

        pdf.set_font("Helvetica", "B", 12); pdf.text(x, y, "Risultati principali"); y -= 14
        pdf.set_font("Helvetica", "", 10)
        lines = [
            f"Distanza: {res['tot_km']} km",
            f"Dislivello +: {int(res['dplus'])} m   Dislivello -: {int(res['dneg'])} m",
            f"Tempo totale: {fmt_hm(res['t_total'])}",
            f"  • Piano: {fmt_hm(res['t_dist'])}   • Salita: {fmt_hm(res['t_up'])}   • Discesa: {fmt_hm(res['t_down'])}",
            f"Calorie stimate: {res['cal_total']} kcal",
            f"Buchi GPX: {int(res['holes'])}",
            f"Lunghezze — Piano: {res['len_flat_km']:.2f} km, Salita: {res['len_up_km']:.2f} km, Discesa: {res['len_down_km']:.2f} km",
            f"Pend. media — Salita: {res['grade_up_pct']:.1f} %, Discesa: {res['grade_down_pct']:.1f} %",
        ]
        for L in lines: pdf.text(x, y, _safe(L)); y -= 12

        ab = [int(round(v)) for v in res["asc_bins_m"]]
        db = [int(round(v)) for v in res["desc_bins_m"]]
        y -= 4
        pdf.set_font("Helvetica", "B", 12); pdf.text(x, y, "Fasce di pendenza (metri)"); y -= 14
        pdf.set_font("Helvetica", "", 10)
        pdf.text(x, y, _safe(f"Salita: <10% {ab[0]} m — 10–20% {ab[1]} m — 20–30% {ab[2]} m — 30–40% {ab[3]} m — >40% {ab[4]} m")); y -= 12
        pdf.text(x, y, _safe(f"Discesa: <10% {db[0]} m — 10–20% {db[1]} m — 20–30% {db[2]} m — 30–40% {db[3]} m — >40% {db[4]} m")); y -= 18

        pdf.set_font("Helvetica", "B", 12); pdf.text(x, y, "Indice di fatica"); y -= 14
        pdf.set_font("Helvetica", "", 10)
        pdf.text(x, y, _safe(f"IF: {fi['IF']} ({fi['cat']}) — IF base: {fi['IF_base']}")); y -= 12
        pdf.text(x, y, _safe(f"Meteo: {fi['M_meteo']}  • Alt: {fi['M_alt']}  • Tec: {fi['M_tech']}  • Zaino: {fi['M_load']}")); y -= 18

        pdf.set_font("Helvetica", "B", 12); pdf.text(x, y, "Parametri e condizioni"); y -= 14
        pdf.set_font("Helvetica", "", 10)
        p1 = f"Min/km (piano): {params['base']} — Min/100m salita: {params['up']} — Min/200m discesa: {params['down']} — Peso: {params['weight']} kg"
        p2 = f"Inverti traccia: {'sì' if params['reverse'] else 'no'} — Zaino: {conds['loadkg']} kg — Tecnica: {conds['tech_it']}"
        p3 = f"Met: T={conds['temp']} °C  U={conds['hum']} %  Vento={conds['wind']} km/h — Prec: {conds['precip_it']} — Fondo: {conds['surface_it']} — Esposizione: {conds['expo_it']}"
        for L in (p1, p2, p3): pdf.text(x, y, _safe(L)); y -= 12

        # Profilo
        y -= 12
        pdf.set_font("Helvetica", "B", 12); pdf.text(x, y, "Profilo altimetrico"); y -= 6
        chart_h = 200; chart_w = W - 2*M; box_y = y - chart_h
        pdf.rect(x, box_y, chart_w, chart_h)

        df = res["df_profile"]
        if len(df) >= 2:
            xs = df["km"].tolist(); ys = df["elev_m"].tolist()
            y_min, y_max = min(ys), max(ys); 
            if y_max - y_min < 1: y_max = y_min + 1
            x0, x1 = xs[0], xs[-1]; 
            if x1 - x0 < 1e-6: x1 = x0 + 1e-6
            def X(u): return x + ((u - x0)/(x1 - x0))*chart_w
            def Y(v): return box_y + ((v - y_min)/(y_max - y_min))*chart_h
            # griglia km
            km0, km1 = int(math.floor(x0)), int(math.ceil(x1))
            for k in range(km0, km1+1):
                gx = X(k); pdf.line(gx, box_y, gx, box_y+chart_h)
            # traccia
            px, py = X(xs[0]), Y(ys[0])
            for u, v in zip(xs[1:], ys[1:]):
                nx, ny = X(u), Y(v); pdf.line(px, py, nx, ny); px, py = nx, ny

        return bytes(pdf.output(dest="S"))

    raise RuntimeError("Nessun backend PDF disponibile (reportlab/fpdf2).")

# ===================== UI Streamlit =====================
st.set_page_config(page_title=APP_TITLE, page_icon="🗺️", layout="wide")
st.title(f"{APP_TITLE} — {APP_VER}")

ensure_defaults()

# Barra superiore
top = st.container()
with top:
    c1, c2, c3 = st.columns([4, 1.2, 1])
    gpx = c1.file_uploader("Carica GPX", type=["gpx"], key="gpx_file")
    _recalc = c2.button("Calcola", use_container_width=True)

    file_bytes = None
    gpx_name = None
    if gpx is not None:
        file_bytes = gpx.getvalue()
        gpx_name = gpx.name
        sig = f"{gpx.name}|{len(file_bytes)}"
        if st.session_state.get("last_gpx_sig") != sig:
            st.session_state["last_gpx_sig"] = sig
            # reset parametri ai default
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
    if gpx is not None:
        c3.caption(f"Selezionato: {gpx.name}")

# Sidebar parametri
with st.sidebar:
    st.header("Impostazioni")
    base    = st.number_input("Min/km (piano)",     min_value=1.0, step=0.5, key="base")
    up      = st.number_input("Min/100 m (salita)", min_value=1.0, step=0.5, key="up")
    down    = st.number_input("Min/200 m (discesa)",min_value=1.0, step=0.5, key="down")
    weight  = st.number_input("Peso (kg)", min_value=30.0, step=1.0, key="weight")
    reverse = st.checkbox("Inverti traccia", key="reverse")

    st.markdown("---")
    st.subheader("Condizioni")
    temp  = st.number_input("Temperatura (°C)", step=1.0, key="temp")
    hum   = st.number_input("Umidità (%)", step=1.0, min_value=0.0, max_value=100.0, key="hum")
    wind  = st.number_input("Vento (km/h)", step=1.0, min_value=0.0, key="wind")
    precip_it = st.selectbox("Precipitazioni",
        ["assenza pioggia","pioviggine","pioggia","pioggia forte","neve fresca","neve profonda"],
        key="precip_sel")
    surface_it = st.selectbox("Fondo",
        ["asciutto","fango","roccia bagnata","neve dura","ghiaccio"],
        key="surface_sel")
    expo_it = st.selectbox("Esposizione",
        ["ombra","misto","pieno sole"],
        key="expo_sel")
    tech_it = st.selectbox("Tecnica",
        ["facile","normale","roccioso","scrambling","neve/ghiaccio"],
        key="tech_sel")
    loadkg = st.number_input("Zaino extra (kg)", step=1.0, min_value=0.0, key="loadkg")

colL, colR = st.columns([1.15, 1])

if not gpx:
    st.info("Carica un file GPX per iniziare.")
else:
    try:
        res = compute_from_gpx_bytes(file_bytes, base, up, down, weight, reverse=reverse)
    except Exception as e:
        st.error(str(e))
    else:
        with colL:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Distanza (km)", res["tot_km"])
            m2.metric("Dislivello + (m)", int(res["dplus"]))
            m3.metric("Dislivello − (m)", int(res["dneg"]))
            m4.metric("Tempo totale", fmt_hm(res["t_total"]))

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Tempo piano:** {fmt_hm(res['t_dist'])}")
            c2.markdown(f"**Tempo salita:** {fmt_hm(res['t_up'])}")
            c3.markdown(f"**Tempo discesa:** {fmt_hm(res['t_down'])}")

            holes = int(res['holes'])
            st.markdown("**Buchi GPX:** <span style='color:{}'>{}</span>".format("red" if holes>0 else "#0b0", holes), unsafe_allow_html=True)

            c4, c5, c6 = st.columns(3)
            c4.markdown(f"**Piano:** {res['len_flat_km']:.2f} km")
            c5.markdown(f"**Salita:** {res['len_up_km']:.2f} km")
            c6.markdown(f"**Discesa:** {res['len_down_km']:.2f} km")

            c7, c8, c9 = st.columns(3)
            c7.markdown(f"**Pend. media salita:** {res['grade_up_pct']:.1f} %")
            c8.markdown(f"**Pend. media discesa:** {res['grade_down_pct']:.1f} %")
            c9.markdown(f"**Calorie stimate:** {res['cal_total']} kcal")

            # fasce pendenza (metri)
            ab = [int(round(x)) for x in res["asc_bins_m"]]
            db = [int(round(x)) for x in res["desc_bins_m"]]
            st.markdown("**Fasce pendenza (metri)**")
            st.markdown(
                f"**Salita:** &lt;10%: {ab[0]} m · 10–20%: {ab[1]} m · 20–30%: {ab[2]} m · 30–40%: {ab[3]} m · &gt;40%: {ab[4]} m",
                unsafe_allow_html=True
            )
            st.markdown(
                f"**Discesa:** &lt;10%: {db[0]} m · 10–20%: {db[1]} m · 20–30%: {db[2]} m · 30–40%: {db[3]} m · &gt;40%: {db[4]} m",
                unsafe_allow_html=True
            )

            # Profilo altimetrico
            dfp = res["df_profile"]
            chart = (
                alt.Chart(dfp)
                .mark_line()
                .encode(
                    x=alt.X("km:Q", axis=alt.Axis(title="Distanza (km)", grid=True)),
                    y=alt.Y("elev_m:Q", axis=alt.Axis(title="Quota (m)", grid=True)),
                    tooltip=[alt.Tooltip("km:Q", format=".2f", title="km"),
                             alt.Tooltip("elev_m:Q", format=".0f", title="m")]
                )
                .properties(height=380)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

            params = dict(base=base, up=up, down=down, weight=weight, reverse=reverse)
            conds  = dict(temp=temp, hum=hum, wind=wind,
                          precip_it=precip_it, surface_it=surface_it, expo_it=expo_it,
                          tech_it=tech_it, loadkg=loadkg)

        with colR:
            st.subheader("Indice di Fatica")
            fi = compute_if_from_res(
                res,
                temp_c=float(temp), humidity_pct=float(hum),
                precip=precip_it, surface=surface_it,
                wind_kmh=float(wind), exposure=expo_it,
                technique_level=tech_it, extra_load_kg=float(loadkg)
            )
            st.metric("Indice di Fatica", f"{fi['IF']}  ({fi['cat']})")
            st.caption(f"IF base: {fi['IF_base']}  ·  Meteo: {fi['M_meteo']}  ·  Alt: {fi['M_alt']}  ·  Tec: {fi['M_tech']}  ·  Zaino: {fi['M_load']}")
            st.caption(f"LCS>=25: {res['lcs25_m']} m · Blocchi>=25: {res['blocks25_count']} · Surge: {res['surge_idx_per_km']}/km")

            # Download CSV profilo
            csv = res["df_profile"].to_csv(index=False).encode("utf-8")
            st.download_button("Scarica profilo (CSV)", csv, file_name="profilo_gpx.csv", mime="text/csv")

            # Stampa / Salva come PDF tramite il browser
st.caption("Suggerimento: per stampare o salvare come PDF usa il comando del browser (Ctrl+P / ⌘P).")
