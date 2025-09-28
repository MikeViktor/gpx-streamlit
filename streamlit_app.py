# streamlit_app.py
import math, xml.etree.ElementTree as ET
import streamlit as st
import pandas as pd

# ---- util & core (versione compatta) ----
def _is_tag(e, name): t=e.tag; return t.endswith('}'+name) or t==name
def dist_km(lat1,lon1,lat2,lon2):
    dy=(lat2-lat1)*111.32
    dx=(lon2-lon1)*111.32*math.cos(math.radians((lat1+lat2)/2))
    return math.hypot(dx,dy)

def parse_gpx(bytes_io):
    root = ET.fromstring(bytes_io.read())
    for wanted in ("trkpt","rtept","wpt"):
        lat,lon,ele=[],[],[]
        for el in root.iter():
            if _is_tag(el,wanted):
                la=el.attrib.get("lat"); lo=el.attrib.get("lon")
                if la is None or lo is None: continue
                z=None
                for ch in el:
                    if _is_tag(ch,"ele"): z=ch.text; break
                if z is None: continue
                try:
                    lat.append(float(la)); lon.append(float(lo)); ele.append(float(z))
                except: pass
        if lat: return lat,lon,ele
    return [],[],[]

def resample_elev(cum_m, ele, step_m=3.0):
    out=[]; t=0.0; j=0; total=cum_m[-1]
    n=int(total//step_m)+1
    for _ in range(n):
        while j<len(cum_m)-1 and cum_m[j+1]<t: j+=1
        if t<=cum_m[0]: out.append(ele[0])
        elif t>=cum_m[-1]: out.append(ele[-1])
        else:
            u=(t-cum_m[j])/(cum_m[j+1]-cum_m[j])
            out.append(ele[j]+u*(ele[j+1]-ele[j]))
        t+=step_m
    return out

def moving_avg(seq,k=3):
    if k<1: k=1
    half=k//2; out=[]; n=len(seq)
    for i in range(n):
        a=max(0,i-half); b=min(n,i+half+1)
        out.append(sum(seq[a:b])/max(1,b-a))
    return out

def compute_from_gpx_bytes(f, base=15.0, up=15.0, down=15.0, reverse=False):
    lat,lon,ele = parse_gpx(f)
    if not ele: raise ValueError("Nessun punto utile con elevazione nel GPX.")
    if reverse: lat,lon,ele = list(reversed(lat)),list(reversed(lon)),list(reversed(ele))

    cum=[0.0]
    for i in range(1,len(lat)):
        cum.append(cum[-1]+dist_km(lat[i-1],lon[i-1],lat[i],lon[i])*1000.0)
    tot_km=cum[-1]/1000.0

    e_res=resample_elev(cum,ele,3.0)
    e_sm =moving_avg(e_res,3)

    dplus=dneg=0.0
    for i in range(1,len(e_sm)):
        d=e_sm[i]-e_sm[i-1]
        if   d>0.25: dplus+=d
        elif d<-0.25:dneg+=(-d)

    t_dist=tot_km*base
    t_up=(dplus/100.0)*up
    t_down=(dneg/200.0)*down
    t_tot=t_dist+t_up+t_down

    holes=sum(1 for i in range(1,len(ele)) if abs(ele[i]-ele[i-1])>=100.0)

    df=pd.DataFrame({
        "km":[c/1000.0 for c in cum],
        "ele":ele
    })
    return {
        "tot_km":round(tot_km,2), "dplus":round(dplus,0), "dneg":round(dneg,0),
        "t_dist":t_dist, "t_up":t_up, "t_down":t_down, "t_tot":t_tot,
        "holes":holes, "df":df
    }

def fmt_hm(m):
    h=int(m//60); mm=int(round(m-h*60))
    if mm==60: h+=1; mm=0
    return f"{h}:{mm:02d}"

# ---- UI ----
st.set_page_config(page_title="Tempo percorrenza sentiero", layout="wide")
st.title("Tempo percorrenza sentiero (web)")

with st.sidebar:
    st.header("Impostazioni")
    base = st.number_input("Min/km (piano)",  min_value=1.0, value=15.0, step=0.5)
    up   = st.number_input("Min/100 m (salita)", min_value=1.0, value=15.0, step=0.5)
    down = st.number_input("Min/200 m (discesa)",min_value=1.0, value=15.0, step=0.5)
    reverse = st.checkbox("Inverti traccia", value=False)
    st.markdown("---")
    gpx = st.file_uploader("Carica GPX", type=["gpx"])

col1,col2 = st.columns([1,1])

if gpx is None:
    st.info("Carica un file GPX per iniziare.")
else:
    try:
        res = compute_from_gpx_bytes(gpx, base, up, down, reverse)
    except Exception as e:
        st.error(str(e))
    else:
        with col1:
            st.metric("Distanza (km)", res["tot_km"])
            st.metric("Dislivello + (m)", int(res["dplus"]))
            st.metric("Dislivello − (m)", int(res["dneg"]))
            st.metric("Tempo totale", fmt_hm(res["t_tot"]))
            st.caption(f"Piano: {fmt_hm(res['t_dist'])} — Salita: {fmt_hm(res['t_up'])} — Discesa: {fmt_hm(res['t_down'])}")
            st.warning(f"Buchi GPX: {res['holes']}" if res["holes"]>0 else "Buchi GPX: 0")

        with col2:
            st.line_chart(res["df"].set_index("km"), y="ele", height=380)
