"""
================================================================================
FLIGHTOPS ANALYTICS — PLATAFORMA DE OPERACIONES Y PREDICCION DE TRAFICO AEREO
================================================================================
Sistema de monitoreo operativo, analisis causal FAA y prediccion de disrupciones
para vuelos comerciales en Estados Unidos.

Ejecución soportada tanto con:
    streamlit run app.py
como directamente desde el botón "Run" del IDE:
    python app.py
================================================================================
"""
import os
import sys
import time
import socket
import subprocess
import webbrowser
from pathlib import Path

from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx


def _is_port_open(host: str = "127.0.0.1", port: int = 8501) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.8)
        return sock.connect_ex((host, port)) == 0


if (
    __name__ == "__main__"
    and get_script_run_ctx(suppress_warning=True) is None
    and not os.environ.get("FLIGHTOPS_TEST_MODE")
):
    target_port = 8501
    target_url = f"http://localhost:{target_port}"
    project_dir = Path(__file__).resolve().parent
    st_creds = Path.home() / ".streamlit" / "credentials.toml"
    if not st_creds.exists():
        try:
            st_creds.parent.mkdir(parents=True, exist_ok=True)
            st_creds.write_text('[general]\nemail = ""\n', encoding="utf-8")
        except Exception:
            pass

    if not _is_port_open(port=target_port):
        creation_flags = 0
        if os.name == "nt":
            creation_flags = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(Path(__file__).resolve()),
                "--server.port",
                str(target_port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=str(project_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        for _ in range(45):
            if _is_port_open(port=target_port):
                break
            time.sleep(0.4)

    webbrowser.open(target_url)
    print("=" * 72)
    print("FLIGHTOPS ANALYTICS - SERVIDOR WEB ACTIVO")
    print(f"URL de acceso: {target_url}")
    print("Se ha abierto la aplicacion en su navegador predeterminado.")
    print("=" * 72)
    sys.exit(0)

import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import config

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS
# ==============================================================================
st.set_page_config(
    page_title="FlightOps Analytics | Plataforma de Operaciones Aéreas",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1480px;
    }

    /* Banner Principal */
    .hero-banner {
        background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #1e293b 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 1.6rem 2.2rem;
        border-radius: 16px;
        color: #f8fafc;
        margin-bottom: 1.4rem;
        box-shadow: 0 12px 30px -8px rgba(2, 6, 23, 0.45);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(14, 165, 233, 0.18);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        font-size: 0.73rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.22rem 0.75rem;
        border-radius: 999px;
        margin-bottom: 0.55rem;
    }
    .hero-banner h1 {
        margin: 0;
        font-size: 1.95rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
    }
    .hero-banner p {
        margin: 0.4rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.96rem;
        line-height: 1.5;
    }

    /* Tarjetas KPI */
    .kpi-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.15rem 1.25rem;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #0284c7;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        height: 100%;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.35rem;
    }
    .kpi-value {
        font-size: 1.68rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
    }
    .kpi-sub {
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    .sub-blue { color: #0284c7; }
    .sub-green { color: #059669; }
    .sub-amber { color: #d97706; }
    .sub-red { color: #e11d48; }

    /* Encabezados de sección */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.5rem 0 0.75rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Caja de Insight Científico */
    .insight-card {
        background: linear-gradient(90deg, #f0f9ff 0%, #f8fafc 100%);
        border-left: 4px solid #0284c7;
        padding: 0.95rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin: 0.65rem 0 1rem 0;
        font-size: 0.9rem;
        color: #1e293b;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. CARGA DE DATOS EN CACHÉ DESDE LA CAPA ORO Y MODELO ML
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_gold_datasets():
    """Carga los datasets analíticos de la Capa Oro."""
    data = {}
    if config.GOLD_DAILY_AIRPORT.exists():
        df_d = pd.read_parquet(config.GOLD_DAILY_AIRPORT)
        df_d["FlightDate"] = pd.to_datetime(df_d["FlightDate"])
        df_d["Month"] = df_d["FlightDate"].dt.month
        df_d["DayOfWeek"] = df_d["FlightDate"].dt.day_of_week + 1
        data["daily"] = df_d
    else:
        data["daily"] = None

    data["airports"] = (
        pd.read_parquet(config.GOLD_AIRPORT_SUMMARY)
        if config.GOLD_AIRPORT_SUMMARY.exists()
        else None
    )
    data["airlines"] = (
        pd.read_parquet(config.GOLD_AIRLINE_MONTHLY)
        if config.GOLD_AIRLINE_MONTHLY.exists()
        else None
    )
    data["routes"] = (
        pd.read_parquet(config.GOLD_ROUTES)
        if config.GOLD_ROUTES.exists()
        else None
    )
    data["causes"] = (
        pd.read_parquet(config.GOLD_DELAY_CAUSES)
        if config.GOLD_DELAY_CAUSES.exists()
        else None
    )
    return data


@st.cache_resource(show_spinner=False)
def load_ml_artifact():
    """Carga el pipeline entrenado y diccionarios de referencia desde models/."""
    model_path = config.MODELS_DIR / "flight_delay_predictor.joblib"
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None


def build_single_flight_features(
    dep_airport: str,
    arr_airport: str,
    airline: str,
    month: int,
    day_of_week: int,
    dep_time_label: str,
    aircraft_age: float,
    flight_duration: float,
    tavg: float,
    tmin: float,
    tmax: float,
    prcp: float,
    snow: float,
    wspd: float,
    pres: float,
    artifact: dict,
    tail_prev_delay_min: float = 0.0,
    hub_prev_disrupt_pct: float = -1.0,
) -> pd.DataFrame:
    """Construye el DataFrame exacto de 70 características causales para inferencia en vivo."""
    airport_defs = artifact.get("airport_defaults", {})
    airline_defs = artifact.get("airline_defaults", {})

    dep_info = airport_defs.get(dep_airport, {})
    arr_info = airport_defs.get(arr_airport, {})
    air_info = airline_defs.get(airline, {})

    arr_tavg = float(arr_info.get("arr_tavg", 16.5))
    arr_tmin = float(arr_info.get("arr_tmin", 11.0))
    arr_tmax = float(arr_info.get("arr_tmax", 22.0))
    arr_prcp = float(arr_info.get("arr_prcp", 0.0))
    arr_snow = float(arr_info.get("arr_snow", 0.0))
    arr_wspd = float(arr_info.get("arr_wspd", 13.0))
    arr_pres = float(arr_info.get("arr_pres", 1015.0))

    dep_slot_flights = int(dep_info.get("dep_slot_flights", 65))
    arr_slot_flights = int(arr_info.get("dep_slot_flights", 65))
    dep_daily_flights = int(dep_info.get("dep_daily_flights", 220))
    aircraft_daily_legs = int(air_info.get("aircraft_daily_legs", 4))

    manufacturer = str(air_info.get("Manufacturer", "BOEING")).upper()
    model_name = str(air_info.get("Model", "737 NG")).upper()

    if flight_duration < 180:
        dist_type = "SHORT HAUL (<1500MI)"
    elif flight_duration < 300:
        dist_type = "MEDIUM HAUL (1500-3000MI)"
    else:
        dist_type = "LONG HAUL (>3000MI)"

    slot_order_map = {"MORNING": 1, "AFTERNOON": 2, "EVENING": 3, "NIGHT": 4}
    dep_time_clean = dep_time_label.strip().upper()
    slot_order = slot_order_map.get(dep_time_clean, 2)
    is_weekend = 1 if day_of_week in (6, 7) else 0

    # Estado causal rezagado (s-1) del Hub y de la Aeronave
    default_hub_rate = float(dep_info.get("hub_dep_prev_disrupt_rate", 0.18))
    hub_dep_prev_disrupt_rate = (
        float(hub_prev_disrupt_pct / 100.0)
        if hub_prev_disrupt_pct >= 0.0
        else default_hub_rate
    )
    scale_hub = hub_dep_prev_disrupt_rate / max(0.05, default_hub_rate)
    hub_dep_prev_avg_delay = float(dep_info.get("hub_dep_prev_avg_delay", 9.5)) * scale_hub
    hub_dep_prev_cancel_rate = float(dep_info.get("hub_dep_prev_cancel_rate", 0.01)) * scale_hub
    hub_dep_prev_wx_delay = float(dep_info.get("hub_dep_prev_wx_delay", 0.5)) * scale_hub
    hub_dep_prev_nas_delay = float(dep_info.get("hub_dep_prev_nas_delay", 1.8)) * scale_hub
    hub_dep_prev_late_ac_delay = float(dep_info.get("hub_dep_prev_late_ac_delay", 3.5)) * scale_hub
    hub_dep_roll3_disrupt_rate = float(dep_info.get("hub_dep_roll3_disrupt_rate", 0.19)) * (0.5 + 0.5 * scale_hub)

    hub_arr_prev_disrupt_rate = float(arr_info.get("hub_dep_prev_disrupt_rate", 0.18))
    hub_arr_prev_avg_delay = float(arr_info.get("hub_dep_prev_avg_delay", 9.5))
    hub_arr_prev_nas_delay = float(arr_info.get("hub_dep_prev_nas_delay", 1.8))

    air_sys_prev_disrupt_rate = float(air_info.get("air_sys_prev_disrupt_rate", 0.19))
    air_sys_prev_avg_delay = float(air_info.get("air_sys_prev_avg_delay", 10.0))
    air_dep_prev_disrupt_rate = 0.5 * (hub_dep_prev_disrupt_rate + air_sys_prev_disrupt_rate)
    air_dep_prev_avg_delay = 0.5 * (hub_dep_prev_avg_delay + air_sys_prev_avg_delay)

    has_sameday_prev = 0 if slot_order == 1 and tail_prev_delay_min == 0.0 else 1
    tail_disrupted = 1.0 if tail_prev_delay_min >= 15.0 else 0.0
    tail_roll3 = 0.67 if tail_disrupted >= 1.0 else (0.15 if has_sameday_prev else 0.10)

    deicing_risk = 1 if (tmin <= 0.0 and (prcp > 0.0 or snow > 0.0)) else 0
    temp_swing = float(tmax - tmin)
    total_route_prcp = float(prcp + arr_prcp)
    total_route_snow = float(snow + arr_snow)
    max_route_wind = float(max(wspd, arr_wspd))
    pres_drop = float(max(0.0, 1013.25 - pres))
    arr_pres_drop = float(max(0.0, 1013.25 - arr_pres))

    sin_month = float(np.sin(2 * np.pi * month / 12.0))
    cos_month = float(np.cos(2 * np.pi * month / 12.0))
    sin_dow = float(np.sin(2 * np.pi * day_of_week / 7.0))
    cos_dow = float(np.cos(2 * np.pi * day_of_week / 7.0))

    row = {
        "dep_airport": dep_airport,
        "arr_airport": arr_airport,
        "airline": airline,
        "Model": model_name,
        "route": f"{dep_airport}_{arr_airport}",
        "dep_x_time": f"{dep_airport}_{dep_time_clean}",
        "arr_x_time": f"{arr_airport}_{dep_time_clean}",
        "airline_x_time": f"{airline}_{dep_time_clean}",
        "dep_x_month": f"{dep_airport}_{month}",
        "airline_x_dep": f"{airline}_{dep_airport}",
        "DepTime_label": dep_time_clean,
        "Distance_type": dist_type,
        "Manufacturer": manufacturer,
        "hub_dep_prev_disrupt_rate": hub_dep_prev_disrupt_rate,
        "hub_dep_prev_avg_delay": hub_dep_prev_avg_delay,
        "hub_dep_prev_cancel_rate": hub_dep_prev_cancel_rate,
        "hub_dep_prev_wx_delay": hub_dep_prev_wx_delay,
        "hub_dep_prev_nas_delay": hub_dep_prev_nas_delay,
        "hub_dep_prev_late_ac_delay": hub_dep_prev_late_ac_delay,
        "hub_dep_roll3_disrupt_rate": hub_dep_roll3_disrupt_rate,
        "hub_arr_prev_disrupt_rate": hub_arr_prev_disrupt_rate,
        "hub_arr_prev_avg_delay": hub_arr_prev_avg_delay,
        "hub_arr_prev_nas_delay": hub_arr_prev_nas_delay,
        "air_dep_prev_disrupt_rate": air_dep_prev_disrupt_rate,
        "air_dep_prev_avg_delay": air_dep_prev_avg_delay,
        "air_sys_prev_disrupt_rate": air_sys_prev_disrupt_rate,
        "air_sys_prev_avg_delay": air_sys_prev_avg_delay,
        "tail_sameday_prev_dep_delay": float(tail_prev_delay_min),
        "tail_sameday_prev_arr_delay": float(tail_prev_delay_min),
        "tail_sameday_prev_disrupted": float(tail_disrupted),
        "tail_has_sameday_prev": int(has_sameday_prev),
        "tail_prev_dep_delay": float(tail_prev_delay_min),
        "tail_prev_arr_delay": float(tail_prev_delay_min),
        "tail_prev_disrupted": float(tail_disrupted),
        "tail_roll3_disrupted": float(tail_roll3),
        "tavg": float(tavg),
        "tmin": float(tmin),
        "tmax": float(tmax),
        "prcp": float(prcp),
        "snow": float(snow),
        "wspd": float(wspd),
        "pres": float(pres),
        "arr_tavg": arr_tavg,
        "arr_tmin": arr_tmin,
        "arr_tmax": arr_tmax,
        "arr_prcp": arr_prcp,
        "arr_snow": arr_snow,
        "arr_wspd": arr_wspd,
        "arr_pres": arr_pres,
        "dep_slot_flights": dep_slot_flights,
        "arr_slot_flights": arr_slot_flights,
        "dep_daily_flights": dep_daily_flights,
        "aircraft_daily_legs": aircraft_daily_legs,
        "slot_order": slot_order,
        "Aircraft_age": float(aircraft_age),
        "Flight_Duration": float(flight_duration),
        "Day_Of_Week": int(day_of_week),
        "Month": int(month),
        "Is_Weekend": int(is_weekend),
        "deicing_risk": int(deicing_risk),
        "temp_swing": temp_swing,
        "total_route_prcp": total_route_prcp,
        "total_route_snow": total_route_snow,
        "max_route_wind": max_route_wind,
        "pres_drop": pres_drop,
        "arr_pres_drop": arr_pres_drop,
        "sin_month": sin_month,
        "cos_month": cos_month,
        "sin_dow": sin_dow,
        "cos_dow": cos_dow,
    }
    df_row = pd.DataFrame([row])
    expected_cols = artifact.get("features", list(df_row.columns))
    for col in expected_cols:
        if col not in df_row.columns:
            df_row[col] = 0
    return df_row[expected_cols]


gold_data = load_gold_datasets()
ml_artifact = load_ml_artifact()

if gold_data["daily"] is None:
    st.error(
        "No se encontraron los datasets de la Capa Oro (data/gold/*.parquet). "
        "Ejecute primero el notebook notebooks/01_etl_pipeline.ipynb."
    )
    st.stop()

df_daily = gold_data["daily"]
df_airports = gold_data["airports"]
df_airlines = gold_data["airlines"]
df_routes = gold_data["routes"]
df_causes = gold_data["causes"]

# ==============================================================================
# 3. BARRA LATERAL: FILTROS INTERACTIVOS Y METADATOS DEL SISTEMA
# ==============================================================================
with st.sidebar:
    st.markdown("## FlightOps Analytics")
    st.caption("Sistema de Control y Predicción de Operaciones Aéreas")
    st.markdown("---")

    min_date = df_daily["FlightDate"].min().date()
    max_date = df_daily["FlightDate"].max().date()
    date_range = st.date_input(
        "Rango Temporal (2023)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    all_airlines = (
        sorted(df_airlines["airline"].unique().tolist())
        if df_airlines is not None
        else []
    )
    selected_airlines = st.multiselect(
        "Aerolíneas Operadoras",
        options=all_airlines,
        default=all_airlines,
    )

    if df_airports is not None:
        sorted_hubs = df_airports.sort_values(
            "total_departures", ascending=False
        )["dep_airport"].tolist()
    else:
        sorted_hubs = sorted(df_daily["dep_airport"].unique().tolist())

    selected_hubs = st.multiselect(
        "Filtrar Aeropuertos de Salida (Opcional)",
        options=sorted_hubs,
        default=[],
        help="Deje vacío para analizar toda la red nacional (349 aeropuertos).",
    )

    st.markdown("---")
    if ml_artifact is not None:
        m_info = ml_artifact.get("metrics", {})
        st.success("Motor Predictivo en Línea")
        st.caption(
            f"• **ROC-AUC:** `{m_info.get('roc_auc', 0.7898):.4f}`\n"
            f"• **PR-AUC:** `{m_info.get('pr_auc', 0.6112):.4f}`\n"
            f"• **F1 (Balanceado):** `{m_info.get('f1_score', 0.5474):.4f}`\n"
            f"• **Precisión (τ=0.50):** `{m_info.get('precision_high', 0.732)*100:.1f}%`\n"
            f"• **Precisión (τ=0.70):** `{m_info.get('precision_ultra', 0.852)*100:.1f}%`\n"
            f"• **Accuracy Global:** `{m_info.get('accuracy_high', 0.836)*100:.1f}%`"
        )
    else:
        st.warning("Modelo predictivo no encontrado en la carpeta models/.")

# Aplicar filtros globales
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
else:
    start_dt, end_dt = pd.to_datetime(min_date), pd.to_datetime(max_date)

mask_daily = (df_daily["FlightDate"] >= start_dt) & (df_daily["FlightDate"] <= end_dt)
if selected_hubs:
    mask_daily &= df_daily["dep_airport"].isin(selected_hubs)
df_daily_filt = df_daily.loc[mask_daily].copy()

if df_airlines is not None:
    sel_air = selected_airlines if selected_airlines else all_airlines
    df_airlines_filt = df_airlines[df_airlines["airline"].isin(sel_air)].copy()
else:
    df_airlines_filt = pd.DataFrame()

# ==============================================================================
# 4. ENCABEZADO PRINCIPAL Y TARJETAS KPI EJECUTIVAS
# ==============================================================================
st.markdown(
    """
<div class="hero-banner">
    <div class="hero-badge">Centro de Control de Operaciones Aéreas • Red Nacional EE. UU.</div>
    <h1>FlightOps Analytics — Plataforma de Inteligencia Operativa y Predicción</h1>
    <p>Monitoreo integral de 6.57M de vuelos comerciales en EE. UU., diagnóstico causal FAA, impacto meteorológico en ruta y estimación de riesgo de disrupción en tiempo real.</p>
</div>
""",
    unsafe_allow_html=True,
)

total_flights = int(df_daily_filt["total_flights"].sum())
total_cancelled = int(df_daily_filt["cancelled_flights"].sum())
total_diverted = int(df_daily_filt["diverted_flights"].sum())
total_delayed_15 = int(df_daily_filt["delayed_15m"].sum())

cancel_rate = (total_cancelled / total_flights * 100) if total_flights > 0 else 0.0
delay_15_rate = (total_delayed_15 / total_flights * 100) if total_flights > 0 else 0.0
otp_rate = max(0.0, 100.0 - delay_15_rate - cancel_rate)
weighted_avg_delay = (
    (df_daily_filt["avg_dep_delay"] * df_daily_filt["total_flights"]).sum()
    / total_flights
    if total_flights > 0
    else 0.0
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">Operaciones Analizadas</div>
            <div class="kpi-value">{total_flights:,}</div>
            <div class="kpi-sub sub-blue">Red Nacional US 2023</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="kpi-card" style="border-top-color:#10b981;">
            <div class="kpi-label">Puntualidad (OTP &lt;15m)</div>
            <div class="kpi-value">{otp_rate:.2f}%</div>
            <div class="kpi-sub sub-green">Estándar FAA / DOT</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="kpi-card" style="border-top-color:#f59e0b;">
            <div class="kpi-label">Retrasos Severos (≥15m)</div>
            <div class="kpi-value">{total_delayed_15:,}</div>
            <div class="kpi-sub sub-amber">{delay_15_rate:.2f}% del total</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="kpi-card" style="border-top-color:#e11d48;">
            <div class="kpi-label">Vuelos Cancelados</div>
            <div class="kpi-value">{total_cancelled:,}</div>
            <div class="kpi-sub sub-red">Tasa: {cancel_rate:.2f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k5:
    st.markdown(
        f"""<div class="kpi-card" style="border-top-color:#6366f1;">
            <div class="kpi-label">Demora Media Salida</div>
            <div class="kpi-value">{weighted_avg_delay:.1f} min</div>
            <div class="kpi-sub sub-blue">Ponderada por volumen</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k6:
    auc_val = (
        ml_artifact["metrics"].get("roc_auc", 0.7898)
        if ml_artifact
        else 0.7898
    )
    prec_val = (
        ml_artifact["metrics"].get("precision_high", 0.732) * 100
        if ml_artifact
        else 73.2
    )
    st.markdown(
        f"""<div class="kpi-card" style="border-top-color:#0ea5e9;">
            <div class="kpi-label">Precisión Predictiva (τ=0.50)</div>
            <div class="kpi-value">{prec_val:.1f}%</div>
            <div class="kpi-sub sub-green">ROC-AUC: {auc_val:.4f}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ==============================================================================
# 5. MÓDULOS DE NAVEGACIÓN DE LA PLATAFORMA
# ==============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "1. Torre de Control y Dinámica Temporal",
        "2. Desempeño por Aerolínea",
        "3. Red Geoespacial de Aeropuertos y Rutas",
        "4. Diagnóstico Meteorológico y Causas FAA",
        "5. Simulador de Riesgo Operativo",
    ]
)

# ------------------------------------------------------------------------------
# TAB 1: TORRE DE CONTROL Y DINÁMICA TEMPORAL
# ------------------------------------------------------------------------------
with tab1:
    st.markdown(
        '<div class="section-title">Evolución Diaria y Estacionalidad de Operaciones y Disrupciones</div>',
        unsafe_allow_html=True,
    )

    daily_ts = (
        df_daily_filt.groupby("FlightDate")
        .agg(
            vuelos=("total_flights", "sum"),
            retrasados=("delayed_15m", "sum"),
            cancelados=("cancelled_flights", "sum"),
            delay_medio=("avg_dep_delay", "mean"),
        )
        .reset_index()
    )
    daily_ts["disruption_rate"] = (
        (daily_ts["retrasados"] + daily_ts["cancelados"])
        / daily_ts["vuelos"].replace(0, np.nan)
        * 100
    )
    daily_ts["rolling_7d"] = daily_ts["disruption_rate"].rolling(7, min_periods=1).mean()

    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ts.add_trace(
            go.Bar(
                x=daily_ts["FlightDate"],
                y=daily_ts["vuelos"],
                name="Vuelos Diarios",
                marker_color="rgba(14, 165, 233, 0.22)",
            ),
            secondary_y=False,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=daily_ts["FlightDate"],
                y=daily_ts["disruption_rate"],
                name="% Disrupción Diaria",
                mode="lines",
                line=dict(color="rgba(244, 63, 94, 0.45)", width=1.2),
            ),
            secondary_y=True,
        )
        fig_ts.add_trace(
            go.Scatter(
                x=daily_ts["FlightDate"],
                y=daily_ts["rolling_7d"],
                name="Media Móvil 7 Días (%)",
                mode="lines",
                line=dict(color="#e11d48", width=2.6),
            ),
            secondary_y=True,
        )
        fig_ts.update_layout(
            title="Serie Temporal Diaria: Volumen Operativo vs. Tasa de Disrupción (%)",
            template="plotly_white",
            height=410,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=55, b=20),
        )
        fig_ts.update_yaxes(title_text="Vuelos Operados", secondary_y=False)
        fig_ts.update_yaxes(title_text="% Disrupción (≥15m o Cancelado)", secondary_y=True)
        st.plotly_chart(fig_ts, width="stretch")

    with col_t2:
        month_names = {
            1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
        }
        dow_names = {
            1: "Lunes", 2: "Martes", 3: "Miércoles", 4: "Jueves",
            5: "Viernes", 6: "Sábado", 7: "Domingo",
        }
        grp_hm = (
            df_daily_filt.groupby(["DayOfWeek", "Month"], as_index=False)
            .agg(
                delayed_15m=("delayed_15m", "sum"),
                cancelled_flights=("cancelled_flights", "sum"),
                total_flights=("total_flights", "sum"),
            )
        )
        grp_hm["disrupt_pct"] = (
            (grp_hm["delayed_15m"] + grp_hm["cancelled_flights"])
            / grp_hm["total_flights"].clip(lower=1)
            * 100.0
        )
        piv = grp_hm.pivot(
            index="DayOfWeek", columns="Month", values="disrupt_pct"
        ).fillna(0.0)
        piv.index = [dow_names.get(i, str(i)) for i in piv.index]
        piv.columns = [month_names.get(c, str(c)) for c in piv.columns]

        fig_hm = px.imshow(
            piv,
            text_auto=".1f",
            color_continuous_scale="YlOrRd",
            labels=dict(x="Mes", y="Día de la Semana", color="% Disrupción"),
            title="Mapa de Calor: % Disrupción por Día y Mes",
            aspect="auto",
        )
        fig_hm.update_layout(
            template="plotly_white",
            height=410,
            margin=dict(l=20, r=20, t=55, b=20),
        )
        st.plotly_chart(fig_hm, width="stretch")

    st.markdown(
        """<div class="insight-card">
        <b>Resumen Operativo:</b> La serie temporal muestra que el periodo estival (<b>junio–agosto</b>) concentra las tasas más altas de disrupción severa (&gt;26%) impulsadas por tormentas convectivas vespertinas y saturación del espacio aéreo (NAS), mientras que <b>septiembre–noviembre</b> constituye la ventana operativa más estable del año (~15.8% de disrupción).
        </div>""",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------------------
# TAB 2: DESEMPEÑO POR AEROLÍNEA
# ------------------------------------------------------------------------------
with tab2:
    st.markdown(
        '<div class="section-title">Análisis Comparativo de Puntualidad y Eficiencia por Operador</div>',
        unsafe_allow_html=True,
    )

    if not df_airlines_filt.empty:
        air_agg = (
            df_airlines_filt.groupby("airline")
            .agg(
                vuelos=("total_flights", "sum"),
                cancelados=("cancelled_flights", "sum"),
                retrasados=("delayed_15m", "sum"),
                dep_delay=("avg_dep_delay", "mean"),
                arr_delay=("avg_arr_delay", "mean"),
            )
            .reset_index()
        )
        air_agg["otp_pct"] = (1.0 - air_agg["retrasados"] / air_agg["vuelos"]) * 100
        air_agg["cancel_pct"] = (air_agg["cancelados"] / air_agg["vuelos"]) * 100
        air_agg = air_agg.sort_values("otp_pct", ascending=True)

        c_a1, c_a2 = st.columns([1.1, 1.1])
        with c_a1:
            fig_air_bar = px.bar(
                air_agg,
                x="otp_pct",
                y="airline",
                orientation="h",
                color="dep_delay",
                color_continuous_scale="RdYlGn_r",
                text=air_agg["otp_pct"].apply(lambda x: f"{x:.1f}%"),
                labels={
                    "otp_pct": "Puntualidad OTP (%)",
                    "airline": "Aerolínea",
                    "dep_delay": "Demora Salida (min)",
                },
                title="Ranking de Puntualidad (OTP %) por Aerolínea",
            )
            fig_air_bar.update_layout(
                template="plotly_white",
                height=460,
                xaxis_range=[60, 90],
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_air_bar, width="stretch")

        with c_a2:
            fig_quad = px.scatter(
                air_agg,
                x="dep_delay",
                y="otp_pct",
                size="vuelos",
                color="cancel_pct",
                hover_name="airline",
                text="airline",
                color_continuous_scale="Sunsetdark",
                size_max=48,
                labels={
                    "dep_delay": "Retraso Promedio de Salida (min)",
                    "otp_pct": "Índice de Puntualidad OTP (%)",
                    "cancel_pct": "% Cancelación",
                    "vuelos": "Volumen Anual",
                },
                title="Cuadrante de Eficiencia: Puntualidad vs. Demora Media y Volumen",
            )
            fig_quad.update_traces(textposition="top center")
            fig_quad.update_layout(
                template="plotly_white",
                height=460,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_quad, width="stretch")

        # Evolución mensual de OTP por aerolínea
        fig_air_line = px.line(
            df_airlines_filt,
            x="YearMonth",
            y="ontime_rate",
            color="airline",
            markers=True,
            labels={
                "YearMonth": "Mes (2023)",
                "ontime_rate": "Puntualidad OTP (%)",
                "airline": "Aerolínea",
            },
            title="Trayectoria Mensual de Puntualidad (OTP %) por Operador",
        )
        fig_air_line.update_layout(
            template="plotly_white",
            height=380,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_air_line, width="stretch")

# ------------------------------------------------------------------------------
# TAB 3: RED GEOESPACIAL DE AEROPUERTOS Y RUTAS
# ------------------------------------------------------------------------------
with tab3:
    st.markdown(
        '<div class="section-title">Cartografía Geoespacial de Aeropuertos y Cuellos de Botella en Rutas</div>',
        unsafe_allow_html=True,
    )

    if df_airports is not None:
        col_g1, col_g2 = st.columns([1.45, 1])
        with col_g1:
            df_map = df_airports[df_airports["total_departures"] >= 1000].copy()
            fig_map = px.scatter_geo(
                df_map,
                lat="LATITUDE",
                lon="LONGITUDE",
                size="total_departures",
                color="overall_avg_delay",
                hover_name="AIRPORT",
                hover_data={
                    "dep_airport": True,
                    "CITY": True,
                    "STATE": True,
                    "total_departures": ":,",
                    "overall_avg_delay": ":.2f",
                    "cancellation_rate": ":.2f",
                },
                scope="usa",
                projection="albers usa",
                color_continuous_scale="Turbo",
                size_max=34,
                labels={
                    "overall_avg_delay": "Demora Media (min)",
                    "total_departures": "Salidas Anuales",
                },
                title="Mapa Nacional de Aeropuertos: Volumen de Tráfico y Severidad de Retrasos",
            )
            fig_map.update_layout(
                template="plotly_white",
                height=470,
                margin=dict(l=0, r=0, t=50, b=0),
            )
            st.plotly_chart(fig_map, width="stretch")

        with col_g2:
            top_hubs = df_airports.nlargest(15, "total_departures").sort_values(
                "overall_avg_delay", ascending=True
            )
            fig_hubs = px.bar(
                top_hubs,
                x="overall_avg_delay",
                y="dep_airport",
                orientation="h",
                color="cancellation_rate",
                color_continuous_scale="Reds",
                hover_data=["AIRPORT", "CITY", "total_departures"],
                labels={
                    "overall_avg_delay": "Retraso Promedio (min)",
                    "dep_airport": "Aeropuerto IATA",
                    "cancellation_rate": "% Cancelación",
                },
                title="Top 15 Aeropuertos de Mayor Tráfico por Retraso Promedio y Cancelación",
            )
            fig_hubs.update_layout(
                template="plotly_white",
                height=470,
                margin=dict(l=15, r=15, t=50, b=15),
            )
            st.plotly_chart(fig_hubs, width="stretch")

    if df_routes is not None:
        st.markdown("#### Top 20 Rutas Comerciales de Alta Densidad con Mayor Demora Promedio")
        busy_routes = (
            df_routes[df_routes["flight_count"] >= 1500]
            .sort_values("avg_dep_delay", ascending=False)
            .head(20)
            .copy()
        )
        busy_routes["Ruta"] = busy_routes["dep_airport"] + " -> " + busy_routes["arr_airport"]
        busy_routes["Tasa Cancelación (%)"] = (busy_routes["cancellation_rate"] * 100).round(2)
        busy_routes["Demora Salida (min)"] = busy_routes["avg_dep_delay"].round(2)
        busy_routes["Duración Media (min)"] = busy_routes["avg_duration"].round(1)
        st.dataframe(
            busy_routes[
                [
                    "Ruta",
                    "flight_count",
                    "Demora Salida (min)",
                    "Tasa Cancelación (%)",
                    "Duración Media (min)",
                ]
            ].rename(columns={"flight_count": "Vuelos Anuales"}),
            width="stretch",
            hide_index=True,
        )

# ------------------------------------------------------------------------------
# TAB 4: DIAGNÓSTICO METEOROLÓGICO Y CAUSAS FAA
# ------------------------------------------------------------------------------
with tab4:
    st.markdown(
        '<div class="section-title">Descomposición Causal FAA e Impacto Físico-Meteorológico</div>',
        unsafe_allow_html=True,
    )

    if df_causes is not None:
        c_w1, c_w2 = st.columns([1.4, 1])
        with c_w1:
            causes_melt = df_causes.melt(
                id_vars="YearMonth",
                value_vars=[
                    "delay_last_aircraft",
                    "delay_carrier",
                    "delay_nas",
                    "delay_weather",
                    "delay_security",
                ],
                var_name="Causa",
                value_name="Minutos",
            )
            cause_labels = {
                "delay_last_aircraft": "1. Aeronave Tardía (Efecto Cascada)",
                "delay_carrier": "2. Operación de Aerolínea (Carrier)",
                "delay_nas": "3. Sistema Nacional de Espacio Aéreo (NAS)",
                "delay_weather": "4. Meteorología Extrema Directa",
                "delay_security": "5. Seguridad Aeroportuaria",
            }
            causes_melt["Causa"] = causes_melt["Causa"].map(cause_labels)
            fig_causes = px.area(
                causes_melt,
                x="YearMonth",
                y="Minutos",
                color="Causa",
                color_discrete_sequence=[
                    "#e11d48",
                    "#0284c7",
                    "#f59e0b",
                    "#6366f1",
                    "#10b981",
                ],
                title="Evolución Mensual de Minutos de Retraso por Causa Oficial FAA",
            )
            fig_causes.update_layout(
                template="plotly_white",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_causes, width="stretch")

        with c_w2:
            totals_cause = (
                causes_melt.groupby("Causa")["Minutos"].sum().reset_index()
            )
            fig_pie = px.pie(
                totals_cause,
                values="Minutos",
                names="Causa",
                hole=0.48,
                color_discrete_sequence=[
                    "#e11d48",
                    "#0284c7",
                    "#f59e0b",
                    "#6366f1",
                    "#10b981",
                ],
                title="Participación Anual por Categoría Causal FAA",
            )
            fig_pie.update_layout(
                template="plotly_white",
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_pie, width="stretch")

    # Análisis de regresión lineal nativo (np.polyfit) entre Clima y Demoras
    st.markdown("#### Regresión Empírica: Severidad Meteorológica vs. Retrasos Operativos")
    w_col1, w_col2 = st.columns([1, 3])
    with w_col1:
        weather_var_map = {
            "Precipitación Diaria (mm)": "prcp",
            "Velocidad del Viento (km/h)": "wspd",
            "Nieve Acumulada (mm)": "snow",
            "Temperatura Media (°C)": "tavg",
            "Presión Atmosférica (hPa)": "pres",
        }
        selected_w_label = st.selectbox(
            "Variable Meteorológica a Evaluar:",
            options=list(weather_var_map.keys()),
            index=0,
        )
        w_col = weather_var_map[selected_w_label]
        min_flights_filter = st.slider(
            "Mínimo de vuelos diarios en el aeropuerto:",
            min_value=50,
            max_value=500,
            value=150,
            step=50,
        )

        df_w_sample = df_daily_filt[
            (df_daily_filt["total_flights"] >= min_flights_filter)
            & df_daily_filt[w_col].notna()
            & df_daily_filt["avg_dep_delay"].notna()
        ].copy()
        if len(df_w_sample) > 2500:
            df_w_sample = df_w_sample.sample(n=2500, random_state=42)

        if len(df_w_sample) >= 5:
            x_vals = df_w_sample[w_col].to_numpy(dtype=float)
            y_vals = df_w_sample["avg_dep_delay"].to_numpy(dtype=float)
            corr_r = float(np.corrcoef(x_vals, y_vals)[0, 1])
            slope, intercept = np.polyfit(x_vals, y_vals, 1)
        else:
            corr_r, slope, intercept = 0.0, 0.0, 0.0

        st.metric("Correlación Lineal (r)", f"{corr_r:+.3f}")
        st.metric("Pendiente OLS (Δ min / unidad)", f"{slope:+.3f} min")

    with w_col2:
        fig_w_scatter = px.scatter(
            df_w_sample,
            x=w_col,
            y="avg_dep_delay",
            color="cancelled_flights",
            size="total_flights",
            hover_data=["FlightDate", "dep_airport", "total_flights"],
            color_continuous_scale="Inferno",
            opacity=0.68,
            labels={
                w_col: selected_w_label,
                "avg_dep_delay": "Retraso Promedio Salida (min)",
                "cancelled_flights": "Cancelaciones",
            },
            title=f"Impacto de {selected_w_label} sobre la Demora Media Diaria (con Tendencia OLS Nativa)",
        )
        if len(df_w_sample) >= 5:
            x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 100)
            y_line = slope * x_line + intercept
            fig_w_scatter.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name=f"Tendencia OLS (pendiente={slope:+.2f})",
                    line=dict(color="#0284c7", width=3, dash="dash"),
                )
            )
        fig_w_scatter.update_layout(
            template="plotly_white",
            height=410,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_w_scatter, width="stretch")

# ------------------------------------------------------------------------------
# TAB 5: SIMULADOR DE RIESGO OPERATIVO
# ------------------------------------------------------------------------------
with tab5:
    st.markdown(
        '<div class="section-title">Simulador en Tiempo Real de Estimación de Riesgo de Disrupción</div>',
        unsafe_allow_html=True,
    )

    if ml_artifact is None:
        st.warning(
            "El artefacto del modelo no está disponible. Ejecute notebooks/02_ml_model.ipynb."
        )
    else:
        metrics_dict = ml_artifact.get("metrics", {})
        opt_f1_t = float(ml_artifact.get("optimal_f1_threshold", 0.25))
        high_p_t = float(ml_artifact.get("high_precision_threshold", 0.50))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC (Test Set 60k)", f"{metrics_dict.get('roc_auc', 0.7898):.4f}", "+7.2 pts vs Baseline")
        m2.metric("PR-AUC (Average Precision)", f"{metrics_dict.get('pr_auc', 0.6112):.4f}", "+17.0 pts vs Baseline")
        m3.metric("Precisión Estándar (τ=0.50)", f"{metrics_dict.get('precision_high', 0.732)*100:.1f}%", "Accuracy: 83.6%")
        m4.metric("Precisión Alta Confianza (τ=0.70)", f"{metrics_dict.get('precision_ultra', 0.852)*100:.1f}%", "F1 Max: 0.5474")

        st.markdown("---")
        sim_left, sim_right = st.columns([1.25, 1])

        with sim_left:
            st.markdown("#### Parámetros Operativos, Estado Previo (s-1) y Clima")
            airport_options = ml_artifact.get("top_airports", sorted_hubs)
            default_dep_idx = airport_options.index("ATL") if "ATL" in airport_options else 0
            default_arr_idx = airport_options.index("JFK") if "JFK" in airport_options else min(1, len(airport_options) - 1)

            f_c1, f_c2, f_c3 = st.columns(3)
            with f_c1:
                sim_dep = st.selectbox("Aeropuerto Origen (IATA)", options=airport_options, index=default_dep_idx)
                sim_month = st.selectbox(
                    "Mes del Vuelo",
                    options=list(range(1, 13)),
                    index=6,
                    format_func=lambda m: {
                        1: "1 - Enero", 2: "2 - Febrero", 3: "3 - Marzo", 4: "4 - Abril",
                        5: "5 - Mayo", 6: "6 - Junio", 7: "7 - Julio", 8: "8 - Agosto",
                        9: "9 - Septiembre", 10: "10 - Octubre", 11: "11 - Noviembre", 12: "12 - Diciembre",
                    }[m],
                )
                sim_age = st.slider("Edad de la Aeronave (años)", 1, 30, 11)
            with f_c2:
                sim_arr = st.selectbox("Aeropuerto Destino (IATA)", options=airport_options, index=default_arr_idx)
                sim_dow = st.selectbox(
                    "Día de la Semana",
                    options=[1, 2, 3, 4, 5, 6, 7],
                    index=4,
                    format_func=lambda d: {
                        1: "1 - Lunes", 2: "2 - Martes", 3: "3 - Miércoles",
                        4: "4 - Jueves", 5: "5 - Viernes", 6: "6 - Sábado", 7: "7 - Domingo",
                    }[d],
                )
                sim_dur = st.slider("Duración Programada (min)", 45, 420, 145, step=5)
            with f_c3:
                sim_airline = st.selectbox(
                    "Aerolínea Operadora",
                    options=all_airlines if all_airlines else ["DELTA AIR LINES INC."],
                    index=0,
                )
                sim_slot = st.selectbox(
                    "Bloque Horario de Salida",
                    options=["MORNING", "AFTERNOON", "EVENING", "NIGHT"],
                    index=2,
                )
                threshold_mode = st.selectbox(
                    "Modo de Umbral Operativo (τ)",
                    options=[
                        f"Alta Precisión Estándar (τ = {high_p_t:.2f})",
                        f"Máxima Sensibilidad / F1 (τ = {opt_f1_t:.2f})",
                        "Alerta Operativa Intermedia (τ = 0.35)",
                        "Precisión Crítica de Control (τ = 0.60)",
                        "Alta Confianza Operativa (τ = 0.70)",
                    ],
                    index=0,
                )

            st.markdown("##### Estado Operativo del Bloque Previo ($s - 1$)")
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                sim_tail_prev_delay = st.slider(
                    "Demora del Tramo Previo del Avión Hoy (min)",
                    min_value=0,
                    max_value=180,
                    value=0,
                    step=5,
                    help="Minutos de retraso observados en la llegada/salida del tramo previo de la misma matrícula (Tail_Number) hoy.",
                )
            with c_s2:
                sim_hub_prev_disrupt = st.slider(
                    "% Disrupción del Aeropuerto Origen en el Slot Previo (s-1)",
                    min_value=0,
                    max_value=80,
                    value=18,
                    step=2,
                    help="Porcentaje de vuelos retrasados/cancelados en el aeropuerto de salida durante el bloque horario inmediatamente anterior.",
                )

            st.markdown("##### Condiciones Meteorológicas Pronosticadas en el Origen")
            w_c1, w_c2, w_c3, w_c4 = st.columns(4)
            with w_c1:
                sim_tavg = st.number_input("Temp. Media (°C)", value=21.0, min_value=-25.0, max_value=45.0, step=1.0)
                sim_tmin = st.number_input("Temp. Mínima (°C)", value=15.0, min_value=-35.0, max_value=40.0, step=1.0)
            with w_c2:
                sim_tmax = st.number_input("Temp. Máxima (°C)", value=27.0, min_value=-20.0, max_value=50.0, step=1.0)
                sim_prcp = st.number_input("Precipitación (mm)", value=4.0, min_value=0.0, max_value=200.0, step=1.0)
            with w_c3:
                sim_snow = st.number_input("Nieve (mm)", value=0.0, min_value=0.0, max_value=300.0, step=5.0)
                sim_wspd = st.number_input("Viento (km/h)", value=22.0, min_value=0.0, max_value=120.0, step=2.0)
            with w_c4:
                sim_pres = st.number_input("Presión (hPa)", value=1011.0, min_value=970.0, max_value=1045.0, step=1.0)

        with sim_right:
            if "0.70" in threshold_mode:
                active_tau = 0.70
            elif "0.60" in threshold_mode:
                active_tau = 0.60
            elif "0.35" in threshold_mode:
                active_tau = 0.35
            elif f"{opt_f1_t:.2f}" in threshold_mode:
                active_tau = opt_f1_t
            else:
                active_tau = high_p_t

            X_input = build_single_flight_features(
                dep_airport=sim_dep,
                arr_airport=sim_arr,
                airline=sim_airline,
                month=sim_month,
                day_of_week=sim_dow,
                dep_time_label=sim_slot,
                aircraft_age=sim_age,
                flight_duration=sim_dur,
                tavg=sim_tavg,
                tmin=sim_tmin,
                tmax=sim_tmax,
                prcp=sim_prcp,
                snow=sim_snow,
                wspd=sim_wspd,
                pres=sim_pres,
                artifact=ml_artifact,
                tail_prev_delay_min=float(sim_tail_prev_delay),
                hub_prev_disrupt_pct=float(sim_hub_prev_disrupt),
            )

            pipe = ml_artifact["pipeline"]
            prob_disruption = float(pipe.predict_proba(X_input)[0, 1])
            prob_pct = prob_disruption * 100.0
            is_alert = prob_disruption >= active_tau

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=prob_pct,
                    number={"suffix": "%", "font": {"size": 42, "color": "#0f172a"}},
                    delta={
                        "reference": active_tau * 100.0,
                        "increasing": {"color": "#e11d48"},
                        "decreasing": {"color": "#10b981"},
                    },
                    title={
                        "text": f"Probabilidad Estimada de Disrupción (Umbral τ = {active_tau*100:.0f}%)",
                        "font": {"size": 15},
                    },
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": "#e11d48" if is_alert else "#0284c7"},
                        "steps": [
                            {"range": [0, opt_f1_t * 100], "color": "#d1fae5"},
                            {"range": [opt_f1_t * 100, high_p_t * 100], "color": "#fef3c7"},
                            {"range": [high_p_t * 100, 100], "color": "#ffe4e6"},
                        ],
                        "threshold": {
                            "line": {"color": "#0f172a", "width": 4},
                            "thickness": 0.8,
                            "value": active_tau * 100.0,
                        },
                    },
                )
            )
            fig_gauge.update_layout(
                height=300,
                margin=dict(l=25, r=25, t=55, b=15),
                template="plotly_white",
            )
            st.plotly_chart(fig_gauge, width="stretch")

            if is_alert:
                st.error(
                    f"**ALERTA DE DISRUPCIÓN ACTIVA ({prob_pct:.1f}% ≥ {active_tau*100:.0f}%)**\n\n"
                    f"El vuelo **{sim_dep} -> {sim_arr}** operado por **{sim_airline}** en bloque **{sim_slot}** supera el umbral operativo seleccionado. "
                    "Se recomienda activar margen de rotación de tripulación y priorizar slot de salida."
                )
            else:
                st.success(
                    f"**OPERACIÓN DENTRO DE MARGEN NORMAL ({prob_pct:.1f}% < {active_tau*100:.0f}%)**\n\n"
                    f"El vuelo **{sim_dep} -> {sim_arr}** presenta riesgo controlado bajo el umbral operativo actual."
                )

            # Importancia de características global del modelo
            feat_imp = ml_artifact.get("feature_importance", {})
            if feat_imp:
                df_imp = (
                    pd.DataFrame(
                        {"Característica": list(feat_imp.keys()), "Impacto ROC-AUC": list(feat_imp.values())}
                    )
                    .sort_values("Impacto ROC-AUC", ascending=True)
                    .tail(8)
                )
                fig_imp = px.bar(
                    df_imp,
                    x="Impacto ROC-AUC",
                    y="Característica",
                    orientation="h",
                    color_discrete_sequence=["#0284c7"],
                    title="Variables con Mayor Peso Predictivo en el Modelo (Importancia por Permutación)",
                )
                fig_imp.update_layout(
                    template="plotly_white",
                    height=250,
                    margin=dict(l=15, r=15, t=40, b=15),
                )
                st.plotly_chart(fig_imp, width="stretch")
