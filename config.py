"""Rutas centralizadas de la arquitectura Medallón.

Todos los notebooks y la app de Streamlit importan estas rutas, de modo que
la estructura del proyecto se define en un único lugar.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# --- Capa BRONCE (raw): datos originales, inmutables -------------------------
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"
BRONZE_AIRPORTS = BRONZE_DIR / "airports_assets"
BRONZE_CANCELLED = BRONZE_DIR / "us_cancelled_div"
BRONZE_FLIGHTS = BRONZE_DIR / "us_flights"

# --- Capa PLATA (cleansed): datos limpios, tipados, en Parquet -----------------
SILVER_DIR = PROJECT_ROOT / "data" / "silver"
SILVER_FLIGHTS = SILVER_DIR / "flights"          # dataset particionado por mes
SILVER_AIRPORTS = SILVER_DIR / "airports.parquet"
SILVER_WEATHER = SILVER_DIR / "weather.parquet"

# --- Capa ORO (aggregated / features): listo para BI y ML ----------------------
GOLD_DIR = PROJECT_ROOT / "data" / "gold"
GOLD_DAILY_AIRPORT = GOLD_DIR / "daily_airport_metrics.parquet"
GOLD_AIRPORT_SUMMARY = GOLD_DIR / "airport_summary.parquet"
GOLD_AIRLINE_MONTHLY = GOLD_DIR / "airline_monthly_metrics.parquet"
GOLD_ROUTES = GOLD_DIR / "route_metrics.parquet"
GOLD_DELAY_CAUSES = GOLD_DIR / "delay_causes_monthly.parquet"
GOLD_ML_FEATURES = GOLD_DIR / "ml_features"      # dataset particionado por mes
GOLD_ML_DATASET = GOLD_DIR / "ml_features_dataset.parquet"

# --- Artefactos de modelos -----------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"


def ensure_dirs() -> None:
    """Crea las carpetas de salida si no existen."""
    for d in (SILVER_DIR, SILVER_FLIGHTS, GOLD_DIR, GOLD_ML_FEATURES, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
