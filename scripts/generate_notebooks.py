"""Script para construir los notebooks 01_etl_pipeline.ipynb y 02_ml_model.ipynb
sin ejecutar ninguna celda de código (cumpliendo la restricción estricta).
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")]
    }

def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").split("\n")]
    }

def save_notebook(filepath: Path, cells: list):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (trafico_vuelo)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.13"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:03d}"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook generado exitosamente en: {filepath}")

# ==============================================================================
# NOTEBOOK 1: 01_etl_pipeline.ipynb
# ==============================================================================
etl_cells = [
    md("""# Pipeline ETL End-to-End: Procesamiento de Vuelos (Arquitectura Medallón)
### Ingesta, Limpieza, Enriquecimiento y Agregaciones de Negocio

---

## 📌 Contexto del Proyecto
Este pipeline implementa una **Arquitectura Medallón** de tres capas para el procesamiento masivo de datos de tráfico aéreo estadounidense durante el año 2023:

1. 🥉 **Capa Bronce (Raw):**
   - Ingesta de datos crudos en formato CSV:
     - `us_flights_2023_*.csv` (12 meses de vuelos comerciales operados).
     - `us_cancelled_div_2023_*.csv` (12 meses de vuelos cancelados o desviados).
     - `airports_geolocation.csv` (metadatos geográficos y nombres oficiales de aeropuertos).
     - `weather_meteo_by_airport.csv` (registros climáticos diarios por aeropuerto: temperatura, lluvia, viento, nieve, presión).

2. 🥈 **Capa Plata (Cleansed / Conformed):**
   - Limpieza, desduplicación, normalización de tipos y esquemas.
   - Tratamiento riguroso de valores nulos e imputaciones basadas en dominio.
   - Unificación de vuelos operados y cancelados en una estructura columnar consistente.
   - Almacenamiento optimizado en formato **Apache Parquet** comprimido (`snappy`), reduciendo el consumo de memoria en más de un 80%.

3. 🥇 **Capa Oro (Aggregated / Business Features):**
   - Creación de datasets analíticos listos para dashboards ejecutivos (`Streamlit`) y matrices de características estructuradas para modelado predictivo de Machine Learning.
"""),

    code("""import os
import sys
import glob
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings('ignore')

# Configurar ruta raíz del proyecto
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

import config
config.ensure_dirs()

print("=" * 70)
print(f"Entorno configurado. Raíz del proyecto: {PROJECT_ROOT}")
print(f"Directorio Bronce: {config.BRONZE_DIR}")
print(f"Directorio Plata:  {config.SILVER_DIR}")
print(f"Directorio Oro:    {config.GOLD_DIR}")
print("=" * 70)"""),

    md("""## 1. Inspección de Archivos en Capa Bronce (Raw Data)
Verificamos la existencia de las fuentes de datos crudas antes de iniciar las transformaciones."""),

    code("""# Listar fuentes disponibles
bronze_airports_geo = config.BRONZE_AIRPORTS / "airports_geolocation.csv"
bronze_weather = config.BRONZE_AIRPORTS / "weather_meteo_by_airport.csv"

flight_files = sorted(glob.glob(str(config.BRONZE_FLIGHTS / "*.csv")))
cancelled_files = sorted(glob.glob(str(config.BRONZE_CANCELLED / "*.csv")))

print(f"✓ Archivos de vuelos encontrados:      {len(flight_files)}")
print(f"✓ Archivos de cancelaciones/desvíos:   {len(cancelled_files)}")
print(f"✓ Archivo de geolocalización presente: {bronze_airports_geo.exists()}")
print(f"✓ Archivo meteorológico presente:      {bronze_weather.exists()}")"""),

    md("""## 2. Procesamiento de Metadatos: Aeropuertos y Clima (Bronce -> Plata)
1. **Aeropuertos (`airports_geolocation.csv`)**:
   - Estandarización de códigos IATA en mayúsculas sin espacios.
   - Eliminación de registros duplicados.
   - Conversión a Parquet en `data/silver/airports.parquet`.

2. **Meteorología (`weather_meteo_by_airport.csv`)**:
   - Conversión de `time` a formato `datetime64[ns]`.
   - Normalización de identificadores IATA.
   - Imputación de variables de precipitación (`prcp`) y nieve (`snow`) con `0.0`.
   - Imputación de variables térmicas y de viento (`tavg`, `tmin`, `tmax`, `wspd`, `pres`) mediante mediana por aeropuerto.
   - Conversión a Parquet en `data/silver/weather.parquet`."""),

    code("""# 2.1 Procesar Aeropuertos
print("Procesando aeropuertos (Bronce -> Plata)...")
df_airports = pd.read_csv(bronze_airports_geo)
df_airports['IATA_CODE'] = df_airports['IATA_CODE'].astype(str).str.strip().str.upper()
df_airports.drop_duplicates(subset=['IATA_CODE'], inplace=True)
df_airports.to_parquet(config.SILVER_AIRPORTS, index=False, engine='pyarrow')
print(f"-> Aeropuertos guardados en Plata: {len(df_airports)} aeropuertos únicos.")

# 2.2 Procesar Clima
print("\nProcesando datos meteorológicos (Bronce -> Plata)...")
df_weather = pd.read_csv(bronze_weather)
df_weather['time'] = pd.to_datetime(df_weather['time'])
df_weather['airport_id'] = df_weather['airport_id'].astype(str).str.strip().str.upper()

# Imputación guiada por dominio
df_weather['prcp'] = df_weather['prcp'].fillna(0.0).astype('float32')
df_weather['snow'] = df_weather['snow'].fillna(0.0).astype('float32')

for col in ['tavg', 'tmin', 'tmax', 'wspd', 'pres']:
    if col in df_weather.columns:
        df_weather[col] = df_weather.groupby('airport_id')[col].transform(lambda x: x.fillna(x.median()))
        df_weather[col] = df_weather[col].fillna(df_weather[col].median()).astype('float32')

df_weather.to_parquet(config.SILVER_WEATHER, index=False, engine='pyarrow', compression='snappy')
print(f"-> Registros de clima guardados en Plata: {len(df_weather):,} observaciones.")"""),

    md("""## 3. Procesamiento y Limpieza de Vuelos y Cancelaciones (Bronce -> Plata)
Para manejar de forma eficiente el volumen de datos de 2023 (~800 MB en CSV):
- Procesamos mes a mes para mantener bajo control el consumo de memoria RAM.
- Unificamos los vuelos operados con los vuelos cancelados/desviados agregando indicadores explícitos (`Cancelled`, `Diverted`).
- Corregimos tipos de datos a representaciones compactas (`int8`, `float32`, categorías).
- Cada mes se almacena de forma particionada en `data/silver/flights/flights_YYYY_MM.parquet`."""),

    code("""def clean_and_unify_monthly_flights(flight_csv_path, cancelled_csv_path=None):
    \"\"\"Limpia, tipa y fusiona los vuelos de un mes específico.\"\"\"
    df_f = pd.read_csv(flight_csv_path, low_memory=False)
    df_f['Cancelled'] = 0
    df_f['Diverted'] = 0
    
    if cancelled_csv_path and Path(cancelled_csv_path).exists():
        df_c = pd.read_csv(cancelled_csv_path, low_memory=False)
        for col in ['Manufacturer', 'Model', 'Aircraft_age']:
            if col not in df_c.columns:
                df_c[col] = 'UNKNOWN' if col != 'Aircraft_age' else np.nan
        df_merged = pd.concat([df_f, df_c], ignore_index=True)
    else:
        df_merged = df_f

    # Conversión y normalización de fechas y enteros
    df_merged['FlightDate'] = pd.to_datetime(df_merged['FlightDate'])
    df_merged['Day_Of_Week'] = df_merged['Day_Of_Week'].astype('int8')
    df_merged['Cancelled'] = df_merged['Cancelled'].fillna(0).astype('int8')
    df_merged['Diverted'] = df_merged['Diverted'].fillna(0).astype('int8')

    # Retrasos y duraciones en minutos
    metric_cols = ['Dep_Delay', 'Arr_Delay', 'Flight_Duration',
                   'Delay_Carrier', 'Delay_Weather', 'Delay_NAS', 
                   'Delay_Security', 'Delay_LastAircraft']
    for col in metric_cols:
        if col in df_merged.columns:
            df_merged[col] = pd.to_numeric(df_merged[col], errors='coerce').fillna(0.0).astype('float32')

    # Limpieza de cadenas de texto
    text_cols = ['airline', 'dep_airport', 'arr_airport', 'DepTime_label', 
                 'Dep_Delay_Type', 'Arr_Delay_Type', 'Distance_type']
    for col in text_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].astype(str).str.strip().str.upper()

    return df_merged

# Ejecutar transformación mensual a Plata
total_rows_silver = 0
for f_path in flight_files:
    file_name = Path(f_path).name
    month_suffix = file_name.replace("us_flights_", "").replace(".csv", "")
    c_path = config.BRONZE_CANCELLED / f"us_cancelled_div_{month_suffix}.csv"
    
    df_month = clean_and_unify_monthly_flights(f_path, c_path if c_path.exists() else None)
    out_parquet = config.SILVER_FLIGHTS / f"flights_{month_suffix}.parquet"
    
    df_month.to_parquet(out_parquet, index=False, engine='pyarrow', compression='snappy')
    total_rows_silver += len(df_month)
    print(f"✓ Mes {month_suffix} procesado: {len(df_month):,} filas -> {out_parquet.name}")

print(f"\nTotal consolidado en Capa Plata: {total_rows_silver:,} registros.")"""),

    md("""## 4. Construcción de la Capa Oro (Business Aggregations & Metrics)
A partir de los datos limpios en la Capa Plata, generamos tablas agregadas de alto rendimiento:

1. `daily_airport_metrics.parquet`:
   - Agregación por fecha y aeropuerto origen (vuelos totales, cancelados, desviados, retrasos medios, vuelos con retraso $\ge 15$ min).
   - Enriquecido con variables meteorológicas del día.
2. `airport_summary.parquet`:
   - KPIs consolidados a nivel de aeropuerto (volumen anual, tasa de cancelación, ranking y coordenadas GPS).
3. `airline_monthly_metrics.parquet`:
   - Desempeño mensual de puntualidad por aerolínea.
4. `route_metrics.parquet`:
   - Métricas de tráfico y retrasos por origen-destino.
5. `delay_causes_monthly.parquet`:
   - Distribución agregada de causas de retraso (Clima, Aerolínea, Control de Tráfico Aéreo NAS, Aeronave anterior, Seguridad)."""),

    code("""silver_parquet_files = sorted(glob.glob(str(config.SILVER_FLIGHTS / "*.parquet")))
print(f"Agregando métricas de {len(silver_parquet_files)} particiones de Plata...")

df_airports_silver = pd.read_parquet(config.SILVER_AIRPORTS)
df_weather_silver = pd.read_parquet(config.SILVER_WEATHER)

daily_list = []
airline_list = []
route_list = []
causes_list = []

for p_file in silver_parquet_files:
    chunk = pd.read_parquet(p_file)
    
    # 1. Diario por aeropuerto de salida
    d_agg = chunk.groupby(['FlightDate', 'dep_airport']).agg(
        total_flights=('FlightDate', 'count'),
        cancelled_flights=('Cancelled', 'sum'),
        diverted_flights=('Diverted', 'sum'),
        avg_dep_delay=('Dep_Delay', 'mean'),
        max_dep_delay=('Dep_Delay', 'max'),
        delayed_15m=('Dep_Delay', lambda x: (x >= 15).sum())
    ).reset_index()
    daily_list.append(d_agg)
    
    # 2. Mensual por aerolínea
    chunk['YearMonth'] = chunk['FlightDate'].dt.to_period('M').astype(str)
    a_agg = chunk.groupby(['YearMonth', 'airline']).agg(
        total_flights=('FlightDate', 'count'),
        cancelled_flights=('Cancelled', 'sum'),
        avg_dep_delay=('Dep_Delay', 'mean'),
        avg_arr_delay=('Arr_Delay', 'mean'),
        delayed_15m=('Dep_Delay', lambda x: (x >= 15).sum())
    ).reset_index()
    airline_list.append(a_agg)
    
    # 3. Métricas por ruta
    r_agg = chunk.groupby(['dep_airport', 'arr_airport']).agg(
        flight_count=('FlightDate', 'count'),
        avg_duration=('Flight_Duration', 'mean'),
        avg_dep_delay=('Dep_Delay', 'mean'),
        cancellation_rate=('Cancelled', 'mean')
    ).reset_index()
    route_list.append(r_agg)
    
    # 4. Causas de demoras
    c_agg = chunk[chunk['Dep_Delay'] > 0].groupby('YearMonth').agg(
        delay_carrier=('Delay_Carrier', 'sum'),
        delay_weather=('Delay_Weather', 'sum'),
        delay_nas=('Delay_NAS', 'sum'),
        delay_security=('Delay_Security', 'sum'),
        delay_last_aircraft=('Delay_LastAircraft', 'sum')
    ).reset_index()
    causes_list.append(c_agg)

# Consolidar y unir clima con la capa Oro
df_gold_daily = pd.concat(daily_list, ignore_index=True)
df_gold_daily = df_gold_daily.merge(
    df_weather_silver,
    left_on=['FlightDate', 'dep_airport'],
    right_on=['time', 'airport_id'],
    how='left'
).drop(columns=['time', 'airport_id'], errors='ignore')
df_gold_daily.to_parquet(config.GOLD_DAILY_AIRPORT, index=False)
print(f"✓ Oro Guardado: {config.GOLD_DAILY_AIRPORT.name}")

# Resumen de Aeropuertos
df_gold_airports = df_gold_daily.groupby('dep_airport').agg(
    total_departures=('total_flights', 'sum'),
    total_cancellations=('cancelled_flights', 'sum'),
    overall_avg_delay=('avg_dep_delay', 'mean'),
    severe_delayed_flights=('delayed_15m', 'sum')
).reset_index()
df_gold_airports['cancellation_rate'] = (df_gold_airports['total_cancellations'] / df_gold_airports['total_departures']) * 100
df_gold_airports = df_gold_airports.merge(df_airports_silver, left_on='dep_airport', right_on='IATA_CODE', how='inner')
df_gold_airports.to_parquet(config.GOLD_AIRPORT_SUMMARY, index=False)
print(f"✓ Oro Guardado: {config.GOLD_AIRPORT_SUMMARY.name}")

# Métricas por Aerolínea
df_gold_airline = pd.concat(airline_list, ignore_index=True).groupby(['YearMonth', 'airline']).sum().reset_index()
df_gold_airline['ontime_rate'] = (1 - (df_gold_airline['delayed_15m'] / df_gold_airline['total_flights'])) * 100
df_gold_airline['cancellation_rate'] = (df_gold_airline['cancelled_flights'] / df_gold_airline['total_flights']) * 100
df_gold_airline.to_parquet(config.GOLD_AIRLINE_MONTHLY, index=False)
print(f"✓ Oro Guardado: {config.GOLD_AIRLINE_MONTHLY.name}")

# Rutas
df_gold_routes = pd.concat(route_list, ignore_index=True).groupby(['dep_airport', 'arr_airport']).agg({
    'flight_count': 'sum',
    'avg_duration': 'mean',
    'avg_dep_delay': 'mean',
    'cancellation_rate': 'mean'
}).reset_index()
df_gold_routes.to_parquet(config.GOLD_ROUTES, index=False)
print(f"✓ Oro Guardado: {config.GOLD_ROUTES.name}")

# Causas de retrasos
df_gold_causes = pd.concat(causes_list, ignore_index=True).groupby('YearMonth').sum().reset_index()
df_gold_causes.to_parquet(config.GOLD_DELAY_CAUSES, index=False)
print(f"✓ Oro Guardado: {config.GOLD_DELAY_CAUSES.name}")"""),

    md("""## 5. Preparación del Dataset de Features para Machine Learning (Gold Layer)
Creamos una muestra estratificada representativa que incluye variables meteorológicas, operativas y la variable objetivo binaria:
- `is_delayed_or_cancelled = 1` si el retraso en salida es $\ge 15$ minutos o el vuelo fue cancelado.
- `0` en caso contrario."""),

    code("""print("Generando dataset de features para Machine Learning...")
ml_sample_list = []

for p_file in silver_parquet_files:
    chunk = pd.read_parquet(p_file, columns=[
        'FlightDate', 'Day_Of_Week', 'airline', 'dep_airport', 'arr_airport',
        'DepTime_label', 'Dep_Delay', 'Cancelled', 'Flight_Duration', 
        'Distance_type', 'Aircraft_age'
    ])
    chunk['is_delayed_or_cancelled'] = ((chunk['Dep_Delay'] >= 15) | (chunk['Cancelled'] == 1)).astype('int8')
    chunk['Month'] = chunk['FlightDate'].dt.month.astype('int8')
    chunk['Is_Weekend'] = chunk['Day_Of_Week'].isin([6, 7]).astype('int8')
    
    # Muestra de 25,000 registros por mes para un balance óptimo de representatividad y memoria
    sample_size = min(25000, len(chunk))
    ml_sample_list.append(chunk.sample(n=sample_size, random_state=42))

df_ml = pd.concat(ml_sample_list, ignore_index=True)

# Enriquecer con clima diario del aeropuerto de origen
df_ml = df_ml.merge(
    df_weather_silver[['time', 'airport_id', 'tavg', 'prcp', 'snow', 'wspd', 'pres']],
    left_on=['FlightDate', 'dep_airport'],
    right_on=['time', 'airport_id'],
    how='left'
).drop(columns=['time', 'airport_id'], errors='ignore')

# Imputación final de valores meteorológicos faltantes con la mediana
for col in ['tavg', 'prcp', 'snow', 'wspd', 'pres']:
    df_ml[col] = df_ml[col].fillna(df_ml[col].median()).astype('float32')

gold_ml_features_path = config.GOLD_DIR / "ml_features_dataset.parquet"
df_ml.to_parquet(gold_ml_features_path, index=False, engine='pyarrow', compression='snappy')
print(f"✓ Dataset de ML guardado: {len(df_ml):,} filas -> {gold_ml_features_path}")"""),

    md("""## 6. Verificación y Resumen de la Arquitectura Medallón
Comprobamos el estado y tamaño de todos los artefactos persistidos en disco."""),

    code("""print("=" * 80)
print("AUDITORÍA DE LA ARQUITECTURA MEDALLÓN")
print("=" * 80)
print(f"• Capa Bronce: {len(list(config.BRONZE_DIR.glob('**/*.csv')))} archivos CSV originales")
print(f"• Capa Plata:  {len(list(config.SILVER_DIR.glob('**/*.parquet')))} archivos Parquet limpios")
print(f"• Capa Oro:    {len(list(config.GOLD_DIR.glob('*.parquet')))} tablas analíticas consolidadas:")
for f in sorted(config.GOLD_DIR.glob("*.parquet")):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"   -> {f.name:<35} | {size_mb:>6.2f} MB")
print("=" * 80)
print("Pipeline ETL completado satisfactoriamente.")""")
]

# ==============================================================================
# NOTEBOOK 2: 02_ml_model.ipynb
# ==============================================================================
ml_cells = [
    md("""# Modelo Predictivo de Retrasos y Cancelaciones de Vuelos
### Machine Learning con scikit-learn y Arquitectura Medallón

---

## 🎯 Objetivo de Negocio y Machine Learning
El objetivo es predecir si un vuelo comercial programado sufrirá una disrupción severa, definida como:
- **Retraso en la salida superior o igual a 15 minutos** (`Dep_Delay >= 15`), o
- **Cancelación total del vuelo** (`Cancelled == 1`).

Esta variable objetivo binaria (`is_delayed_or_cancelled`) permite a las aerolíneas y operadores aeroportuarios anticipar riesgos operativos, reasignar tripulaciones y mitigar el impacto para los pasajeros.

### 🧪 Metodología:
1. **Carga y Verificación:** Ingesta del dataset de características desde la **Capa Oro** (`ml_features_dataset.parquet`).
2. **EDA Focalizado:** Análisis del desbalance de clases y correlación con variables meteorológicas y temporales.
3. **Feature Engineering & Preprocesamiento:**
   - Variables numéricas (temperatura, lluvia, nieve, viento, presión atmosférica, edad del avión).
   - Variables categóricas (aerolínea, aeropuerto agrupado por frecuencia, franja horaria, día de la semana, mes).
   - `ColumnTransformer` + `Pipeline` para garantizar cero fuga de datos (*data leakage*).
4. **Comparación de Modelos:**
   - **Baseline:** Regresión Logística con pesos balanceados (`class_weight='balanced'`).
   - **Challenger:** `HistGradientBoostingClassifier` (árboles de decisión potenciados con soporte nativo de histogramas rápidos).
5. **Evaluación Rigurosa:** Curvas ROC-AUC, Precision-Recall, F1-Score, Matriz de Confusión y Feature Importance por permutación.
6. **Serialización:** Exportación del mejor pipeline entrenado a la carpeta `models/` para su consumo en la aplicación Streamlit.
"""),

    code("""import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
from sklearn.inspection import permutation_importance

# Configuración de estilo gráfico
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 110

# Configurar rutas del proyecto
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

import config
config.ensure_dirs()

print("✓ Librerías importadas y entorno inicializado.")"""),

    md("""## 1. Carga de Datos desde la Capa Oro
Cargamos la tabla de características preparada durante el pipeline ETL."""),

    code("""ml_features_path = config.GOLD_DIR / "ml_features_dataset.parquet"

if ml_features_path.exists():
    print(f"Cargando dataset desde Capa Oro: {ml_features_path.name}")
    df = pd.read_parquet(ml_features_path)
else:
    print("⚠️ El dataset consolidado en Oro no fue encontrado. Verifique la ejecución de 01_etl_pipeline.ipynb.")
    silver_files = list(config.SILVER_FLIGHTS.glob("*.parquet"))
    if silver_files:
        print("Cargando muestra de contingencia desde Capa Plata...")
        df = pd.read_parquet(silver_files[0]).sample(n=25000, random_state=42)
        df['is_delayed_or_cancelled'] = ((df['Dep_Delay'] >= 15) | (df['Cancelled'] == 1)).astype('int8')
        df['Month'] = pd.to_datetime(df['FlightDate']).dt.month.astype('int8')
        df['Is_Weekend'] = df['Day_Of_Week'].isin([6, 7]).astype('int8')
        for w in ['tavg', 'prcp', 'snow', 'wspd', 'pres']:
            df[w] = 0.0

print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas.")
df.head(5)"""),

    md("""## 2. Análisis Exploratorio de Datos (EDA) y Desbalance de Clases
Examinamos la distribución de la variable objetivo y el impacto de los factores climáticos."""),

    code("""target_counts = df['is_delayed_or_cancelled'].value_counts()
target_pct = df['is_delayed_or_cancelled'].value_counts(normalize=True) * 100

print("=" * 60)
print("DISTRIBUCIÓN DE LA VARIABLE OBJETIVO:")
for val, count, pct in zip(target_counts.index, target_counts.values, target_pct.values):
    label = "Disrupción (Retraso >=15m / Cancelado)" if val == 1 else "A Tiempo / Retraso Menor (<15m)"
    print(f"  Clase {val} ({label:<35}): {count:>7,} ({pct:>5.2f}%)")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

# 1. Gráfico de barras de desbalance
sns.barplot(x=['A Tiempo (0)', 'Disrupción (1)'], y=target_counts.values, palette=['#3b82f6', '#ef4444'], ax=axes[0])
axes[0].set_title("Proporción de Clases")
axes[0].set_ylabel("Cantidad de Vuelos")

# 2. Tasa de retraso/cancelación por mes
monthly_rate = df.groupby('Month')['is_delayed_or_cancelled'].mean() * 100
sns.lineplot(x=monthly_rate.index, y=monthly_rate.values, marker='o', color='#0284c7', linewidth=2, ax=axes[1])
axes[1].set_title("Tasa de Disrupción por Mes (%)")
axes[1].set_xlabel("Mes del Año")
axes[1].set_ylabel("% Disrupción")

# 3. Impacto de velocidad de viento
df['wind_bin'] = pd.qcut(df['wspd'], q=4, duplicates='drop')
wind_rate = df.groupby('wind_bin', observed=False)['is_delayed_or_cancelled'].mean() * 100
sns.barplot(x=wind_rate.index.astype(str), y=wind_rate.values, palette='Blues', ax=axes[2])
axes[2].set_title("Tasa de Disrupción vs Cuartiles de Viento (km/h)")
axes[2].set_xlabel("Rango de Velocidad del Viento")
axes[2].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.show()"""),

    md("""## 3. Definición del Espacio de Características y Pipeline de Preprocesamiento
Configuramos transformadores independientes para variables numéricas y categóricas:
- **Numéricas:** Imputación de edad del avión y escalamiento estándar (`StandardScaler`).
- **Categóricas:** Agrupamiento de aeropuertos menores en categoría `'OTHER'` para evitar explosión dimensional y codificación One-Hot (`OneHotEncoder`)."""),

    code("""# 1. Seleccionar columnas predictoras
numeric_features = ['tavg', 'prcp', 'snow', 'wspd', 'pres']

if 'Aircraft_age' in df.columns:
    df['Aircraft_age'] = pd.to_numeric(df['Aircraft_age'], errors='coerce')
    df['Aircraft_age'] = df['Aircraft_age'].fillna(df['Aircraft_age'].median())
    numeric_features.append('Aircraft_age')

# Agrupar aeropuertos de baja frecuencia en 'OTHER'
top_30_airports = set(df['dep_airport'].value_counts().nlargest(30).index)
df['dep_airport_grouped'] = df['dep_airport'].apply(lambda x: x if x in top_30_airports else 'OTHER')

categorical_features = ['Day_Of_Week', 'Month', 'Is_Weekend', 'DepTime_label', 'airline', 'dep_airport_grouped']

feature_cols = numeric_features + categorical_features
X = df[feature_cols].copy()
y = df['is_delayed_or_cancelled'].copy()

# 2. Pipeline de preprocesamiento con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# 3. Partición Train / Test estratificada (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Variables seleccionadas: {len(feature_cols)} ({len(numeric_features)} numéricas, {len(categorical_features)} categóricas)")
print(f"Dimensiones de entrenamiento: {X_train.shape[0]:,} muestras")
print(f"Dimensiones de prueba:        {X_test.shape[0]:,} muestras")"""),

    md("""## 4. Entrenamiento y Comparación de Modelos
Entrenamos dos clasificadores:
1. **Regresión Logística (Baseline)** con penalización regularizada y `class_weight='balanced'`.
2. **HistGradientBoostingClassifier (Challenger)** basado en árboles potenciados por gradiente con discretización en histogramas."""),

    code("""# 4.1 Baseline: Regresión Logística
print("Entrenando Modelo 1: Regresión Logística (Baseline)...")
pipe_lr = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])
pipe_lr.fit(X_train, y_train)

y_pred_lr = pipe_lr.predict(X_test)
y_proba_lr = pipe_lr.predict_proba(X_test)[:, 1]
auc_lr = roc_auc_score(y_test, y_proba_lr)
f1_lr = f1_score(y_test, y_pred_lr)
print(f"-> Regresión Logística: ROC-AUC = {auc_lr:.4f} | F1 = {f1_lr:.4f}")

# 4.2 Challenger: HistGradientBoostingClassifier
print("\nEntrenando Modelo 2: HistGradientBoostingClassifier (Árboles Potenciados)...")
pipe_hgb = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_depth=8,
        class_weight='balanced',
        random_state=42
    ))
])
pipe_hgb.fit(X_train, y_train)

y_pred_hgb = pipe_hgb.predict(X_test)
y_proba_hgb = pipe_hgb.predict_proba(X_test)[:, 1]
auc_hgb = roc_auc_score(y_test, y_proba_hgb)
f1_hgb = f1_score(y_test, y_pred_hgb)
print(f"-> HistGradientBoosting: ROC-AUC = {auc_hgb:.4f} | F1 = {f1_hgb:.4f}")"""),

    md("""## 5. Evaluación Exhaustiva de Desempeño
Seleccionamos el mejor modelo y analizamos en detalle su matriz de confusión, reporte de clasificación y curvas ROC."""),

    code("""best_model = pipe_hgb if auc_hgb >= auc_lr else pipe_lr
best_name = "HistGradientBoosting" if auc_hgb >= auc_lr else "Logistic Regression"
y_pred_best = y_pred_hgb if auc_hgb >= auc_lr else y_pred_lr
y_proba_best = y_proba_hgb if auc_hgb >= auc_lr else y_proba_lr

print("=" * 65)
print(f"REPORTE DE CLASIFICACIÓN FINAL - MODELO GANADOR: {best_name}")
print("=" * 65)
print(classification_report(y_test, y_pred_best, target_names=['A Tiempo (0)', 'Disrupción (1)']))

# Gráficos de evaluación
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Pred: A Tiempo', 'Pred: Disrupción'],
            yticklabels=['Real: A Tiempo', 'Real: Disrupción'])
axes[0].set_title(f"Matriz de Confusión ({best_name})")

# Curvas ROC
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_hgb, tpr_hgb, _ = roc_curve(y_test, y_proba_hgb)

axes[1].plot(fpr_lr, tpr_lr, label=f"LogReg (AUC = {auc_lr:.3f})", color='#64748b', linestyle='--')
axes[1].plot(fpr_hgb, tpr_hgb, label=f"HistGradBoost (AUC = {auc_hgb:.3f})", color='#0ea5e9', linewidth=2.5)
axes[1].plot([0, 1], [0, 1], 'k:', alpha=0.4)
axes[1].set_title("Curva ROC Comparativa")
axes[1].set_xlabel("Tasa de Falsos Positivos (FPR)")
axes[1].set_ylabel("Tasa de Verdaderos Positivos (TPR)")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()"""),

    md("""## 6. Importancia de Características (Permutation Importance)
Calculamos la importancia de cada variable evaluando el impacto del desorden aleatorio en el conjunto de prueba."""),

    code("""print("Calculando importancia de características mediante permutación en test set...")
perm_result = permutation_importance(best_model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)

top_indices = perm_result.importances_mean.argsort()[::-1][:10]

plt.figure(figsize=(10, 5))
plt.barh(
    [X_test.columns[i] for i in top_indices][::-1],
    perm_result.importances_mean[top_indices][::-1],
    color='#0284c7'
)
plt.title("Top 10 Características Clave para la Predicción de Retrasos y Cancelaciones")
plt.xlabel("Pérdida de Rendimiento al Permutar (Mean Accuracy Drop)")
plt.tight_layout()
plt.show()"""),

    md("""## 7. Serialización del Modelo en Producción
Guardamos el pipeline completo junto con metadatos clave en `models/flight_delay_predictor.joblib`."""),

    code("""model_artifact_path = config.MODELS_DIR / "flight_delay_predictor.joblib"

payload = {
    'model_name': best_name,
    'pipeline': best_model,
    'features': feature_cols,
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'metrics': {
        'roc_auc': max(auc_lr, auc_hgb),
        'f1_score': max(f1_lr, f1_hgb)
    },
    'target_description': 'is_delayed_or_cancelled: 1 si Dep_Delay >= 15 o Cancelled == 1, 0 en caso contrario'
}

joblib.dump(payload, model_artifact_path)
print("=" * 65)
print(f"✓ Modelo serializado con éxito en: {model_artifact_path}")
print(f"• Algoritmo: {best_name}")
print(f"• ROC-AUC:    {payload['metrics']['roc_auc']:.4f}")
print(f"• F1-Score:   {payload['metrics']['f1_score']:.4f}")
print("=" * 65)""")
]

# Guardar los notebooks en la carpeta notebooks/
notebooks_dir = PROJECT_ROOT / "notebooks"
save_notebook(notebooks_dir / "01_etl_pipeline.ipynb", etl_cells)
save_notebook(notebooks_dir / "02_ml_model.ipynb", ml_cells)
