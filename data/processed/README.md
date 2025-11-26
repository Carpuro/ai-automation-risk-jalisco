# Datos Procesados

Este directorio contiene datos **transformados y limpios** listos para análisis.

## 📊 Archivos Generados

Los archivos aquí son **generados automáticamente** por los scripts de procesamiento.

### Formato Parquet (Recomendado)

```
data/processed/
├── occupations_clean.parquet
├── risk_analysis_results.parquet
└── features_engineered.parquet
```

**Ventajas:**
- Compresión eficiente
- Lectura rápida en PySpark
- Preserva tipos de datos
- Columnar storage

### Formato CSV (Para análisis externo)

```
data/processed/
├── occupations_clean.csv
├── risk_analysis_results.csv
└── features_engineered.csv
```

**Uso:** Análisis en Excel, Tableau, Power BI

## 🔄 Cómo se Generan

### Opción 1: Pipeline Completo

```bash
python src/main.py --mode real \
  --occupation-data data/raw/onet/Occupation_Data.txt \
  --employment-data data/raw/enoe_jalisco.csv
```

### Opción 2: Notebook

Ejecutar `notebooks/automation_risk_analysis.ipynb` hasta la sección de Feature Engineering.

### Opción 3: Módulos Individuales

```python
from data_loader import load_onet_occupations, load_enoe_jalisco
from data_preprocessing import preprocess_pipeline
from feature_engineering import feature_engineering_pipeline

# Cargar
df_onet = load_onet_occupations(spark, 'data/raw/onet/Occupation_Data.txt')
df_enoe = load_enoe_jalisco(spark, 'data/raw/enoe_jalisco.csv')

# Integrar (simplificado)
df = df_onet.join(df_enoe, how='inner')

# Preprocesar
df_clean = preprocess_pipeline(df)

# Feature engineering
df_features = feature_engineering_pipeline(df_clean)

# Guardar
df_features.to_spark().write.parquet('data/processed/occupations_clean.parquet')
```

## 📁 Estructura de Archivos

### occupations_clean.parquet
Dataset limpio después de preprocesamiento.

**Transformaciones aplicadas:**
- Valores faltantes manejados
- Duplicados eliminados
- Outliers filtrados (opcional)
- Tipos de datos corregidos

### features_engineered.parquet
Dataset con features adicionales creados.

**Features nuevos incluyen:**
- `routine_index` - Índice de rutinización
- `cognitive_demand` - Demanda cognitiva
- `social_interaction` - Interacción social
- `creativity` - Nivel de creatividad
- `complexity_index` - Complejidad general
- `automation_susceptibility` - Susceptibilidad a automatización
- Features temporales (proyecciones 2025-2030)
- Agregaciones por sector

### risk_analysis_results.parquet
Resultados completos del análisis de riesgo.

**Columnas adicionales:**
- `automation_risk` - Riesgo calculado (0-1)
- `risk_category` - Alto/Medio/Bajo
- `sector_avg_risk` - Riesgo promedio del sector
- `risk_deviation_from_sector` - Desviación vs sector

## 🚫 No Subir a Git

Estos archivos **NO** deben subirse a Git porque:
- Son grandes (>50MB)
- Se pueden regenerar
- Pueden contener datos sensibles

Están incluidos en `.gitignore`.

## ✅ Validación

Para verificar integridad de archivos procesados:

```python
import pyspark.pandas as ps

# Cargar
df = ps.read_parquet('data/processed/risk_analysis_results.parquet')

# Validar
print(f"Filas: {len(df):,}")
print(f"Columnas: {len(df.columns)}")
print(f"Valores nulos: {df.isnull().sum().sum()}")

# Verificar columnas clave
required_cols = ['occupation_id', 'automation_risk', 'risk_category']
for col in required_cols:
    assert col in df.columns, f"Falta columna: {col}"

print("✓ Dataset válido")
```

## 🔄 Regenerar Datos

Si los archivos se corrompen o se pierden:

```bash
# Borrar archivos procesados
rm -rf data/processed/*.parquet

# Regenerar
python src/main.py --mode real \
  --occupation-data data/raw/onet/Occupation_Data.txt \
  --employment-data data/raw/enoe_jalisco.csv
```

---

**Última actualización:** Noviembre 2025  
**Generados por:** Pipeline de procesamiento PySpark
