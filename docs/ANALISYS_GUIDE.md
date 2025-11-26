# 📊 Guía de Análisis Paso a Paso

## Modelo Predictivo de Riesgo de Sustitución Laboral por IA - Jalisco

**Autor:** Carlos Pulido Rosas  
**Proyecto de Tesis:** Maestría en Ciencias de los Datos - CUCEA

---

## 🎯 Objetivo de esta Guía

Esta guía te llevará paso a paso por todo el proceso de análisis, desde la carga de datos hasta la generación de resultados y visualizaciones. Está diseñada para ser seguida tanto en **Jupyter Notebook** como en **scripts Python**.

---

## 📋 Tabla de Contenidos

1. [Preparación del Entorno](#1-preparación-del-entorno)
2. [Carga de Datos](#2-carga-de-datos)
3. [Exploración Inicial](#3-exploración-inicial)
4. [Limpieza y Preprocesamiento](#4-limpieza-y-preprocesamiento)
5. [Feature Engineering](#5-feature-engineering)
6. [Análisis de Riesgo de Automatización](#6-análisis-de-riesgo-de-automatización)
7. [Análisis por Dimensiones](#7-análisis-por-dimensiones)
8. [Visualizaciones](#8-visualizaciones)
9. [Modelado Predictivo](#9-modelado-predictivo)
10. [Interpretación de Resultados](#10-interpretación-de-resultados)
11. [Generación de Reportes](#11-generación-de-reportes)

---

## 1. Preparación del Entorno

### 1.1 Verificar Instalación

```bash
# Activar entorno
conda activate ai_automation_thesis

# Verificar que todo funcione
python verify_setup.py
```

**Salida esperada:**
```
✓ Python 3.10.x compatible
✓ pyspark 3.5.0
✓ pandas 2.1.4
✓ Spark funcionando correctamente
```

### 1.2 Iniciar Jupyter Notebook

```bash
jupyter notebook
```

### 1.3 Importaciones Iniciales

```python
# Importaciones básicas
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession
import pyspark.pandas as ps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Configurar visualizaciones
%matplotlib inline
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Librerías importadas exitosamente")
```

### 1.4 Crear Spark Session

```python
from src.data_loader import create_spark_session

# Crear sesión
spark = create_spark_session(
    app_name="AI_Automation_Risk_Jalisco",
    memory="8g"
)

print(f"✓ Spark Session creada")
print(f"  Versión: {spark.version}")
print(f"  Master: {spark.sparkContext.master}")
```

---

## 2. Carga de Datos

### 2.1 Opción A: Usar Datos Reales (Recomendado para Tesis Final)

#### Paso 1: Cargar O*NET

```python
from src.data_loader import load_onet_occupations

# Cargar datos de ocupaciones
onet_occupations = load_onet_occupations(
    spark, 
    'data/raw/onet/Occupation Data.txt'
)

# Cargar habilidades
onet_skills = load_onet_occupations(
    spark,
    'data/raw/onet/Skills.txt'
)

# Cargar actividades
onet_activities = load_onet_occupations(
    spark,
    'data/raw/onet/Work Activities.txt'
)

print(f"✓ Datos O*NET cargados")
```

#### Paso 2: Cargar ENOE Jalisco

```python
from src.data_loader import load_enoe_jalisco

# Cargar datos de empleo
enoe_jalisco = load_enoe_jalisco(
    spark,
    'data/raw/enoe_jalisco.csv',
    encoding='latin1'
)

print(f"✓ Datos ENOE Jalisco cargados")
```

#### Paso 3: Integrar Fuentes

```python
# Mapear SOC a SINCO
mapping = spark.read.csv(
    'data/mappings/soc_sinco_mapping.csv',
    header=True,
    inferSchema=True
)

# Hacer joins
onet_full = onet_occupations.join(
    onet_skills, 
    on='O*NET-SOC Code', 
    how='inner'
).join(
    onet_activities,
    on='O*NET-SOC Code',
    how='inner'
)

# Mapear a SINCO
onet_mapped = onet_full.join(
    mapping,
    onet_full['O*NET-SOC Code'] == mapping['soc_code'],
    how='inner'
)

# Integrar con ENOE
df_integrated = onet_mapped.join(
    enoe_jalisco,
    onet_mapped['sinco_code'] == enoe_jalisco['clase2'],
    how='inner'
)

print(f"✓ Datos integrados: {df_integrated.count():,} registros")
```

### 2.2 Opción B: Usar Datos Simulados (Para Desarrollo/Pruebas)

```python
from src.data_loader import load_sample_data

# Generar datos de muestra
df_spark = load_sample_data(spark, n_occupations=200)

print(f"✓ Datos simulados generados: {df_spark.count():,} ocupaciones")
```

### 2.3 Convertir a pyspark.pandas

```python
from src.data_loader import convert_to_pandas_api

# Convertir para facilitar manipulación
df = convert_to_pandas_api(df_spark)

print(f"✓ DataFrame convertido a pyspark.pandas")
print(f"  Shape: {df.shape}")
print(f"  Columnas: {len(df.columns)}")
```

---

## 3. Exploración Inicial

### 3.1 Vista General del Dataset

```python
# Información básica
print("="*80)
print("INFORMACIÓN GENERAL DEL DATASET")
print("="*80)
print(f"\nDimensiones: {df.shape}")
print(f"Filas: {df.shape[0]:,}")
print(f"Columnas: {df.shape[1]}")

# Primeras filas
print("\nPrimeras 5 ocupaciones:")
print(df.head())

# Tipos de datos
print("\nTipos de datos:")
print(df.dtypes)
```

### 3.2 Estadísticas Descriptivas

```python
# Columnas numéricas
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*80)
print(df[numeric_cols].describe())
```

### 3.3 Análisis de Valores Faltantes

```python
from src.data_preprocessing import analyze_missing_values

# Analizar nulos
missing_df = analyze_missing_values(df)

if len(missing_df) > 0:
    print("\nColumnas con valores faltantes:")
    print(missing_df.head(10))
else:
    print("\n✓ No hay valores faltantes")
```

### 3.4 Distribuciones Básicas

```python
# Distribución por sector
print("\nDistribución por Sector Económico:")
print(df['sector'].value_counts())

# Distribución por nivel educativo
print("\nDistribución por Nivel Educativo:")
print(df['education_level'].value_counts().sort_index())

# Estadísticas de trabajadores
print(f"\nTrabajadores en Jalisco:")
print(f"  Total: {df['workers_jalisco'].sum():,}")
print(f"  Promedio por ocupación: {df['workers_jalisco'].mean():,.0f}")
print(f"  Mediana: {df['workers_jalisco'].median():,.0f}")
```

---

## 4. Limpieza y Preprocesamiento

### 4.1 Manejar Valores Faltantes

```python
from src.data_preprocessing import handle_missing_values

# Limpiar datos
df_clean = handle_missing_values(df, strategy='auto')

print(f"✓ Datos limpios: {df_clean.shape[0]:,} filas")
```

### 4.2 Eliminar Duplicados

```python
from src.data_preprocessing import remove_duplicates

# Quitar duplicados
df_clean = remove_duplicates(df_clean, subset=['occupation_name'])

print(f"✓ Duplicados eliminados")
```

### 4.3 Filtrar Outliers (Opcional)

```python
from src.data_preprocessing import filter_outliers

# Filtrar valores extremos en salarios
outlier_cols = ['avg_salary_mxn', 'workers_jalisco']
df_clean = filter_outliers(
    df_clean, 
    columns=outlier_cols,
    method='iqr',
    threshold=3.0
)

print(f"✓ Outliers filtrados")
```

### 4.4 Validar Calidad

```python
from src.data_preprocessing import validate_data_quality

# Validar
metrics = validate_data_quality(df_clean)

print("\nMÉTRICAS DE CALIDAD:")
for key, value in metrics.items():
    print(f"  {key}: {value}")
```

---

## 5. Feature Engineering

### 5.1 Crear Features Base

```python
# Índice de rutinización (si no existe)
if 'routine_index' not in df_clean.columns:
    df_clean['routine_index'] = (
        df_clean.get('task_repetitive', 50) * 0.4 +
        df_clean.get('task_predictable', 50) * 0.35 +
        df_clean.get('task_structured', 50) * 0.25
    )

# Demanda cognitiva (si no existe)
if 'cognitive_demand' not in df_clean.columns:
    cognitive_cols = [c for c in df_clean.columns if 'cognitive' in c.lower() or 'thinking' in c.lower()]
    if cognitive_cols:
        df_clean['cognitive_demand'] = df_clean[cognitive_cols].mean(axis=1)
    else:
        df_clean['cognitive_demand'] = 50

# Interacción social (si no existe)
if 'social_interaction' not in df_clean.columns:
    social_cols = [c for c in df_clean.columns if 'social' in c.lower() or 'communication' in c.lower()]
    if social_cols:
        df_clean['social_interaction'] = df_clean[social_cols].mean(axis=1)
    else:
        df_clean['social_interaction'] = 50

# Creatividad (si no existe)
if 'creativity' not in df_clean.columns:
    creative_cols = [c for c in df_clean.columns if 'creative' in c.lower() or 'original' in c.lower()]
    if creative_cols:
        df_clean['creativity'] = df_clean[creative_cols].mean(axis=1)
    else:
        df_clean['creativity'] = 50

print("✓ Features base creados")
```

### 5.2 Features Derivados

```python
# Ratio salario/educación
df_clean['salary_education_ratio'] = df_clean['avg_salary_mxn'] / (df_clean['education_level'] + 1)

# Índice de complejidad
df_clean['complexity_index'] = (
    df_clean['cognitive_demand'] * 0.5 +
    df_clean['creativity'] * 0.3 +
    (100 - df_clean['routine_index']) * 0.2
)

# Categoría de trabajadores
def categorize_workers(count):
    if count < 1000:
        return 'Pequeña'
    elif count < 10000:
        return 'Mediana'
    else:
        return 'Grande'

df_clean['occupation_size'] = df_clean['workers_jalisco'].apply(categorize_workers)

print("✓ Features derivados creados")
```

### 5.3 Normalizar Features (Opcional)

```python
from src.data_preprocessing import normalize_columns

# Normalizar para modelado
norm_cols = ['routine_index', 'cognitive_demand', 'social_interaction', 'creativity']
df_clean = normalize_columns(df_clean, columns=norm_cols, method='minmax')

print("✓ Features normalizados")
```

---

## 6. Análisis de Riesgo de Automatización

### 6.1 Calcular Riesgo Base

```python
from src.automation_analyzer import AutomationRiskAnalyzer

# Crear analizador
analyzer = AutomationRiskAnalyzer()

# Calcular riesgo
df_risk = analyzer.calculate_automation_risk(
    df_clean, 
    method='frey_osborne'
)

print("✓ Riesgo de automatización calculado")
print(f"\nDistribución de riesgo:")
print(df_risk['automation_risk'].describe())
```

### 6.2 Categorizar Riesgo

```python
# Categorizar en Alto/Medio/Bajo
df_risk = analyzer.categorize_risk(
    df_risk,
    thresholds=(0.30, 0.70)
)

print("\nDistribución por categoría de riesgo:")
print(df_risk['risk_category'].value_counts())
print("\nPorcentajes:")
print(df_risk['risk_category'].value_counts(normalize=True) * 100)
```

### 6.3 Top Ocupaciones en Riesgo

```python
# Top 20 más riesgosas
top_risk = analyzer.identify_top_at_risk(df_risk, n=20)

print("\n" + "="*80)
print("TOP 20 OCUPACIONES CON MAYOR RIESGO DE AUTOMATIZACIÓN")
print("="*80)
print(top_risk[['occupation_name', 'sector', 'automation_risk', 'workers_jalisco']])
```

### 6.4 Ocupaciones Más Seguras

```python
# Top 20 más seguras
low_risk = analyzer.identify_low_risk_occupations(df_risk, n=20)

print("\n" + "="*80)
print("TOP 20 OCUPACIONES MÁS SEGURAS")
print("="*80)
print(low_risk[['occupation_name', 'sector', 'automation_risk']])
```

---

## 7. Análisis por Dimensiones

### 7.1 Análisis por Sector Económico

```python
# Análisis por sector
sector_analysis = analyzer.analyze_by_sector(df_risk)

print("\n" + "="*80)
print("ANÁLISIS POR SECTOR ECONÓMICO")
print("="*80)
print(sector_analysis.sort_values('risk_mean', ascending=False))
```

**Interpretación:**
- Sectores con `risk_mean` > 0.70: Alto riesgo
- Sectores con más trabajadores requieren más atención

### 7.2 Análisis por Nivel Educativo

```python
# Análisis por educación
education_analysis = analyzer.analyze_by_education(df_risk)

print("\n" + "="*80)
print("ANÁLISIS POR NIVEL EDUCATIVO")
print("="*80)
print(education_analysis)
```

**Pregunta clave:** ¿Educación superior protege contra automatización?

### 7.3 Impacto Económico

```python
# Calcular impacto
impact = analyzer.calculate_economic_impact(df_risk)

print("\n" + "="*80)
print("IMPACTO ECONÓMICO DE LA AUTOMATIZACIÓN")
print("="*80)
print(f"Total de trabajadores: {impact['total_workers']:,}")
print(f"Trabajadores en alto riesgo: {impact['workers_high_risk']:,} ({impact['pct_workers_at_risk']:.1f}%)")
if impact['salary_at_risk_mxn']:
    print(f"Masa salarial en riesgo: ${impact['salary_at_risk_mxn']:,.2f} MXN")
```

---

## 8. Visualizaciones

### 8.1 Distribución de Riesgo

```python
# Histograma
plt.figure(figsize=(12, 6))
plt.hist(df_risk['automation_risk'], bins=50, edgecolor='black', alpha=0.7)
plt.axvline(0.30, color='green', linestyle='--', label='Umbral Bajo-Medio')
plt.axvline(0.70, color='red', linestyle='--', label='Umbral Medio-Alto')
plt.xlabel('Riesgo de Automatización')
plt.ylabel('Frecuencia')
plt.title('Distribución del Riesgo de Automatización en Jalisco')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 8.2 Riesgo por Sector

```python
# Gráfico de barras
sector_risk = df_risk.groupby('sector')['automation_risk'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sector_risk.plot(kind='barh', color='coral')
plt.xlabel('Riesgo Promedio')
plt.title('Riesgo de Automatización por Sector Económico')
plt.axvline(0.70, color='red', linestyle='--', alpha=0.5, label='Alto Riesgo')
plt.axvline(0.30, color='green', linestyle='--', alpha=0.5, label='Bajo Riesgo')
plt.legend()
plt.tight_layout()
plt.show()
```

### 8.3 Scatter: Salario vs Riesgo

```python
# Scatter plot interactivo
fig = px.scatter(
    df_risk.to_pandas(),
    x='avg_salary_mxn',
    y='automation_risk',
    color='sector',
    size='workers_jalisco',
    hover_data=['occupation_name'],
    title='Relación entre Salario y Riesgo de Automatización',
    labels={
        'avg_salary_mxn': 'Salario Promedio (MXN)',
        'automation_risk': 'Riesgo de Automatización'
    }
)
fig.show()
```

### 8.4 Mapa de Calor: Sector vs Educación

```python
# Heatmap
pivot_data = df_risk.groupby(['sector', 'education_level'])['automation_risk'].mean().reset_index()
pivot_table = pivot_data.pivot(index='sector', columns='education_level', values='automation_risk')

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn_r', cbar_kws={'label': 'Riesgo'})
plt.title('Riesgo de Automatización: Sector vs Nivel Educativo')
plt.xlabel('Nivel Educativo')
plt.ylabel('Sector Económico')
plt.tight_layout()
plt.show()
```

### 8.5 Treemap: Impacto por Sector

```python
# Treemap de trabajadores en riesgo
sector_impact = df_risk[df_risk['automation_risk'] >= 0.70].groupby('sector').agg({
    'workers_jalisco': 'sum',
    'automation_risk': 'mean'
}).reset_index()

fig = px.treemap(
    sector_impact,
    path=['sector'],
    values='workers_jalisco',
    color='automation_risk',
    color_continuous_scale='Reds',
    title='Trabajadores en Alto Riesgo por Sector'
)
fig.show()
```

---

## 9. Modelado Predictivo

### 9.1 Preparar Datos para Modelado

```python
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline

# Convertir de vuelta a Spark para MLlib
df_spark_ml = df_risk.to_spark()

# Seleccionar features
feature_cols = [
    'routine_index',
    'cognitive_demand', 
    'social_interaction',
    'creativity',
    'education_level',
    'avg_salary_mxn'
]

# Ensamblar features
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol='features',
    handleInvalid='skip'
)

# Indexar target
indexer = StringIndexer(
    inputCol='risk_category',
    outputCol='label'
)

print("✓ Pipeline de features configurado")
```

### 9.2 Split Train/Test

```python
# Dividir datos
train_df, test_df = df_spark_ml.randomSplit([0.8, 0.2], seed=42)

print(f"✓ Datos divididos:")
print(f"  Entrenamiento: {train_df.count():,} registros")
print(f"  Prueba: {test_df.count():,} registros")
```

### 9.3 Entrenar Modelo de Clasificación

```python
from pyspark.ml.classification import RandomForestClassifier

# Crear modelo
rf = RandomForestClassifier(
    featuresCol='features',
    labelCol='label',
    numTrees=100,
    maxDepth=10,
    seed=42
)

# Pipeline completo
pipeline = Pipeline(stages=[assembler, indexer, rf])

# Entrenar
print("Entrenando modelo...")
model = pipeline.fit(train_df)

print("✓ Modelo entrenado")
```

### 9.4 Evaluar Modelo

```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Predecir
predictions = model.transform(test_df)

# Evaluar
evaluator = MulticlassClassificationEvaluator(
    labelCol='label',
    predictionCol='prediction',
    metricName='accuracy'
)

accuracy = evaluator.evaluate(predictions)
print(f"\n✓ Accuracy: {accuracy:.3f}")

# Otras métricas
for metric in ['weightedPrecision', 'weightedRecall', 'f1']:
    evaluator.setMetricName(metric)
    score = evaluator.evaluate(predictions)
    print(f"  {metric}: {score:.3f}")
```

### 9.5 Feature Importance

```python
# Importancia de features
rf_model = model.stages[-1]
importances = rf_model.featureImportances

feature_importance_df = ps.DataFrame({
    'feature': feature_cols,
    'importance': importances.toArray()
}).sort_values('importance', ascending=False)

print("\n" + "="*80)
print("IMPORTANCIA DE FEATURES")
print("="*80)
print(feature_importance_df)

# Visualizar
plt.figure(figsize=(10, 6))
feature_importance_df.plot(x='feature', y='importance', kind='barh', legend=False)
plt.xlabel('Importancia')
plt.title('Importancia de Características en el Modelo')
plt.tight_layout()
plt.show()
```

---

## 10. Interpretación de Resultados

### 10.1 Hallazgos Clave

```python
print("\n" + "="*80)
print("HALLAZGOS CLAVE DEL ANÁLISIS")
print("="*80)

# 1. Distribución general
total_high = len(df_risk[df_risk['automation_risk'] >= 0.70])
pct_high = (total_high / len(df_risk)) * 100
print(f"\n1. DISTRIBUCIÓN DE RIESGO")
print(f"   - {pct_high:.1f}% de ocupaciones en alto riesgo")

# 2. Sector más afectado
top_sector = sector_analysis.iloc[0]
print(f"\n2. SECTOR MÁS AFECTADO")
print(f"   - {top_sector['sector']}: {top_sector['risk_mean']:.3f} riesgo promedio")
print(f"   - {top_sector['total_workers']:,} trabajadores afectados")

# 3. Correlación educación-riesgo
corr = df_risk[['education_level', 'automation_risk']].corr().iloc[0, 1]
print(f"\n3. EDUCACIÓN vs RIESGO")
print(f"   - Correlación: {corr:.3f}")
if corr < -0.3:
    print(f"   - Mayor educación → Menor riesgo ✓")
else:
    print(f"   - Relación débil entre educación y riesgo")

# 4. Ocupación más riesgosa
most_risky = df_risk.nlargest(1, 'automation_risk').iloc[0]
print(f"\n4. OCUPACIÓN MÁS RIESGOSA")
print(f"   - {most_risky['occupation_name']}")
print(f"   - Riesgo: {most_risky['automation_risk']:.3f}")
print(f"   - Trabajadores: {most_risky['workers_jalisco']:,}")
```

### 10.2 Implicaciones para Política Pública

```python
print("\n" + "="*80)
print("RECOMENDACIONES DE POLÍTICA PÚBLICA")
print("="*80)

# Sectores prioritarios
print("\n1. SECTORES PRIORITARIOS PARA INTERVENCIÓN:")
priority_sectors = sector_analysis.head(3)
for idx, row in priority_sectors.iterrows():
    print(f"   - {row['sector']}: {row['total_workers']:,} trabajadores en riesgo")

# Programas de reconversión
high_risk_workers = impact['workers_high_risk']
print(f"\n2. PROGRAMAS DE RECONVERSIÓN LABORAL:")
print(f"   - Meta: {high_risk_workers:,} trabajadores")
print(f"   - Presupuesto estimado: ${high_risk_workers * 50000:,.2f} MXN")
print(f"     (asumiendo $50,000 MXN por trabajador)")

# Educación continua
print(f"\n3. EDUCACIÓN Y CAPACITACIÓN:")
print(f"   - Enfoque en habilidades no automatizables:")
print(f"     • Pensamiento crítico y creativo")
print(f"     • Inteligencia emocional y social")
print(f"     • Resolución de problemas complejos")
```

---

## 11. Generación de Reportes

### 11.1 Reporte de Texto

```python
from src.automation_analyzer import generate_risk_report

# Generar reporte
report_text = generate_risk_report(
    df_risk,
    output_path='outputs/reports/risk_analysis_report.txt'
)

print("✓ Reporte generado: outputs/reports/risk_analysis_report.txt")
```

### 11.2 Exportar Resultados

```python
# Guardar dataset procesado
df_risk.to_spark().write.mode('overwrite').parquet('outputs/processed/risk_analysis_results.parquet')

# Guardar en CSV para Excel
df_risk.to_pandas().to_csv('outputs/reports/risk_analysis_results.csv', index=False)

print("✓ Resultados exportados")
```

### 11.3 Guardar Modelo

```python
# Guardar modelo entrenado
model.write().overwrite().save('outputs/models/automation_risk_model')

print("✓ Modelo guardado: outputs/models/automation_risk_model")
```

---

## 🎯 Checklist de Análisis Completo

Verifica que hayas completado todos los pasos:

- [ ] ✅ Entorno configurado y verificado
- [ ] ✅ Datos cargados (reales o simulados)
- [ ] ✅ Exploración inicial completada
- [ ] ✅ Datos limpios y preprocesados
- [ ] ✅ Features engineered
- [ ] ✅ Riesgo de automatización calculado
- [ ] ✅ Análisis por sector realizado
- [ ] ✅ Análisis por educación realizado
- [ ] ✅ Impacto económico cuantificado
- [ ] ✅ Visualizaciones generadas (8+)
- [ ] ✅ Modelo predictivo entrenado
- [ ] ✅ Modelo evaluado (accuracy, F1, etc.)
- [ ] ✅ Feature importance analizado
- [ ] ✅ Resultados interpretados
- [ ] ✅ Recomendaciones formuladas
- [ ] ✅ Reportes generados
- [ ] ✅ Resultados exportados
- [ ] ✅ Modelo guardado

---

## 📚 Recursos Adicionales

### Troubleshooting Común

**Error: "Memory overflow"**
```python
# Reducir particiones
df_spark = df_spark.repartition(10)
```

**Error: "PyArrow conversion failed"**
```python
# Usar conversión explícita
df_pandas = df.to_pandas()
```

**Visualizaciones no aparecen**
```python
# En Jupyter
%matplotlib inline
plt.show()
```

### Próximos Pasos

1. **Validación con expertos:** Presentar resultados a especialistas del mercado laboral
2. **Análisis temporal:** Incorporar datos históricos para tendencias
3. **Análisis geográfico:** Desglosar por municipios de Jalisco
4. **Escenarios:** Modelar diferentes escenarios de adopción de IA

---

## 📧 Contacto y Soporte

**Carlos Pulido Rosas**  
📧 carlos.pulido.rosas@gmail.com  
🎓 CUCEA - Universidad de Guadalajara

---

**Versión:** 1.0  
**Última actualización:** Noviembre 2025  
**Autor:** Carlos Pulido Rosas