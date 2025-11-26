# Outputs - Resultados del Análisis

Este directorio contiene todos los **resultados generados** por el análisis.

## 📁 Estructura

```
outputs/
├── models/          # Modelos ML entrenados
├── visualizations/  # Gráficos y dashboards
└── reports/         # Reportes de texto y CSV
```

## 🤖 models/

Modelos de Machine Learning entrenados.

### Archivos típicos:
```
models/
├── automation_risk_model/      # Random Forest (PySpark MLlib)
├── rf_model_20251126.pkl       # Pickle (scikit-learn)
└── model_metadata.json         # Metadatos del modelo
```

### Uso:
```python
from pyspark.ml import PipelineModel

# Cargar modelo
model = PipelineModel.load('outputs/models/automation_risk_model')

# Predecir
predictions = model.transform(new_data)
```

## 📊 visualizations/

Gráficos y visualizaciones generadas.

### Archivos típicos:
```
visualizations/
├── 01_risk_distribution.png
├── 02_risk_by_sector.png
├── 03_salary_vs_risk.html          # Interactivo (Plotly)
├── 04_education_vs_risk.png
├── 05_heatmap_sector_education.png
├── 06_treemap_workers_risk.html    # Interactivo
├── 07_temporal_projections.png
├── 08_correlation_matrix.png
└── 09_top_occupations_risk.png
```

### Generar visualizaciones:
```python
from visualizations import create_dashboard

# Generar todas las visualizaciones
create_dashboard(df_risk, output_dir='outputs/visualizations/')
```

## 📝 reports/

Reportes de texto, CSVs y documentos.

### Archivos típicos:
```
reports/
├── risk_analysis_report_20251126.txt    # Reporte completo
├── risk_analysis_results_20251126.csv   # Para Excel
├── sector_analysis.csv
├── education_analysis.csv
└── economic_impact_summary.txt
```

### Estructura de reporte de texto:

```
================================================================================
REPORTE DE ANÁLISIS DE RIESGO DE AUTOMATIZACIÓN
================================================================================

1. RESUMEN EJECUTIVO
   - Ocupaciones analizadas: 450
   - Trabajadores en Jalisco: 2,450,000
   - Trabajadores en alto riesgo: 735,000 (30%)

2. OCUPACIONES MÁS RIESGOSAS
   Top 20 con mayor probabilidad de automatización

3. ANÁLISIS POR SECTOR
   Sectores ordenados por riesgo promedio

4. ANÁLISIS POR EDUCACIÓN
   Correlación entre nivel educativo y riesgo

5. IMPACTO ECONÓMICO
   Masa salarial en riesgo, presupuesto de reconversión

6. RECOMENDACIONES
   Políticas públicas sugeridas
```

## 🚫 No Subir a Git

Estos archivos **NO** deben subirse a Git:
- Son generados automáticamente
- Pueden ser grandes (>10MB)
- Se pueden regenerar fácilmente

Todos están en `.gitignore`.

## ✅ Cómo Generar

### Opción 1: Script principal
```bash
python src/main.py --mode sample --output outputs/
```

### Opción 2: Notebook
Ejecutar `notebooks/automation_risk_analysis.ipynb` completamente.

### Opción 3: Funciones individuales
```python
from automation_analyzer import generate_risk_report
from visualizations import create_dashboard

# Reporte
generate_risk_report(df_risk, output_path='outputs/reports/report.txt')

# Visualizaciones
create_dashboard(df_risk, output_dir='outputs/visualizations/')

# Guardar CSV
df_risk.to_pandas().to_csv('outputs/reports/results.csv', index=False)
```

## 📅 Convención de Nombres

Usar timestamp en nombres de archivo:

```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"risk_analysis_report_{timestamp}.txt"
```

## 🔍 Inspeccionar Outputs

```bash
# Ver archivos generados
ls -lh outputs/models/
ls -lh outputs/visualizations/
ls -lh outputs/reports/

# Contar archivos
find outputs/ -type f | wc -l

# Ver tamaño total
du -sh outputs/
```

## 🧹 Limpiar Outputs

Para limpiar todos los outputs generados:

```bash
# ⚠️ CUIDADO: Esto borra todo
rm -rf outputs/models/*
rm -rf outputs/visualizations/*
rm -rf outputs/reports/*

# Mantener .gitkeep
touch outputs/models/.gitkeep
touch outputs/visualizations/.gitkeep
touch outputs/reports/.gitkeep
```

## 📦 Compartir Resultados

Para compartir resultados con colaboradores:

### Opción 1: Comprimir outputs
```bash
tar -czf outputs_20251126.tar.gz outputs/
```

### Opción 2: Subir a cloud
```bash
# Google Drive, Dropbox, etc.
# O usar Git LFS para archivos grandes
```

### Opción 3: GitHub Release
Crear un release en GitHub con los outputs como assets.

---

**Nota:** Estos archivos se regeneran cada vez que ejecutas el análisis. Mantén versiones importantes con timestamps.
