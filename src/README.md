# Módulos de Código Fuente

Este directorio contiene todos los módulos Python del proyecto.

## 📁 Estructura

```
src/
├── automation_analyzer.py   # Análisis de riesgo de automatización
├── data_loader.py           # Carga de datos (O*NET, ENOE, simulados)
├── data_preprocessing.py    # Limpieza y preprocesamiento
├── feature_engineering.py   # Creación de features
├── visualizations.py        # Generación de gráficos
└── main.py                  # Script ejecutable principal
```

---

## 📦 Módulos

### 1. data_loader.py

**Propósito:** Carga de datos desde múltiples fuentes.

**Funciones principales:**
- `create_spark_session()` - Inicializa Spark con configuración óptima
- `load_sample_data()` - Genera datos simulados para pruebas
- `load_onet_occupations()` - Carga datos de O*NET
- `load_enoe_jalisco()` - Carga datos de ENOE (Jalisco)
- `load_mapping_table()` - Carga mapeo SOC-SINCO
- `convert_to_pandas_api()` - Convierte a pyspark.pandas
- `save_dataset()` - Guarda dataset en parquet/CSV

**Ejemplo de uso:**
```python
from data_loader import create_spark_session, load_sample_data

# Crear Spark
spark = create_spark_session(memory="8g")

# Cargar datos simulados
df = load_sample_data(spark, n_occupations=200)

print(f"Dataset cargado: {df.count()} ocupaciones")
```

**Dependencias:** `pyspark`, `pyspark.pandas`, `numpy`, `pandas`

---

### 2. data_preprocessing.py

**Propósito:** Limpieza y validación de datos.

**Funciones principales:**
- `analyze_missing_values()` - Identifica valores faltantes
- `handle_missing_values()` - Maneja nulos (estrategias: auto, drop, fill)
- `remove_duplicates()` - Elimina duplicados
- `filter_outliers()` - Filtra valores atípicos (IQR, z-score)
- `normalize_columns()` - Normaliza features (MinMax, Standard)
- `validate_data_quality()` - Calcula métricas de calidad
- `preprocess_pipeline()` - Pipeline completo de preprocesamiento

**Ejemplo de uso:**
```python
from data_preprocessing import preprocess_pipeline

# Pipeline completo
df_clean = preprocess_pipeline(df, config={
    'handle_missing': 'auto',
    'remove_duplicates': True,
    'filter_outliers': False
})

print(f"Datos limpios: {df_clean.shape}")
```

**Dependencias:** `pyspark.pandas`, `numpy`

---

### 3. automation_analyzer.py ⭐ CORE

**Propósito:** Análisis de riesgo de automatización laboral.

**Clase principal:** `AutomationRiskAnalyzer`

**Métodos principales:**
- `calculate_routine_index()` - Índice de rutinización (0-100)
- `calculate_cognitive_demand()` - Demanda cognitiva (0-100)
- `calculate_social_interaction()` - Interacción social (0-100)
- `calculate_creativity_level()` - Nivel de creatividad (0-100)
- `calculate_automation_risk()` - **Riesgo de automatización (0-1)**
- `categorize_risk()` - Clasifica en Alto/Medio/Bajo
- `analyze_by_sector()` - Análisis por sector económico
- `analyze_by_education()` - Análisis por nivel educativo
- `calculate_economic_impact()` - Impacto económico estimado
- `identify_top_at_risk()` - Top N ocupaciones en riesgo
- `identify_low_risk_occupations()` - Ocupaciones más seguras
- `generate_risk_report()` - Genera reporte de texto

**Ejemplo de uso:**
```python
from automation_analyzer import AutomationRiskAnalyzer

# Crear analizador
analyzer = AutomationRiskAnalyzer()

# Calcular riesgo
df_risk = analyzer.calculate_automation_risk(df, method='frey_osborne')

# Categorizar
df_risk = analyzer.categorize_risk(df_risk, thresholds=(0.30, 0.70))

# Ver distribución
print(df_risk['risk_category'].value_counts())
```

**Metodología:**
- Basado en **Frey & Osborne (2017)**
- Pesos: Rutina (40%), Cognitivo (-25%), Social (-20%), Creatividad (-15%)

**Dependencias:** `pyspark.pandas`, `numpy`

---

### 4. feature_engineering.py

**Propósito:** Creación de features derivados para modelado.

**Funciones principales:**
- `create_routine_index()` - Índice de rutinización
- `create_cognitive_demand_index()` - Índice cognitivo
- `create_social_interaction_index()` - Índice social
- `create_creativity_index()` - Índice de creatividad
- `create_complexity_index()` - Complejidad general
- `create_education_categories()` - Categorías educativas
- `create_salary_categories()` - Quintiles salariales
- `create_occupation_size_categories()` - Tamaño de ocupación
- `create_derived_ratios()` - Ratios útiles
- `create_automation_susceptibility_score()` - Score de susceptibilidad
- `create_temporal_features()` - Proyecciones temporales
- `create_sector_aggregations()` - Agregaciones por sector
- `feature_engineering_pipeline()` - **Pipeline completo**

**Ejemplo de uso:**
```python
from feature_engineering import feature_engineering_pipeline

# Crear todos los features
df_features = feature_engineering_pipeline(df_clean, config={
    'create_indices': True,
    'create_categories': True,
    'create_ratios': True,
    'create_temporal': True,
    'create_sector_agg': True
})

print(f"Features creados: {df_features.shape[1]} columnas")
```

**Features generados:** 20+ nuevas columnas

**Dependencias:** `pyspark.pandas`, `numpy`

---

### 5. visualizations.py

**Propósito:** Generación de visualizaciones del análisis.

**Funciones principales:**
- `plot_risk_distribution()` - Histograma de riesgo
- `plot_risk_by_sector()` - Riesgo por sector (barras)
- `plot_salary_vs_risk()` - Scatter salario vs riesgo (Plotly)
- `plot_education_vs_risk()` - Boxplot educación vs riesgo
- `plot_heatmap_sector_education()` - Heatmap sector × educación
- `plot_treemap_workers_at_risk()` - Treemap de impacto (Plotly)
- `plot_temporal_projections()` - Serie temporal 2025-2030
- `plot_correlation_matrix()` - Matriz de correlación
- `plot_top_occupations_at_risk()` - Top ocupaciones en riesgo
- `create_dashboard()` - **Genera todas las visualizaciones**

**Ejemplo de uso:**
```python
from visualizations import create_dashboard

# Generar dashboard completo (9 gráficos)
create_dashboard(df_risk, output_dir='outputs/visualizations/')

print("✓ Dashboard generado en outputs/visualizations/")
```

**Visualizaciones generadas:**
1. Distribución de riesgo (PNG)
2. Riesgo por sector (PNG)
3. Salario vs riesgo (HTML interactivo)
4. Educación vs riesgo (PNG)
5. Heatmap sector-educación (PNG)
6. Treemap trabajadores (HTML interactivo)
7. Proyecciones temporales (PNG)
8. Matriz de correlación (PNG)
9. Top ocupaciones (PNG)

**Dependencias:** `matplotlib`, `seaborn`, `plotly`, `numpy`, `pandas`

---

### 6. main.py 🚀

**Propósito:** Script ejecutable principal - Punto de entrada del proyecto.

**Características:**
- ✅ CLI con argparse
- ✅ 2 modos: `sample` (simulado) y `real` (O*NET + ENOE)
- ✅ Pipeline completo en 7 pasos
- ✅ Logging detallado
- ✅ Manejo de errores
- ✅ Exportación de resultados

**Uso:**
```bash
# Modo simulado (pruebas)
python src/main.py --mode sample --n-occupations 200

# Modo real (producción)
python src/main.py --mode real \
  --occupation-data data/raw/onet/Occupation_Data.txt \
  --employment-data data/raw/enoe_jalisco.csv

# Sin visualizaciones (más rápido)
python src/main.py --mode sample --no-visualizations

# Especificar memoria
python src/main.py --mode sample --memory 16g
```

**Pipeline de 7 pasos:**
1. Inicialización (Spark Session)
2. Carga de datos
3. Preprocesamiento
4. Feature engineering
5. Análisis de riesgo
6. Visualizaciones
7. Reportes y exportación

**Argumentos disponibles:**
```
--mode              sample o real (default: sample)
--n-occupations     Número de ocupaciones simuladas (default: 200)
--occupation-data   Ruta a datos O*NET
--employment-data   Ruta a datos ENOE
--output            Directorio de salida (default: outputs/)
--no-visualizations Saltar visualizaciones
--no-model          Saltar entrenamiento de modelo
--memory            Memoria de Spark (default: 8g)
```

**Dependencias:** Todos los módulos anteriores

---

## 🔄 Flujo de Trabajo Típico

### Opción 1: Usar main.py (Recomendado)

```bash
python src/main.py --mode sample
```

### Opción 2: Importar módulos individuales

```python
import sys
sys.path.append('src')

from data_loader import create_spark_session, load_sample_data
from data_preprocessing import preprocess_pipeline
from feature_engineering import feature_engineering_pipeline
from automation_analyzer import AutomationRiskAnalyzer
from visualizations import create_dashboard

# 1. Spark
spark = create_spark_session()

# 2. Datos
df = load_sample_data(spark, n_occupations=100)

# 3. Preprocesar
df_clean = preprocess_pipeline(df)

# 4. Features
df_features = feature_engineering_pipeline(df_clean)

# 5. Riesgo
analyzer = AutomationRiskAnalyzer()
df_risk = analyzer.calculate_automation_risk(df_features)
df_risk = analyzer.categorize_risk(df_risk)

# 6. Visualizar
create_dashboard(df_risk, output_dir='outputs/visualizations/')

# 7. Analizar
sector_analysis = analyzer.analyze_by_sector(df_risk)
print(sector_analysis)
```

---

## 📊 Dependencias Totales

### Python Packages Required:
```
pyspark==3.5.0
pandas==2.1.4
numpy==1.26.4
pyarrow==14.0.1
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0
scikit-learn==1.3.2
```

### Instalación:
```bash
pip install -r requirements.txt
# O
conda env create -f environment.yml
```

---

## 🧪 Testing

### Probar módulo individual:

```python
# Probar data_loader
python -c "from data_loader import create_spark_session; spark = create_spark_session(); print('✓ OK')"

# Probar automation_analyzer
python -c "from automation_analyzer import AutomationRiskAnalyzer; a = AutomationRiskAnalyzer(); print('✓ OK')"
```

### Probar pipeline completo:

```bash
python src/main.py --mode sample --n-occupations 10 --no-visualizations
```

---

## 📝 Convenciones de Código

### Estilo:
- **PEP 8** compliant
- Líneas máximo 100 caracteres
- Docstrings en español (Google style)

### Ejemplo de docstring:
```python
def calculate_risk(df, method='frey_osborne'):
    """
    Calcula riesgo de automatización.
    
    Parameters:
    -----------
    df : pyspark.pandas.DataFrame
        DataFrame con características de ocupaciones
    method : str, default 'frey_osborne'
        Método de cálculo ('frey_osborne', 'task_based', 'hybrid')
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con columna 'automation_risk' agregada
        
    Examples:
    ---------
    >>> df_risk = calculate_risk(df, method='frey_osborne')
    >>> print(df_risk['automation_risk'].mean())
    0.543
    """
    pass
```

---

## 🐛 Debugging

### Logging:

Todos los módulos usan `logging`:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Procesando datos...")
logger.warning("Valores faltantes detectados")
logger.error("Error en cálculo")
```

### Ver logs:
```bash
python src/main.py --mode sample 2>&1 | tee logs/run.log
```

---

## 🔧 Troubleshooting

### Error: "Module not found"
```bash
# Agregar src al PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# O en el script
import sys
sys.path.append('src')
```

### Error: "Spark memory overflow"
```bash
# Reducir ocupaciones
python src/main.py --mode sample --n-occupations 50

# O aumentar memoria
python src/main.py --mode sample --memory 16g
```

### Error: "NumPy incompatible"
```bash
pip install "numpy<2.0" --force-reinstall
```

---

## 📚 Recursos Adicionales

### Documentación del Proyecto:
- [README.md](../README.md) - Visión general
- [METHODOLOGY.md](../docs/METHODOLOGY.md) - Metodología completa
- [ANALYSIS_GUIDE.md](../docs/ANALYSIS_GUIDE.md) - Guía paso a paso

### Referencias Externas:
- **PySpark:** https://spark.apache.org/docs/latest/api/python/
- **Pandas API on Spark:** https://spark.apache.org/docs/latest/api/python/user_guide/pandas_on_spark/
- **Frey & Osborne (2017):** The future of employment

---

## 🤝 Contribuir

Para agregar nuevos módulos o funciones:

1. Seguir PEP 8
2. Agregar docstrings completos
3. Agregar type hints cuando sea posible
4. Crear tests unitarios
5. Actualizar este README

Ver [CONTRIBUTING.md](../CONTRIBUTING.md) para más detalles.

---

## 📊 Estadísticas del Código

| Módulo | Líneas | Funciones | Complejidad |
|--------|--------|-----------|-------------|
| data_loader.py | ~300 | 10+ | Baja |
| data_preprocessing.py | ~350 | 8+ | Media |
| automation_analyzer.py | ~600 | 15+ | Alta |
| feature_engineering.py | ~550 | 12+ | Media |
| visualizations.py | ~600 | 10+ | Media |
| main.py | ~350 | 3+ | Media |
| **TOTAL** | **~2,750** | **60+** | - |

---

## ✅ Checklist de Desarrollo

Al modificar código:

- [ ] Código sigue PEP 8
- [ ] Docstrings agregados/actualizados
- [ ] Type hints agregados
- [ ] Logging implementado
- [ ] Manejo de errores robusto
- [ ] Probado localmente
- [ ] README actualizado (este archivo)

---

**Última actualización:** Noviembre 2025  
**Autor:** Carlos Pulido Rosas  
**Versión:** 1.0