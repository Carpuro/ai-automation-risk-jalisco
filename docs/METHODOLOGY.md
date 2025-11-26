# 📊 Metodología del Proyecto

## Modelo Predictivo de Riesgo de Sustitución Laboral por IA

**Autor:** Carlos Pulido Rosas  
**Tesis:** Maestría en Ciencias de los Datos - CUCEA

---

## 1. Marco Metodológico

### 1.1 Tipo de Investigación

- **Tipo:** Investigación cuantitativa con enfoque predictivo
- **Alcance:** Descriptivo-explicativo-predictivo
- **Diseño:** No experimental, transversal con proyección temporal
- **Enfoque:** Data Science y Machine Learning aplicado

### 1.2 Pregunta de Investigación

**¿Cuáles ocupaciones en Jalisco tienen mayor riesgo de sustitución por Inteligencia Artificial en el período 2025-2030, considerando características ocupacionales y tendencias socioeconómicas?**

### 1.3 Hipótesis

**H1:** Ocupaciones con alto grado de rutinización de tareas tienen mayor probabilidad de automatización.

**H2:** El nivel educativo requerido está inversamente correlacionado con el riesgo de automatización.

**H3:** Sectores de servicios administrativos y manufactura presentan mayor riesgo que sectores creativos y de alta especialización.

---

## 2. Diseño Metodológico

### 2.1 Fases del Proyecto

```
Fase 1: Recolección de Datos (Semanas 1-3)
├── Obtención datos O*NET
├── Descarga ENOE Jalisco
├── Búsqueda fuentes complementarias
└── Documentación de fuentes

Fase 2: Preprocesamiento (Semanas 4-6)
├── Limpieza de datos
├── Integración de fuentes
├── Normalización
├── Manejo de valores faltantes
└── Validación de calidad

Fase 3: Análisis Exploratorio (Semanas 7-9)
├── Estadísticas descriptivas
├── Análisis univariado
├── Análisis bivariado
├── Identificación de patrones
└── Visualizaciones iniciales

Fase 4: Feature Engineering (Semanas 10-12)
├── Creación de índices
├── Transformación de variables
├── Selección de features
└── Ingeniería de características

Fase 5: Modelado (Semanas 13-16)
├── Selección de algoritmos
├── Entrenamiento de modelos
├── Validación cruzada
├── Optimización de hiperparámetros
└── Ensamble de modelos

Fase 6: Evaluación y Resultados (Semanas 17-18)
├── Métricas de desempeño
├── Interpretación de resultados
├── Validación con expertos
└── Análisis de sensibilidad

Fase 7: Documentación (Semanas 19-20)
├── Reporte técnico
├── Visualizaciones finales
├── Presentación de resultados
└── Tesis escrita
```

---

## 3. Fuentes de Datos

### 3.1 Datos Primarios

#### O*NET Database
- **Descripción:** Base de datos de características ocupacionales
- **Variables clave:**
  - Habilidades requeridas (100+ categorías)
  - Conocimientos necesarios
  - Actividades laborales
  - Contexto laboral
  - Características del trabajo
  
#### ENOE - Jalisco
- **Descripción:** Encuesta de empleo en Jalisco
- **Variables clave:**
  - Ocupación (clasificación SINCO)
  - Número de trabajadores
  - Salarios
  - Nivel educativo
  - Sector económico
  - Municipio

### 3.2 Datos Secundarios

#### Estudios de Automatización
- Frey & Osborne (2013) - Probabilidades de automatización
- McKinsey Global Institute - Índices de automatización
- World Economic Forum - Future of Jobs Report

#### Datos Económicos
- PIB sectorial Jalisco (INEGI)
- Inversión en tecnología por sector
- Tendencias de empleo histórico

---

## 4. Variables del Estudio

### 4.1 Variable Dependiente

**Riesgo de Automatización (automation_risk)**
- Tipo: Numérica continua [0, 1]
- Definición: Probabilidad de que una ocupación sea automatizada
- Categorización:
  - Alto: > 0.70
  - Medio: 0.30 - 0.70
  - Bajo: < 0.30

### 4.2 Variables Independientes

#### Características de Tareas (Task-based)
1. **task_routine_index** - Índice de rutinización (0-100)
2. **task_cognitive_demand** - Demanda cognitiva (0-100)
3. **task_manual** - Intensidad manual (0-100)
4. **task_social_interaction** - Interacción social (0-100)
5. **task_creativity** - Creatividad requerida (0-100)

#### Habilidades Requeridas
1. **skills_technical** - Habilidades técnicas (0-100)
2. **skills_analytical** - Pensamiento analítico (0-100)
3. **skills_interpersonal** - Habilidades sociales (0-100)
4. **skills_management** - Gestión (0-100)

#### Contexto Laboral
1. **education_level** - Nivel educativo (ordinal)
   - 1: Sin educación formal
   - 2: Primaria
   - 3: Secundaria
   - 4: Preparatoria
   - 5: Universidad
   - 6: Posgrado

2. **avg_salary** - Salario promedio (MXN/mes)
3. **workers_count** - Número de trabajadores
4. **sector** - Sector económico (categórica)

#### Variables de Control
1. **year** - Año de referencia
2. **region** - Región en Jalisco
3. **company_size** - Tamaño de empresa

---

## 5. Proceso de Análisis con PySpark

### 5.1 Arquitectura de Procesamiento

```python
# Pipeline de procesamiento
raw_data → cleaning → feature_engineering → modeling → evaluation
```

### 5.2 Configuración de Spark

```python
spark = SparkSession.builder \
    .appName("AI_Automation_Risk_Analysis") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()
```

### 5.3 Carga de Datos

```python
# O*NET data
onet_df = spark.read.csv("data/raw/onet_occupations.csv", 
                         header=True, inferSchema=True)

# ENOE Jalisco
enoe_df = spark.read.csv("data/raw/enoe_jalisco.csv",
                         header=True, inferSchema=True,
                         encoding='latin1')

# Join datasets
combined_df = onet_df.join(enoe_df, 
                           onet_df.soc_code == enoe_df.occupation_code,
                           'inner')
```

### 5.4 Preprocesamiento

```python
# Convertir a pyspark.pandas
import pyspark.pandas as ps
df_ps = combined_df.pandas_api()

# Limpieza
df_clean = df_ps.dropna(subset=['occupation_name', 'workers_count'])
df_clean = df_clean[df_clean['workers_count'] > 0]

# Normalización
from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
```

### 5.5 Feature Engineering

```python
# Índice de rutinización
df_ps['routine_index'] = (
    df_ps['task_repetitive'] * 0.4 +
    df_ps['task_predictable'] * 0.3 +
    df_ps['task_structured'] * 0.3
)

# Índice de automatización base (método Frey-Osborne)
df_ps['automation_base'] = (
    df_ps['routine_index'] * 0.5 -
    df_ps['creativity'] * 0.25 -
    df_ps['social_intelligence'] * 0.25
)

# Categorización de riesgo
def categorize_risk(score):
    if score >= 0.70:
        return 'Alto'
    elif score >= 0.30:
        return 'Medio'
    else:
        return 'Bajo'

df_ps['risk_category'] = df_ps['automation_base'].apply(categorize_risk)
```

---

## 6. Modelado Predictivo

### 6.1 Algoritmos a Utilizar

#### Modelos de Clasificación
1. **Random Forest Classifier** (Spark MLlib)
   - Ventaja: Interpreta importancia de features
   - Hiperparámetros: numTrees, maxDepth

2. **Gradient Boosting Trees** (Spark MLlib)
   - Ventaja: Alto desempeño predictivo
   - Hiperparámetros: maxIter, stepSize

3. **Logistic Regression** (Spark MLlib)
   - Ventaja: Baseline interpretable
   - Hiperparámetros: maxIter, regParam

#### Modelos de Regresión
1. **Linear Regression** (Spark MLlib)
   - Para predecir probabilidad continua

2. **Random Forest Regressor**
   - Para predicciones robustas

### 6.2 Pipeline de Modelado

```python
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, StringIndexer

# Preparar features
feature_cols = ['routine_index', 'education_level', 'avg_salary',
                'task_cognitive', 'skills_technical']

assembler = VectorAssembler(inputCols=feature_cols, 
                            outputCol="features")

# Indexar target
indexer = StringIndexer(inputCol="risk_category", 
                       outputCol="label")

# Modelo
rf = RandomForestClassifier(featuresCol="features",
                            labelCol="label",
                            numTrees=100,
                            maxDepth=10)

# Pipeline completo
pipeline = Pipeline(stages=[assembler, indexer, rf])
```

### 6.3 Validación

```python
# Split train/test
train_df, test_df = df_spark.randomSplit([0.8, 0.2], seed=42)

# Entrenar
model = pipeline.fit(train_df)

# Predecir
predictions = model.transform(test_df)

# Evaluar
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy:.3f}")
```

---

## 7. Métricas de Evaluación

### 7.1 Métricas de Clasificación

- **Accuracy**: Precisión general
- **Precision**: Por clase (Alto/Medio/Bajo)
- **Recall**: Sensibilidad por clase
- **F1-Score**: Media armónica
- **Matriz de Confusión**: Errores por clase
- **ROC-AUC**: Curva ROC multiclase

### 7.2 Métricas de Regresión

- **RMSE**: Error cuadrático medio
- **MAE**: Error absoluto medio
- **R²**: Coeficiente de determinación

### 7.3 Validación Externa

- Comparación con estudios previos (Frey-Osborne)
- Validación con expertos del mercado laboral
- Análisis de casos específicos

---

## 8. Análisis de Resultados

### 8.1 Interpretación de Importancia de Features

```python
# Feature importance
feature_importance = model.stages[-1].featureImportances

# Ordenar
importance_df = ps.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance.toArray()
}).sort_values('importance', ascending=False)
```

### 8.2 Análisis de Sensibilidad

- Variación de parámetros
- Análisis "what-if"
- Escenarios optimista/pesimista

### 8.3 Proyecciones Temporales

```python
# Proyección 2025-2030
years = range(2025, 2031)
for year in years:
    projected_risk = base_risk * (1 + annual_growth_rate) ** (year - 2025)
```

---

## 9. Visualizaciones

### 9.1 Visualizaciones Exploratorias

1. Distribución de ocupaciones por sector
2. Correlaciones entre variables
3. Box plots por nivel educativo
4. Scatter plots salario vs riesgo

### 9.2 Visualizaciones de Resultados

1. Mapa de calor: Riesgo por sector y educación
2. Treemap: Impacto por número de trabajadores
3. Serie temporal: Proyecciones 2025-2030
4. Red: Relaciones entre habilidades
5. Mapa geográfico: Jalisco por municipio

### 9.3 Dashboard Interactivo

```python
import plotly.express as px
import plotly.graph_objects as go

# Dashboard con Plotly
fig = go.Figure()
fig.add_trace(go.Bar(...))
fig.show()
```

---

## 10. Consideraciones Éticas

### 10.1 Privacidad de Datos

- Datos agregados (no individuales)
- Anonimización de información sensible
- Cumplimiento con INAI

### 10.2 Sesgo en Datos

- Verificación de representatividad
- Análisis de equidad por género
- Consideración de grupos vulnerables

### 10.3 Uso Responsable de Resultados

- Recomendaciones de política pública
- Enfoque en reconversión laboral
- No estigmatización de ocupaciones

---

## 11. Limitaciones del Estudio

### 11.1 Limitaciones de Datos

- Actualización de datos O*NET (anual)
- Cobertura ENOE (no todas las ocupaciones)
- Cambio tecnológico acelerado

### 11.2 Limitaciones Metodológicas

- Proyecciones basadas en tendencias actuales
- Factores externos no considerados
- Alcance geográfico limitado a Jalisco

### 11.3 Limitaciones Técnicas

- Capacidad computacional
- Disponibilidad de datos históricos
- Modelos simplificados de adopción tecnológica

---

## 12. Cronograma

|  Fase  | Duración  |      Entregables       |
|--------|-----------|------------------------|
| Fase 1 | 3 semanas | Datasets integrados    |
| Fase 2 | 3 semanas | Datos limpios          |
| Fase 3 | 3 semanas | Reporte exploratorio   |
| Fase 4 | 3 semanas | Features engineered    |
| Fase 5 | 4 semanas | Modelos entrenados     |
| Fase 6 | 2 semanas | Evaluación completa    |
| Fase 7 | 2 semanas | Tesis y presentación   |

**Total:** 20 semanas (~5 meses)

---

## 13. Referencias Metodológicas

### Artículos Clave

1. Frey, C. B., & Osborne, M. A. (2017). The future of employment: How susceptible are jobs to computerisation? *Technological Forecasting and Social Change*, 114, 254-280.

2. Arntz, M., Gregory, T., & Zierahn, U. (2016). The risk of automation for jobs in OECD countries: A comparative analysis. *OECD Social, Employment and Migration Working Papers*, No. 189.

3. Autor, D. H., Levy, F., & Murnane, R. J. (2003). The skill content of recent technological change: An empirical exploration. *The Quarterly Journal of Economics*, 118(4), 1279-1333.

### Recursos Técnicos

- PySpark Documentation: https://spark.apache.org/docs/latest/api/python/
- O*NET Database: https://www.onetcenter.org/
- INEGI Metodología ENOE: https://www.inegi.org.mx/programas/enoe/

---

## 14. Contribución Esperada

### 14.1 Contribución Académica

- Aplicación de Big Data a mercado laboral mexicano
- Metodología replicable para otras entidades
- Integración de fuentes heterogéneas

### 14.2 Contribución Práctica

- Herramienta de diagnóstico para política pública
- Identificación de necesidades de capacitación
- Planeación educativa basada en evidencia

### 14.3 Contribución Tecnológica

- Pipeline escalable con PySpark
- Código abierto y documentado
- Dashboard interactivo para stakeholders

--- Fin del Documento ---