# 🤖 Análisis de Riesgo de Automatización Laboral con PySpark

## Modelo Predictivo de Sustitución Laboral por IA - Jalisco 2025-2030

**Autor:** Carlos Pulido Rosas  
**Institución:** CUCEA - Universidad de Guadalajara  
**Programa:** Maestría en Ciencias de los Datos  
**Línea LGAC:** SMART DATA  
**Fecha:** Junio 2025

---

## 📋 Descripción del Proyecto

Este proyecto implementa un análisis exploratorio y predictivo usando **PySpark** para evaluar el riesgo de sustitución laboral por Inteligencia Artificial en el estado de Jalisco, México.

### Objetivo General

Desarrollar un modelo predictivo que identifique ocupaciones con alto riesgo de automatización mediante análisis de características laborales y tendencias socioeconómicas usando Big Data.

---

## 🎯 Objetivos Específicos

1. **Carga y procesamiento** de datos de ocupaciones (ENOE, O*NET, INEGI)
2. **Análisis exploratorio** de características de automatización
3. **Identificación de patrones** de riesgo laboral por sector
4. **Visualización** de tendencias de automatización 2025-2030
5. **Modelado predictivo** usando Spark MLlib

---

## 📊 Fuentes de Datos

### Datos Principales

1. **O*NET Database** (Occupational Information Network)
   - Características de ocupaciones
   - Habilidades requeridas
   - Tareas automatizables
   - URL: https://www.onetcenter.org/database.html

2. **INEGI - ENOE** (Encuesta Nacional de Ocupación y Empleo)
   - Empleo por ocupación en Jalisco
   - Datos socioeconómicos
   - URL: https://www.inegi.org.mx/programas/enoe/

3. **McKinsey/Frey-Osborne** (Opcional)
   - Índices de automatización por ocupación
   - Probabilidades de automatización

### Estructura Esperada de Datos

```
occupation_id | occupation_name | sector | automation_risk | 
workers_jalisco | avg_salary | education_level | skills_required |
task_routine | task_cognitive | task_manual
```

---

## 🛠️ Tecnologías Utilizadas

- **Apache Spark 3.5+** - Procesamiento distribuido
- **PySpark** - API Python para Spark
- **pyspark.pandas** - Manipulación de datos
- **Spark MLlib** - Machine Learning
- **Matplotlib/Seaborn/Plotly** - Visualizaciones

---

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.10 o 3.11
- Conda/Anaconda
- 8GB RAM mínimo

### Instalación en 3 Pasos

```bash
# 1. Crear entorno
conda env create -f environment.yml
conda activate ai_automation_thesis

# 2. Verificar instalación
python verify_setup.py

# 3. Ejecutar análisis
jupyter notebook notebooks/automation_risk_analysis.ipynb
```

---

## 📁 Estructura del Proyecto

```
pyspark-ai-automation-thesis/
│
├── 📄 README.md
├── 📄 environment.yml
├── 📄 requirements.txt
├── 📄 verify_setup.py
│
├── 📁 notebooks/
│   └── automation_risk_analysis.ipynb
│
├── 📁 src/
│   ├── data_loader.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── automation_analyzer.py
│   ├── visualizations.py
│   └── main.py
│
├── 📁 data/
│   ├── raw/              # Datos originales
│   ├── processed/        # Datos procesados
│   └── external/         # Datos externos
│
├── 📁 outputs/
│   ├── models/           # Modelos entrenados
│   ├── visualizations/   # Gráficos
│   └── reports/          # Reportes
│
└── 📁 docs/
    ├── METHODOLOGY.md
    ├── DATA_SOURCES.md
    └── ANALYSIS_GUIDE.md
```

---

## 📈 Análisis Implementados

### 1. Análisis Exploratorio (30%)

- Distribución de ocupaciones por sector
- Análisis de salarios y empleo
- Correlación entre variables
- Estadísticas descriptivas

### 2. Análisis de Riesgo (40%)

- **Riesgo por ocupación** - Probabilidad de automatización
- **Riesgo por sector** - Impacto sectorial
- **Riesgo geográfico** - Zonas de Jalisco más afectadas
- **Análisis temporal** - Proyecciones 2025-2030

### 3. Feature Engineering (20%)

- Índice de automatización compuesto
- Categorización de habilidades
- Rutinización de tareas
- Impacto económico estimado

### 4. Modelado Predictivo (10%)

- Clasificación de riesgo (Alto/Medio/Bajo)
- Regresión para probabilidad de automatización
- Clustering de ocupaciones similares

---

## 🎨 Visualizaciones Incluidas

1. **Mapa de Calor** - Riesgo por sector y nivel educativo
2. **Scatter Plot** - Salario vs Riesgo de automatización
3. **Barras** - Top ocupaciones en riesgo
4. **Serie Temporal** - Proyección 2025-2030
5. **Treemap** - Impacto por sector económico
6. **Red** - Relaciones entre habilidades y automatización
7. **Boxplot** - Distribución de riesgo por educación
8. **Mapa Geográfico** - Jalisco por municipio

---

## 🔬 Metodología

### Fase 1: Recolección de Datos
- Descarga de datasets O*NET
- Extracción datos ENOE Jalisco
- Integración de fuentes

### Fase 2: Preprocesamiento
- Limpieza de datos
- Normalización
- Manejo de valores faltantes
- Feature scaling

### Fase 3: Análisis Exploratorio
- Estadísticas descriptivas
- Visualizaciones
- Identificación de patrones

### Fase 4: Feature Engineering
- Creación de índices
- Transformaciones
- Selección de features

### Fase 5: Modelado
- Entrenamiento modelos
- Validación
- Optimización

### Fase 6: Interpretación
- Análisis de resultados
- Recomendaciones
- Visualización de insights

---

## 📊 Métricas de Evaluación del Modelo

- **Accuracy** - Precisión general
- **Precision/Recall** - Por clase de riesgo
- **F1-Score** - Balance precision/recall
- **ROC-AUC** - Capacidad discriminativa
- **RMSE** - Error en predicciones numéricas

---

## 🎓 Resultados Esperados

1. **Identificación** de ocupaciones de alto riesgo en Jalisco
2. **Cuantificación** del impacto laboral por sector
3. **Proyecciones** de automatización 2025-2030
4. **Recomendaciones** de política pública
5. **Modelo predictivo** replicable para otras regiones

---

## 📝 Uso del Proyecto

### Análisis Completo (Jupyter Notebook)

```bash
jupyter notebook notebooks/automation_risk_analysis.ipynb
```

### Script Automatizado

```bash
python src/main.py \
    --occupation-data data/raw/onet_occupations.csv \
    --employment-data data/raw/enoe_jalisco.csv \
    --output outputs/results
```

### Análisis por Módulo

```python
# Cargar datos
from src.data_loader import load_occupation_data
df = load_occupation_data('data/raw/onet_occupations.csv')

# Análisis de riesgo
from src.automation_analyzer import calculate_automation_risk
risk_df = calculate_automation_risk(df)

# Visualizar
from src.visualizations import plot_risk_heatmap
plot_risk_heatmap(risk_df)
```

---

## 🔧 Configuración del Entorno

### Opción A: Conda (Recomendado)

```bash
conda env create -f environment.yml
conda activate ai_automation_thesis
```

### Opción B: pip

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📚 Documentación Adicional

| Documento | Descripción |
|-----------|-------------|
| `METHODOLOGY.md` | Metodología detallada |
| `DATA_SOURCES.md` | Guía de fuentes de datos |
| `ANALYSIS_GUIDE.md` | Guía de análisis paso a paso |
| `API_REFERENCE.md` | Referencia de funciones |

---

## 🆘 Soporte y Troubleshooting

### Problemas Comunes

**Error: "PyArrow not found"**
```bash
pip install pyarrow>=4.0.0
```

**Error: "NumPy incompatible"**
```bash
pip install numpy==1.26.4
```

**Kernel muere en Jupyter**
```bash
# Usar Python 3.10
conda create -n ai_automation_thesis python=3.10
```

---

## 📞 Contacto

**Carlos Pulido Rosas**  
📧 carlos.pulido.rosas@gmail.com  
📱 +52 33 1030 5580  
🎓 CUCEA - Universidad de Guadalajara  
🔗 [GitHub](https://github.com/carpuro)
🔗 [LinkedIn](https://www.linkedin.com/in/carlos-pulido-489700132/)

---

## 🙏 Referencias

1. Frey, C. B., & Osborne, M. A. (2017). *The future of employment*
2. McKinsey Global Institute (2023). *AI and automation impact*
3. O*NET Program (2024). *Occupational Information Network*
4. INEGI (2024). *Encuesta Nacional de Ocupación y Empleo*

---

## 📄 Licencia

Este proyecto es parte de una tesis de maestría y está disponible para fines académicos y de investigación.

---

## ✅ Checklist de Desarrollo

- [x] Configuración del entorno
- [x] Carga de datos O*NET
- [ ] Integración datos ENOE
- [ ] Análisis exploratorio completo
- [ ] Feature engineering
- [ ] Modelo predictivo
- [ ] Visualizaciones interactivas
- [ ] Documentación completa
- [ ] Validación de resultados
- [ ] Presentación de resultados

---

## 🎯 Keywords

`PySpark` `Machine Learning` `Automatización Laboral` `Inteligencia Artificial` 
`Análisis Predictivo` `Big Data` `Jalisco` `Sustitución Laboral` `O*NET` 
`ENOE` `Data Science` `Spark MLlib` `Tesis` `CUCEA`

--- Versión del Documento ---

**Versión:** 1.0  
**Última actualización:** Noviembre 2025  
**Estado:** ✅ En desarrollo

--- Fin del Documento ---