# 🚀 Inicio Rápido - Quick Start

Guía de 5 minutos para empezar a usar el proyecto.

## ⚡ Instalación Rápida

### Paso 1: Clonar el Repositorio
```bash
git clone https://github.com/Carpuro/ai-automation-risk-jalisco.git
cd ai-automation-risk-jalisco
```

### Paso 2: Crear Entorno
```bash
# Opción A: Con Conda (RECOMENDADO)
conda env create -f environment.yml
conda activate ai_automation_thesis

# Opción B: Con pip
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Paso 3: Verificar Instalación
```bash
python verify_setup.py
```

✅ **Deberías ver:**
```
✓ Python version compatible
✓ All dependencies installed
✓ Spark working correctly
✓ pyspark.pandas available
✓ Jupyter available
```

---

## 🎯 3 Formas de Usar el Proyecto

### 1️⃣ Jupyter Notebook (RECOMENDADO para Exploración)

```bash
jupyter notebook
# Abrir: notebooks/automation_risk_analysis.ipynb
# Ejecutar: Kernel → Restart & Run All
```

**Ventajas:**
- ✅ Interactivo
- ✅ Visualizaciones en vivo
- ✅ Documentado paso a paso
- ✅ Ideal para aprendizaje

### 2️⃣ Script Python (Para Producción)

```bash
# Con datos simulados (rápido, para pruebas)
python src/main.py --mode sample --n-occupations 200

# Con datos reales (requiere descargar O*NET + ENOE)
python src/main.py --mode real \
  --occupation-data data/raw/onet/Occupation_Data.txt \
  --employment-data data/raw/enoe_jalisco.csv
```

**Ventajas:**
- ✅ Automatizado
- ✅ Reproducible
- ✅ Fácil de programar (cron jobs)

### 3️⃣ Módulos Individuales (Para Desarrollo)

```python
import sys
sys.path.append('src')

from data_loader import create_spark_session, load_sample_data
from automation_analyzer import AutomationRiskAnalyzer

# Crear Spark
spark = create_spark_session()

# Cargar datos
df = load_sample_data(spark, n_occupations=100)

# Analizar riesgo
analyzer = AutomationRiskAnalyzer()
df_risk = analyzer.calculate_automation_risk(df)

print(df_risk.head())
```

**Ventajas:**
- ✅ Flexible
- ✅ Ideal para experimentar
- ✅ Fácil debugging

---

## 📊 Ejemplo Rápido (5 líneas)

```python
from data_loader import create_spark_session, load_sample_data
from automation_analyzer import AutomationRiskAnalyzer

spark = create_spark_session()
df = load_sample_data(spark, n_occupations=100)
analyzer = AutomationRiskAnalyzer()
df_risk = analyzer.calculate_automation_risk(df)
print(f"Ocupaciones en alto riesgo: {(df_risk['automation_risk'] >= 0.70).sum()}")
```

---

## 🎨 Generar Visualizaciones

```python
from visualizations import create_dashboard

# Genera 9 gráficos automáticamente
create_dashboard(df_risk, output_dir='outputs/visualizations/')
```

**Outputs:**
- `01_risk_distribution.png` - Histograma
- `02_risk_by_sector.png` - Barras por sector
- `03_salary_vs_risk.html` - Scatter interactivo
- ... (6 más)

---

## 📁 ¿Dónde Están los Archivos?

```
📦 Proyecto
├── 📓 notebooks/automation_risk_analysis.ipynb  ← Empieza aquí
├── 🐍 src/main.py                               ← O aquí (CLI)
├── 📊 data/sample/occupations_sample.csv        ← Datos de prueba
└── 📈 outputs/                                  ← Resultados aquí
```

---

## ❓ Troubleshooting Rápido

### Error: "NumPy incompatible"
```bash
pip install "numpy<2.0" --force-reinstall
```

### Error: "PyArrow not found"
```bash
pip install pyarrow>=4.0.0
```

### Jupyter Kernel muere
```bash
# Usar Python 3.10 o 3.11 (NO 3.13)
conda create -n ai_automation_thesis python=3.10
conda activate ai_automation_thesis
pip install -r requirements.txt
```

### Visualizaciones no aparecen
```python
# En Jupyter
%matplotlib inline
import matplotlib.pyplot as plt
plt.show()
```

---

## 📚 Siguientes Pasos

1. ✅ **Instalar y verificar** (arriba)
2. 📓 **Ejecutar notebook** completo
3. 📊 **Ver visualizaciones** en `outputs/visualizations/`
4. 📄 **Leer documentación** en `docs/`
5. 🔧 **Personalizar** para tus datos

---

## 🎓 Recursos Útiles

### Documentación del Proyecto
- 📖 [README.md](README.md) - Visión general
- 📋 [METHODOLOGY.md](docs/METHODOLOGY.md) - Metodología completa
- 📊 [DATA_SOURCES.md](docs/DATA_SOURCES.md) - Fuentes de datos
- 🛠️ [ANALYSIS_GUIDE.md](docs/ANALYSIS_GUIDE.md) - Guía paso a paso

### Tutoriales Externos
- **PySpark:** https://spark.apache.org/docs/latest/api/python/
- **O*NET:** https://www.onetcenter.org/overview.html
- **ENOE:** https://www.inegi.org.mx/programas/enoe/

---

## 💡 Tips Rápidos

### Usar menos memoria
```python
# Reducir número de ocupaciones
df = load_sample_data(spark, n_occupations=50)
```

### Ejecutar sin visualizaciones (más rápido)
```bash
python src/main.py --mode sample --no-visualizations
```

### Ver solo top riesgos
```python
top_20 = df_risk.nlargest(20, 'automation_risk')
print(top_20[['occupation_name', 'automation_risk']])
```

---

## 🤝 ¿Necesitas Ayuda?

1. **Issues:** https://github.com/Carpuro/ai-automation-risk-jalisco/issues
2. **Email:** carlos.pulido.rosas@gmail.com
3. **Documentación:** Lee `docs/ANALYSIS_GUIDE.md`

---

## ✨ ¡Listo!

Ahora tienes todo para empezar. **Ejecuta el notebook** y explora:

```bash
jupyter notebook notebooks/automation_risk_analysis.ipynb
```

¡Buena suerte con tu análisis! 🚀

---

**Tiempo estimado:** 5-10 minutos para setup, 30-60 minutos para análisis completo.
