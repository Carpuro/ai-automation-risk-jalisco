# Datos de Ejemplo

Este directorio contiene datos simulados para pruebas y desarrollo.

## 📁 Archivos

### occupations_sample.csv

Dataset simulado de ocupaciones con características para análisis de automatización.

**Columnas:**
- `occupation_id` - ID único de la ocupación (1-100)
- `occupation_name` - Nombre de la ocupación
- `sector` - Sector económico
- `routine_index` - Índice de rutinización (0-100)
- `cognitive_demand` - Demanda cognitiva (0-100)
- `social_interaction` - Nivel de interacción social (0-100)
- `creativity` - Nivel de creatividad requerida (0-100)
- `education_level` - Nivel educativo requerido (1-6)
- `avg_salary_mxn` - Salario promedio mensual en MXN
- `workers_jalisco` - Número de trabajadores en Jalisco
- `automation_risk` - Riesgo de automatización calculado (0-1)

**Tamaño:** 100 ocupaciones

**Uso:**
```python
import pandas as pd

df = pd.read_csv('data/sample/occupations_sample.csv')
print(df.head())
```

## ⚠️ Nota Importante

Estos datos son **simulados** y generados aleatoriamente para:
- Desarrollo y pruebas
- Demostración de funcionalidad
- Validación de código

**NO usar para:**
- Análisis real
- Toma de decisiones
- Publicaciones académicas
- Reportes oficiales

## 🎯 Para Producción

Para análisis real, reemplazar con:
1. **O*NET Database** - https://www.onetcenter.org/database.html
2. **INEGI ENOE** - https://www.inegi.org.mx/programas/enoe/

## 📊 Características de los Datos Simulados

- **Distribución de sectores:** Ponderada según economía típica
- **Salarios:** Distribución log-normal
- **Trabajadores:** Distribución log-normal
- **Riesgo:** Calculado con fórmula basada en Frey-Osborne
- **Seed:** 42 (reproducible)

## 🔄 Regenerar Datos

Para regenerar con diferentes parámetros:

```python
from data_loader import load_sample_data

df = load_sample_data(spark, n_occupations=200)
df.to_pandas().to_csv('data/sample/occupations_sample.csv', index=False)
```

---

**Generado:** Noviembre 2025  
**Autor:** Carlos Pulido Rosas
