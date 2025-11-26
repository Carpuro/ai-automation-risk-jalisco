# 📊 Guía de Fuentes de Datos

## Modelo Predictivo de Sustitución Laboral por IA - Jalisco

**Proyecto de Tesis:** Carlos Pulido Rosas  
**CUCEA - Universidad de Guadalajara**

---

## 1. Fuentes de Datos Primarias

### 1.1 O*NET Database (Occupational Information Network)

#### Descripción General
- **Proveedor:** U.S. Department of Labor / Employment and Training Administration
- **URL:** https://www.onetcenter.org/database.html
- **Actualización:** Anual
- **Cobertura:** 1,000+ ocupaciones estandarizadas
- **Licencia:** Dominio público (uso libre)

#### Datasets Clave

##### 1.1.1 Occupation Data
**Archivo:** `Occupation Data.txt`
```
Contenido:
- O*NET-SOC Code: Código estándar de ocupación
- Title: Nombre de la ocupación
- Description: Descripción detallada
```

##### 1.1.2 Skills
**Archivo:** `Skills.txt`
```
Habilidades medidas (35 categorías):
- Active Learning
- Critical Thinking
- Complex Problem Solving
- Programming
- Mathematics
- Science
- Social Perceptiveness
- Coordination
- etc.

Escala: 0-100
```

##### 1.1.3 Abilities
**Archivo:** `Abilities.txt`
```
52 habilidades cognitivas, físicas y sensoriales:
- Oral Comprehension
- Written Comprehension
- Deductive Reasoning
- Mathematical Reasoning
- Manual Dexterity
- etc.

Escala: 0-100
```

##### 1.1.4 Work Activities
**Archivo:** `Work Activities.txt`
```
41 actividades laborales:
- Making Decisions
- Analyzing Data
- Interacting With Computers
- Communicating with Supervisors
- Performing Repetitive Tasks
- etc.

Importancia: 1-5
Nivel: 0-100
```

##### 1.1.5 Work Context
**Archivo:** `Work Context.txt`
```
Contexto laboral (57 variables):
- Degree of Automation
- Structured versus Unstructured Work
- Freedom to Make Decisions
- Face-to-Face Discussions
- Telephone Conversations
- etc.

Escala: 1-5 o categórica
```

##### 1.1.6 Education
**Archivo:** `Education, Training, and Experience.txt`
```
Requisitos educativos:
- Required Level of Education
- Experience Required
- On-the-Job Training Needed

Categorías: 1-12 (desde ninguno hasta doctorado)
```

#### Descarga de O*NET

**Método 1: Descarga Manual**
```
1. Visita: https://www.onetcenter.org/database.html
2. Sección "Database Releases"
3. Download "All Files (ZIP format)"
4. Descomprime en data/raw/onet/
```

**Método 2: Descarga Programática**
```python
import requests
import zipfile
import os

url = "https://www.onetcenter.org/dl_files/database/db_28_3_text.zip"
output_path = "data/raw/onet.zip"

# Descargar
response = requests.get(url)
with open(output_path, 'wb') as f:
    f.write(response.content)

# Descomprimir
with zipfile.ZipFile(output_path, 'r') as zip_ref:
    zip_ref.extractall("data/raw/onet/")
```

#### Estructura de Archivos O*NET

```
data/raw/onet/
├── Occupation Data.txt
├── Skills.txt
├── Abilities.txt
├── Work Activities.txt
├── Work Context.txt
├── Work Styles.txt
├── Education, Training, and Experience.txt
├── Job Zones.txt
├── Knowledge.txt
└── README.txt
```

#### Mapeo SOC a SINCO

```python
# Diccionario de mapeo SOC (USA) → SINCO (México)
soc_to_sinco = {
    '11-1011.00': '1110',  # Chief Executives → Directores generales
    '15-1252.00': '2613',  # Software Developers → Desarrolladores de software
    '29-1141.00': '2231',  # Registered Nurses → Enfermeros
    # ... más mapeos
}
```

---

### 1.2 ENOE - Encuesta Nacional de Ocupación y Empleo (Jalisco)

#### Descripción General
- **Proveedor:** INEGI (Instituto Nacional de Estadística y Geografía)
- **URL:** https://www.inegi.org.mx/programas/enoe/15ymas/
- **Actualización:** Trimestral
- **Cobertura:** Nacional (filtrar por Jalisco)
- **Licencia:** Uso libre con atribución

#### Variables Clave

```
Demográficas:
- ent: Entidad federativa (14 = Jalisco)
- mun: Municipio
- edad: Edad
- sexo: Sexo (1=Hombre, 2=Mujer)

Educación:
- niv_edu: Nivel educativo
  1 = Sin instrucción
  2 = Preescolar
  3 = Primaria incompleta
  4 = Primaria completa
  5 = Secundaria incompleta
  6 = Secundaria completa
  7 = Preparatoria incompleta
  8 = Preparatoria completa
  9 = Universidad incompleta
  10 = Universidad completa
  11 = Maestría
  12 = Doctorado

Ocupación:
- clase2: Ocupación SINCO (4 dígitos)
- p3: Actividad económica
- rama: Rama de actividad

Ingresos:
- ing_x_hrs: Ingreso por hora trabajada
- hrsocup: Horas trabajadas

Posición:
- pos_ocu: Posición en la ocupación
  1 = Trabajador subordinado
  2 = Empleador
  3 = Trabajador por cuenta propia
  4 = Trabajador sin pago
```

#### Descarga de ENOE

**Método 1: Portal INEGI**
```
1. Visita: https://www.inegi.org.mx/sistemas/olap/proyectos/bd/encuestas/hogares/enoe/2019_PE_ED15/bd/enoe_n_pob_bd.asp
2. Selecciona trimestre más reciente
3. Filtra por entidad: 14 (Jalisco)
4. Descarga formato CSV
```

**Método 2: API INEGI**
```python
import requests
import pandas as pd

# Token API INEGI (requiere registro)
token = "TU_TOKEN_AQUI"

# Consultar datos
url = f"https://www.inegi.org.mx/app/api/indicadores/desarrolladores/jsonxml/INDICATOR/ID/es/{token}/?type=json"

response = requests.get(url)
data = response.json()
```

#### Procesamiento ENOE para Jalisco

```python
import pyspark.pandas as ps

# Cargar ENOE
enoe = ps.read_csv('data/raw/enoe_raw.csv', encoding='latin1')

# Filtrar Jalisco
jalisco = enoe[enoe['ent'] == 14]

# Agrupar por ocupación
ocupacion_summary = jalisco.groupby('clase2').agg({
    'sexo': 'count',  # Número de trabajadores
    'ing_x_hrs': 'mean',  # Salario promedio
    'edad': 'mean',  # Edad promedio
    'niv_edu': 'mode',  # Nivel educativo más común
    'hrsocup': 'mean'  # Horas trabajadas promedio
}).rename(columns={'sexo': 'num_trabajadores'})
```

---

## 2. Fuentes de Datos Secundarias

### 2.1 Estudios de Automatización

#### 2.1.1 Frey & Osborne (2013)
**Título:** "The Future of Employment"
**Datos:** Probabilidades de automatización por ocupación SOC

```
Descarga: https://www.oxfordmartin.ox.ac.uk/downloads/academic/The_Future_of_Employment.pdf

Estructura:
- SOC code
- Occupation name
- Probability of computerization (0-1)
- Computerisable: 1 = High risk (>0.7), 0 = Low risk (<0.3)
```

**Integración:**
```python
frey_osborne = ps.read_csv('data/external/frey_osborne_2013.csv')

# Merge con O*NET
combined = onet.merge(frey_osborne, 
                      left_on='soc_code',
                      right_on='SOC code',
                      how='left')
```

#### 2.1.2 McKinsey Global Institute
**Datos:** Porcentaje de actividades automatizables por ocupación

```
Fuente: "A Future That Works: Automation, Employment, and Productivity" (2017)

Variables:
- Occupation
- Current activities time spent (%)
- Technical automation potential (%)
- Time saved with automation
```

#### 2.1.3 OECD - Risk of Automation
**Datos:** Índice de riesgo por país y ocupación

```
Fuente: "Automation, skills use and training" (Nedelkoska & Quintini, 2018)

Variables:
- Country
- ISCO occupation code
- Risk of automation (%)
- Task content indicators
```

---

### 2.2 Datos Económicos de Jalisco

#### 2.2.1 PIB por Sector
**Fuente:** INEGI - Producto Interno Bruto por Entidad Federativa

```python
# Descargar
url = "https://www.inegi.org.mx/app/api/..."
pib_jalisco = ps.read_excel('data/external/pib_jalisco.xlsx')

# Variables:
# - Sector económico
# - PIB (millones de pesos)
# - Crecimiento anual (%)
# - Año
```

#### 2.2.2 Inversión en Tecnología
**Fuente:** Secretaría de Economía Jalisco

```
Variables:
- Sector
- Inversión en I+D (millones MXN)
- Inversión en automatización
- Adopción de IA (1-5)
```

---

## 3. Integración de Fuentes

### 3.1 Pipeline de Integración

```python
from pyspark.sql import SparkSession
import pyspark.pandas as ps

# 1. Cargar O*NET
onet = ps.read_csv('data/raw/onet/Occupation Data.txt', sep='\t')
skills = ps.read_csv('data/raw/onet/Skills.txt', sep='\t')
work_activities = ps.read_csv('data/raw/onet/Work Activities.txt', sep='\t')

# 2. Merge O*NET datasets
onet_full = onet.merge(skills, on='O*NET-SOC Code') \
                .merge(work_activities, on='O*NET-SOC Code')

# 3. Cargar ENOE Jalisco
enoe_jalisco = ps.read_csv('data/raw/enoe_jalisco.csv')

# 4. Mapear SOC → SINCO
mapping = ps.read_csv('data/mappings/soc_sinco_mapping.csv')
onet_mapped = onet_full.merge(mapping, on='O*NET-SOC Code')

# 5. Integrar con ENOE
final_dataset = onet_mapped.merge(
    enoe_jalisco,
    left_on='SINCO_code',
    right_on='clase2',
    how='inner'
)

# 6. Agregar datos de automatización
frey_data = ps.read_csv('data/external/frey_osborne.csv')
final_dataset = final_dataset.merge(frey_data, on='O*NET-SOC Code')

# 7. Guardar dataset integrado
final_dataset.to_spark().write.parquet('data/processed/integrated_dataset.parquet')
```

### 3.2 Esquema del Dataset Final

```python
final_dataset.printSchema()
```

```
root
 |-- occupation_id: string (O*NET-SOC Code)
 |-- occupation_name: string
 |-- sector: string
 |-- 
 |-- # Características de O*NET
 |-- skill_critical_thinking: double
 |-- skill_programming: double
 |-- skill_social_perceptiveness: double
 |-- ability_oral_comprehension: double
 |-- activity_analyzing_data: double
 |-- activity_making_decisions: double
 |-- context_degree_automation: integer
 |-- context_repetitive_tasks: integer
 |-- required_education: integer
 |-- 
 |-- # Datos de ENOE Jalisco
 |-- workers_jalisco: integer
 |-- avg_salary_mxn: double
 |-- avg_age: double
 |-- predominant_education: integer
 |-- avg_hours_worked: double
 |-- municipality: string
 |-- 
 |-- # Índices de automatización
 |-- frey_osborne_prob: double (0-1)
 |-- mckinsey_automation_potential: double (0-100)
 |-- 
 |-- # Features engineered
 |-- routine_index: double
 |-- cognitive_demand: double
 |-- social_interaction: double
 |-- automation_risk: double (0-1)
 |-- risk_category: string (Alto/Medio/Bajo)
```

---

## 4. Calidad y Validación de Datos

### 4.1 Checklist de Calidad

```python
def validate_dataset(df):
    """Valida calidad del dataset"""
    
    checks = {
        'Total registros': len(df),
        'Columnas': len(df.columns),
        'Valores nulos (%)': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        'Duplicados': df.duplicated().sum(),
        'Ocupaciones únicas': df['occupation_name'].nunique(),
        'Rango salarios': f"${df['avg_salary_mxn'].min():.2f} - ${df['avg_salary_mxn'].max():.2f}",
        'Rango riesgo': f"{df['automation_risk'].min():.3f} - {df['automation_risk'].max():.3f}"
    }
    
    for check, result in checks.items():
        print(f"✓ {check}: {result}")
    
    return checks
```

### 4.2 Estadísticas Esperadas

```
✓ Total registros: 300-500 ocupaciones
✓ Cobertura Jalisco: 80%+ de empleo
✓ Valores nulos: <5%
✓ Rango automation_risk: 0.0 - 1.0
✓ Distribución educación: Similar a población real
```

---

## 5. Datos de Ejemplo (Simulados)

Para pruebas y desarrollo, puedes generar datos sintéticos:

```python
import numpy as np
import pandas as pd

# Generar 100 ocupaciones simuladas
np.random.seed(42)

n = 100
data = {
    'occupation_id': [f'OCC-{i:03d}' for i in range(n)],
    'occupation_name': [f'Ocupación {i}' for i in range(n)],
    'sector': np.random.choice(['Manufactura', 'Servicios', 'Comercio', 'Gobierno'], n),
    'routine_index': np.random.uniform(20, 95, n),
    'cognitive_demand': np.random.uniform(30, 90, n),
    'social_interaction': np.random.uniform(10, 85, n),
    'education_level': np.random.choice([3, 4, 5, 6], n),
    'workers_jalisco': np.random.randint(100, 50000, n),
    'avg_salary_mxn': np.random.uniform(5000, 50000, n),
    'automation_risk': np.random.uniform(0.1, 0.9, n)
}

df_sample = pd.DataFrame(data)
df_sample.to_csv('data/sample/ocupaciones_simuladas.csv', index=False)
```

---

## 6. Actualización de Datos

### Frecuencia Recomendada

| Fuente | Frecuencia | Última actualización |
|--------|-----------|---------------------|
| O*NET | Anual | Diciembre 2024 |
| ENOE | Trimestral | Q3 2024 |
| Frey-Osborne | Estático | 2013 |
| PIB Jalisco | Anual | 2023 |

### Script de Actualización

```python
def update_datasets():
    """Actualiza todos los datasets"""
    
    print("Actualizando datos...")
    
    # 1. Verificar nuevas versiones O*NET
    check_onet_updates()
    
    # 2. Descargar último trimestre ENOE
    download_latest_enoe()
    
    # 3. Re-ejecutar integración
    integrate_sources()
    
    # 4. Validar calidad
    validate_dataset()
    
    print("✓ Actualización completa")
```

---

## 7. Repositorios de Datos

### Repositorios Públicos Recomendados

1. **Datos Abiertos México**
   - https://datos.gob.mx/
   - Datasets gubernamentales

2. **Kaggle Datasets**
   - https://www.kaggle.com/datasets
   - Búsqueda: "employment automation", "jobs ai"

3. **UCI Machine Learning Repository**
   - https://archive.ics.uci.edu/ml/
   - Datasets de investigación

4. **GitHub - Awesome Public Datasets**
   - https://github.com/awesomedata/awesome-public-datasets
   - Colección curada

---

## 8. Licencias y Atribución

### O*NET
```
"This application uses information from O*NET OnLine. 
O*NET is a trademark of the U.S. Department of Labor."
```

### INEGI
```
"Fuente: INEGI. Encuesta Nacional de Ocupación y Empleo (ENOE), [Trimestre/Año]."
```

### Frey & Osborne
```
"Frey, C. B., & Osborne, M. A. (2017). The future of employment: 
How susceptible are jobs to computerisation? 
Technological forecasting and social change, 114, 254-280."
```

--- Fin del Documento ---