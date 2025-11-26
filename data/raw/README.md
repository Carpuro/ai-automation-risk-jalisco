# Datos Crudos (Raw Data)

Este directorio debe contener los datos originales sin procesar.

## 📥 Datos Requeridos

### 1. O*NET Database

Descargar desde: https://www.onetcenter.org/database.html

**Archivos necesarios:**
```
data/raw/onet/
├── Occupation Data.txt
├── Skills.txt
├── Abilities.txt
├── Work Activities.txt
├── Work Context.txt
└── Knowledge.txt
```

**Formato:** Tab-separated values (TSV)  
**Encoding:** UTF-8

### 2. INEGI ENOE - Jalisco

Descargar desde: https://www.inegi.org.mx/programas/enoe/

**Archivo:**
```
data/raw/enoe_jalisco.csv
```

**Filtrar por:**
- `ent` = 14 (Jalisco)
- Trimestre más reciente

**Columnas clave:**
- `clase2` - Código SINCO
- `pos_ocu` - Posición en la ocupación
- `ingocup` - Ingreso por ocupación
- `nivel` - Nivel educativo

### 3. Estudios de Automatización (Opcional)

```
data/raw/frey_osborne_2013.csv
data/raw/mckinsey_automation.csv
```

## ⚠️ Importante

- **NO subir** estos archivos a Git (están en `.gitignore`)
- Son archivos grandes (>100MB en algunos casos)
- Descargar localmente para cada usuario
- Verificar licencias de uso

## 📝 Instrucciones de Descarga

### O*NET Database

1. Ir a https://www.onetcenter.org/database.html
2. Seleccionar "Download Database"
3. Elegir formato "Tab-Delimited Text"
4. Descargar y extraer en `data/raw/onet/`

### ENOE INEGI

1. Ir a https://www.inegi.org.mx/programas/enoe/
2. Seleccionar "Microdatos"
3. Descargar trimestre más reciente
4. Filtrar registros donde `ent = 14` (Jalisco)
5. Guardar como `data/raw/enoe_jalisco.csv`

## 🔍 Validación

Después de descargar, verificar con:

```python
import os

required_files = [
    'data/raw/onet/Occupation Data.txt',
    'data/raw/enoe_jalisco.csv'
]

for file in required_files:
    if os.path.exists(file):
        print(f"✓ {file}")
    else:
        print(f"✗ {file} - FALTA")
```

---

**Nota:** Este directorio contiene datos originales **inmutables**. No modificar archivos aquí, usar `data/processed/` para datos transformados.
