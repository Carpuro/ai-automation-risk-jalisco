# Mapeos de Clasificación de Ocupaciones

Este directorio contiene archivos de mapeo entre sistemas de clasificación de ocupaciones.

## 📁 Archivos

### soc_sinco_mapping.csv

Mapeo entre **SOC (USA)** y **SINCO (México)**.

**Sistemas de clasificación:**
- **SOC** (Standard Occupational Classification) - Estados Unidos
- **SINCO** (Sistema Nacional de Clasificación de Ocupaciones) - México

**Columnas:**
- `soc_code` - Código SOC de 8 dígitos (ej. "15-1211.00")
- `soc_title` - Título de la ocupación en SOC
- `sinco_code` - Código SINCO de 4 dígitos (ej. "2121")
- `sinco_title` - Título de la ocupación en SINCO

**Tamaño:** 60 mapeos

**Uso:**
```python
import pandas as pd

mapping = pd.read_csv('data/mappings/soc_sinco_mapping.csv')

# Buscar equivalente SINCO para un código SOC
soc_code = '15-1211.00'
sinco_equiv = mapping[mapping['soc_code'] == soc_code]
print(sinco_equiv)
```

## 🌐 Fuentes

### O*NET Database (SOC)
- **URL:** https://www.onetcenter.org/
- **Versión:** 28.2 (2024)
- **Mantenido por:** U.S. Department of Labor
- **Actualización:** Anual

### INEGI SINCO (México)
- **URL:** https://www.inegi.org.mx/app/scian/
- **Versión:** 2011 (vigente)
- **Mantenido por:** INEGI
- **Cobertura:** México

## ⚠️ Limitaciones

Este mapeo es **simplificado** y cubre solo ocupaciones comunes. Para un mapeo completo:

1. **Oficial INEGI:** Consultar tablas de correspondencia oficiales
2. **O*NET-SOC:** Usar crosswalks oficiales
3. **Validación manual:** Requerida para casos específicos

## 📊 Estructura de Códigos

### SOC (8 dígitos)
```
XX-XXXX.XX
│  │    └─ Ocupación detallada (00-99)
│  └────── Grupo de ocupaciones (4 dígitos)
└───────── Grupo mayor (2 dígitos)
```

Ejemplos:
- `15-1211.00` - Computer Systems Analysts
- `29-1141.00` - Registered Nurses
- `41-2011.00` - Cashiers

### SINCO (4 dígitos)
```
XXXX
││└└── Ocupación específica
│└──── Subgrupo
└───── Grupo principal
```

Ejemplos:
- `2121` - Analistas de sistemas
- `3221` - Enfermeras generales
- `4211` - Cajeros

## 🔄 Actualizar Mapeo

Para agregar más mapeos:

```python
import pandas as pd

# Cargar mapeo existente
mapping = pd.read_csv('data/mappings/soc_sinco_mapping.csv')

# Agregar nuevos mapeos
new_mapping = pd.DataFrame({
    'soc_code': ['XX-XXXX.XX'],
    'soc_title': ['Título SOC'],
    'sinco_code': ['XXXX'],
    'sinco_title': ['Título SINCO']
})

# Combinar y guardar
updated_mapping = pd.concat([mapping, new_mapping], ignore_index=True)
updated_mapping.to_csv('data/mappings/soc_sinco_mapping.csv', index=False)
```

## 📚 Referencias

1. **O*NET OnLine:** https://www.onetonline.org/
2. **INEGI SINCO:** https://www.inegi.org.mx/contenidos/productos/prod_serv/contenidos/espanol/bvinegi/productos/nueva_estruc/702825198701.pdf
3. **BLS SOC:** https://www.bls.gov/soc/

## 🤝 Contribuir

Si tienes correcciones o mejoras al mapeo, por favor:
1. Verifica con fuentes oficiales
2. Abre un issue
3. Envía un PR con la actualización

---

**Última actualización:** Noviembre 2025  
**Autor:** Carlos Pulido Rosas  
**Versión:** 1.0
