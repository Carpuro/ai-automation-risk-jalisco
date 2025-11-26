# Guía de Contribución

¡Gracias por tu interés en contribuir al proyecto de Análisis de Riesgo de Automatización Laboral! 

## 📋 Cómo Contribuir

### 1. Fork del Repositorio

1. Haz fork del repositorio
2. Clona tu fork localmente:
```bash
git clone https://github.com/Carpuro/ai-automation-risk-jalisco.git
cd ai-automation-risk-jalisco
```

### 2. Configurar Entorno

```bash
# Crear entorno conda
conda env create -f environment.yml
conda activate ai_automation_thesis

# Verificar instalación
python verify_setup.py
```

### 3. Crear una Rama

```bash
git checkout -b feature/nombre-de-tu-feature
```

Usa prefijos descriptivos:
- `feature/` - Nueva funcionalidad
- `fix/` - Corrección de bugs
- `docs/` - Documentación
- `refactor/` - Refactorización de código
- `test/` - Tests

### 4. Realizar Cambios

#### Código Python
- Sigue PEP 8
- Documenta funciones con docstrings
- Agrega type hints cuando sea posible
- Mantén funciones pequeñas y enfocadas

```python
def calculate_risk(df: ps.DataFrame, method: str = 'frey_osborne') -> ps.DataFrame:
    """
    Calcula riesgo de automatización.
    
    Parameters:
    -----------
    df : pyspark.pandas.DataFrame
        DataFrame con características de ocupaciones
    method : str
        Método de cálculo ('frey_osborne', 'task_based', 'hybrid')
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con columna 'automation_risk' agregada
    """
    # Tu código aquí
    pass
```

#### Tests
Si agregas nueva funcionalidad, incluye tests:

```python
# En tests/test_automation_analyzer.py
def test_calculate_risk():
    df = load_sample_data(spark, n_occupations=10)
    analyzer = AutomationRiskAnalyzer()
    df_risk = analyzer.calculate_automation_risk(df)
    
    assert 'automation_risk' in df_risk.columns
    assert df_risk['automation_risk'].min() >= 0
    assert df_risk['automation_risk'].max() <= 1
```

### 5. Commit

Usa mensajes claros y descriptivos:

```bash
git add .
git commit -m "feat: agregar análisis por municipio en Jalisco"
```

**Formato de commits:**
- `feat:` Nueva funcionalidad
- `fix:` Corrección de bug
- `docs:` Cambios en documentación
- `style:` Formato, sin cambio de lógica
- `refactor:` Refactorización
- `test:` Agregar o modificar tests
- `chore:` Mantenimiento

### 6. Push y Pull Request

```bash
git push origin feature/nombre-de-tu-feature
```

Luego crea un Pull Request en GitHub con:
- Título descriptivo
- Descripción detallada de los cambios
- Referencias a issues relacionados
- Screenshots (si aplica)

## 🎯 Áreas de Contribución

### Prioridad Alta
- [ ] Integración con datos reales de O*NET y ENOE
- [ ] Validación del modelo con datos históricos
- [ ] Análisis geográfico por municipios
- [ ] Dashboard interactivo con Streamlit/Dash

### Prioridad Media
- [ ] Más algoritmos de ML (XGBoost, LightGBM)
- [ ] Análisis de sensibilidad
- [ ] Exportar reportes a PDF
- [ ] API REST para predicciones

### Prioridad Baja
- [ ] Internacionalización (i18n)
- [ ] Más visualizaciones
- [ ] Optimización de performance
- [ ] Documentación adicional

## 📝 Estándares de Código

### Python
- **Estilo:** PEP 8
- **Longitud de línea:** Máximo 100 caracteres
- **Imports:** Ordenados alfabéticamente
- **Docstrings:** Google style

### Documentación
- README en español e inglés
- Comentarios en español
- Docstrings en español
- Ejemplos de uso en notebooks

### Tests
- Usar pytest
- Cobertura mínima: 70%
- Tests unitarios para funciones core
- Tests de integración para pipelines

## 🐛 Reportar Bugs

Usa el template de issue:

**Descripción del bug:**
Describe claramente el problema.

**Pasos para reproducir:**
1. Paso 1
2. Paso 2
3. ...

**Comportamiento esperado:**
Qué debería pasar.

**Screenshots:**
Si aplica.

**Entorno:**
- OS: [ej. Windows 11, macOS 14]
- Python version: [ej. 3.10.8]
- PySpark version: [ej. 3.5.0]

## 💡 Sugerir Features

Crea un issue con:
- Título claro
- Descripción del feature
- Por qué es útil
- Ejemplo de uso propuesto
- Alternativas consideradas

## 📧 Contacto

**Carlos Pulido Rosas**  
📧 carlos.pulido.rosas@gmail.com  
🎓 CUCEA - Universidad de Guadalajara

## 🙏 Agradecimientos

Gracias a todos los contribuidores que ayudan a mejorar este proyecto:

<!-- ALL-CONTRIBUTORS-LIST:START -->
<!-- Aquí se agregarán automáticamente los contribuidores -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## 📜 Código de Conducta

Este proyecto adhiere al Contributor Covenant Code of Conduct. Al participar, se espera que respetes este código.

### Nuestros Estándares

**Ejemplos de comportamiento que contribuyen a crear un ambiente positivo:**
- Usar lenguaje acogedor e inclusivo
- Respetar puntos de vista y experiencias diferentes
- Aceptar críticas constructivas
- Enfocarse en lo mejor para la comunidad
- Mostrar empatía hacia otros miembros

**Ejemplos de comportamiento inaceptable:**
- Uso de lenguaje o imágenes sexualizadas
- Trolling, insultos o ataques personales
- Acoso público o privado
- Publicar información privada sin permiso
- Conducta no ética o no profesional

## ✅ Checklist antes de PR

- [ ] El código sigue el estilo PEP 8
- [ ] He agregado docstrings a funciones nuevas
- [ ] He agregado tests para nueva funcionalidad
- [ ] Todos los tests pasan (`pytest`)
- [ ] He actualizado la documentación
- [ ] He agregado ejemplo de uso si aplica
- [ ] Mi commit message es descriptivo
- [ ] He verificado que no rompo funcionalidad existente

---

**¡Gracias por contribuir!** 🚀
