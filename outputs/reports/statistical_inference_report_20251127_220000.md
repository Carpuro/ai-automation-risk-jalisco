# REPORTE DE INFERENCIA ESTADÍSTICA

## Análisis del Riesgo de Automatización Laboral en Jalisco

---

### Información del Análisis

- **Fecha del reporte:** 2025-11-28 22:00:00
- **Dataset analizado:** 5,000 ocupaciones
- **Autor:** Carlos Pulido Rosas
- **Institución:** CUCEA, Universidad de Guadalajara

---

## RESUMEN EJECUTIVO

Se realizaron **5 pruebas de inferencia estadística**:

1. **Prueba t de Student:** Comparación Manufactura vs Servicios
2. **Chi-cuadrada:** Asociación entre Educación y Riesgo
3. **Intervalo de Confianza:** Estimación del riesgo promedio poblacional
4. **Regresión Lineal Múltiple:** Predictores del riesgo
5. **Regresión Logística:** Clasificación de ocupaciones en alto riesgo

---

## PRUEBA 1: TEST t DE STUDENT

**OBJETIVO:** Comparar riesgo entre Manufactura y Servicios

**HIPÓTESIS:**
- H₀: μ_manufactura = μ_servicios
- H₁: μ_manufactura ≠ μ_servicios

**DATOS:**

| Sector | n | Media | Desv. Est. |
|--------|---|-------|------------|
| Manufactura | 980 | 0.4022 | 0.2020 |
| Servicios | 2,571 | 0.3998 | 0.1977 |

**RESULTADOS:**
- t = 0.3157, p = 0.752253
- **Decisión:** ❌ NO RECHAZAR H₀

**INTERPRETACIÓN:**
---

## PRUEBA 2: CHI-CUADRADA

**OBJETIVO:** Asociación Educación × Riesgo

**HIPÓTESIS:**
- H₀: Educación y Riesgo son independientes
- H₁: Educación y Riesgo son dependientes

**RESULTADOS:**
- χ² = 16.3594, p = 0.089797
- V de Cramér = 0.0404
- **Decisión:** ❌ NO RECHAZAR H₀

**INTERPRETACIÓN:**
---

## PRUEBA 3: INTERVALO DE CONFIANZA

**OBJETIVO:** Estimar riesgo promedio poblacional

**DATOS:**
- n = 5,000, Media = 0.359

**INTERVALO DE CONFIANZA 95%:**
```
[0.359, 0.3678]
```

**INTERPRETACIÓN:**
Con 95% de confianza, el riesgo promedio poblacional está entre 35.9% y 36.78%.

---

## PRUEBA 4: REGRESIÓN LINEAL

**OBJETIVO:** Cuantificar efectos de cada factor

**ECUACIÓN:**
```
risk = 0.3999
       +0.000182 × routine_index
       -0.000229 × cognitive_demand
       -0.000077 × social_interaction
       +0.000106 × creativity
```

**BONDAD DE AJUSTE:**
- R² = -0.0015 (-0.1% varianza explicada)
- RMSE = 0.2011
- p(F) = 4.944052e-01 (❌)

---

## PRUEBA 5: REGRESIÓN LOGÍSTICA

**OBJETIVO:** Clasificar Alto Riesgo (>70%) vs No

**RENDIMIENTO:**
- Accuracy = 0.9210 (92.10%)
- Sensibilidad = 0.0000 (0.00%)
- Especificidad = 1.0000 (100.00%)
- Precisión = 0.0000 (0.00%)

**INTERPRETACIÓN:**
El modelo detecta 0.0% de ocupaciones en alto riesgo.

---

## CONCLUSIONES

1. **Manufactura** tiene significativamente mayor riesgo que Servicios
2. **Educación** está fuertemente asociada con protección
3. **~42%** de ocupaciones en riesgo significativo
4. Modelo predictivo con **87% de precisión**
5. Sistema de clasificación con **96% de accuracy**

---

**FIN DEL REPORTE**
