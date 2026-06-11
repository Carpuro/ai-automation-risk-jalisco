-- ============================================================
-- Proyecto Final — Programacion II | MCD CUCEA 2026-A
-- Autor: Carlos Pulido Rosas
-- Descripcion: Consultas JOIN y creacion de VIEW analitica
--              sobre el riesgo de automatizacion en Jalisco
-- Base de datos: ai_automation_risk_jalisco (SQL Server)
-- ============================================================

USE ai_automation_risk_jalisco;
GO

-- ============================================================
-- JOIN 1 — INNER JOIN
-- Pregunta de tesis: ¿Que porcentaje de trabajadores en Jalisco
-- tiene datos de tareas O*NET disponibles (crosswalk completo)?
-- Se usa INNER JOIN porque solo nos interesan los registros con
-- coincidencia en ambas tablas (trabajadores CON ocupacion mapeada).
-- ============================================================
SELECT
    o.occupation_name                           AS ocupacion,
    o.sector,
    o.frey_osborne_score                        AS riesgo_frey_osborne,
    t.escoacum                                  AS nivel_educativo,
    t.ingocup                                   AS ingreso_mensual,
    t.formalidad,
    t.edad
FROM dbo.trabajadores t
INNER JOIN dbo.ocupaciones_onet o
    ON t.sinco_code = o.sinco_code
ORDER BY o.frey_osborne_score DESC;
GO

-- ============================================================
-- JOIN 2 — LEFT JOIN
-- Pregunta de tesis: ¿Que ocupaciones quedan sin cobertura O*NET
-- (sin crosswalk SOC→SINCO completo)?
-- Se usa LEFT JOIN para conservar TODOS los trabajadores, incluso
-- los que no tienen ocupacion mapeada en O*NET — precisamente
-- los mas vulnerables del sector informal y agropecuario.
-- ============================================================
SELECT
    t.id_trabajador,
    t.sinco_code,
    t.scian_sector,
    t.formalidad,
    o.occupation_name                           AS ocupacion_onet,
    o.frey_osborne_score                        AS riesgo,
    CASE
        WHEN o.sinco_code IS NULL THEN 'Sin crosswalk O*NET'
        ELSE 'Mapeado'
    END                                         AS estado_cobertura
FROM dbo.trabajadores t
LEFT JOIN dbo.ocupaciones_onet o
    ON t.sinco_code = o.sinco_code
ORDER BY estado_cobertura, t.scian_sector;
GO

-- ============================================================
-- JOIN 3 — INNER JOIN + GROUP BY + HAVING
-- Pregunta de tesis: ¿Que sectores economicos concentran el mayor
-- riesgo promedio de automatizacion en Jalisco?
-- Se agrega GROUP BY para consolidar por sector y HAVING para
-- filtrar solo sectores con muestra representativa (>= 20 trab).
-- ============================================================
SELECT
    o.sector,
    COUNT(t.id_trabajador)                      AS total_trabajadores,
    ROUND(AVG(o.frey_osborne_score), 4)         AS riesgo_promedio,
    ROUND(AVG(t.ingocup), 2)                    AS ingreso_promedio,
    ROUND(AVG(CAST(t.escoacum AS FLOAT)), 2)    AS educacion_promedio,
    SUM(CASE WHEN t.formalidad = 'Informal'
             THEN 1 ELSE 0 END)                 AS trabajadores_informales
FROM dbo.trabajadores t
INNER JOIN dbo.ocupaciones_onet o
    ON t.sinco_code = o.sinco_code
GROUP BY o.sector
HAVING COUNT(t.id_trabajador) >= 20
ORDER BY riesgo_promedio DESC;
GO

-- ============================================================
-- VIEW: vista_riesgo_jalisco
-- Consolida el perfil completo del trabajador (variables
-- socioeconomicas ENOE + descriptores de tareas O*NET) en una
-- sola consulta lista para Orange y Python.
-- Incluye campo calculado riesgo_categoria (CASE WHEN).
-- ============================================================
IF OBJECT_ID('dbo.vista_riesgo_jalisco', 'V') IS NOT NULL
    DROP VIEW dbo.vista_riesgo_jalisco;
GO

CREATE VIEW dbo.vista_riesgo_jalisco AS
WITH sector_factors AS (
    -- Factores de ajuste por sector: misma ocupacion tiene distinto perfil
    -- segun el contexto sectorial donde opera (Acemoglu & Restrepo 2018)
    SELECT
        id_trabajador,
        CASE scian_sector
            WHEN 'Manufactura'  THEN  0.24 WHEN 'Agricultura'  THEN  0.16
            WHEN 'Construccion' THEN  0.10 WHEN 'Transporte'   THEN  0.06
            WHEN 'Comercio'     THEN -0.12 WHEN 'Servicios'    THEN -0.14
            WHEN 'Tecnologia'   THEN -0.16 WHEN 'Finanzas'     THEN -0.20
            WHEN 'Salud'        THEN -0.25 WHEN 'Educacion'    THEN -0.28
            ELSE 0.0 END                                    AS f_risk,
        CASE scian_sector
            WHEN 'Manufactura'  THEN  13.0 WHEN 'Agricultura'  THEN   9.0
            WHEN 'Construccion' THEN   6.0 WHEN 'Transporte'   THEN   4.0
            WHEN 'Comercio'     THEN  -3.0 WHEN 'Servicios'    THEN  -5.0
            WHEN 'Tecnologia'   THEN  -7.0 WHEN 'Finanzas'     THEN  -9.0
            WHEN 'Salud'        THEN -10.0 WHEN 'Educacion'    THEN -13.0
            ELSE 0.0 END                                    AS f_rti,
        -- Demanda cognitiva: manufactura/agri es mas mecanica; salud/educ es mas cognitiva
        CASE scian_sector
            WHEN 'Manufactura'  THEN -22.0 WHEN 'Agricultura'  THEN -18.0
            WHEN 'Construccion' THEN -12.0 WHEN 'Transporte'   THEN  -6.0
            WHEN 'Comercio'     THEN  -2.0 WHEN 'Servicios'    THEN   2.0
            WHEN 'Tecnologia'   THEN  12.0 WHEN 'Finanzas'     THEN  14.0
            WHEN 'Salud'        THEN  18.0 WHEN 'Educacion'    THEN  22.0
            ELSE 0.0 END                                    AS f_cog,
        -- Interaccion social: manufactura/agri es individual; salud/educ es relacional
        CASE scian_sector
            WHEN 'Manufactura'  THEN -18.0 WHEN 'Agricultura'  THEN -14.0
            WHEN 'Construccion' THEN  -8.0 WHEN 'Transporte'   THEN  -4.0
            WHEN 'Comercio'     THEN   8.0 WHEN 'Servicios'    THEN   6.0
            WHEN 'Tecnologia'   THEN   4.0 WHEN 'Finanzas'     THEN   6.0
            WHEN 'Salud'        THEN  22.0 WHEN 'Educacion'    THEN  18.0
            ELSE 0.0 END                                    AS f_soc,
        -- Factor educacion: baja escolaridad = mayor riesgo de automatizacion
        CASE
            WHEN escoacum = 1 THEN  0.06 WHEN escoacum = 2 THEN  0.04
            WHEN escoacum = 3 THEN  0.02 WHEN escoacum = 4 THEN  0.00
            WHEN escoacum = 5 THEN -0.04 WHEN escoacum = 6 THEN -0.07
            ELSE 0.0 END                                    AS f_edu,
        -- Factor ingreso: bajo ingreso = mayor riesgo (Frey & Osborne 2013)
        CASE
            WHEN ingocup <  5000 THEN  0.05
            WHEN ingocup < 10000 THEN  0.02
            WHEN ingocup < 20000 THEN  0.00
            WHEN ingocup < 35000 THEN -0.03
            ELSE                       -0.05
        END                                                 AS f_ing
    FROM dbo.trabajadores
)
SELECT
    t.id_trabajador,
    t.escoacum                                  AS nivel_educativo,
    t.ingocup                                   AS ingreso_mensual,
    t.scian_sector                              AS sector,
    t.formalidad,
    t.edad,
    t.tamanio_empresa,
    t.ira,
    o.occupation_name                           AS ocupacion,
    o.frey_osborne_score,
    -- Block 2 — Task profile ajustado por contexto sectorial
    ROUND(CASE WHEN o.rti + sf.f_rti > 96 THEN 96
               WHEN o.rti + sf.f_rti < 4  THEN 4
               ELSE o.rti + sf.f_rti END, 2)    AS rti,
    ROUND(CASE WHEN o.cognitive_demand + sf.f_cog > 96 THEN 96
               WHEN o.cognitive_demand + sf.f_cog < 4  THEN 4
               ELSE o.cognitive_demand + sf.f_cog END, 2) AS cognitive_demand,
    ROUND(CASE WHEN o.social_interaction + sf.f_soc > 96 THEN 96
               WHEN o.social_interaction + sf.f_soc < 4  THEN 4
               ELSE o.social_interaction + sf.f_soc END, 2) AS social_interaction,
    o.creativity,
    -- automation_risk: score O*NET + ajuste sectorial + educacion + ingreso
    ROUND(CASE
        WHEN o.automation_risk + sf.f_risk + sf.f_edu + sf.f_ing > 0.99 THEN 0.99
        WHEN o.automation_risk + sf.f_risk + sf.f_edu + sf.f_ing < 0.01 THEN 0.01
        ELSE o.automation_risk + sf.f_risk + sf.f_edu + sf.f_ing
    END, 4)                                     AS automation_risk,
    -- riesgo_categoria derivada del mismo valor clippeado que automation_risk
    CASE
        WHEN CASE
                 WHEN o.automation_risk + sf.f_risk + sf.f_edu + sf.f_ing > 0.99 THEN 0.99
                 WHEN o.automation_risk + sf.f_risk + sf.f_edu + sf.f_ing < 0.01 THEN 0.01
                 ELSE o.automation_risk + sf.f_risk + sf.f_edu + sf.f_ing
             END >= 0.70 THEN 'Alto'
        WHEN CASE
                 WHEN o.automation_risk + sf.f_risk + sf.f_edu + sf.f_ing > 0.99 THEN 0.99
                 WHEN o.automation_risk + sf.f_risk + sf.f_edu + sf.f_ing < 0.01 THEN 0.01
                 ELSE o.automation_risk + sf.f_risk + sf.f_edu + sf.f_ing
             END >= 0.40 THEN 'Medio'
        ELSE 'Bajo'
    END                                         AS riesgo_categoria,
    -- Block 3 — LLM Exposure (Phase 2, nivel ocupacion)
    -- Eloundou et al. 2024 via IDB SINCO crosswalk (arXiv inestabilidad: NBER w35110)
    o.gpt_exposure_score,
    -- Arora et al. 2025 arXiv:2510.13369 — robusto, sin sesgo de modelo anotador
    o.moravec_index,
    -- arXiv:2604.04464 — factibilidad tecnica x (1 - compliance_friction)
    -- Servicios informal: mayor riesgo real que Finanzas/Salud pese a similar gpt_exposure
    o.dual_factor_score,
    -- MIT Iceberg Index arXiv:2510.25137 — valor salarial expuesto (H1 evidence)
    o.iceberg_score,
    -- ILO/World Bank WP121 2024 — buffer/bottleneck digital divide Jalisco rural
    o.digital_access
FROM dbo.trabajadores t
INNER JOIN dbo.ocupaciones_onet o ON t.sinco_code = o.sinco_code
INNER JOIN sector_factors sf      ON t.id_trabajador = sf.id_trabajador;
GO

-- Verificar la vista
SELECT TOP 10 * FROM dbo.vista_riesgo_jalisco ORDER BY frey_osborne_score DESC;
GO

PRINT 'JOINs y VIEW creados exitosamente';
GO
