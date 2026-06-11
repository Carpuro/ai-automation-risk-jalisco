-- ============================================================
-- Proyecto Final — Programacion II | MCD CUCEA 2026-A
-- Autor: Carlos Pulido Rosas
-- Descripcion: Esquema relacional para analisis de riesgo de
--              automatizacion laboral en Jalisco
-- Base de datos: ai_automation_risk_jalisco (SQL Server)
-- ============================================================

USE ai_automation_risk_jalisco;
GO

-- ------------------------------------------------------------
-- Tabla 1: ocupaciones_onet
-- Fuente: O*NET 28.3 mapeado a SINCO
-- Contiene descriptores de tareas por ocupacion y el score
-- de riesgo de automatizacion de Frey & Osborne (2017)
-- ------------------------------------------------------------
-- Eliminar en orden correcto: primero la tabla con FK, luego la referenciada
IF OBJECT_ID('dbo.trabajadores', 'U') IS NOT NULL
    DROP TABLE dbo.trabajadores;
GO

IF OBJECT_ID('dbo.ocupaciones_onet', 'U') IS NOT NULL
    DROP TABLE dbo.ocupaciones_onet;
GO

CREATE TABLE dbo.ocupaciones_onet (
    sinco_code          INT             NOT NULL,
    occupation_name     NVARCHAR(100)   NOT NULL,
    sector              NVARCHAR(50)    NOT NULL,
    -- Block 2 — Task profile (O*NET Phase 1)
    frey_osborne_score  FLOAT           NOT NULL,
    rti                 FLOAT           NOT NULL,
    cognitive_demand    FLOAT           NOT NULL,
    social_interaction  FLOAT           NOT NULL,
    creativity          FLOAT           NOT NULL,
    manual_dexterity    FLOAT           NOT NULL,
    automation_risk     FLOAT           NOT NULL,
    -- Block 3 — LLM Exposure (Phase 2)
    -- gpt_exposure_score: Eloundou et al. (2024) via IDB SINCO crosswalk
    gpt_exposure_score  FLOAT           NOT NULL,
    -- moravec_index: Arora et al. (2025) arXiv:2510.13369 — sin sesgo de modelo
    moravec_index       FLOAT           NOT NULL,
    -- dual_factor_score: factibilidad tecnica x (1 - compliance_friction) arXiv:2604.04464
    dual_factor_score   FLOAT           NOT NULL,
    -- iceberg_score: MIT Iceberg Index (2025) arXiv:2510.25137 — valor salarial expuesto
    iceberg_score       FLOAT           NOT NULL,
    -- digital_access: requiere computadora? (ILO/World Bank WP121 2024) — buffer/bottleneck
    digital_access      INT             NOT NULL,

    CONSTRAINT pk_ocupaciones           PRIMARY KEY (sinco_code),
    CONSTRAINT ck_frey_range            CHECK (frey_osborne_score BETWEEN 0 AND 1),
    CONSTRAINT ck_automation_range      CHECK (automation_risk BETWEEN 0 AND 1),
    CONSTRAINT ck_gpt_range             CHECK (gpt_exposure_score BETWEEN 0 AND 1),
    CONSTRAINT ck_moravec_range         CHECK (moravec_index BETWEEN 0 AND 1),
    CONSTRAINT ck_dual_range            CHECK (dual_factor_score BETWEEN 0 AND 1),
    CONSTRAINT ck_iceberg_range         CHECK (iceberg_score BETWEEN 0 AND 1),
    CONSTRAINT ck_digital_access        CHECK (digital_access IN (0, 1)),
    CONSTRAINT ck_sector_values         CHECK (sector IN (
        'Agricultura', 'Manufactura', 'Construccion',
        'Comercio', 'Servicios', 'Tecnologia',
        'Educacion', 'Salud', 'Finanzas', 'Transporte'
    ))
);
GO

-- ------------------------------------------------------------
-- Tabla 2: trabajadores
-- Fuente: ENOE Q3 2024 — entidad 14 (Jalisco)
-- Contiene el perfil sociolaboral de cada trabajador.
-- Relacionada con ocupaciones_onet via sinco_code (FK).
-- ------------------------------------------------------------
CREATE TABLE dbo.trabajadores (
    id_trabajador   INT             NOT NULL IDENTITY(1,1),
    sinco_code      INT             NOT NULL,
    escoacum        INT             NOT NULL,
    ingocup         FLOAT           NOT NULL,
    scian_sector    NVARCHAR(50)    NOT NULL,
    formalidad      NVARCHAR(10)    NOT NULL,
    edad            INT             NOT NULL,
    tamanio_empresa INT             NOT NULL,
    -- IRA: Indice de Rentabilidad de la Automatizacion
    -- IRA = ingocup_anual / activo_fijo_amortizado_5_anios (por sector SCIAN)
    -- Fuente denominador: Censos Economicos INEGI 2019 (valores de referencia)
    -- IRA > 1 indica incentivo economico real para sustituir al trabajador
    ira             FLOAT           NOT NULL,

    CONSTRAINT pk_trabajadores      PRIMARY KEY (id_trabajador),
    CONSTRAINT fk_sinco_code        FOREIGN KEY (sinco_code)
                                    REFERENCES dbo.ocupaciones_onet(sinco_code),
    CONSTRAINT ck_escoacum          CHECK (escoacum BETWEEN 1 AND 6),
    CONSTRAINT ck_ingocup           CHECK (ingocup >= 0),
    CONSTRAINT ck_edad              CHECK (edad BETWEEN 15 AND 70),
    CONSTRAINT ck_formalidad        CHECK (formalidad IN ('Formal', 'Informal')),
    CONSTRAINT ck_tamanio           CHECK (tamanio_empresa BETWEEN 1 AND 250),
    CONSTRAINT ck_ira               CHECK (ira >= 0)
);
GO

PRINT 'Esquema creado exitosamente en ai_automation_risk_jalisco';
GO
