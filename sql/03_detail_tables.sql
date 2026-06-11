-- 10_schema_expansion.sql
-- Expande la base de datos ai_automation_risk_jalisco con todas las fuentes disponibles
-- Ejecutar en orden desde localhost (homelab) o via 10_load_all_data.py

USE ai_automation_risk_jalisco;
GO

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. ENOE SDEMT — todos los trabajadores de Jalisco (13,839)
-- ─────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID('enoe_sdemt_full', 'U') IS NULL
CREATE TABLE enoe_sdemt_full (
    -- Clave compuesta ENOE
    ent         TINYINT,
    con         NVARCHAR(5),
    upm         NVARCHAR(10),
    d_sem       TINYINT,
    n_pro_viv   TINYINT,
    v_sel       TINYINT,
    n_hog       TINYINT,
    h_mud       TINYINT,
    n_ent       TINYINT,
    per         TINYINT,
    n_ren       TINYINT,
    -- Variables sociodemograficas
    sex         TINYINT,
    eda         TINYINT,
    -- Variables educativas
    niv_ins     TINYINT,
    anios_esc   TINYINT,
    cs_p17      TINYINT,
    -- Variables laborales
    clase1      TINYINT,
    pos_ocu     TINYINT,
    seg_soc     TINYINT,
    c_ocu11c    NVARCHAR(5),
    scian       NVARCHAR(5),
    rama_est2   TINYINT,
    emple7c     TINYINT,
    t_tra       TINYINT,
    -- Variables de ingreso
    ingocup     FLOAT,
    ing7c       TINYINT,
    ing_x_hrs   FLOAT,
    hrsocup     FLOAT,
    -- Ubicacion
    t_loc_tri   TINYINT,
    ur          TINYINT,
    -- Pesos muestrales
    fac_tri     FLOAT,
    CONSTRAINT PK_enoe_sdemt PRIMARY KEY (ent, con, upm, d_sem, n_pro_viv, v_sel, n_hog, h_mud, n_ent, per, n_ren)
);
GO

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. ENOE COE1 — condiciones de trabajo (p5f = condiciones fisicas)
-- ─────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID('enoe_coe1_jalisco', 'U') IS NULL
CREATE TABLE enoe_coe1_jalisco (
    -- Clave de join con enoe_sdemt_full
    ent         TINYINT,
    con         NVARCHAR(5),
    upm         NVARCHAR(10),
    d_sem       TINYINT,
    n_pro_viv   TINYINT,
    v_sel       TINYINT,
    n_hog       TINYINT,
    h_mud       TINYINT,
    n_ent       TINYINT,
    per         TINYINT,
    n_ren       TINYINT,
    -- Tipo de lugar de trabajo
    tipo_local              TINYINT,   -- p1: 1=casa, 2=local, 3=vía pública, 4=campo, 5=otro
    -- Contrato
    tipo_contrato           TINYINT,   -- p2c: 1=indefinido, 2=temporal, 3=sin contrato
    contrato_escrito        TINYINT,   -- p3i: 1=si, 2=no
    -- Tamaño del establecimiento (continuo)
    tam_establecimiento_n   INT,       -- p3h: numero de trabajadores
    -- Horas trabajadas
    horas_semana            FLOAT,     -- p5b_thrs
    -- Condiciones fisicas del trabajo (p5f1-p5f15)
    -- 1=si, 2=no, 9=no especificado
    cond_exterior     TINYINT,  -- p5f1:  al aire libre, condiciones climaticas
    cond_de_pie       TINYINT,  -- p5f2:  de pie la mayor parte del tiempo
    cond_peso         TINYINT,  -- p5f3:  levanta o mueve objetos pesados
    cond_repetitivo   TINYINT,  -- p5f4:  movimientos repetitivos
    cond_maquinaria   TINYINT,  -- p5f5:  opera maquinaria fija
    cond_transporte   TINYINT,  -- p5f6:  opera equipo de transporte
    cond_computadora  TINYINT,  -- p5f7:  usa computadora / smartphone / tableta
    cond_clientes     TINYINT,  -- p5f8:  atiende clientes directamente
    cond_vigilancia   TINYINT,  -- p5f9:  protege o vigila
    cond_temperatura  TINYINT,  -- p5f10: altas temperaturas
    cond_peligroso    TINYINT,  -- p5f11: sustancias peligrosas
    cond_altura       TINYINT,  -- p5f12: trabajo en altura
    cond_herramientas TINYINT,  -- p5f13: herramientas manuales
    cond_elaborar     TINYINT,  -- p5f14: elabora o transforma productos
    cond_reparacion   TINYINT,  -- p5f15: reparacion o mantenimiento
    cond_ninguna      TINYINT,  -- p5f99: ninguna de las anteriores
    CONSTRAINT PK_enoe_coe1 PRIMARY KEY (ent, con, upm, d_sem, n_pro_viv, v_sel, n_hog, h_mud, n_ent, per, n_ren)
);
GO

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. O*NET Task Ratings
-- ─────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID('onet_task_ratings', 'U') IS NULL
CREATE TABLE onet_task_ratings (
    onetsoc_code    NVARCHAR(10)  NOT NULL,
    task_id         INT           NOT NULL,
    task_type       NVARCHAR(15),          -- Core / Supplemental
    scale_id        NVARCHAR(5)   NOT NULL, -- IM=Importance FT=Frequency RL=Relevance RT=Required Level
    data_value      FLOAT,
    n_value         INT,
    standard_error  FLOAT,
    recommend_suppress CHAR(1),
    date_updated    DATE,
    INDEX IX_onet_task_soc (onetsoc_code),
    INDEX IX_onet_task_id  (task_id)
);
GO

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. O*NET Technology Skills
-- ─────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID('onet_technology_skills', 'U') IS NULL
CREATE TABLE onet_technology_skills (
    onetsoc_code        NVARCHAR(10),
    example             NVARCHAR(500),
    commodity_code      INT,
    commodity_title     NVARCHAR(200),
    hot_technology      CHAR(1),           -- Y / N
    INDEX IX_onet_tech_soc (onetsoc_code)
);
GO

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. O*NET Task Statements
-- ─────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID('onet_task_statements', 'U') IS NULL
CREATE TABLE onet_task_statements (
    task_id         INT           NOT NULL,
    onetsoc_code    NVARCHAR(10)  NOT NULL,
    task            NVARCHAR(MAX),
    task_type       NVARCHAR(15),
    incumbents_responding INT,
    date_updated    DATE,
    CONSTRAINT PK_onet_tasks PRIMARY KEY (task_id),
    INDEX IX_onet_task_stmt_soc (onetsoc_code)
);
GO

-- ─────────────────────────────────────────────────────────────────────────────
-- 6. O*NET Emerging Tasks
-- ─────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID('onet_emerging_tasks', 'U') IS NULL
CREATE TABLE onet_emerging_tasks (
    onetsoc_code    NVARCHAR(10),
    task_id         INT,
    task            NVARCHAR(MAX),
    date_updated    DATE,
    INDEX IX_onet_emerging_soc (onetsoc_code)
);
GO

-- ─────────────────────────────────────────────────────────────────────────────
-- 7. O*NET Skills
-- ─────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID('onet_skills', 'U') IS NULL
CREATE TABLE onet_skills (
    onetsoc_code    NVARCHAR(10),
    element_id      NVARCHAR(20),
    element_name    NVARCHAR(200),
    scale_id        NVARCHAR(5),   -- IM / LV
    data_value      FLOAT,
    n_value         INT,
    standard_error  FLOAT,
    recommend_suppress CHAR(1),
    date_updated    DATE,
    INDEX IX_onet_skills_soc (onetsoc_code),
    INDEX IX_onet_skills_elem (element_id)
);
GO

-- ─────────────────────────────────────────────────────────────────────────────
-- 8. O*NET Work Context
-- ─────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID('onet_work_context', 'U') IS NULL
CREATE TABLE onet_work_context (
    onetsoc_code    NVARCHAR(10),
    element_id      NVARCHAR(20),
    element_name    NVARCHAR(200),
    scale_id        NVARCHAR(5),
    data_value      FLOAT,
    n_value         INT,
    standard_error  FLOAT,
    recommend_suppress CHAR(1),
    date_updated    DATE,
    INDEX IX_onet_wc_soc (onetsoc_code)
);
GO

-- ─────────────────────────────────────────────────────────────────────────────
-- 9. O*NET Education, Training & Experience
-- ─────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID('onet_education_training', 'U') IS NULL
CREATE TABLE onet_education_training (
    onetsoc_code    NVARCHAR(10),
    element_id      NVARCHAR(20),
    element_name    NVARCHAR(200),
    scale_id        NVARCHAR(5),
    category        INT,
    data_value      FLOAT,
    n_value         INT,
    standard_error  FLOAT,
    recommend_suppress CHAR(1),
    date_updated    DATE,
    INDEX IX_onet_edu_soc (onetsoc_code)
);
GO

-- ─────────────────────────────────────────────────────────────────────────────
-- 10. O*NET Work Activities (detalle — complementa los scores agregados)
-- ─────────────────────────────────────────────────────────────────────────────
IF OBJECT_ID('onet_work_activities_detail', 'U') IS NULL
CREATE TABLE onet_work_activities_detail (
    onetsoc_code    NVARCHAR(10),
    element_id      NVARCHAR(20),
    element_name    NVARCHAR(200),
    scale_id        NVARCHAR(5),   -- IM / LV
    data_value      FLOAT,
    n_value         INT,
    standard_error  FLOAT,
    recommend_suppress CHAR(1),
    date_updated    DATE,
    INDEX IX_onet_wa_soc  (onetsoc_code),
    INDEX IX_onet_wa_elem (element_id)
);
GO

PRINT 'Schema expansion completado: 9 tablas nuevas creadas.';
