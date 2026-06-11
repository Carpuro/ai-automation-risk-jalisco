"""
Data Loader Module - AI Automation Risk Analysis
=================================================
Módulo para cargar y configurar datos de ocupaciones y automatización.

Autor: Carlos Pulido Rosas
Proyecto: Modelo Predictivo de Sustitución Laboral por IA - Jalisco
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pyspark.pandas as ps
import logging
import os
import sys

# Configurar Python para Spark (crítico en Windows)
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_spark_session(app_name="AI_Automation_Risk_Analysis", memory="8g"):
    """
    Crea y configura una Spark Session optimizada para análisis de automatización.
    
    Parameters:
    -----------
    app_name : str
        Nombre de la aplicación Spark
    memory : str
        Memoria asignada (driver y executor)
        
    Returns:
    --------
    SparkSession
        Sesión de Spark configurada
    """
    try:
        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.memory", memory) \
            .config("spark.executor.memory", memory) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .getOrCreate()
        
        # Habilitar operaciones entre diferentes DataFrames (necesario para feature engineering)
        ps.set_option('compute.ops_on_diff_frames', True)
        
        logger.info(f"✓ Spark Session '{app_name}' creada exitosamente")
        logger.info(f"  Versión de Spark: {spark.version}")
        logger.info(f"  Memoria asignada: {memory}")
        
        return spark
        
    except Exception as e:
        logger.error(f"Error creando Spark Session: {str(e)}")
        raise


def load_onet_occupations(spark, file_path, encoding='utf-8'):
    """
    Carga datos de ocupaciones de O*NET.
    
    Parameters:
    -----------
    spark : SparkSession
        Sesión activa de Spark
    file_path : str
        Ruta al archivo de ocupaciones (generalmente tab-separated)
    encoding : str
        Codificación del archivo
        
    Returns:
    --------
    DataFrame
        Spark DataFrame con datos de O*NET
    """
    try:
        logger.info(f"Cargando datos O*NET desde: {file_path}")
        
        df_spark = spark.read.csv(
            file_path,
            sep='\t',
            header=True,
            inferSchema=True,
            encoding=encoding
        )
        
        num_rows = df_spark.count()
        num_cols = len(df_spark.columns)
        
        logger.info(f"✓ Datos O*NET cargados exitosamente")
        logger.info(f"  Ocupaciones: {num_rows:,}")
        logger.info(f"  Columnas: {num_cols}")
        
        return df_spark
        
    except Exception as e:
        logger.error(f"Error cargando datos O*NET: {str(e)}")
        raise


def load_enoe_jalisco(spark, file_path, encoding='latin1'):
    """
    Carga datos de ENOE filtrados para Jalisco.
    
    Parameters:
    -----------
    spark : SparkSession
        Sesión activa de Spark
    file_path : str
        Ruta al archivo ENOE
    encoding : str
        Codificación del archivo (típicamente latin1 para INEGI)
        
    Returns:
    --------
    DataFrame
        Spark DataFrame con datos de ENOE Jalisco
    """
    try:
        logger.info(f"Cargando datos ENOE Jalisco desde: {file_path}")
        
        df_spark = spark.read.csv(
            file_path,
            header=True,
            inferSchema=True,
            encoding=encoding
        )
        
        # Filtrar solo Jalisco (entidad 14)
        if 'ent' in df_spark.columns:
            df_jalisco = df_spark.filter(col('ent') == 14)
            num_rows = df_jalisco.count()
            logger.info(f"✓ Datos ENOE Jalisco cargados (filtrados por entidad=14)")
        else:
            df_jalisco = df_spark
            num_rows = df_jalisco.count()
            logger.info(f"✓ Datos ENOE cargados (sin filtro de entidad)")
        
        num_cols = len(df_jalisco.columns)
        
        logger.info(f"  Registros: {num_rows:,}")
        logger.info(f"  Columnas: {num_cols}")
        
        return df_jalisco
        
    except Exception as e:
        logger.error(f"Error cargando datos ENOE: {str(e)}")
        raise


def load_automation_studies(spark, frey_osborne_path=None, mckinsey_path=None):
    """
    Carga datos de estudios de automatización (Frey-Osborne, McKinsey).
    
    Parameters:
    -----------
    spark : SparkSession
        Sesión activa de Spark
    frey_osborne_path : str, optional
        Ruta a datos de Frey & Osborne
    mckinsey_path : str, optional
        Ruta a datos de McKinsey
        
    Returns:
    --------
    dict
        Diccionario con DataFrames de cada estudio
    """
    studies = {}
    
    if frey_osborne_path:
        try:
            logger.info("Cargando estudio Frey & Osborne...")
            df_frey = spark.read.csv(
                frey_osborne_path,
                header=True,
                inferSchema=True
            )
            studies['frey_osborne'] = df_frey
            logger.info(f"✓ Frey & Osborne: {df_frey.count():,} ocupaciones")
        except Exception as e:
            logger.warning(f"No se pudo cargar Frey & Osborne: {str(e)}")
    
    if mckinsey_path:
        try:
            logger.info("Cargando estudio McKinsey...")
            df_mckinsey = spark.read.csv(
                mckinsey_path,
                header=True,
                inferSchema=True
            )
            studies['mckinsey'] = df_mckinsey
            logger.info(f"✓ McKinsey: {df_mckinsey.count():,} ocupaciones")
        except Exception as e:
            logger.warning(f"No se pudo cargar McKinsey: {str(e)}")
    
    return studies


def load_mapping_table(spark, mapping_path):
    """
    Carga tabla de mapeo SOC → SINCO.
    
    Parameters:
    -----------
    spark : SparkSession
        Sesión activa de Spark
    mapping_path : str
        Ruta a archivo de mapeo
        
    Returns:
    --------
    DataFrame
        DataFrame con mapeo SOC-SINCO
    """
    try:
        logger.info(f"Cargando tabla de mapeo SOC-SINCO...")
        
        df_mapping = spark.read.csv(
            mapping_path,
            header=True,
            inferSchema=True
        )
        
        logger.info(f"✓ Mapeo cargado: {df_mapping.count():,} registros")
        
        return df_mapping
        
    except Exception as e:
        logger.error(f"Error cargando mapeo: {str(e)}")
        raise


def convert_to_pandas_api(df_spark):
    """
    Convierte un Spark DataFrame a pyspark.pandas DataFrame.
    
    Parameters:
    -----------
    df_spark : DataFrame
        Spark DataFrame a convertir
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame convertido con API de pandas
    """
    try:
        df_ps = df_spark.pandas_api()
        logger.info(f"✓ DataFrame convertido a pyspark.pandas")
        logger.info(f"  Tipo: {type(df_ps)}")
        logger.info(f"  Shape: {df_ps.shape}")
        
        return df_ps
        
    except Exception as e:
        logger.error(f"Error en conversión: {str(e)}")
        raise


def get_data_info(df_ps, name="Dataset"):
    """
    Muestra información general del DataFrame.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame a analizar
    name : str
        Nombre del dataset para logging
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"INFORMACIÓN DEL {name.upper()}")
    logger.info(f"{'='*80}")
    
    logger.info(f"\nDimensiones: {df_ps.shape}")
    logger.info(f"Columnas totales: {len(df_ps.columns)}")
    
    logger.info(f"\nPrimeras columnas:")
    logger.info(df_ps.columns[:10].tolist())
    
    logger.info(f"\nTipos de datos (primeras 10):")
    dtypes_dict = dict(df_ps.dtypes.to_pandas()[:10])
    for col, dtype in dtypes_dict.items():
        logger.info(f"  {col}: {dtype}")
    
    logger.info(f"{'='*80}\n")


def load_sample_data(spark, n_occupations=5000):
    """
    Genera datos de muestra REALISTAS para pruebas y desarrollo.
    
    Genera ocupaciones con características coherentes basadas en ~120 perfiles reales
    organizados en 6 estratos ocupacionales (20 templates por estrato).
    
    Parameters:
    -----------
    spark : SparkSession
        Sesión activa de Spark
    n_occupations : int
        Número de ocupaciones a generar (default: 5000)
        
    Returns:
    --------
    DataFrame
        Spark DataFrame con datos simulados realistas
    """
    import numpy as np
    import pandas as pd
    
    logger.info(f"Generando {n_occupations:,} ocupaciones simuladas con ~120 perfiles realistas...")
    
    np.random.seed(42)
    
    # Templates expandidos: ~120 ocupaciones organizadas por estrato
    # Formato: (Nombre, Sector, Educación, Rutina, Cognitivo, Social, Creatividad, Salario_Min, Salario_Max)
    
    occupation_templates = [
        # ==================== ESTRATO 1: DIRECTIVOS Y ALTA GERENCIA (~20) ====================
        # Alto cognitivo (80-95), Bajo rutina (10-35), Alto salario (25k-60k)
        ('Director General', 'Servicios', 6, 15, 90, 75, 80, 35000, 60000),
        ('Director de Finanzas', 'Servicios', 6, 18, 92, 65, 75, 32000, 58000),
        ('Director de Operaciones', 'Manufactura', 6, 20, 88, 70, 72, 30000, 55000),
        ('Director de Recursos Humanos', 'Servicios', 6, 22, 85, 85, 78, 28000, 52000),
        ('Director de Marketing', 'Comercio', 6, 20, 87, 82, 88, 30000, 56000),
        ('Director de TI', 'Servicios', 6, 18, 92, 68, 85, 32000, 58000),
        ('Gerente General', 'Servicios', 5, 25, 85, 78, 75, 28000, 50000),
        ('Gerente de Planta', 'Manufactura', 5, 28, 82, 72, 70, 26000, 48000),
        ('Gerente de Ventas Regional', 'Comercio', 5, 25, 80, 88, 75, 28000, 52000),
        ('Gerente de Proyectos', 'Construcción', 5, 30, 83, 75, 72, 25000, 48000),
        ('Gerente de Compras', 'Comercio', 5, 32, 78, 70, 68, 24000, 45000),
        ('Gerente de Producción', 'Manufactura', 5, 30, 80, 68, 70, 25000, 47000),
        ('Gerente de Calidad', 'Manufactura', 5, 28, 82, 65, 72, 24000, 46000),
        ('Gerente de Logística', 'Comercio', 5, 32, 78, 70, 65, 23000, 44000),
        ('Gerente de Innovación', 'Servicios', 6, 20, 88, 72, 90, 30000, 55000),
        ('Gerente de Desarrollo', 'Servicios', 6, 22, 86, 70, 85, 28000, 52000),
        ('Subdirector Administrativo', 'Gobierno', 5, 30, 80, 72, 70, 25000, 48000),
        ('Subdirector de Educación', 'Gobierno', 6, 28, 85, 80, 75, 26000, 50000),
        ('Coordinador Regional', 'Gobierno', 5, 32, 78, 75, 68, 24000, 46000),
        ('Jefe de Departamento', 'Gobierno', 5, 35, 75, 70, 65, 22000, 42000),
        
        # ==================== ESTRATO 2: PROFESIONISTAS (~20) ====================
        # Alto cognitivo (75-92), Medio-bajo rutina (20-40), Salario medio-alto (18k-40k)
        ('Ingeniero de Software', 'Servicios', 6, 25, 90, 50, 88, 25000, 45000),
        ('Científico de Datos', 'Servicios', 6, 22, 92, 48, 85, 28000, 48000),
        ('Ingeniero DevOps', 'Servicios', 6, 28, 88, 52, 82, 24000, 44000),
        ('Arquitecto de Soluciones', 'Servicios', 6, 24, 90, 55, 86, 26000, 46000),
        ('Médico Especialista', 'Servicios', 6, 30, 92, 78, 75, 32000, 60000),
        ('Médico General', 'Servicios', 6, 32, 88, 80, 72, 28000, 50000),
        ('Cirujano', 'Servicios', 6, 28, 94, 75, 80, 35000, 65000),
        ('Pediatra', 'Servicios', 6, 30, 90, 85, 75, 30000, 55000),
        ('Abogado Corporativo', 'Servicios', 6, 28, 90, 75, 78, 28000, 52000),
        ('Abogado Litigante', 'Servicios', 6, 30, 88, 82, 80, 26000, 50000),
        ('Notario Público', 'Servicios', 6, 35, 85, 70, 65, 30000, 60000),
        ('Contador Público', 'Servicios', 5, 40, 82, 62, 58, 18000, 35000),
        ('Auditor', 'Servicios', 5, 42, 84, 60, 60, 20000, 38000),
        ('Arquitecto', 'Construcción', 6, 22, 88, 68, 92, 24000, 48000),
        ('Ingeniero Civil', 'Construcción', 5, 28, 86, 70, 75, 24000, 46000),
        ('Ingeniero Industrial', 'Manufactura', 5, 32, 84, 65, 72, 22000, 42000),
        ('Ingeniero Químico', 'Manufactura', 6, 30, 88, 55, 78, 24000, 44000),
        ('Biólogo', 'Servicios', 6, 32, 86, 62, 80, 20000, 38000),
        ('Químico Farmacéutico', 'Manufactura', 6, 35, 85, 58, 75, 22000, 40000),
        ('Veterinario', 'Servicios', 6, 32, 84, 75, 72, 20000, 40000),
        
        # ==================== ESTRATO 3: TÉCNICOS Y ESPECIALISTAS (~20) ====================
        # Medio cognitivo (65-80), Medio rutina (35-55), Salario medio (12k-28k)
        ('Técnico en Sistemas', 'Servicios', 4, 42, 76, 58, 72, 16000, 28000),
        ('Técnico de Redes', 'Servicios', 4, 45, 74, 55, 70, 15000, 26000),
        ('Técnico de Soporte IT', 'Servicios', 4, 48, 72, 62, 65, 14000, 24000),
        ('Desarrollador Web Jr', 'Servicios', 4, 38, 78, 52, 80, 18000, 32000),
        ('Enfermero Especializado', 'Servicios', 4, 45, 80, 88, 68, 14000, 26000),
        ('Enfermero General', 'Servicios', 4, 50, 76, 85, 65, 12000, 24000),
        ('Técnico de Radiología', 'Servicios', 4, 55, 75, 68, 62, 14000, 26000),
        ('Técnico de Laboratorio', 'Servicios', 4, 58, 74, 55, 65, 13000, 24000),
        ('Paramédico', 'Servicios', 4, 52, 78, 82, 62, 12000, 23000),
        ('Electricista Industrial', 'Manufactura', 3, 42, 72, 62, 70, 12000, 24000),
        ('Electromecánico', 'Manufactura', 3, 45, 70, 58, 68, 11000, 23000),
        ('Técnico Mecánico', 'Manufactura', 3, 48, 68, 60, 66, 11000, 22000),
        ('Técnico Automotriz', 'Servicios', 3, 50, 70, 65, 68, 12000, 24000),
        ('Soldador Especializado', 'Manufactura', 3, 45, 65, 45, 75, 11000, 22000),
        ('Técnico en Refrigeración', 'Servicios', 3, 48, 68, 62, 65, 10000, 21000),
        ('Diseñador Gráfico', 'Servicios', 4, 35, 75, 62, 90, 13000, 26000),
        ('Diseñador UX/UI', 'Servicios', 4, 32, 78, 65, 88, 16000, 30000),
        ('Diseñador Industrial', 'Manufactura', 4, 38, 76, 60, 86, 14000, 28000),
        ('Fotógrafo Profesional', 'Servicios', 4, 40, 72, 70, 88, 12000, 26000),
        ('Editor de Video', 'Servicios', 4, 42, 74, 58, 85, 13000, 25000),
        
        # ==================== ESTRATO 4: APOYO ADMINISTRATIVO (~20) ====================
        # Medio-bajo cognitivo (55-70), Alto rutina (65-85), Salario medio-bajo (7k-17k)
        ('Asistente Ejecutivo', 'Servicios', 3, 65, 68, 80, 52, 10000, 18000),
        ('Secretaria Ejecutiva', 'Servicios', 3, 70, 65, 78, 48, 9000, 16000),
        ('Secretaria', 'Servicios', 3, 75, 62, 75, 45, 8000, 14000),
        ('Recepcionista', 'Servicios', 2, 78, 58, 85, 38, 7500, 13000),
        ('Telefonista', 'Servicios', 2, 82, 55, 75, 32, 7000, 12000),
        ('Asistente Administrativo', 'Servicios', 3, 72, 64, 72, 46, 8500, 15000),
        ('Auxiliar Administrativo', 'Servicios', 2, 78, 60, 68, 42, 7500, 13000),
        ('Asistente Contable', 'Servicios', 3, 70, 68, 62, 50, 9000, 17000),
        ('Auxiliar de Nómina', 'Servicios', 3, 75, 65, 58, 45, 8500, 15000),
        ('Cajero de Banco', 'Servicios', 3, 82, 66, 72, 35, 9000, 16000),
        ('Cajero Pagador', 'Servicios', 2, 85, 62, 70, 32, 8000, 14000),
        ('Archivista', 'Servicios', 2, 88, 52, 48, 28, 7000, 12000),
        ('Capturista de Datos', 'Servicios', 2, 85, 55, 45, 30, 7500, 13000),
        ('Operador de Call Center', 'Servicios', 2, 80, 60, 78, 35, 8000, 14000),
        ('Supervisor de Call Center', 'Servicios', 3, 72, 68, 82, 55, 11000, 20000),
        ('Coordinador Administrativo', 'Servicios', 3, 68, 70, 75, 58, 12000, 22000),
        ('Asistente de Recursos Humanos', 'Servicios', 3, 70, 66, 78, 52, 9500, 17000),
        ('Asistente de Compras', 'Comercio', 3, 72, 64, 65, 48, 9000, 16000),
        ('Almacenista', 'Comercio', 2, 82, 58, 52, 35, 7500, 13000),
        ('Auxiliar de Inventario', 'Comercio', 2, 85, 56, 48, 32, 7000, 12000),
        
        # ==================== ESTRATO 5: VENTAS Y SERVICIOS (~20) ====================
        # Medio-bajo cognitivo (50-70), Medio-alto rutina (55-75), Alto social (75-95)
        ('Ejecutivo de Cuentas', 'Comercio', 4, 50, 72, 92, 75, 14000, 30000),
        ('Gerente de Tienda', 'Comercio', 4, 55, 70, 88, 68, 13000, 26000),
        ('Supervisor de Ventas', 'Comercio', 3, 58, 68, 90, 65, 12000, 24000),
        ('Vendedor de Autos', 'Comercio', 3, 52, 68, 95, 70, 12000, 28000),
        ('Vendedor de Seguros', 'Servicios', 3, 55, 70, 92, 68, 11000, 25000),
        ('Agente Inmobiliario', 'Servicios', 3, 58, 72, 90, 72, 12000, 30000),
        ('Agente de Viajes', 'Servicios', 3, 60, 66, 88, 65, 9000, 18000),
        ('Vendedor de Tienda', 'Comercio', 2, 68, 58, 92, 58, 7500, 15000),
        ('Demostrador de Productos', 'Comercio', 2, 70, 55, 90, 60, 7000, 14000),
        ('Promotor de Ventas', 'Comercio', 2, 65, 58, 88, 62, 7500, 16000),
        ('Cajero de Supermercado', 'Comercio', 2, 88, 50, 78, 28, 6500, 11000),
        ('Despachador de Mostrador', 'Comercio', 2, 75, 54, 82, 45, 7000, 13000),
        ('Mesero', 'Servicios', 2, 72, 52, 90, 52, 6000, 11000),
        ('Bartender', 'Servicios', 2, 68, 58, 88, 68, 7500, 14000),
        ('Chef', 'Servicios', 3, 58, 72, 78, 88, 12000, 24000),
        ('Cocinero', 'Servicios', 2, 70, 60, 72, 72, 8000, 16000),
        ('Ayudante de Cocina', 'Servicios', 1, 85, 45, 65, 48, 6000, 10000),
        ('Peluquero', 'Servicios', 2, 68, 55, 88, 75, 7000, 14000),
        ('Estilista', 'Servicios', 3, 62, 62, 90, 82, 8500, 18000),
        ('Cosmetóloga', 'Servicios', 2, 70, 58, 85, 78, 7500, 15000),
        
        # ==================== ESTRATO 6: OPERARIOS Y TRABAJADORES (~20) ====================
        # Bajo cognitivo (35-55), Muy alto rutina (75-98), Salario bajo (5k-14k)
        ('Operador de Máquina CNC', 'Manufactura', 2, 85, 52, 40, 35, 8500, 15000),
        ('Operador de Torno', 'Manufactura', 2, 82, 54, 42, 40, 8000, 14000),
        ('Operador de Prensa', 'Manufactura', 2, 88, 48, 38, 32, 7500, 13000),
        ('Operador de Inyección', 'Manufactura', 2, 90, 46, 35, 30, 7500, 13000),
        ('Ensamblador de Línea', 'Manufactura', 2, 92, 42, 35, 25, 7000, 12000),
        ('Ensamblador de Electrónicos', 'Manufactura', 2, 88, 48, 38, 35, 7500, 13000),
        ('Inspector de Calidad', 'Manufactura', 3, 78, 62, 45, 48, 9000, 16000),
        ('Empacador', 'Manufactura', 1, 95, 32, 32, 18, 6000, 10000),
        ('Etiquetador', 'Manufactura', 1, 92, 35, 35, 20, 6000, 10000),
        ('Albañil', 'Construcción', 2, 75, 52, 58, 55, 8000, 15000),
        ('Ayudante de Albañil', 'Construcción', 1, 85, 42, 52, 45, 6500, 11000),
        ('Carpintero', 'Construcción', 2, 70, 58, 55, 75, 8500, 16000),
        ('Plomero', 'Construcción', 2, 72, 56, 62, 68, 9000, 17000),
        ('Pintor de Construcción', 'Construcción', 2, 78, 48, 52, 70, 7500, 14000),
        ('Conductor de Autobús', 'Servicios', 2, 82, 52, 75, 38, 8500, 15000),
        ('Conductor de Camión', 'Servicios', 2, 85, 50, 65, 35, 8000, 15000),
        ('Taxista', 'Servicios', 2, 80, 48, 78, 40, 6500, 12000),
        ('Repartidor', 'Comercio', 2, 85, 45, 70, 35, 7000, 13000),
        ('Mensajero', 'Servicios', 1, 88, 42, 68, 32, 6500, 11000),
        ('Guardia de Seguridad', 'Servicios', 2, 80, 45, 68, 30, 7500, 13000),
        ('Vigilante', 'Servicios', 2, 85, 42, 62, 28, 7000, 12000),
        ('Conserje', 'Servicios', 1, 90, 38, 58, 25, 6000, 10000),
        ('Jardinero', 'Servicios', 1, 85, 42, 52, 58, 6500, 11000),
        ('Trabajador Agrícola', 'Agricultura', 1, 90, 38, 42, 38, 5500, 9500),
        ('Operador de Tractor', 'Agricultura', 2, 88, 45, 45, 42, 6500, 11000),
    ]
    
    logger.info(f"  Usando {len(occupation_templates)} templates base organizados en 6 estratos")
    
    # Generar ocupaciones basadas en templates
    occupations = []
    base_templates = len(occupation_templates)
    
    for i in range(n_occupations):
        # Seleccionar template base
        template_idx = i % base_templates
        name_base, sector, edu, routine, cognitive, social, creative, sal_min, sal_max = occupation_templates[template_idx]
        
        # Agregar variación para crear diversidad
        variation_suffix = f" - Var {(i // base_templates) + 1}" if i >= base_templates else ""
        
        # Agregar ruido realista a las métricas (mantener coherencia del perfil)
        routine_var = np.clip(routine + np.random.normal(0, 4), 10, 100)
        cognitive_var = np.clip(cognitive + np.random.normal(0, 3.5), 20, 100)
        social_var = np.clip(social + np.random.normal(0, 4), 10, 100)
        creative_var = np.clip(creative + np.random.normal(0, 4), 10, 100)
        
        # Salario con distribución log-normal más realista
        salary = np.clip(np.random.uniform(sal_min, sal_max) * np.random.lognormal(0, 0.12), 5000, 100000)
        
        # Número de trabajadores: más concentrado en ocupaciones comunes
        if routine_var > 80:  # Ocupaciones muy rutinarias = muchos trabajadores
            workers = np.random.randint(2000, 25000)
        elif routine_var > 65:  # Ocupaciones rutinarias = trabajadores moderados
            workers = np.random.randint(800, 15000)
        elif cognitive_var > 85:  # Ocupaciones muy especializadas = pocos trabajadores
            workers = np.random.randint(100, 2500)
        else:
            workers = np.random.randint(400, 8000)
        
        occupations.append({
            'occupation_id': f'OCC-{i:05d}',
            'occupation_name': name_base + variation_suffix,
            'soc_code': f'{np.random.randint(11,99)}-{np.random.randint(1000,9999)}.00',
            'sector': sector,
            'routine_index': routine_var,
            'cognitive_demand': cognitive_var,
            'social_interaction': social_var,
            'creativity': creative_var,
            'education_level': edu + np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]),
            'workers_jalisco': workers,
            'avg_salary_mxn': salary,
            'skill_critical_thinking': cognitive_var + np.random.normal(0, 4),
            'skill_programming': np.clip(
                np.random.uniform(15, 92) if any(tech in name_base for tech in ['Software', 'Datos', 'DevOps', 'Web', 'TI', 'Sistemas']) 
                else np.random.uniform(5, 35), 
                5, 95
            ),
            'skill_social_perceptiveness': social_var + np.random.normal(0, 4),
        })
    
    # Calcular riesgo de automatización basado en fórmula coherente
    for occ in occupations:
        # Fórmula Frey-Osborne adaptada
        risk = (
            occ['routine_index'] * 0.40 +
            (100 - occ['cognitive_demand']) * 0.25 +
            (100 - occ['social_interaction']) * 0.20 +
            (100 - occ['creativity']) * 0.15
        ) / 100
        
        occ['automation_risk'] = np.clip(risk + np.random.normal(0, 0.04), 0.05, 0.95)
    
    # Convertir a DataFrame
    df_pandas = pd.DataFrame(occupations)
    
    # Redondear valores
    df_pandas['routine_index'] = df_pandas['routine_index'].round(1)
    df_pandas['cognitive_demand'] = df_pandas['cognitive_demand'].round(1)
    df_pandas['social_interaction'] = df_pandas['social_interaction'].round(1)
    df_pandas['creativity'] = df_pandas['creativity'].round(1)
    df_pandas['avg_salary_mxn'] = df_pandas['avg_salary_mxn'].round(0)
    df_pandas['automation_risk'] = df_pandas['automation_risk'].round(3)
    df_pandas['skill_critical_thinking'] = np.clip(df_pandas['skill_critical_thinking'].round(1), 10, 100)
    df_pandas['skill_programming'] = np.clip(df_pandas['skill_programming'].round(1), 5, 95)
    df_pandas['skill_social_perceptiveness'] = np.clip(df_pandas['skill_social_perceptiveness'].round(1), 10, 100)
    df_pandas['education_level'] = df_pandas['education_level'].clip(1, 6).astype(int)
    
    # Convertir a Spark DataFrame
    df_spark = spark.createDataFrame(df_pandas)
    
    logger.info(f"✓ Datos simulados generados: {n_occupations:,} ocupaciones")
    logger.info(f"  Distribución por sector:")
    sector_counts = df_pandas['sector'].value_counts()
    for sector, count in sector_counts.items():
        logger.info(f"    {sector}: {count:,} ({count/len(df_pandas)*100:.1f}%)")
    logger.info(f"  Riesgo promedio: {df_pandas['automation_risk'].mean():.3f}")
    logger.info(f"  Riesgo alto (>0.7): {(df_pandas['automation_risk'] > 0.7).sum():,} ({(df_pandas['automation_risk'] > 0.7).sum()/len(df_pandas)*100:.1f}%)")
    logger.info(f"  Salario promedio: ${df_pandas['avg_salary_mxn'].mean():,.0f} MXN")
    
    return df_spark

    """
    Genera datos de muestra REALISTAS para pruebas y desarrollo.
    
    Genera ocupaciones con características coherentes basadas en perfiles reales.
    Por defecto genera 5,000 ocupaciones para análisis robusto.
    
    Parameters:
    -----------
    spark : SparkSession
        Sesión activa de Spark
    n_occupations : int
        Número de ocupaciones a generar (default: 5000)
        
    Returns:
    --------
    DataFrame
        Spark DataFrame con datos simulados realistas
    """
    import numpy as np
    import pandas as pd
    
    logger.info(f"Generando {n_occupations:,} ocupaciones simuladas con perfiles realistas...")
    
    np.random.seed(42)
    
    # Nombres de ocupaciones realistas (expandidos)
    occupation_templates = [
        # Directivos y gerentes (Alto cognitivo, Bajo rutina, Alto salario)
        ('Director General', 'Gobierno', 5, 15, 85, 70, 75, 15000, 35000),
        ('Gerente de Recursos Humanos', 'Servicios', 5, 25, 80, 85, 65, 18000, 32000),
        ('Gerente de Ventas', 'Comercio', 4, 30, 75, 90, 70, 20000, 40000),
        ('Director Financiero', 'Servicios', 6, 20, 90, 65, 60, 25000, 50000),
        ('Gerente de Operaciones', 'Manufactura', 5, 35, 80, 70, 65, 22000, 42000),
        
        # Profesionistas (Alto cognitivo, Medio-bajo rutina, Salario medio-alto)
        ('Ingeniero de Software', 'Servicios', 6, 25, 88, 45, 85, 25000, 55000),
        ('Médico General', 'Servicios', 6, 30, 92, 75, 70, 30000, 60000),
        ('Abogado', 'Servicios', 6, 28, 90, 80, 75, 28000, 58000),
        ('Contador Público', 'Servicios', 5, 45, 85, 60, 55, 18000, 35000),
        ('Arquitecto', 'Construcción', 6, 20, 88, 65, 92, 22000, 48000),
        ('Profesor Universitario', 'Gobierno', 6, 35, 90, 85, 80, 20000, 38000),
        ('Ingeniero Civil', 'Construcción', 5, 30, 86, 70, 75, 24000, 50000),
        ('Químico', 'Manufactura', 6, 35, 88, 50, 80, 20000, 42000),
        ('Biólogo', 'Servicios', 6, 32, 87, 60, 82, 18000, 38000),
        ('Psicólogo Clínico', 'Servicios', 6, 25, 85, 90, 78, 16000, 35000),
        
        # Técnicos y especialistas (Medio cognitivo, Medio rutina, Salario medio)
        ('Técnico en Sistemas', 'Servicios', 4, 40, 75, 55, 70, 15000, 28000),
        ('Enfermero', 'Servicios', 4, 50, 78, 85, 65, 12000, 25000),
        ('Electricista', 'Construcción', 3, 45, 72, 60, 68, 10000, 22000),
        ('Técnico Mecánico', 'Manufactura', 3, 48, 70, 55, 65, 11000, 24000),
        ('Diseñador Gráfico', 'Servicios', 4, 30, 75, 60, 90, 12000, 26000),
        ('Paramédico', 'Servicios', 4, 55, 80, 80, 60, 10000, 22000),
        ('Técnico de Laboratorio', 'Servicios', 4, 60, 78, 50, 68, 11000, 23000),
        ('Desarrollador Web', 'Servicios', 4, 35, 82, 50, 85, 18000, 35000),
        ('Analista de Datos', 'Servicios', 5, 38, 85, 55, 78, 20000, 40000),
        ('Técnico de Radiología', 'Servicios', 4, 58, 76, 65, 60, 13000, 26000),
        
        # Administrativos (Medio-bajo cognitivo, Alto rutina, Salario medio-bajo)
        ('Asistente Administrativo', 'Servicios', 3, 70, 65, 75, 45, 8000, 15000),
        ('Secretaria', 'Servicios', 3, 75, 62, 80, 40, 7500, 14000),
        ('Recepcionista', 'Servicios', 2, 80, 55, 85, 35, 6500, 12000),
        ('Cajero de Banco', 'Servicios', 3, 85, 68, 70, 30, 8500, 16000),
        ('Asistente Contable', 'Servicios', 3, 72, 70, 60, 48, 9000, 17000),
        ('Archivista', 'Servicios', 2, 88, 50, 45, 25, 6000, 11000),
        ('Operador de Call Center', 'Servicios', 2, 82, 58, 75, 32, 7000, 13000),
        
        # Ventas y comercio (Medio-bajo cognitivo, Medio rutina, Alto social)
        ('Vendedor de Tienda', 'Comercio', 2, 65, 55, 92, 55, 6500, 14000),
        ('Agente de Ventas', 'Comercio', 3, 58, 68, 90, 65, 8000, 20000),
        ('Cajero de Supermercado', 'Comercio', 2, 90, 48, 75, 25, 6000, 10000),
        ('Vendedor de Autos', 'Comercio', 3, 55, 70, 95, 68, 10000, 25000),
        ('Promotor de Ventas', 'Comercio', 2, 68, 60, 88, 60, 7000, 15000),
        ('Ejecutivo de Cuenta', 'Comercio', 4, 50, 75, 92, 72, 12000, 28000),
        
        # Servicios personales (Bajo cognitivo, Alto rutina, Alto social)
        ('Mesero', 'Servicios', 2, 75, 48, 90, 50, 5500, 10000),
        ('Chef', 'Servicios', 3, 60, 70, 75, 88, 10000, 22000),
        ('Peluquero', 'Servicios', 2, 70, 52, 85, 72, 6000, 13000),
        ('Conductor de Taxi', 'Servicios', 2, 78, 45, 75, 38, 5000, 11000),
        ('Guardia de Seguridad', 'Servicios', 2, 82, 42, 65, 28, 6500, 12000),
        ('Conserje', 'Servicios', 1, 88, 35, 55, 22, 5000, 9000),
        ('Jardinero', 'Servicios', 1, 85, 38, 48, 55, 5500, 10000),
        
        # Manufactura y producción (Bajo cognitivo, Muy alto rutina)
        ('Operador de Máquina', 'Manufactura', 2, 92, 42, 35, 20, 7000, 13000),
        ('Ensamblador', 'Manufactura', 2, 95, 38, 32, 18, 6500, 12000),
        ('Soldador', 'Manufactura', 3, 80, 55, 40, 60, 8500, 16000),
        ('Tornero', 'Manufactura', 3, 82, 52, 38, 58, 8000, 15000),
        ('Inspector de Calidad', 'Manufactura', 3, 75, 62, 42, 48, 9000, 17000),
        ('Empacador', 'Manufactura', 1, 98, 25, 28, 15, 5500, 9500),
        
        # Construcción (Medio-bajo cognitivo, Medio rutina, Físico)
        ('Albañil', 'Construcción', 2, 75, 48, 55, 52, 7500, 14000),
        ('Carpintero', 'Construcción', 2, 70, 58, 52, 72, 8000, 16000),
        ('Plomero', 'Construcción', 2, 72, 55, 60, 65, 8500, 17000),
        ('Pintor', 'Construcción', 2, 78, 45, 50, 68, 7000, 14000),
        ('Supervisor de Obra', 'Construcción', 3, 55, 70, 75, 65, 12000, 25000),
        
        # Agricultura (Bajo cognitivo, Alto rutina, Bajo salario)
        ('Agricultor', 'Agricultura', 1, 85, 40, 45, 48, 5000, 10000),
        ('Trabajador Agrícola', 'Agricultura', 1, 90, 32, 38, 35, 4500, 8500),
        ('Operador de Tractor', 'Agricultura', 2, 88, 45, 42, 40, 6000, 11000),
        
        # Transporte (Bajo-medio cognitivo, Alto rutina)
        ('Conductor de Autobús', 'Servicios', 2, 85, 50, 72, 35, 7500, 14000),
        ('Repartidor', 'Comercio', 2, 88, 42, 68, 32, 6500, 12000),
        ('Operador de Montacargas', 'Manufactura', 2, 90, 40, 35, 30, 7000, 13000),
    ]
    
    # Generar ocupaciones basadas en templates
    occupations = []
    base_templates = len(occupation_templates)
    
    for i in range(n_occupations):
        # Seleccionar template base
        template_idx = i % base_templates
        name_base, sector, edu, routine, cognitive, social, creative, sal_min, sal_max = occupation_templates[template_idx]
        
        # Agregar variación para crear diversidad
        variation_suffix = f" - Especialidad {(i // base_templates) + 1}" if i >= base_templates else ""
        
        # Agregar ruido realista a las métricas
        routine_var = np.clip(routine + np.random.normal(0, 5), 10, 100)
        cognitive_var = np.clip(cognitive + np.random.normal(0, 4), 20, 100)
        social_var = np.clip(social + np.random.normal(0, 5), 10, 100)
        creative_var = np.clip(creative + np.random.normal(0, 5), 10, 100)
        
        # Salario con distribución log-normal más realista
        salary = np.clip(np.random.uniform(sal_min, sal_max) * np.random.lognormal(0, 0.15), 5000, 100000)
        
        # Número de trabajadores: más concentrado en ocupaciones comunes
        if routine_var > 75:  # Ocupaciones rutinarias = más trabajadores
            workers = np.random.randint(1000, 20000)
        elif cognitive_var > 80:  # Ocupaciones especializadas = menos trabajadores
            workers = np.random.randint(100, 3000)
        else:
            workers = np.random.randint(500, 10000)
        
        occupations.append({
            'occupation_id': f'OCC-{i:05d}',
            'occupation_name': name_base + variation_suffix,
            'soc_code': f'{np.random.randint(11,99)}-{np.random.randint(1000,9999)}.00',
            'sector': sector,
            'routine_index': routine_var,
            'cognitive_demand': cognitive_var,
            'social_interaction': social_var,
            'creativity': creative_var,
            'education_level': edu + np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2]),
            'workers_jalisco': workers,
            'avg_salary_mxn': salary,
            'skill_critical_thinking': cognitive_var + np.random.normal(0, 5),
            'skill_programming': np.clip(np.random.uniform(10, 90) if 'Software' in name_base or 'Datos' in name_base else np.random.uniform(5, 40), 5, 95),
            'skill_social_perceptiveness': social_var + np.random.normal(0, 5),
        })
    
    # Calcular riesgo de automatización basado en fórmula coherente
    for occ in occupations:
        # Fórmula Frey-Osborne adaptada
        risk = (
            occ['routine_index'] * 0.40 +
            (100 - occ['cognitive_demand']) * 0.25 +
            (100 - occ['social_interaction']) * 0.20 +
            (100 - occ['creativity']) * 0.15
        ) / 100
        
        occ['automation_risk'] = np.clip(risk + np.random.normal(0, 0.05), 0.05, 0.95)
    
    # Convertir a DataFrame
    df_pandas = pd.DataFrame(occupations)
    
    # Redondear valores
    df_pandas['routine_index'] = df_pandas['routine_index'].round(1)
    df_pandas['cognitive_demand'] = df_pandas['cognitive_demand'].round(1)
    df_pandas['social_interaction'] = df_pandas['social_interaction'].round(1)
    df_pandas['creativity'] = df_pandas['creativity'].round(1)
    df_pandas['avg_salary_mxn'] = df_pandas['avg_salary_mxn'].round(0)
    df_pandas['automation_risk'] = df_pandas['automation_risk'].round(3)
    df_pandas['skill_critical_thinking'] = np.clip(df_pandas['skill_critical_thinking'].round(1), 10, 100)
    df_pandas['skill_programming'] = np.clip(df_pandas['skill_programming'].round(1), 5, 95)
    df_pandas['skill_social_perceptiveness'] = np.clip(df_pandas['skill_social_perceptiveness'].round(1), 10, 100)
    df_pandas['education_level'] = df_pandas['education_level'].clip(1, 6).astype(int)
    
    # Convertir a Spark DataFrame
    df_spark = spark.createDataFrame(df_pandas)
    
    logger.info(f"✓ Datos simulados generados: {n_occupations:,} ocupaciones")
    logger.info(f"  Distribución por sector:")
    sector_counts = df_pandas['sector'].value_counts()
    for sector, count in sector_counts.items():
        logger.info(f"    {sector}: {count:,} ({count/len(df_pandas)*100:.1f}%)")
    logger.info(f"  Riesgo promedio: {df_pandas['automation_risk'].mean():.3f}")
    logger.info(f"  Riesgo alto (>0.7): {(df_pandas['automation_risk'] > 0.7).sum():,} ({(df_pandas['automation_risk'] > 0.7).sum()/len(df_pandas)*100:.1f}%)")
    
    return df_spark


def save_dataset(df_ps, output_path, format='parquet'):
    """
    Guarda el dataset procesado.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame a guardar
    output_path : str
        Ruta de salida
    format : str
        Formato de archivo ('parquet', 'csv')
    """
    try:
        logger.info(f"Guardando dataset en: {output_path}")
        
        df_spark = df_ps.to_spark()
        
        if format == 'parquet':
            df_spark.write.mode('overwrite').parquet(output_path)
        elif format == 'csv':
            df_spark.write.mode('overwrite') \
                .option('header', 'true') \
                .csv(output_path)
        else:
            raise ValueError(f"Formato no soportado: {format}")
        
        logger.info(f"✓ Dataset guardado exitosamente")
        
    except Exception as e:
        logger.error(f"Error guardando dataset: {str(e)}")
        raise


# Función auxiliar para pruebas
def test_data_loader():
    """Prueba las funciones del módulo"""
    print("\n" + "="*80)
    print("PROBANDO DATA LOADER MODULE")
    print("="*80 + "\n")
    
    # Crear Spark Session
    spark = create_spark_session("Test_Data_Loader")
    
    # Generar datos de muestra
    df_sample = load_sample_data(spark, n_occupations=50)
    
    # Convertir a pandas API
    df_ps = convert_to_pandas_api(df_sample)
    
    # Mostrar info
    get_data_info(df_ps, "Datos de Prueba")
    
    # Mostrar primeras filas
    print("Primeras 5 filas:")
    print(df_ps.head())
    
    # Cerrar Spark
    spark.stop()
    
    print("\n✓ Prueba completada exitosamente\n")


def download_onet_data(output_dir='data/raw/onet'):
    """
    Descarga automáticamente la base de datos de O*NET.
    
    Parameters:
    -----------
    output_dir : str
        Directorio donde guardar los archivos
        
    Returns:
    --------
    bool
        True si la descarga fue exitosa
    """
    import urllib.request
    import zipfile
    import shutil
    
    logger.info("Descargando O*NET Database...")
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # URL de descarga (O*NET 28.3 - Excel/Text files)
    # Nota: Esta URL puede cambiar. Verificar en https://www.onetcenter.org/database.html
    onet_url = "https://www.onetcenter.org/dl_files/database/db_28_3_text.zip"
    zip_path = os.path.join(output_dir, "onet_database.zip")
    
    try:
        # Descargar archivo
        logger.info(f"  Descargando desde: {onet_url}")
        urllib.request.urlretrieve(onet_url, zip_path)
        logger.info(f"  ✓ Descarga completada: {zip_path}")
        
        # Extraer ZIP
        logger.info("  Extrayendo archivos...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        logger.info(f"  ✓ Archivos extraídos en: {output_dir}")
        
        # Eliminar ZIP
        os.remove(zip_path)
        logger.info("  ✓ Archivo ZIP eliminado")
        
        # Verificar archivos esenciales
        essential_files = [
            'Occupation Data.txt',
            'Skills.txt',
            'Abilities.txt',
            'Work Activities.txt'
        ]
        
        found_files = []
        for file in essential_files:
            file_path = os.path.join(output_dir, file)
            if os.path.exists(file_path):
                found_files.append(file)
        
        logger.info(f"  ✓ Archivos encontrados: {len(found_files)}/{len(essential_files)}")
        
        return len(found_files) > 0
        
    except Exception as e:
        logger.error(f"  ✗ Error descargando O*NET: {str(e)}")
        logger.info("  Por favor descarga manualmente desde: https://www.onetcenter.org/database.html")
        return False


def create_sample_enoe_data(output_path='data/raw/enoe_jalisco.csv', n_records=10000):
    """
    Crea un dataset de ejemplo simulado de ENOE para Jalisco.
    
    IMPORTANTE: Este es solo un ejemplo para pruebas.
    Para análisis real, descargar desde: https://www.inegi.org.mx/programas/enoe/
    
    Parameters:
    -----------
    output_path : str
        Ruta donde guardar el CSV
    n_records : int
        Número de registros a generar
        
    Returns:
    --------
    str
        Ruta del archivo creado
    """
    import pandas as pd
    import numpy as np
    
    logger.info(f"Generando dataset de ejemplo ENOE (n={n_records:,})...")
    logger.warning("⚠️  IMPORTANTE: Estos son datos SIMULADOS para pruebas")
    logger.warning("    Para análisis real, descargar ENOE desde INEGI")
    
    np.random.seed(42)
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Códigos SINCO comunes (simplificados)
    sinco_codes = [
        '1111', '1112', '1113',  # Directivos
        '2111', '2121', '2131',  # Profesionistas
        '3111', '3121', '3131',  # Técnicos
        '4111', '4121', '4131',  # Apoyo administrativo
        '5111', '5121', '5131',  # Comercio
        '6111', '6121', '6131',  # Servicios
        '7111', '7121', '7131',  # Agricultura
        '8111', '8121', '8131',  # Manufactura
        '9111', '9121', '9131'   # Operadores
    ]
    
    # Generar datos
    df_enoe = pd.DataFrame({
        'ent': [14] * n_records,  # Jalisco
        'clase2': np.random.choice(sinco_codes, n_records),
        'pos_ocu': np.random.choice([1, 2, 3, 4], n_records, p=[0.5, 0.3, 0.15, 0.05]),
        'nivel': np.random.choice([1, 2, 3, 4, 5, 6], n_records, p=[0.1, 0.2, 0.25, 0.25, 0.15, 0.05]),
        'ingocup': np.random.lognormal(mean=8.5, sigma=0.8, size=n_records).astype(int),
        'sex': np.random.choice([1, 2], n_records),
        'eda': np.random.randint(18, 65, n_records),
        'anios_esc': np.random.randint(0, 20, n_records),
        'rama': np.random.choice(range(1, 21), n_records)
    })
    
    # Guardar
    df_enoe.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"  ✓ Dataset de ejemplo creado: {output_path}")
    logger.info(f"  Registros: {len(df_enoe):,}")
    logger.info(f"  Columnas: {list(df_enoe.columns)}")
    
    return output_path


def auto_download_data(force_download=False):
    """
    Descarga automáticamente todos los datos necesarios.
    
    Parameters:
    -----------
    force_download : bool
        Si True, descarga aunque los archivos ya existan
        
    Returns:
    --------
    dict
        Rutas de los archivos descargados
    """
    logger.info("\n" + "="*80)
    logger.info("DESCARGA AUTOMÁTICA DE DATOS")
    logger.info("="*80)
    
    paths = {
        'onet_dir': 'data/raw/onet',
        'enoe_file': 'data/raw/enoe_jalisco.csv'
    }
    
    # 1. Verificar O*NET
    onet_file = os.path.join(paths['onet_dir'], 'Occupation Data.txt')
    if os.path.exists(onet_file) and not force_download:
        logger.info("\n✓ O*NET ya existe")
        logger.info(f"  Ubicación: {onet_file}")
    else:
        logger.info("\n📥 Descargando O*NET...")
        success = download_onet_data(paths['onet_dir'])
        if success:
            logger.info("  ✓ O*NET descargado exitosamente")
        else:
            logger.warning("  ⚠️  Descarga de O*NET falló")
            logger.info("  Se usarán datos simulados alternativos")
    
    # 2. Verificar ENOE
    if os.path.exists(paths['enoe_file']) and not force_download:
        logger.info("\n✓ ENOE ya existe")
        logger.info(f"  Ubicación: {paths['enoe_file']}")
    else:
        logger.info("\n📊 Creando dataset de ejemplo ENOE...")
        logger.warning("  ⚠️  NOTA: Estos son datos SIMULADOS")
        logger.warning("  Para datos reales, descargar desde:")
        logger.warning("  https://www.inegi.org.mx/programas/enoe/")
        
        create_sample_enoe_data(paths['enoe_file'], n_records=10000)
    
    logger.info("\n" + "="*80)
    logger.info("✓ DESCARGA COMPLETADA")
    logger.info("="*80)
    logger.info(f"\nArchivos disponibles:")
    logger.info(f"  O*NET: {paths['onet_dir']}")
    logger.info(f"  ENOE: {paths['enoe_file']}")
    
    return paths


if __name__ == "__main__":
    # Ejecutar prueba si se corre directamente
    test_data_loader()