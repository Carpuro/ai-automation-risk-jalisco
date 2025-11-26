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
    for col, dtype in list(df_ps.dtypes.items())[:10]:
        logger.info(f"  {col}: {dtype}")
    
    logger.info(f"{'='*80}\n")


def load_sample_data(spark, n_occupations=100):
    """
    Genera datos de muestra simulados para pruebas.
    
    Parameters:
    -----------
    spark : SparkSession
        Sesión activa de Spark
    n_occupations : int
        Número de ocupaciones a generar
        
    Returns:
    --------
    DataFrame
        Spark DataFrame con datos simulados
    """
    import numpy as np
    import pandas as pd
    
    logger.info(f"Generando {n_occupations} ocupaciones simuladas...")
    
    np.random.seed(42)
    
    # Generar datos sintéticos
    data = {
        'occupation_id': [f'OCC-{i:04d}' for i in range(n_occupations)],
        'occupation_name': [f'Ocupación {i}' for i in range(n_occupations)],
        'soc_code': [f'{np.random.randint(11,99)}-{np.random.randint(1000,9999)}.00' 
                     for _ in range(n_occupations)],
        'sector': np.random.choice(['Manufactura', 'Servicios', 'Comercio', 
                                   'Gobierno', 'Construcción', 'Agricultura'], 
                                  n_occupations),
        'routine_index': np.random.uniform(20, 95, n_occupations),
        'cognitive_demand': np.random.uniform(30, 90, n_occupations),
        'social_interaction': np.random.uniform(10, 85, n_occupations),
        'creativity': np.random.uniform(15, 95, n_occupations),
        'education_level': np.random.choice([3, 4, 5, 6], n_occupations),
        'workers_jalisco': np.random.randint(100, 50000, n_occupations),
        'avg_salary_mxn': np.random.uniform(5000, 50000, n_occupations),
        'skill_critical_thinking': np.random.uniform(30, 95, n_occupations),
        'skill_programming': np.random.uniform(10, 90, n_occupations),
        'skill_social_perceptiveness': np.random.uniform(20, 95, n_occupations),
        'automation_risk': np.random.uniform(0.1, 0.9, n_occupations)
    }
    
    # Convertir a pandas luego a Spark
    df_pandas = pd.DataFrame(data)
    df_spark = spark.createDataFrame(df_pandas)
    
    logger.info(f"✓ Datos simulados generados: {n_occupations} ocupaciones")
    
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


if __name__ == "__main__":
    # Ejecutar prueba si se corre directamente
    test_data_loader()