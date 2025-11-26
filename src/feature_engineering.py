"""
Feature Engineering Module
===========================
Módulo para creación y transformación de features para análisis de automatización.

Autor: Carlos Pulido Rosas
Proyecto: Modelo Predictivo de Sustitución Laboral por IA - Jalisco
"""

import pyspark.pandas as ps
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_routine_index(df_ps):
    """
    Crea índice de rutinización de tareas.
    
    Combina múltiples indicadores de rutina en un solo índice compuesto.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con características de ocupaciones
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con columna 'routine_index' agregada
    """
    logger.info("Creando índice de rutinización...")
    
    df_result = df_ps.copy()
    
    # Componentes del índice (con pesos)
    components = []
    weights = []
    
    # Tareas repetitivas (40%)
    if 'task_repetitive' in df_ps.columns:
        components.append(df_ps['task_repetitive'])
        weights.append(0.40)
    
    # Tareas predecibles (35%)
    if 'task_predictable' in df_ps.columns:
        components.append(df_ps['task_predictable'])
        weights.append(0.35)
    
    # Tareas estructuradas (25%)
    if 'task_structured' in df_ps.columns:
        components.append(df_ps['task_structured'])
        weights.append(0.25)
    
    # Calcular índice ponderado
    if components:
        # Normalizar pesos
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Calcular índice
        routine_index = sum(c * w for c, w in zip(components, normalized_weights))
        df_result['routine_index'] = routine_index
        
        logger.info(f"✓ Índice de rutinización creado (basado en {len(components)} componentes)")
    else:
        # Usar valor por defecto si no hay componentes
        logger.warning("No se encontraron componentes de rutinización, usando valor por defecto")
        df_result['routine_index'] = 50.0
    
    return df_result


def create_cognitive_demand_index(df_ps):
    """
    Crea índice de demanda cognitiva.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con características
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con columna 'cognitive_demand' agregada
    """
    logger.info("Creando índice de demanda cognitiva...")
    
    df_result = df_ps.copy()
    
    # Habilidades cognitivas relevantes
    cognitive_skills = [
        'skill_critical_thinking',
        'skill_complex_problem_solving',
        'skill_analytical',
        'ability_deductive_reasoning',
        'ability_inductive_reasoning',
        'ability_mathematical_reasoning'
    ]
    
    # Buscar columnas disponibles
    available_skills = [col for col in cognitive_skills if col in df_ps.columns]
    
    if available_skills:
        # Promedio de habilidades cognitivas
        df_result['cognitive_demand'] = df_ps[available_skills].mean(axis=1)
        logger.info(f"✓ Índice de demanda cognitiva creado (basado en {len(available_skills)} habilidades)")
    else:
        logger.warning("No se encontraron habilidades cognitivas, usando valor por defecto")
        df_result['cognitive_demand'] = 50.0
    
    return df_result


def create_social_interaction_index(df_ps):
    """
    Crea índice de interacción social.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con características
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con columna 'social_interaction' agregada
    """
    logger.info("Creando índice de interacción social...")
    
    df_result = df_ps.copy()
    
    # Actividades sociales relevantes
    social_activities = [
        'activity_communicating_supervisors',
        'activity_communicating_coworkers',
        'activity_interacting_public',
        'skill_social_perceptiveness',
        'skill_persuasion',
        'skill_negotiation',
        'skill_service_orientation'
    ]
    
    # Buscar columnas disponibles
    available_activities = [col for col in social_activities if col in df_ps.columns]
    
    if available_activities:
        df_result['social_interaction'] = df_ps[available_activities].mean(axis=1)
        logger.info(f"✓ Índice de interacción social creado (basado en {len(available_activities)} actividades)")
    else:
        logger.warning("No se encontraron actividades sociales, usando valor por defecto")
        df_result['social_interaction'] = 50.0
    
    return df_result


def create_creativity_index(df_ps):
    """
    Crea índice de creatividad.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con características
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con columna 'creativity' agregada
    """
    logger.info("Creando índice de creatividad...")
    
    df_result = df_ps.copy()
    
    # Habilidades creativas relevantes
    creative_skills = [
        'skill_originality',
        'ability_fluency_ideas',
        'activity_thinking_creatively',
        'work_style_innovation',
        'skill_design'
    ]
    
    # Buscar columnas disponibles
    available_skills = [col for col in creative_skills if col in df_ps.columns]
    
    if available_skills:
        df_result['creativity'] = df_ps[available_skills].mean(axis=1)
        logger.info(f"✓ Índice de creatividad creado (basado en {len(available_skills)} habilidades)")
    else:
        logger.warning("No se encontraron habilidades creativas, usando valor por defecto")
        df_result['creativity'] = 50.0
    
    return df_result


def create_complexity_index(df_ps):
    """
    Crea índice de complejidad general de la ocupación.
    
    Combina demanda cognitiva, creatividad y rutina inversa.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con índices base
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con columna 'complexity_index' agregada
    """
    logger.info("Creando índice de complejidad...")
    
    df_result = df_ps.copy()
    
    # Verificar que existan los índices necesarios
    required_cols = ['cognitive_demand', 'creativity', 'routine_index']
    
    if all(col in df_ps.columns for col in required_cols):
        df_result['complexity_index'] = (
            df_result['cognitive_demand'] * 0.4 +
            df_result['creativity'] * 0.3 +
            (100 - df_result['routine_index']) * 0.3
        )
        logger.info("✓ Índice de complejidad creado")
    else:
        logger.warning(f"Faltan columnas requeridas: {required_cols}")
        df_result['complexity_index'] = 50.0
    
    return df_result


def create_education_categories(df_ps):
    """
    Categoriza niveles educativos en grupos.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con columna 'education_level'
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con columna 'education_category' agregada
    """
    logger.info("Creando categorías educativas...")
    
    df_result = df_ps.copy()
    
    if 'education_level' in df_ps.columns:
        def categorize_education(level):
            if level <= 2:
                return 'Básica'
            elif level <= 4:
                return 'Media'
            elif level == 5:
                return 'Superior'
            else:
                return 'Posgrado'
        
        df_result['education_category'] = df_result['education_level'].apply(categorize_education)
        logger.info("✓ Categorías educativas creadas")
    else:
        logger.warning("Columna 'education_level' no encontrada")
    
    return df_result


def create_salary_categories(df_ps):
    """
    Categoriza salarios en quintiles.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con columna 'avg_salary_mxn'
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con columna 'salary_category' agregada
    """
    logger.info("Creando categorías salariales...")
    
    df_result = df_ps.copy()
    
    if 'avg_salary_mxn' in df_ps.columns:
        # Calcular quintiles
        df_result['salary_category'] = ps.qcut(
            df_result['avg_salary_mxn'],
            q=5,
            labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto']
        )
        logger.info("✓ Categorías salariales creadas (5 quintiles)")
    else:
        logger.warning("Columna 'avg_salary_mxn' no encontrada")
    
    return df_result


def create_occupation_size_categories(df_ps):
    """
    Categoriza ocupaciones por tamaño (número de trabajadores).
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con columna 'workers_jalisco'
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con columna 'occupation_size' agregada
    """
    logger.info("Creando categorías de tamaño de ocupación...")
    
    df_result = df_ps.copy()
    
    if 'workers_jalisco' in df_ps.columns:
        def categorize_size(count):
            if count < 1000:
                return 'Pequeña'
            elif count < 5000:
                return 'Mediana'
            elif count < 20000:
                return 'Grande'
            else:
                return 'Muy Grande'
        
        df_result['occupation_size'] = df_result['workers_jalisco'].apply(categorize_size)
        logger.info("✓ Categorías de tamaño creadas")
    else:
        logger.warning("Columna 'workers_jalisco' no encontrada")
    
    return df_result


def create_derived_ratios(df_ps):
    """
    Crea ratios derivados útiles para análisis.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con columnas base
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con ratios agregados
    """
    logger.info("Creando ratios derivados...")
    
    df_result = df_ps.copy()
    
    # Ratio salario/educación
    if 'avg_salary_mxn' in df_ps.columns and 'education_level' in df_ps.columns:
        df_result['salary_education_ratio'] = df_result['avg_salary_mxn'] / (df_result['education_level'] + 1)
        logger.info("  ✓ salary_education_ratio creado")
    
    # Ratio complejidad/rutina
    if 'complexity_index' in df_ps.columns and 'routine_index' in df_ps.columns:
        df_result['complexity_routine_ratio'] = df_result['complexity_index'] / (df_result['routine_index'] + 1)
        logger.info("  ✓ complexity_routine_ratio creado")
    
    # Ratio social/cognitivo
    if 'social_interaction' in df_ps.columns and 'cognitive_demand' in df_ps.columns:
        df_result['social_cognitive_ratio'] = df_result['social_interaction'] / (df_result['cognitive_demand'] + 1)
        logger.info("  ✓ social_cognitive_ratio creado")
    
    logger.info("✓ Ratios derivados creados")
    
    return df_result


def create_automation_susceptibility_score(df_ps):
    """
    Crea un score de susceptibilidad a automatización basado en múltiples factores.
    
    Este es diferente al automation_risk porque considera más dimensiones.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con índices base
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con columna 'automation_susceptibility' agregada
    """
    logger.info("Creando score de susceptibilidad a automatización...")
    
    df_result = df_ps.copy()
    
    # Factores que aumentan susceptibilidad
    positive_factors = []
    positive_weights = []
    
    if 'routine_index' in df_ps.columns:
        positive_factors.append(df_ps['routine_index'])
        positive_weights.append(0.40)
    
    # Factores que disminuyen susceptibilidad
    negative_factors = []
    negative_weights = []
    
    if 'cognitive_demand' in df_ps.columns:
        negative_factors.append(df_ps['cognitive_demand'])
        negative_weights.append(0.25)
    
    if 'creativity' in df_ps.columns:
        negative_factors.append(df_ps['creativity'])
        negative_weights.append(0.20)
    
    if 'social_interaction' in df_ps.columns:
        negative_factors.append(df_ps['social_interaction'])
        negative_weights.append(0.15)
    
    # Calcular score
    if positive_factors and negative_factors:
        positive_score = sum(f * w for f, w in zip(positive_factors, positive_weights))
        negative_score = sum(f * w for f, w in zip(negative_factors, negative_weights))
        
        # Normalizar a escala 0-100
        susceptibility = (positive_score - negative_score + 100) / 2
        df_result['automation_susceptibility'] = susceptibility.clip(0, 100)
        
        logger.info("✓ Score de susceptibilidad creado")
    else:
        logger.warning("Factores insuficientes para calcular susceptibilidad")
        df_result['automation_susceptibility'] = 50.0
    
    return df_result


def create_temporal_features(df_ps):
    """
    Crea features temporales para proyecciones.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con datos base
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con features temporales
    """
    logger.info("Creando features temporales...")
    
    df_result = df_ps.copy()
    
    # Año base
    df_result['base_year'] = 2025
    
    # Proyecciones (asumiendo crecimiento exponencial del riesgo)
    if 'automation_risk' in df_ps.columns:
        # Tasa de crecimiento anual estimada
        annual_growth_rate = 0.05  # 5% anual
        
        for year in [2026, 2027, 2028, 2029, 2030]:
            years_ahead = year - 2025
            df_result[f'risk_projection_{year}'] = df_result['automation_risk'] * (1 + annual_growth_rate) ** years_ahead
            df_result[f'risk_projection_{year}'] = df_result[f'risk_projection_{year}'].clip(0, 1)
        
        logger.info("✓ Proyecciones temporales creadas (2025-2030)")
    
    return df_result


def create_sector_aggregations(df_ps):
    """
    Crea aggregaciones a nivel de sector.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con columna 'sector'
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con features de sector agregadas
    """
    logger.info("Creando aggregaciones por sector...")
    
    df_result = df_ps.copy()
    
    if 'sector' not in df_ps.columns:
        logger.warning("Columna 'sector' no encontrada")
        return df_result
    
    # Calcular estadísticas por sector
    sector_stats = df_ps.groupby('sector').agg({
        'automation_risk': ['mean', 'std'],
        'workers_jalisco': 'sum',
        'avg_salary_mxn': 'mean'
    })
    
    # Aplanar columnas multiindex
    sector_stats.columns = ['_'.join(col).strip() for col in sector_stats.columns.values]
    sector_stats = sector_stats.reset_index()
    
    # Renombrar columnas
    sector_stats.columns = [
        'sector', 
        'sector_avg_risk', 
        'sector_std_risk',
        'sector_total_workers',
        'sector_avg_salary'
    ]
    
    # Merge con dataframe original
    df_result = df_result.merge(sector_stats, on='sector', how='left')
    
    # Feature: Desviación del riesgo respecto al promedio del sector
    if 'automation_risk' in df_result.columns:
        df_result['risk_deviation_from_sector'] = df_result['automation_risk'] - df_result['sector_avg_risk']
    
    logger.info("✓ Aggregaciones por sector creadas")
    
    return df_result


def feature_engineering_pipeline(df_ps, config=None):
    """
    Pipeline completo de feature engineering.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame original
    config : dict, optional
        Configuración de features a crear
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con todos los features creados
    """
    logger.info("\n" + "="*80)
    logger.info("INICIANDO PIPELINE DE FEATURE ENGINEERING")
    logger.info("="*80 + "\n")
    
    if config is None:
        config = {
            'create_indices': True,
            'create_categories': True,
            'create_ratios': True,
            'create_temporal': True,
            'create_sector_agg': True
        }
    
    df_result = df_ps.copy()
    
    # 1. Crear índices base
    if config.get('create_indices', True):
        logger.info("Paso 1/5: Creando índices base...")
        df_result = create_routine_index(df_result)
        df_result = create_cognitive_demand_index(df_result)
        df_result = create_social_interaction_index(df_result)
        df_result = create_creativity_index(df_result)
        df_result = create_complexity_index(df_result)
        df_result = create_automation_susceptibility_score(df_result)
    
    # 2. Crear categorías
    if config.get('create_categories', True):
        logger.info("\nPaso 2/5: Creando categorías...")
        df_result = create_education_categories(df_result)
        df_result = create_salary_categories(df_result)
        df_result = create_occupation_size_categories(df_result)
    
    # 3. Crear ratios derivados
    if config.get('create_ratios', True):
        logger.info("\nPaso 3/5: Creando ratios derivados...")
        df_result = create_derived_ratios(df_result)
    
    # 4. Crear features temporales
    if config.get('create_temporal', True):
        logger.info("\nPaso 4/5: Creando features temporales...")
        df_result = create_temporal_features(df_result)
    
    # 5. Crear aggregaciones por sector
    if config.get('create_sector_agg', True):
        logger.info("\nPaso 5/5: Creando aggregaciones por sector...")
        df_result = create_sector_aggregations(df_result)
    
    logger.info("\n" + "="*80)
    logger.info("✓ PIPELINE DE FEATURE ENGINEERING COMPLETADO")
    logger.info("="*80)
    logger.info(f"\nFeatures totales: {len(df_result.columns)}")
    logger.info(f"Features nuevos: {len(df_result.columns) - len(df_ps.columns)}")
    
    return df_result


if __name__ == "__main__":
    print("Feature Engineering Module")
    print("Este módulo debe ser importado, no ejecutado directamente.")