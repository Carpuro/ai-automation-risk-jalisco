"""
Data Preprocessing Module
==========================
Módulo para limpieza y preprocesamiento de datos de automatización.

Autor: Carlos Pulido Rosas
Proyecto: Modelo Predictivo de Sustitución Laboral por IA - Jalisco
"""

import pyspark.pandas as ps
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_missing_values(df_ps):
    """
    Analiza valores faltantes en el dataset.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame a analizar
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        Resumen de valores faltantes
    """
    logger.info("Analizando valores faltantes...")
    
    missing_count = df_ps.isnull().sum()
    missing_pct = (missing_count / len(df_ps)) * 100
    
    missing_df = ps.DataFrame({
        'Column': missing_count.index,
        'Missing_Count': missing_count.values,
        'Missing_Percent': missing_pct.values
    }).sort_values('Missing_Count', ascending=False)
    
    # Filtrar solo columnas con valores faltantes
    missing_df = missing_df[missing_df['Missing_Count'] > 0]
    
    if len(missing_df) > 0:
        logger.info(f"  Columnas con valores faltantes: {len(missing_df)}")
        logger.info(f"  Total de valores faltantes: {missing_count.sum():,}")
    else:
        logger.info("  ✓ No hay valores faltantes")
    
    return missing_df


def handle_missing_values(df_ps, strategy='auto'):
    """
    Maneja valores faltantes según estrategia definida.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame a limpiar
    strategy : str
        Estrategia: 'auto', 'drop', 'fill_mean', 'fill_median', 'fill_mode'
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame limpio
    """
    logger.info(f"Manejando valores faltantes (estrategia: {strategy})...")
    
    df_clean = df_ps.copy()
    initial_rows = len(df_clean)
    
    if strategy == 'auto':
        # Estrategia automática inteligente
        for col in df_clean.columns:
            missing_pct = (df_clean[col].isnull().sum() / len(df_clean)) * 100
            
            if missing_pct > 50:
                # Eliminar columna si >50% faltante
                logger.info(f"  Eliminando columna '{col}' ({missing_pct:.1f}% faltante)")
                df_clean = df_clean.drop(columns=[col])
            
            elif missing_pct > 0:
                # Rellenar según tipo de dato
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown')
    
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
        
    elif strategy == 'fill_mean':
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
        
    elif strategy == 'fill_median':
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
        
    elif strategy == 'fill_mode':
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 0
                df_clean[col] = df_clean[col].fillna(mode_val)
    
    final_rows = len(df_clean)
    logger.info(f"  ✓ Filas: {initial_rows:,} → {final_rows:,} (eliminadas: {initial_rows - final_rows:,})")
    
    return df_clean


def remove_duplicates(df_ps, subset=None):
    """
    Elimina filas duplicadas.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame a limpiar
    subset : list, optional
        Columnas a considerar para duplicados
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame sin duplicados
    """
    logger.info("Eliminando duplicados...")
    
    initial_rows = len(df_ps)
    
    if subset:
        df_clean = df_ps.drop_duplicates(subset=subset)
    else:
        df_clean = df_ps.drop_duplicates()
    
    final_rows = len(df_clean)
    duplicates_removed = initial_rows - final_rows
    
    logger.info(f"  ✓ Duplicados eliminados: {duplicates_removed:,}")
    
    return df_clean


def filter_outliers(df_ps, columns, method='iqr', threshold=1.5):
    """
    Filtra outliers en columnas numéricas.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame a limpiar
    columns : list
        Columnas numéricas a analizar
    method : str
        Método: 'iqr' o 'zscore'
    threshold : float
        Umbral para IQR o z-score
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame sin outliers
    """
    logger.info(f"Filtrando outliers (método: {method})...")
    
    df_clean = df_ps.copy()
    initial_rows = len(df_clean)
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df_clean = df_clean[
                (df_clean[col] >= lower_bound) & 
                (df_clean[col] <= upper_bound)
            ]
        
        elif method == 'zscore':
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            
            df_clean = df_clean[
                abs((df_clean[col] - mean) / std) <= threshold
            ]
    
    final_rows = len(df_clean)
    outliers_removed = initial_rows - final_rows
    
    logger.info(f"  ✓ Outliers eliminados: {outliers_removed:,}")
    
    return df_clean


def normalize_columns(df_ps, columns, method='minmax'):
    """
    Normaliza columnas numéricas.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame a normalizar
    columns : list
        Columnas a normalizar
    method : str
        Método: 'minmax' o 'standard'
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame con columnas normalizadas
    """
    logger.info(f"Normalizando columnas (método: {method})...")
    
    df_norm = df_ps.copy()
    
    for col in columns:
        if col not in df_norm.columns:
            continue
        
        if method == 'minmax':
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            df_norm[f'{col}_norm'] = (df_norm[col] - min_val) / (max_val - min_val)
        
        elif method == 'standard':
            mean = df_norm[col].mean()
            std = df_norm[col].std()
            df_norm[f'{col}_norm'] = (df_norm[col] - mean) / std
    
    logger.info(f"  ✓ {len(columns)} columnas normalizadas")
    
    return df_norm


def validate_data_quality(df_ps):
    """
    Valida calidad general del dataset.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame a validar
        
    Returns:
    --------
    dict
        Métricas de calidad
    """
    logger.info("Validando calidad de datos...")
    
    metrics = {
        'total_rows': len(df_ps),
        'total_columns': len(df_ps.columns),
        'missing_values': int(df_ps.isnull().sum().sum()),
        'missing_percent': float((df_ps.isnull().sum().sum() / (len(df_ps) * len(df_ps.columns))) * 100),
        'duplicates': int(df_ps.duplicated().sum()),
        'memory_usage_mb': float(df_ps.memory_usage(deep=True).sum() / 1024**2)
    }
    
    # Validaciones específicas
    checks = []
    
    if metrics['missing_percent'] < 5:
        checks.append(('Missing values', 'PASS', f"{metrics['missing_percent']:.2f}%"))
    else:
        checks.append(('Missing values', 'WARN', f"{metrics['missing_percent']:.2f}%"))
    
    if metrics['duplicates'] == 0:
        checks.append(('Duplicates', 'PASS', '0'))
    else:
        checks.append(('Duplicates', 'WARN', str(metrics['duplicates'])))
    
    logger.info("  Métricas de calidad:")
    for check, status, value in checks:
        symbol = '✓' if status == 'PASS' else '⚠'
        logger.info(f"    {symbol} {check}: {value}")
    
    return metrics


def preprocess_pipeline(df_ps, config=None):
    """
    Pipeline completo de preprocesamiento.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame original
    config : dict, optional
        Configuración de preprocesamiento
        
    Returns:
    --------
    pyspark.pandas.DataFrame
        DataFrame preprocesado
    """
    logger.info("\n" + "="*80)
    logger.info("INICIANDO PIPELINE DE PREPROCESAMIENTO")
    logger.info("="*80 + "\n")
    
    if config is None:
        config = {
            'handle_missing': 'auto',
            'remove_duplicates': True,
            'filter_outliers': False,
            'normalize': False
        }
    
    df_processed = df_ps.copy()
    
    # 1. Analizar valores faltantes
    missing_df = analyze_missing_values(df_processed)
    
    # 2. Manejar valores faltantes
    if config.get('handle_missing'):
        df_processed = handle_missing_values(df_processed, strategy=config['handle_missing'])
    
    # 3. Eliminar duplicados
    if config.get('remove_duplicates'):
        df_processed = remove_duplicates(df_processed)
    
    # 4. Filtrar outliers (opcional)
    if config.get('filter_outliers') and config.get('outlier_columns'):
        df_processed = filter_outliers(df_processed, config['outlier_columns'])
    
    # 5. Normalizar (opcional)
    if config.get('normalize') and config.get('normalize_columns'):
        df_processed = normalize_columns(df_processed, config['normalize_columns'])
    
    # 6. Validar calidad final
    metrics = validate_data_quality(df_processed)
    
    logger.info("\n" + "="*80)
    logger.info("✓ PIPELINE DE PREPROCESAMIENTO COMPLETADO")
    logger.info("="*80 + "\n")
    
    return df_processed


if __name__ == "__main__":
    print("Data Preprocessing Module")
    print("Este módulo debe ser importado, no ejecutado directamente.")