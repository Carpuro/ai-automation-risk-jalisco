"""
Main Script
===========
Script principal para ejecutar el análisis completo de automatización laboral.

Uso:
    python main.py --mode sample
    python main.py --mode real --occupation-data data/raw/onet.csv --employment-data data/raw/enoe.csv

Autor: Carlos Pulido Rosas
Proyecto: Modelo Predictivo de Sustitución Laboral por IA - Jalisco
"""

import argparse
import sys
import os
import logging
import warnings
from datetime import datetime

# Suprimir warnings de PySpark y bibliotecas relacionadas
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', module='pyspark')
warnings.filterwarnings('ignore', module='py4j')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar módulos del proyecto
from data_loader import (
    create_spark_session,
    load_sample_data,
    load_onet_occupations,
    load_enoe_jalisco,
    convert_to_pandas_api,
    save_dataset
)

from data_preprocessing import (
    preprocess_pipeline,
    validate_data_quality
)

from feature_engineering import (
    feature_engineering_pipeline
)

from automation_analyzer import (
    AutomationRiskAnalyzer,
    generate_risk_report
)

from visualizations import create_dashboard


def parse_arguments():
    """
    Parsea argumentos de línea de comandos.
    
    Returns:
    --------
    argparse.Namespace
        Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description='Análisis de Riesgo de Automatización Laboral en Jalisco',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos de uso:
  # Usar datos simulados (recomendado para pruebas)
  python main.py --mode sample --n-occupations 200

  # Usar datos reales
  python main.py --mode real \\
    --occupation-data data/raw/onet/Occupation_Data.txt \\
    --employment-data data/raw/enoe_jalisco.csv

  # Especificar directorio de salida
  python main.py --mode sample --output outputs/run_2025_11_26

  # Sin visualizaciones (más rápido)
  python main.py --mode sample --no-visualizations
        '''
    )
    
    # Modo de ejecución
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sample', 'real'],
        default='sample',
        help='Modo de ejecución: "sample" (datos simulados) o "real" (datos reales)'
    )
    
    # Datos de entrada (solo para modo real)
    parser.add_argument(
        '--occupation-data',
        type=str,
        help='Ruta a archivo de datos de ocupaciones (O*NET)'
    )
    
    parser.add_argument(
        '--employment-data',
        type=str,
        help='Ruta a archivo de datos de empleo (ENOE Jalisco)'
    )
    
    # Configuración de datos simulados
    parser.add_argument(
        '--n-occupations',
        type=int,
        default=200,
        help='Número de ocupaciones simuladas (solo modo sample)'
    )
    
    # Directorios de salida
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Directorio base para guardar resultados'
    )
    
    # Opciones de procesamiento
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Saltar generación de visualizaciones (más rápido)'
    )
    
    parser.add_argument(
        '--no-model',
        action='store_true',
        help='Saltar entrenamiento de modelo predictivo'
    )
    
    # Configuración de Spark
    parser.add_argument(
        '--memory',
        type=str,
        default='8g',
        help='Memoria asignada a Spark (ej: 4g, 8g, 16g)'
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """
    Valida los argumentos de entrada.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Argumentos parseados
        
    Returns:
    --------
    bool
        True si válido, False en caso contrario
    """
    if args.mode == 'real':
        if not args.occupation_data:
            logger.error("Modo 'real' requiere --occupation-data")
            return False
        
        if not args.employment_data:
            logger.error("Modo 'real' requiere --employment-data")
            return False
        
        if not os.path.exists(args.occupation_data):
            logger.error(f"Archivo no encontrado: {args.occupation_data}")
            return False
        
        if not os.path.exists(args.employment_data):
            logger.error(f"Archivo no encontrado: {args.employment_data}")
            return False
    
    return True


def main():
    """
    Función principal del script.
    """
    # Banner
    print("\n" + "="*80)
    print("ANÁLISIS DE RIESGO DE AUTOMATIZACIÓN LABORAL")
    print("Modelo Predictivo para Jalisco, México (2025-2030)")
    print("Autor: Carlos Pulido Rosas - CUCEA, UdeG")
    print("="*80 + "\n")
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Validar inputs
    if not validate_inputs(args):
        sys.exit(1)
    
    # Crear directorios de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = args.output
    output_processed = os.path.join(output_base, 'processed')
    output_reports = os.path.join(output_base, 'reports')
    output_viz = os.path.join(output_base, 'visualizations')
    output_models = os.path.join(output_base, 'models')
    
    os.makedirs(output_processed, exist_ok=True)
    os.makedirs(output_reports, exist_ok=True)
    os.makedirs(output_viz, exist_ok=True)
    os.makedirs(output_models, exist_ok=True)
    
    logger.info(f"Directorios de salida creados en: {output_base}")
    
    try:
        # ====================================================================
        # PASO 1: INICIALIZACIÓN
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 1/7: INICIALIZACIÓN")
        logger.info("="*80)
        
        # Crear Spark Session
        spark = create_spark_session(
            app_name="AI_Automation_Risk_Jalisco",
            memory=args.memory
        )
        
        # Suprimir logs verbosos de Spark
        spark.sparkContext.setLogLevel("ERROR")
        
        # ====================================================================
        # PASO 2: CARGA DE DATOS
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 2/7: CARGA DE DATOS")
        logger.info("="*80)
        
        if args.mode == 'sample':
            logger.info(f"Modo: Datos simulados (n={args.n_occupations})")
            df_spark = load_sample_data(spark, n_occupations=args.n_occupations)
        
        elif args.mode == 'real':
            logger.info("Modo: Datos reales")
            
            # Cargar O*NET
            df_onet = load_onet_occupations(spark, args.occupation_data)
            
            # Cargar ENOE
            df_enoe = load_enoe_jalisco(spark, args.employment_data)
            
            # Integrar (simplificado - en producción usar mapeo SOC-SINCO)
            df_spark = df_onet.join(df_enoe, how='inner')
            logger.info("✓ Datos integrados")
        
        # Convertir a pyspark.pandas
        df = convert_to_pandas_api(df_spark)
        logger.info(f"✓ Dataset cargado: {df.shape[0]:,} ocupaciones, {df.shape[1]} columnas")
        
        # ====================================================================
        # PASO 3: PREPROCESAMIENTO
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 3/7: PREPROCESAMIENTO")
        logger.info("="*80)
        
        df_clean = preprocess_pipeline(df, config={
            'handle_missing': 'auto',
            'remove_duplicates': True,
            'filter_outliers': False
        })
        
        logger.info(f"✓ Datos limpios: {df_clean.shape[0]:,} ocupaciones")
        
        # ====================================================================
        # PASO 4: FEATURE ENGINEERING
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 4/7: FEATURE ENGINEERING")
        logger.info("="*80)
        
        df_features = feature_engineering_pipeline(df_clean, config={
            'create_indices': True,
            'create_categories': True,
            'create_ratios': True,
            'create_temporal': True,
            'create_sector_agg': True
        })
        
        logger.info(f"✓ Features creados: {df_features.shape[1]} columnas totales")
        
        # ====================================================================
        # PASO 5: ANÁLISIS DE RIESGO
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 5/7: ANÁLISIS DE RIESGO DE AUTOMATIZACIÓN")
        logger.info("="*80)
        
        # Crear analizador
        analyzer = AutomationRiskAnalyzer()
        
        # Calcular riesgo
        df_risk = analyzer.calculate_automation_risk(
            df_features,
            method='frey_osborne'
        )
        
        # Categorizar
        df_risk = analyzer.categorize_risk(df_risk, thresholds=(0.30, 0.70))
        
        # Análisis por dimensiones
        logger.info("\nAnálisis por dimensiones:")
        
        sector_analysis = analyzer.analyze_by_sector(df_risk)
        education_analysis = analyzer.analyze_by_education(df_risk)
        impact = analyzer.calculate_economic_impact(df_risk)
        
        logger.info(f"  Trabajadores en alto riesgo: {impact['workers_high_risk']:,} ({impact['pct_workers_at_risk']:.1f}%)")
        
        # ====================================================================
        # PASO 6: VISUALIZACIONES
        # ====================================================================
        if not args.no_visualizations:
            logger.info("\n" + "="*80)
            logger.info("PASO 6/7: GENERACIÓN DE VISUALIZACIONES")
            logger.info("="*80)
            
            # Dashboard completo (14 visualizaciones)
            create_dashboard(df_risk, output_dir=output_viz)
            
            logger.info(f"✓ Visualizaciones guardadas en: {output_viz}")
        else:
            logger.info("\nPASO 6/7: VISUALIZACIONES - OMITIDAS")
        
        # ====================================================================
        # PASO 7: REPORTES Y EXPORTACIÓN
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("PASO 7/7: GENERACIÓN DE REPORTES")
        logger.info("="*80)
        
        # Generar reporte de texto
        report_path = os.path.join(output_reports, f'risk_analysis_report_{timestamp}.txt')
        report_text = generate_risk_report(df_risk, output_path=report_path)
        
        # Guardar dataset procesado
        processed_path = os.path.join(output_processed, f'risk_analysis_results_{timestamp}.parquet')
        save_dataset(df_risk, processed_path, format='parquet')
        
        # Exportar CSV para Excel
        csv_path = os.path.join(output_reports, f'risk_analysis_results_{timestamp}.csv')
        df_risk.to_pandas().to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"✓ Reporte de texto: {report_path}")
        logger.info(f"✓ Datos procesados: {processed_path}")
        logger.info(f"✓ CSV exportado: {csv_path}")
        
        # ====================================================================
        # RESUMEN FINAL
        # ====================================================================
        logger.info("\n" + "="*80)
        logger.info("ANÁLISIS COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        
        print("\n📊 RESUMEN DE RESULTADOS:")
        print(f"   • Ocupaciones analizadas: {len(df_risk):,}")
        print(f"   • Trabajadores totales: {impact['total_workers']:,}")
        print(f"   • Trabajadores en alto riesgo: {impact['workers_high_risk']:,} ({impact['pct_workers_at_risk']:.1f}%)")
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   • Reporte: {report_path}")
        print(f"   • Dataset: {processed_path}")
        print(f"   • CSV: {csv_path}")
        if not args.no_visualizations:
            print(f"   • Visualizaciones: {output_viz}/")
        
        print(f"\n✅ Análisis finalizado correctamente")
        print("="*80 + "\n")
        
        # Cerrar Spark
        spark.stop()
        
        return 0
    
    except Exception as e:
        logger.error(f"\n❌ ERROR DURANTE LA EJECUCIÓN: {str(e)}")
        logger.exception("Detalles del error:")
        
        # Cerrar Spark si existe
        try:
            spark.stop()
        except:
            pass
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)