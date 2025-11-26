"""
Script de Verificación del Entorno
====================================
Proyecto: Modelo Predictivo de Sustitución Laboral por IA
Autor: Carlos Pulido Rosas

Ejecutar: python verify_setup.py
"""

import sys

def check_python_version():
    """Verifica la versión de Python"""
    version = sys.version_info
    print(f"Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor in [10, 11]:
        print("✓ Versión de Python compatible\n")
        return True
    else:
        print("⚠ Se recomienda Python 3.10 o 3.11\n")
        return False

def check_dependencies():
    """Verifica las dependencias del proyecto"""
    required = {
        'pyspark': '3.5.0',
        'pandas': '2.1.0',
        'numpy': '1.26.0',
        'pyarrow': '14.0.0',
        'sklearn': '1.3.0',
        'matplotlib': '3.8.0',
        'seaborn': '0.13.0',
        'plotly': '5.18.0',
        'jupyter': '1.0.0'
    }
    
    print("="*80)
    print("VERIFICACIÓN DE DEPENDENCIAS")
    print("="*80)
    print()
    
    all_ok = True
    
    for package, min_version in required.items():
        try:
            # Importar el paquete
            if package == 'sklearn':
                mod = __import__('sklearn')
            else:
                mod = __import__(package)
            
            # Obtener versión
            version = getattr(mod, '__version__', 'unknown')
            
            # Verificar versión mínima
            status = "✓" if version >= min_version or version == 'unknown' else "⚠"
            
            print(f"{status} {package:20} {version:15} (requerido: >={min_version})")
            
        except ImportError:
            print(f"✗ {package:20} NO INSTALADO")
            all_ok = False
    
    print()
    print("="*80)
    
    return all_ok

def check_spark():
    """Verifica que Spark funcione correctamente"""
    print("\nVERIFICANDO SPARK...")
    print("-"*80)
    
    try:
        from pyspark.sql import SparkSession
        
        spark = SparkSession.builder \
            .appName("VerificationTest") \
            .master("local[*]") \
            .getOrCreate()
        
        # Test simple
        data = [("Alice", 1), ("Bob", 2)]
        df = spark.createDataFrame(data, ["name", "value"])
        count = df.count()
        
        spark.stop()
        
        print(f"✓ Spark funcionando correctamente (test: {count} registros)")
        return True
        
    except Exception as e:
        print(f"✗ Error en Spark: {e}")
        return False

def check_pyspark_pandas():
    """Verifica que pyspark.pandas funcione"""
    print("\nVERIFICANDO PYSPARK.PANDAS...")
    print("-"*80)
    
    try:
        import pyspark.pandas as ps
        
        # Test simple
        df = ps.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = df.sum().sum()
        
        print(f"✓ pyspark.pandas funcionando (test sum: {result})")
        return True
        
    except Exception as e:
        print(f"✗ Error en pyspark.pandas: {e}")
        return False

def check_jupyter():
    """Verifica que Jupyter esté disponible"""
    print("\nVERIFICANDO JUPYTER...")
    print("-"*80)
    
    try:
        import notebook
        import ipykernel
        
        print(f"✓ Jupyter Notebook {notebook.__version__}")
        print(f"✓ IPyKernel {ipykernel.__version__}")
        return True
        
    except Exception as e:
        print(f"✗ Error en Jupyter: {e}")
        return False

def print_next_steps(all_passed):
    """Imprime los siguientes pasos"""
    print("\n")
    print("="*80)
    
    if all_passed:
        print("✅ ENTORNO CONFIGURADO CORRECTAMENTE")
        print("="*80)
        print("\n📋 SIGUIENTES PASOS:")
        print()
        print("1. Coloca tus datos en la carpeta 'data/raw/'")
        print("2. Ejecuta el análisis:")
        print("   jupyter notebook notebooks/automation_risk_analysis.ipynb")
        print()
        print("3. O ejecuta el script principal:")
        print("   python src/main.py --occupation-data data/raw/occupations.csv")
        print()
    else:
        print("❌ CONFIGURACIÓN INCOMPLETA")
        print("="*80)
        print("\n📋 ACCIÓN REQUERIDA:")
        print()
        print("Instala las dependencias faltantes:")
        print()
        print("Opción 1 (Conda - Recomendado):")
        print("   conda env create -f environment.yml")
        print("   conda activate ai_automation_thesis")
        print()
        print("Opción 2 (pip):")
        print("   pip install -r requirements.txt")
        print()
        print("Luego ejecuta este script nuevamente:")
        print("   python verify_setup.py")
        print()
    
    print("="*80)
    print()

def main():
    """Función principal"""
    print()
    print("="*80)
    print("VERIFICACIÓN DEL ENTORNO")
    print("Proyecto: Modelo Predictivo de Sustitución Laboral por IA")
    print("="*80)
    print()
    
    # Verificar Python
    python_ok = check_python_version()
    
    # Verificar dependencias
    deps_ok = check_dependencies()
    
    # Verificar Spark
    spark_ok = check_spark()
    
    # Verificar pyspark.pandas
    pandas_ok = check_pyspark_pandas()
    
    # Verificar Jupyter
    jupyter_ok = check_jupyter()
    
    # Resultado final
    all_passed = python_ok and deps_ok and spark_ok and pandas_ok and jupyter_ok
    
    print_next_steps(all_passed)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)