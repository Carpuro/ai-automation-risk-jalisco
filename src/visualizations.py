"""
Visualizations Module
=====================
Módulo para crear visualizaciones del análisis de automatización laboral.

Autor: Carlos Pulido Rosas
Proyecto: Modelo Predictivo de Sustitución Laboral por IA - Jalisco
"""

import pyspark.pandas as ps
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_risk_distribution(df_ps, save_path=None):
    """
    Visualiza la distribución del riesgo de automatización.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con columna 'automation_risk'
    save_path : str, optional
        Ruta para guardar la figura
    """
    logger.info("Generando gráfico: Distribución de riesgo...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histograma
    ax.hist(df_ps['automation_risk'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Líneas de umbral
    ax.axvline(0.30, color='green', linestyle='--', linewidth=2, label='Umbral Bajo-Medio (0.30)')
    ax.axvline(0.70, color='red', linestyle='--', linewidth=2, label='Umbral Medio-Alto (0.70)')
    
    # Estadísticas
    mean_risk = df_ps['automation_risk'].mean()
    median_risk = df_ps['automation_risk'].median()
    ax.axvline(mean_risk, color='orange', linestyle=':', linewidth=2, label=f'Media ({mean_risk:.3f})')
    ax.axvline(median_risk, color='purple', linestyle=':', linewidth=2, label=f'Mediana ({median_risk:.3f})')
    
    ax.set_xlabel('Riesgo de Automatización', fontsize=12)
    ax.set_ylabel('Frecuencia', fontsize=12)
    ax.set_title('Distribución del Riesgo de Automatización en Jalisco', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Guardado en: {save_path}")
    
    plt.show()


def plot_risk_by_sector(df_ps, top_n=15, save_path=None):
    """
    Visualiza riesgo promedio por sector económico.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con columnas 'sector' y 'automation_risk'
    top_n : int
        Número de sectores a mostrar
    save_path : str, optional
        Ruta para guardar
    """
    logger.info(f"Generando gráfico: Riesgo por sector (top {top_n})...")
    
    # Calcular riesgo promedio por sector
    sector_risk = df_ps.groupby('sector')['automation_risk'].mean().sort_values(ascending=True).tail(top_n)
    
    # Colores según nivel de riesgo
    colors = ['red' if x >= 0.70 else 'orange' if x >= 0.30 else 'green' for x in sector_risk.values]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sector_risk.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
    
    # Líneas de referencia
    ax.axvline(0.70, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Alto Riesgo')
    ax.axvline(0.30, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Bajo Riesgo')
    
    ax.set_xlabel('Riesgo Promedio de Automatización', fontsize=12)
    ax.set_ylabel('Sector Económico', fontsize=12)
    ax.set_title(f'Riesgo de Automatización por Sector Económico (Top {top_n})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Guardado en: {save_path}")
    
    plt.show()


def plot_salary_vs_risk(df_ps, save_path=None):
    """
    Scatter plot: Salario vs Riesgo de automatización.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con datos
    save_path : str, optional
        Ruta para guardar
    """
    logger.info("Generando gráfico: Salario vs Riesgo (scatter)...")
    
    # Convertir a pandas para Plotly
    df_plot = df_ps[['occupation_name', 'avg_salary_mxn', 'automation_risk', 
                     'sector', 'workers_jalisco']].to_pandas()
    
    fig = px.scatter(
        df_plot,
        x='avg_salary_mxn',
        y='automation_risk',
        color='sector',
        size='workers_jalisco',
        hover_data=['occupation_name'],
        title='Relación entre Salario y Riesgo de Automatización',
        labels={
            'avg_salary_mxn': 'Salario Promedio (MXN/mes)',
            'automation_risk': 'Riesgo de Automatización',
            'sector': 'Sector',
            'workers_jalisco': 'Trabajadores'
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    # Líneas de referencia
    fig.add_hline(y=0.70, line_dash="dash", line_color="red", annotation_text="Alto Riesgo")
    fig.add_hline(y=0.30, line_dash="dash", line_color="green", annotation_text="Bajo Riesgo")
    
    fig.update_layout(
        height=600,
        font=dict(size=12),
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"  ✓ Guardado en: {save_path}")
    
    fig.show()


def plot_education_vs_risk(df_ps, save_path=None):
    """
    Box plot: Nivel educativo vs Riesgo.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con datos
    save_path : str, optional
        Ruta para guardar
    """
    logger.info("Generando gráfico: Educación vs Riesgo (boxplot)...")
    
    # Etiquetas de educación
    edu_labels = {
        1: 'Sin educación',
        2: 'Primaria',
        3: 'Secundaria',
        4: 'Preparatoria',
        5: 'Universidad',
        6: 'Posgrado'
    }
    
    df_plot = df_ps[['education_level', 'automation_risk']].to_pandas()
    df_plot['education_label'] = df_plot['education_level'].map(edu_labels)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(
        data=df_plot,
        x='education_label',
        y='automation_risk',
        palette='RdYlGn_r',
        ax=ax
    )
    
    # Líneas de referencia
    ax.axhline(0.70, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Alto Riesgo')
    ax.axhline(0.30, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Bajo Riesgo')
    
    ax.set_xlabel('Nivel Educativo', fontsize=12)
    ax.set_ylabel('Riesgo de Automatización', fontsize=12)
    ax.set_title('Distribución del Riesgo por Nivel Educativo', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Guardado en: {save_path}")
    
    plt.show()


def plot_heatmap_sector_education(df_ps, save_path=None):
    """
    Heatmap: Riesgo por Sector vs Nivel Educativo.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con datos
    save_path : str, optional
        Ruta para guardar
    """
    logger.info("Generando gráfico: Heatmap Sector vs Educación...")
    
    # Crear pivot table
    pivot_data = df_ps.groupby(['sector', 'education_level'])['automation_risk'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='sector', columns='education_level', values='automation_risk')
    
    # Convertir a pandas
    pivot_table = pivot_table.to_pandas()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        center=0.5,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Riesgo de Automatización'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_xlabel('Nivel Educativo', fontsize=12)
    ax.set_ylabel('Sector Económico', fontsize=12)
    ax.set_title('Mapa de Calor: Riesgo por Sector y Nivel Educativo', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Guardado en: {save_path}")
    
    plt.show()


def plot_treemap_workers_at_risk(df_ps, save_path=None):
    """
    Treemap: Trabajadores en alto riesgo por sector.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con datos
    save_path : str, optional
        Ruta para guardar
    """
    logger.info("Generando gráfico: Treemap de trabajadores en riesgo...")
    
    # Filtrar alto riesgo
    high_risk = df_ps[df_ps['automation_risk'] >= 0.70]
    
    # Agregar por sector
    sector_impact = high_risk.groupby('sector').agg({
        'workers_jalisco': 'sum',
        'automation_risk': 'mean'
    }).reset_index().to_pandas()
    
    fig = px.treemap(
        sector_impact,
        path=['sector'],
        values='workers_jalisco',
        color='automation_risk',
        color_continuous_scale='Reds',
        title='Trabajadores en Alto Riesgo por Sector Económico',
        labels={'workers_jalisco': 'Trabajadores', 'automation_risk': 'Riesgo Promedio'}
    )
    
    fig.update_layout(height=600)
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"  ✓ Guardado en: {save_path}")
    
    fig.show()


def plot_temporal_projections(df_ps, save_path=None):
    """
    Serie temporal: Proyecciones 2025-2030.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con proyecciones temporales
    save_path : str, optional
        Ruta para guardar
    """
    logger.info("Generando gráfico: Proyecciones temporales 2025-2030...")
    
    years = [2025, 2026, 2027, 2028, 2029, 2030]
    
    # Calcular porcentaje en alto riesgo por año
    high_risk_pct = []
    
    for year in years:
        if year == 2025:
            col = 'automation_risk'
        else:
            col = f'risk_projection_{year}'
        
        if col in df_ps.columns:
            pct = (df_ps[col] >= 0.70).sum() / len(df_ps) * 100
            high_risk_pct.append(pct)
        else:
            high_risk_pct.append(None)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(years, high_risk_pct, marker='o', linewidth=2, markersize=10, color='crimson')
    ax.fill_between(years, high_risk_pct, alpha=0.3, color='crimson')
    
    ax.set_xlabel('Año', fontsize=12)
    ax.set_ylabel('Ocupaciones en Alto Riesgo (%)', fontsize=12)
    ax.set_title('Proyección de Ocupaciones en Alto Riesgo (2025-2030)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(years)
    
    # Anotaciones
    for i, (year, pct) in enumerate(zip(years, high_risk_pct)):
        if pct is not None:
            ax.annotate(f'{pct:.1f}%', 
                       xy=(year, pct), 
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=10,
                       fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Guardado en: {save_path}")
    
    plt.show()


def plot_correlation_matrix(df_ps, save_path=None):
    """
    Matriz de correlación de variables clave.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con datos
    save_path : str, optional
        Ruta para guardar
    """
    logger.info("Generando gráfico: Matriz de correlación...")
    
    # Seleccionar columnas numéricas relevantes
    numeric_cols = [
        'automation_risk',
        'routine_index',
        'cognitive_demand',
        'social_interaction',
        'creativity',
        'education_level',
        'avg_salary_mxn',
        'workers_jalisco'
    ]
    
    # Filtrar solo columnas que existen
    available_cols = [col for col in numeric_cols if col in df_ps.columns]
    
    # Calcular correlación
    corr_matrix = df_ps[available_cols].corr().to_pandas()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlación'},
        ax=ax
    )
    
    ax.set_title('Matriz de Correlación de Variables Clave', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Guardado en: {save_path}")
    
    plt.show()


def plot_top_occupations_at_risk(df_ps, n=20, save_path=None):
    """
    Top N ocupaciones con mayor riesgo.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con datos
    n : int
        Número de ocupaciones a mostrar
    save_path : str, optional
        Ruta para guardar
    """
    logger.info(f"Generando gráfico: Top {n} ocupaciones en riesgo...")
    
    # Top N ocupaciones
    top_risk = df_ps.nlargest(n, 'automation_risk')[
        ['occupation_name', 'automation_risk', 'workers_jalisco', 'sector']
    ].to_pandas()
    
    # Ordenar de menor a mayor para que el más riesgoso esté arriba
    top_risk = top_risk.sort_values('automation_risk', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    bars = ax.barh(
        range(len(top_risk)),
        top_risk['automation_risk'],
        color='darkred',
        edgecolor='black'
    )
    
    # Tamaño de burbuja proporcional a trabajadores
    for i, (idx, row) in enumerate(top_risk.iterrows()):
        size = np.sqrt(row['workers_jalisco']) / 20
        ax.scatter(row['automation_risk'], i, s=size, color='gold', 
                  edgecolor='black', zorder=3, alpha=0.7)
    
    ax.set_yticks(range(len(top_risk)))
    ax.set_yticklabels(top_risk['occupation_name'], fontsize=10)
    ax.set_xlabel('Riesgo de Automatización', fontsize=12)
    ax.set_title(f'Top {n} Ocupaciones con Mayor Riesgo de Automatización', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Guardado en: {save_path}")
    
    plt.show()


def create_dashboard(df_ps, output_dir='outputs/visualizations/'):
    """
    Crea un dashboard completo con todas las visualizaciones.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con datos procesados
    output_dir : str
        Directorio para guardar visualizaciones
    """
    logger.info("\n" + "="*80)
    logger.info("GENERANDO DASHBOARD COMPLETO")
    logger.info("="*80 + "\n")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Distribución de riesgo
    plot_risk_distribution(df_ps, save_path=f'{output_dir}01_risk_distribution.png')
    
    # 2. Riesgo por sector
    plot_risk_by_sector(df_ps, save_path=f'{output_dir}02_risk_by_sector.png')
    
    # 3. Salario vs Riesgo
    plot_salary_vs_risk(df_ps, save_path=f'{output_dir}03_salary_vs_risk.html')
    
    # 4. Educación vs Riesgo
    plot_education_vs_risk(df_ps, save_path=f'{output_dir}04_education_vs_risk.png')
    
    # 5. Heatmap
    plot_heatmap_sector_education(df_ps, save_path=f'{output_dir}05_heatmap_sector_education.png')
    
    # 6. Treemap
    plot_treemap_workers_at_risk(df_ps, save_path=f'{output_dir}06_treemap_workers_risk.html')
    
    # 7. Proyecciones temporales
    if 'risk_projection_2026' in df_ps.columns:
        plot_temporal_projections(df_ps, save_path=f'{output_dir}07_temporal_projections.png')
    
    # 8. Correlaciones
    plot_correlation_matrix(df_ps, save_path=f'{output_dir}08_correlation_matrix.png')
    
    # 9. Top ocupaciones
    plot_top_occupations_at_risk(df_ps, save_path=f'{output_dir}09_top_occupations_risk.png')
    
    logger.info("\n" + "="*80)
    logger.info("✓ DASHBOARD COMPLETO GENERADO")
    logger.info("="*80)
    logger.info(f"\nArchivos guardados en: {output_dir}")


if __name__ == "__main__":
    print("Visualizations Module")
    print("Este módulo debe ser importado, no ejecutado directamente.")