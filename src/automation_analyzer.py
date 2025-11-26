"""
Automation Analyzer Module
===========================
Módulo central para calcular riesgo de automatización laboral por IA.

Basado en metodología Frey-Osborne y adaptado para Jalisco, México.

Autor: Carlos Pulido Rosas
Proyecto: Modelo Predictivo de Sustitución Laboral por IA - Jalisco
"""

import pyspark.pandas as ps
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomationRiskAnalyzer:
    """
    Clase para analizar riesgo de automatización de ocupaciones.
    """
    
    def __init__(self):
        """Inicializa el analizador"""
        self.weights = {
            'routine': 0.40,
            'cognitive': -0.25,
            'social': -0.20,
            'creativity': -0.15
        }
        logger.info("✓ AutomationRiskAnalyzer inicializado")
    
    def calculate_routine_index(self, df_ps):
        """
        Calcula índice de rutinización de tareas.
        
        Basado en:
        - Repetitividad de tareas
        - Predictibilidad del trabajo
        - Estructuración de actividades
        
        Parameters:
        -----------
        df_ps : pyspark.pandas.DataFrame
            DataFrame con características de ocupaciones
            
        Returns:
        --------
        pyspark.pandas.Series
            Índice de rutinización (0-100)
        """
        logger.info("Calculando índice de rutinización...")
        
        # Si ya existe, usar directamente
        if 'routine_index' in df_ps.columns:
            return df_ps['routine_index']
        
        # Calcular basado en componentes
        components = []
        weights = []
        
        if 'task_repetitive' in df_ps.columns:
            components.append(df_ps['task_repetitive'])
            weights.append(0.40)
        
        if 'task_predictable' in df_ps.columns:
            components.append(df_ps['task_predictable'])
            weights.append(0.35)
        
        if 'task_structured' in df_ps.columns:
            components.append(df_ps['task_structured'])
            weights.append(0.25)
        
        if components:
            routine_index = sum(c * w for c, w in zip(components, weights))
            logger.info("✓ Índice de rutinización calculado")
            return routine_index
        else:
            logger.warning("No se encontraron componentes de rutinización, usando valores por defecto")
            return df_ps.get('routine_index', 50)
    
    def calculate_cognitive_demand(self, df_ps):
        """
        Calcula demanda cognitiva de la ocupación.
        
        Mayor demanda cognitiva = Menor riesgo de automatización
        
        Parameters:
        -----------
        df_ps : pyspark.pandas.DataFrame
            DataFrame con características
            
        Returns:
        --------
        pyspark.pandas.Series
            Demanda cognitiva (0-100)
        """
        logger.info("Calculando demanda cognitiva...")
        
        if 'cognitive_demand' in df_ps.columns:
            return df_ps['cognitive_demand']
        
        # Calcular basado en habilidades cognitivas
        components = []
        
        cognitive_skills = [
            'skill_critical_thinking',
            'skill_complex_problem_solving',
            'skill_analytical',
            'ability_deductive_reasoning',
            'ability_inductive_reasoning'
        ]
        
        for skill in cognitive_skills:
            if skill in df_ps.columns:
                components.append(df_ps[skill])
        
        if components:
            cognitive_demand = sum(components) / len(components)
            logger.info("✓ Demanda cognitiva calculada")
            return cognitive_demand
        else:
            logger.warning("No se encontraron habilidades cognitivas, usando valores por defecto")
            return df_ps.get('cognitive_demand', 50)
    
    def calculate_social_interaction(self, df_ps):
        """
        Calcula nivel de interacción social requerida.
        
        Mayor interacción social = Menor riesgo de automatización
        
        Parameters:
        -----------
        df_ps : pyspark.pandas.DataFrame
            DataFrame con características
            
        Returns:
        --------
        pyspark.pandas.Series
            Nivel de interacción social (0-100)
        """
        logger.info("Calculando nivel de interacción social...")
        
        if 'social_interaction' in df_ps.columns:
            return df_ps['social_interaction']
        
        # Calcular basado en actividades sociales
        components = []
        
        social_activities = [
            'activity_communicating_supervisors',
            'activity_communicating_coworkers',
            'activity_interacting_public',
            'skill_social_perceptiveness',
            'skill_persuasion',
            'skill_negotiation'
        ]
        
        for activity in social_activities:
            if activity in df_ps.columns:
                components.append(df_ps[activity])
        
        if components:
            social_interaction = sum(components) / len(components)
            logger.info("✓ Interacción social calculada")
            return social_interaction
        else:
            logger.warning("No se encontraron actividades sociales, usando valores por defecto")
            return df_ps.get('social_interaction', 50)
    
    def calculate_creativity_level(self, df_ps):
        """
        Calcula nivel de creatividad requerida.
        
        Mayor creatividad = Menor riesgo de automatización
        
        Parameters:
        -----------
        df_ps : pyspark.pandas.DataFrame
            DataFrame con características
            
        Returns:
        --------
        pyspark.pandas.Series
            Nivel de creatividad (0-100)
        """
        logger.info("Calculando nivel de creatividad...")
        
        if 'creativity' in df_ps.columns:
            return df_ps['creativity']
        
        # Calcular basado en habilidades creativas
        components = []
        
        creative_skills = [
            'skill_originality',
            'ability_fluency_ideas',
            'activity_thinking_creatively',
            'work_style_innovation'
        ]
        
        for skill in creative_skills:
            if skill in df_ps.columns:
                components.append(df_ps[skill])
        
        if components:
            creativity = sum(components) / len(components)
            logger.info("✓ Creatividad calculada")
            return creativity
        else:
            logger.warning("No se encontraron habilidades creativas, usando valores por defecto")
            return df_ps.get('creativity', 50)
    
    def calculate_automation_risk(self, df_ps, method='frey_osborne'):
        """
        Calcula riesgo de automatización usando diferentes metodologías.
        
        Parameters:
        -----------
        df_ps : pyspark.pandas.DataFrame
            DataFrame con características de ocupaciones
        method : str
            Método a usar: 'frey_osborne', 'task_based', 'hybrid'
            
        Returns:
        --------
        pyspark.pandas.DataFrame
            DataFrame con columna 'automation_risk' agregada
        """
        logger.info(f"Calculando riesgo de automatización (método: {method})...")
        
        df_result = df_ps.copy()
        
        if method == 'frey_osborne':
            # Método Frey & Osborne simplificado
            routine = self.calculate_routine_index(df_ps)
            cognitive = self.calculate_cognitive_demand(df_ps)
            social = self.calculate_social_interaction(df_ps)
            creativity = self.calculate_creativity_level(df_ps)
            
            # Normalizar a 0-1
            routine_norm = routine / 100
            cognitive_norm = cognitive / 100
            social_norm = social / 100
            creativity_norm = creativity / 100
            
            # Calcular riesgo ponderado
            risk = (
                routine_norm * self.weights['routine'] +
                cognitive_norm * self.weights['cognitive'] +
                social_norm * self.weights['social'] +
                creativity_norm * self.weights['creativity']
            )
            
            # Normalizar a rango 0-1
            risk = (risk - risk.min()) / (risk.max() - risk.min())
            
        elif method == 'task_based':
            # Método basado en tareas
            if 'task_manual' in df_ps.columns and 'task_cognitive' in df_ps.columns:
                manual = df_ps['task_manual'] / 100
                cognitive = df_ps['task_cognitive'] / 100
                risk = manual * 0.6 + (1 - cognitive) * 0.4
            else:
                logger.warning("Columnas de tareas no encontradas, usando método frey_osborne")
                return self.calculate_automation_risk(df_ps, method='frey_osborne')
        
        elif method == 'hybrid':
            # Método híbrido (combina ambos)
            risk_fo = self.calculate_automation_risk(df_ps, method='frey_osborne')['automation_risk']
            risk_tb = self.calculate_automation_risk(df_ps, method='task_based')['automation_risk']
            risk = (risk_fo + risk_tb) / 2
        
        else:
            raise ValueError(f"Método no reconocido: {method}")
        
        df_result['automation_risk'] = risk
        
        logger.info(f"✓ Riesgo de automatización calculado")
        logger.info(f"  Rango: {risk.min():.3f} - {risk.max():.3f}")
        logger.info(f"  Media: {risk.mean():.3f}")
        logger.info(f"  Mediana: {risk.median():.3f}")
        
        return df_result
    
    def categorize_risk(self, df_ps, thresholds=(0.30, 0.70)):
        """
        Categoriza el riesgo en Alto/Medio/Bajo.
        
        Parameters:
        -----------
        df_ps : pyspark.pandas.DataFrame
            DataFrame con columna 'automation_risk'
        thresholds : tuple
            Umbrales (bajo_medio, medio_alto)
            
        Returns:
        --------
        pyspark.pandas.DataFrame
            DataFrame con columna 'risk_category'
        """
        logger.info("Categorizando riesgo...")
        
        df_result = df_ps.copy()
        
        def categorize(risk):
            if risk >= thresholds[1]:
                return 'Alto'
            elif risk >= thresholds[0]:
                return 'Medio'
            else:
                return 'Bajo'
        
        df_result['risk_category'] = df_result['automation_risk'].apply(categorize)
        
        # Estadísticas por categoría
        counts = df_result['risk_category'].value_counts()
        logger.info("✓ Categorización completada:")
        for category, count in counts.items():
            pct = (count / len(df_result)) * 100
            logger.info(f"  {category}: {count} ({pct:.1f}%)")
        
        return df_result
    
    def analyze_by_sector(self, df_ps):
        """
        Analiza riesgo de automatización por sector económico.
        
        Parameters:
        -----------
        df_ps : pyspark.pandas.DataFrame
            DataFrame con columnas 'sector' y 'automation_risk'
            
        Returns:
        --------
        pyspark.pandas.DataFrame
            Análisis agregado por sector
        """
        logger.info("Analizando por sector económico...")
        
        if 'sector' not in df_ps.columns:
            logger.warning("Columna 'sector' no encontrada")
            return None
        
        sector_analysis = df_ps.groupby('sector').agg({
            'automation_risk': ['mean', 'median', 'std', 'min', 'max'],
            'workers_jalisco': 'sum',
            'occupation_name': 'count'
        }).reset_index()
        
        sector_analysis.columns = [
            'sector', 'risk_mean', 'risk_median', 'risk_std',
            'risk_min', 'risk_max', 'total_workers', 'num_occupations'
        ]
        
        # Ordenar por riesgo promedio
        sector_analysis = sector_analysis.sort_values('risk_mean', ascending=False)
        
        logger.info("✓ Análisis por sector completado")
        logger.info(f"  Sectores analizados: {len(sector_analysis)}")
        
        return sector_analysis
    
    def analyze_by_education(self, df_ps):
        """
        Analiza riesgo por nivel educativo.
        
        Parameters:
        -----------
        df_ps : pyspark.pandas.DataFrame
            DataFrame con columnas 'education_level' y 'automation_risk'
            
        Returns:
        --------
        pyspark.pandas.DataFrame
            Análisis agregado por nivel educativo
        """
        logger.info("Analizando por nivel educativo...")
        
        if 'education_level' not in df_ps.columns:
            logger.warning("Columna 'education_level' no encontrada")
            return None
        
        education_analysis = df_ps.groupby('education_level').agg({
            'automation_risk': ['mean', 'median', 'std'],
            'workers_jalisco': 'sum',
            'occupation_name': 'count',
            'avg_salary_mxn': 'mean'
        }).reset_index()
        
        education_analysis.columns = [
            'education_level', 'risk_mean', 'risk_median', 'risk_std',
            'total_workers', 'num_occupations', 'avg_salary'
        ]
        
        # Etiquetas de educación
        edu_labels = {
            1: 'Sin educación',
            2: 'Primaria',
            3: 'Secundaria',
            4: 'Preparatoria',
            5: 'Universidad',
            6: 'Posgrado'
        }
        
        education_analysis['education_label'] = education_analysis['education_level'].apply(
            lambda x: edu_labels.get(x, f'Nivel {x}')
        )
        
        # Ordenar por nivel educativo
        education_analysis = education_analysis.sort_values('education_level')
        
        logger.info("✓ Análisis por educación completado")
        
        return education_analysis
    
    def calculate_economic_impact(self, df_ps):
        """
        Calcula impacto económico potencial de la automatización.
        
        Parameters:
        -----------
        df_ps : pyspark.pandas.DataFrame
            DataFrame con datos de ocupaciones
            
        Returns:
        --------
        dict
            Métricas de impacto económico
        """
        logger.info("Calculando impacto económico...")
        
        # Trabajadores en alto riesgo
        high_risk = df_ps[df_ps['automation_risk'] >= 0.70]
        workers_high_risk = high_risk['workers_jalisco'].sum()
        
        # Trabajadores totales
        total_workers = df_ps['workers_jalisco'].sum()
        
        # Porcentaje en riesgo
        pct_at_risk = (workers_high_risk / total_workers) * 100
        
        # Masa salarial en riesgo
        if 'avg_salary_mxn' in df_ps.columns:
            salary_at_risk = (high_risk['workers_jalisco'] * high_risk['avg_salary_mxn']).sum()
            total_salary = (df_ps['workers_jalisco'] * df_ps['avg_salary_mxn']).sum()
            pct_salary_risk = (salary_at_risk / total_salary) * 100
        else:
            salary_at_risk = None
            pct_salary_risk = None
        
        impact = {
            'total_workers': int(total_workers),
            'workers_high_risk': int(workers_high_risk),
            'pct_workers_at_risk': float(pct_at_risk),
            'salary_at_risk_mxn': float(salary_at_risk) if salary_at_risk else None,
            'pct_salary_at_risk': float(pct_salary_risk) if pct_salary_risk else None
        }
        
        logger.info("✓ Impacto económico calculado:")
        logger.info(f"  Trabajadores en alto riesgo: {workers_high_risk:,} ({pct_at_risk:.1f}%)")
        if salary_at_risk:
            logger.info(f"  Masa salarial en riesgo: ${salary_at_risk:,.2f} MXN ({pct_salary_risk:.1f}%)")
        
        return impact
    
    def identify_top_at_risk(self, df_ps, n=20):
        """
        Identifica las ocupaciones con mayor riesgo.
        
        Parameters:
        -----------
        df_ps : pyspark.pandas.DataFrame
            DataFrame con datos de ocupaciones
        n : int
            Número de ocupaciones a retornar
            
        Returns:
        --------
        pyspark.pandas.DataFrame
            Top N ocupaciones en riesgo
        """
        logger.info(f"Identificando top {n} ocupaciones en riesgo...")
        
        top_risk = df_ps.nlargest(n, 'automation_risk')[
            ['occupation_name', 'sector', 'automation_risk', 
             'workers_jalisco', 'avg_salary_mxn', 'education_level']
        ]
        
        logger.info(f"✓ Top {n} ocupaciones identificadas")
        
        return top_risk
    
    def identify_low_risk_occupations(self, df_ps, n=20):
        """
        Identifica ocupaciones con menor riesgo (más seguras).
        
        Parameters:
        -----------
        df_ps : pyspark.pandas.DataFrame
            DataFrame con datos
        n : int
            Número de ocupaciones
            
        Returns:
        --------
        pyspark.pandas.DataFrame
            Top N ocupaciones seguras
        """
        logger.info(f"Identificando top {n} ocupaciones más seguras...")
        
        low_risk = df_ps.nsmallest(n, 'automation_risk')[
            ['occupation_name', 'sector', 'automation_risk',
             'workers_jalisco', 'avg_salary_mxn', 'education_level']
        ]
        
        logger.info(f"✓ Top {n} ocupaciones seguras identificadas")
        
        return low_risk


def generate_risk_report(df_ps, output_path='outputs/reports/risk_report.txt'):
    """
    Genera un reporte completo de análisis de riesgo.
    
    Parameters:
    -----------
    df_ps : pyspark.pandas.DataFrame
        DataFrame con análisis completo
    output_path : str
        Ruta para guardar el reporte
    """
    logger.info("Generando reporte de riesgo...")
    
    analyzer = AutomationRiskAnalyzer()
    
    # Análisis
    impact = analyzer.calculate_economic_impact(df_ps)
    sector_analysis = analyzer.analyze_by_sector(df_ps)
    edu_analysis = analyzer.analyze_by_education(df_ps)
    top_risk = analyzer.identify_top_at_risk(df_ps, n=10)
    low_risk = analyzer.identify_low_risk_occupations(df_ps, n=10)
    
    # Generar reporte
    report = []
    report.append("="*80)
    report.append("REPORTE DE ANÁLISIS DE RIESGO DE AUTOMATIZACIÓN")
    report.append("Jalisco, México - 2025-2030")
    report.append("="*80)
    report.append("")
    
    report.append("1. IMPACTO ECONÓMICO")
    report.append("-"*80)
    report.append(f"Trabajadores totales: {impact['total_workers']:,}")
    report.append(f"Trabajadores en alto riesgo: {impact['workers_high_risk']:,} ({impact['pct_workers_at_risk']:.1f}%)")
    if impact['salary_at_risk_mxn']:
        report.append(f"Masa salarial en riesgo: ${impact['salary_at_risk_mxn']:,.2f} MXN")
    report.append("")
    
    report.append("2. TOP 10 OCUPACIONES EN RIESGO")
    report.append("-"*80)
    for idx, row in top_risk.iterrows():
        report.append(f"{row['occupation_name']}: {row['automation_risk']:.3f} ({row['workers_jalisco']:,} trabajadores)")
    report.append("")
    
    report.append("3. TOP 10 OCUPACIONES MÁS SEGURAS")
    report.append("-"*80)
    for idx, row in low_risk.iterrows():
        report.append(f"{row['occupation_name']}: {row['automation_risk']:.3f}")
    report.append("")
    
    report.append("="*80)
    
    # Guardar reporte
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logger.info(f"✓ Reporte generado: {output_path}")
    
    return '\n'.join(report)


if __name__ == "__main__":
    print("Automation Analyzer Module")
    print("Este módulo debe ser importado, no ejecutado directamente.")