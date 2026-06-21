import configparser
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.problem import antenaparcheInteger
from jmetal.util.observer import ProgressBarObserver, LivePlotObserver, GuardarResultadosFisicosObserver, AntenaGraficasObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.generator import InjectorFromFile, RandomGenerator

"""
MÓDULO PRINCIPAL DE ORQUESTACIÓN Y LANZAMIENTO (ORQUESTADOR SMPSO)

Este script constituye el punto de entrada ejecutable para la variante metaheurística
basada en Inteligencia de Enjambres (SI). Configura y ejecuta el algoritmo SMPSO
(Speed-constrained Multi-objective Particle Swarm Optimization), parametrizando la
cinemática de las partículas, la gestión de velocidades acotadas y la persistencia
de frentes no dominados en un archivo externo limitado por distancia de hacinamiento.
"""

if __name__ == "__main__":
    
    # =============================================================================
    # 1. CARGA DE CONFIGURACIÓN Y DESACOPLAMIENTO DE PARÁMETROS
    # =============================================================================
    print("--> Leyendo archivo de configuración...")
    config = configparser.ConfigParser()
    # Interfaz de lectura para el aislamiento de hiperparámetros operativos del enjambre
    config.read('conf_SMPSO.txt', encoding='utf-8')
    
    # --- 1. EXTRACCIÓN DE PARÁMETROS DEL TXT ---
    
    # Mapeo del diccionario de directorios del lazo local de control COM para CST
    rutas_cst = {
        'carpeta_cst': config.get('DIRECTORIOS_CST', 'carpeta_cst'),
        'ruta_plantilla_macro': config.get('DIRECTORIOS_CST', 'ruta_plantilla_macro'),
        'carpeta_tiempos': config.get('DIRECTORIOS_CST', 'carpeta_tiempos')
    }
    # Estructuración de las rutas de persistencia de telemetría y frentes de Pareto
    dir_vars = config.get('DIRECTORIOS', 'carpeta_resultados_vars')
    dir_cst = config.get('DIRECTORIOS', 'carpeta_graficas_cst')
    dir_pareto = config.get('DIRECTORIOS', 'carpeta_graficas_pareto')
    
    # Parámetros cinemáticos y de dimensionamiento del enjambre de partículas
    max_evaluations = config.getint('ALGORITMO', 'max_evaluaciones')
    swrm_size = config.getint('ALGORITMO', 'tamaño_enjambre')
    capacidad_arch = config.getint('ALGORITMO', 'capacidad_archivo')

    # Configuración del operador de perturbación estocástica (Mutación del Enjambre)
    factor_mut = config.getfloat('OPERADORES', 'factor_prob_mutacion')
    dist_index_mut = config.getfloat('OPERADORES', 'dist_index_mutacion')
    
    # Frecuencia de muestreo integrada para los observadores del sistema
    frecuencia_obs = config.getint('OBSERVERS', 'frecuencia_guardado')
    
    # =============================================================================
    # 2. INSTANCIACIÓN DEL PROBLEMA Y ARQUITECTURA WARM-START
    # =============================================================================
    # Carga del problema en espacio entero cuantizado (Evaluación multifrecuencia)
    problem = antenaparcheInteger(rutas_config=rutas_cst)
    
    # Lógica de inyección genotípica adaptativa (Reinicio guiado o Warm-Start)
    usar_rescate = config.getboolean('WARM_START', 'usar_rescate')
    if usar_rescate:
        # Inyección posicional: inicializa el enjambre a partir de coordenadas precalculadas
        ruta_rescate_VAR = config.get('WARM_START', 'ruta_archivo_VAR')
        ruta_rescate_FUN = config.get('WARM_START', 'ruta_archivo_FUN')
        print(f"--> Modo Warm-Start ACTIVADO. Cargando desde: {ruta_rescate_VAR}")
        generador_inicial = InjectorFromFile(ruta_var=ruta_rescate_VAR,ruta_fun=ruta_rescate_FUN , problem=problem)
    else:
        # Inicialización canónica: distribución pseudoaleatoria uniforme de las partículas
        print("--> Modo Warm-Start DESACTIVADO. Generando población aleatoria.")
        generador_inicial = RandomGenerator()

    # =============================================================================
    # 3. CONFIGURACIÓN E INSTANCIACIÓN DEL ENJAMBRE DE PARTÍCULAS (SMPSO)
    # =============================================================================
    algorithm = SMPSO(
        problem=problem,
        swarm_size=swrm_size,
        # Mutación polinómica: evita el estancamiento cinemático por velocidad cero
        mutation=IntegerPolynomialMutation(
            probability=factor_mut / problem.number_of_variables(), 
            distribution_index=dist_index_mut
        ),        
        # Archivo externo no dominado regido por el estimador de densidad de hacinamiento
        leaders=CrowdingDistanceArchive(capacidad_arch),
        # Criterio de parada acotado al presupuesto temporal de simulaciones tridimensionales
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        swarm_generator=generador_inicial
    )
    
    # =============================================================================
    # 4. ENLACE Y REGISTRO DE MÓDULOS DE MONOTORIZACIÓN (PATRÓN OBSERVER)
    # =============================================================================
    
    # Enlace del monitor de progreso de evaluaciones analíticas en consola
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    
    # Persistencia de telemetría física: vuelca las posiciones factibles del enjambre a CSV
    saver = GuardarResultadosFisicosObserver(
        frequency=frecuencia_obs, 
        algorithm=algorithm,
        steps_del_problema=problem.steps,
        output_directory=dir_vars
    )
    algorithm.observable.register(observer=saver)
    
    # Observador de variables CAD: almacena los históricos estructurales de la geometría
    plotter_antena = AntenaGraficasObserver(frequency=frecuencia_obs, output_directory=dir_cst)
    algorithm.observable.register(observer=plotter_antena)
    
    # Renderizado en tiempo real de la frontera de Pareto empírica y control de penalizaciones
    plotter_pareto = LivePlotObserver(
        frequency=frecuencia_obs, 
        algorithm=algorithm,
        max_evaluations=max_evaluations,  
        output_directory=dir_pareto,
        objective_labels=["S11 (dB) [Minimizar]", "Axial Ratio (dB) [Minimizar]"],
        penalty_threshold=1000.0
    )
    algorithm.observable.register(observer=plotter_pareto)
                        
    # =============================================================================
    # 5. LANZAMIENTO DEL PROCESO CINEMÁTICO Y EXTRACCIÓN DE MÉTRICAS
    # =============================================================================         
    algorithm.run()
    front = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")