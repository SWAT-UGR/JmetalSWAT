import configparser
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import IntegerSBXCrossover
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.problem import antenaparcheIntegerMono
from jmetal.util.observer import ProgressBarObserver, LivePlotObserver, GuardarResultadosFisicosObserver, AntenaGraficasObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.generator import InjectorFromFile, RandomGenerator

"""
MÓDULO PRINCIPAL DE ORQUESTACIÓN Y LANZAMIENTO (ORQUESTADOR NSGA-II)

Este script actúa como el punto de entrada de ingeniería para la ejecución del bucle
de optimización estocástica global. Coordina el desacoplamiento de hiperparámetros 
mediante archivos de configuración estructurados, inicializa el escenario monofrecuencia 
basado en variables cuantizadas discretas, parametriza los operadores genéticos 
de explotación y exploración, y registra múltiples observadores concurrentes para la 
trazabilidad y monitorización en tiempo real del frente de Pareto resultante.
"""

if __name__ == "__main__":
    
    # =============================================================================
    # 1. CARGA DE CONFIGURACIÓN Y DESACOPLAMIENTO DE PARÁMETROS
    # =============================================================================
    print("--> Leyendo archivo de configuración...")
    config = configparser.ConfigParser()
    # Lectura del fichero plano de directivas para evitar el hardcoding en el flujo lógico
    config.read('conf_NSGAII_monofreq.txt', encoding='utf-8')
    
    # --- 1. EXTRACCIÓN DE PARÁMETROS DEL TXT ---
    
    # Mapeo del diccionario de rutas relativas y absolutas del entorno de simulación 3D CST
    rutas_cst = {
        'carpeta_cst': config.get('DIRECTORIOS_CST', 'carpeta_cst'),
        'ruta_plantilla_macro': config.get('DIRECTORIOS_CST', 'ruta_plantilla_macro'),
        'carpeta_tiempos': config.get('DIRECTORIOS_CST', 'carpeta_tiempos')
    }
    # Aislamiento de directorios destinados a la persistencia de datos y gráficos
    dir_vars = config.get('DIRECTORIOS', 'carpeta_resultados_vars')
    dir_cst = config.get('DIRECTORIOS', 'carpeta_graficas_cst')
    dir_pareto = config.get('DIRECTORIOS', 'carpeta_graficas_pareto')
    
    # Extracción de dimensiones poblacionales y cotas del presupuesto computacional
    max_evaluations = config.getint('ALGORITMO', 'max_evaluaciones')
    pop_size = config.getint('ALGORITMO', 'tamaño_poblacion')
    offspring_size = config.getint('ALGORITMO', 'tamaño_descendencia')
    
    # Configuración de los hiperparámetros del operador unario de mutación polinómica
    factor_mut = config.getfloat('OPERADORES', 'factor_prob_mutacion')
    dist_index_mut = config.getfloat('OPERADORES', 'dist_index_mutacion')
    
    # Parámetros operativos para el operador binario de recombinación simulada binaria (SBX)
    prob_cross = config.getfloat('OPERADORES', 'prob_cruce')
    dist_index_cross = config.getfloat('OPERADORES', 'dist_index_cruce')
    
    # Frecuencia de muestreo y actualización para los observadores del sistema
    frecuencia_obs = config.getint('OBSERVERS', 'frecuencia_guardado')
    
    # =============================================================================
    # 2. INSTANCIACIÓN DEL PROBLEMA Y ESTRATEGIA DE INICIALIZACIÓN
    # =============================================================================
    problem = antenaparcheIntegerMono(rutas_config=rutas_cst)
    
    # Lógica de rescate inteligente (Warm-Start / Reinicio Guiado del Algoritmo)
    usar_rescate = config.getboolean('WARM_START', 'usar_rescate')
    if usar_rescate:
        # Modo de inyección: inicializa la población cargando una base de datos genotípica previa
        ruta_rescate_VAR = config.get('WARM_START', 'ruta_archivo_VAR')
        ruta_rescate_FUN = config.get('WARM_START', 'ruta_archivo_FUN')
        print(f"--> Modo Warm-Start ACTIVADO. Cargando desde: {ruta_rescate_VAR}")
        generador_inicial = InjectorFromFile(ruta_var=ruta_rescate_VAR,ruta_fun=ruta_rescate_FUN , problem=problem)
    else:
        # Modo estocástico: inicializa la población mediante distribución uniforme aleatoria
        print("--> Modo Warm-Start DESACTIVADO. Generando población aleatoria.")
        generador_inicial = RandomGenerator()


    # =============================================================================
    # 3. CONFIGURACIÓN E INSTANCIACIÓN DEL ALGORITMO EVOLUTIVO (NSGA-II)
    # =============================================================================
    algorithm = NSGAII(
        problem=problem,
        population_size=pop_size,
        population_generator=generador_inicial,
        offspring_population_size=offspring_size,
        # Mutación polinómica adaptada a índices discretos mediante el escalado de variables
        mutation=IntegerPolynomialMutation(
            probability=factor_mut / problem.number_of_variables(), 
            distribution_index=dist_index_mut
        ),
        # Cruce SBX avanzado para mantener la correlación de vecindad paramétrica en enteros
        crossover=IntegerSBXCrossover(
            probability=prob_cross, 
            distribution_index=dist_index_cross
        ),
        # Criterio de parada determinista condicionado al presupuesto analítico global
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )
    
    # =============================================================================
    # 4. ENLACE Y REGISTRO DE MÓDULOS DE MONOTORIZACIÓN (PATRÓN OBSERVER)
    # =============================================================================
    
    # Registro del observador de consola para el seguimiento de la velocidad de cómputo
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    
    # Observador de persistencia física: vuelca los vectores numéricos válidos en bruto a archivos CSV
    saver = GuardarResultadosFisicosObserver(
        frequency=frecuencia_obs, 
        algorithm=algorithm,
        steps_del_problema=problem.steps,
        output_directory=dir_vars
    )
    algorithm.observable.register(observer=saver)
    
    # Observador gráfico CAD: genera capturas físicas e histogramas del estado geométrico de la antena
    plotter_antena = AntenaGraficasObserver(frequency=frecuencia_obs, output_directory=dir_cst)
    algorithm.observable.register(observer=plotter_antena)
    
    # Observador dinámico de Pareto: renderiza interactivamente la evolución de la frontera no dominada
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
    # 5. DISPARO DEL PROCESO EVOLUTIVO Y LOG DE SALIDA EN TERMINAL
    # =============================================================================         
    algorithm.run()
    front = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
