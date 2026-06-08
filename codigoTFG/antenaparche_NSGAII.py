import configparser
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import IntegerSBXCrossover
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.problem import antenaparcheIntegerMono
from jmetal.util.observer import ProgressBarObserver, LivePlotObserver, GuardarResultadosFisicosObserver, AntenaGraficasObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.generator import InjectorFromFile, RandomGenerator
"""

"""

if __name__ == "__main__":
    
    print("--> Leyendo archivo de configuración...")
    config = configparser.ConfigParser()
    # Leemos el archivo txt (asegúrate de que esté en la misma carpeta)
    config.read('conf_NSGAII_monofreq.txt', encoding='utf-8')
    
    # --- 1. EXTRACCIÓN DE PARÁMETROS DEL TXT ---
    
    # Rutas de CST
    rutas_cst = {
        'carpeta_cst': config.get('DIRECTORIOS_CST', 'carpeta_cst'),
        'ruta_plantilla_macro': config.get('DIRECTORIOS_CST', 'ruta_plantilla_macro'),
        'carpeta_tiempos': config.get('DIRECTORIOS_CST', 'carpeta_tiempos')
    }
    # Directorios
    dir_vars = config.get('DIRECTORIOS', 'carpeta_resultados_vars')
    dir_cst = config.get('DIRECTORIOS', 'carpeta_graficas_cst')
    dir_pareto = config.get('DIRECTORIOS', 'carpeta_graficas_pareto')
    
    # Algoritmo
    max_evaluations = config.getint('ALGORITMO', 'max_evaluaciones')
    pop_size = config.getint('ALGORITMO', 'tamaño_poblacion')
    offspring_size = config.getint('ALGORITMO', 'tamaño_descendencia')
    
    # Operadores
    factor_mut = config.getfloat('OPERADORES', 'factor_prob_mutacion')
    dist_index_mut = config.getfloat('OPERADORES', 'dist_index_mutacion')
    
    # ---> NUEVO: Operadores de Cruce <---
    prob_cross = config.getfloat('OPERADORES', 'prob_cruce')
    dist_index_cross = config.getfloat('OPERADORES', 'dist_index_cruce')
    
    # Observers
    frecuencia_obs = config.getint('OBSERVERS', 'frecuencia_guardado')
    
    # --- 2. CONFIGURACIÓN DEL PROBLEMA Y GENERADOR ---
    problem = antenaparcheIntegerMono(rutas_config=rutas_cst)
    
    # Lógica de rescate inteligente
    usar_rescate = config.getboolean('WARM_START', 'usar_rescate')
    if usar_rescate:
        ruta_rescate_VAR = config.get('WARM_START', 'ruta_archivo_VAR')
        ruta_rescate_FUN = config.get('WARM_START', 'ruta_archivo_FUN')
        print(f"--> Modo Warm-Start ACTIVADO. Cargando desde: {ruta_rescate_VAR}")
        generador_inicial = InjectorFromFile(ruta_var=ruta_rescate_VAR,ruta_fun=ruta_rescate_FUN , problem=problem)
    else:
        print("--> Modo Warm-Start DESACTIVADO. Generando población aleatoria.")
        generador_inicial = RandomGenerator()


    algorithm = NSGAII(
        problem=problem,
        population_size=pop_size,
        population_generator=generador_inicial,
        offspring_population_size=offspring_size,
        mutation=IntegerPolynomialMutation(
            probability=factor_mut / problem.number_of_variables(), 
            distribution_index=dist_index_mut
        ),
        crossover=IntegerSBXCrossover(
            probability=prob_cross, 
            distribution_index=dist_index_cross
        ),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )
    
    # --- 4. REGISTRO DE OBSERVERS ---
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    
    saver = GuardarResultadosFisicosObserver(
        frequency=frecuencia_obs, 
        algorithm=algorithm,
        steps_del_problema=problem.steps,
        output_directory=dir_vars
    )
    algorithm.observable.register(observer=saver)
    
    plotter_antena = AntenaGraficasObserver(frequency=frecuencia_obs, output_directory=dir_cst)
    algorithm.observable.register(observer=plotter_antena)
    
    plotter_pareto = LivePlotObserver(
        frequency=frecuencia_obs, 
        max_evaluations=max_evaluations,  
        output_directory=dir_pareto,
        objective_labels=["S11 (dB) [Minimizar]", "Axial Ratio (dB) [Minimizar]"],
        penalty_threshold=1000.0
    )
    algorithm.observable.register(observer=plotter_pareto)
                        
    # --- 5. EJECUCIÓN ---          
    algorithm.run()
    front = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
