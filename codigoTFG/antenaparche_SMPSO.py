import configparser
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.problem import antenaparcheInteger
from jmetal.util.observer import ProgressBarObserver, LivePlotObserver, GuardarResultadosFisicosObserver, AntenaGraficasObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.generator import InjectorFromFile, RandomGenerator
"""

"""

if __name__ == "__main__":
    
    print("--> Leyendo archivo de configuración...")
    config = configparser.ConfigParser()
    # Leemos el archivo txt (asegúrate de que esté en la misma carpeta)
    config.read('conf_SMPSO.txt', encoding='utf-8')
    
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
    swrm_size = config.getint('ALGORITMO', 'tamaño_enjambre')
    
    # Observers
    frecuencia_obs = config.getint('OBSERVERS', 'frecuencia_guardado')
    
    # --- 2. CONFIGURACIÓN DEL PROBLEMA Y GENERADOR ---
    problem = antenaparcheInteger(rutas_config=rutas_cst)
    
    # Lógica de rescate inteligente
    usar_rescate = config.getboolean('WARM_START', 'usar_rescate')
    if usar_rescate:
        ruta_rescate = config.get('WARM_START', 'ruta_archivo_rescate')
        print(f"--> Modo Warm-Start ACTIVADO. Cargando desde: {ruta_rescate}")
        generador_inicial = InjectorFromFile(ruta_archivo=ruta_rescate, problem=problem)
    else:
        print("--> Modo Warm-Start DESACTIVADO. Generando población aleatoria.")
        generador_inicial = RandomGenerator()

    algorithm = SMPSO(
        problem=problem,
        swarm_size=swrm_size,
        mutation=IntegerPolynomialMutation(probability=3.0 / problem.number_of_variables(), distribution_index=5),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )
    
    # --- 4. REGISTRO DE OBSERVERS ---
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    
    saver = GuardarResultadosFisicosObserver(
        frequency=frecuencia_obs, 
        steps_del_problema=problem.steps,
        output_directory=dir_vars
    )
    algorithm.observable.register(observer=saver)
    
    plotter_antena = AntenaGraficasObserver(frequency=frecuencia_obs, output_directory=dir_cst)
    algorithm.observable.register(observer=plotter_antena)
    
    plotter_pareto = LivePlotObserver(
        frequency=frecuencia_obs, 
        max_evaluations=max_evaluations,  
        output_directory=dir_pareto
    )
    algorithm.observable.register(observer=plotter_pareto)
                        
    # --- 5. EJECUCIÓN ---          
    algorithm.run()
    front = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
