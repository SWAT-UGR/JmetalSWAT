from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import IntegerSBXCrossover
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.problem import antenaparcheInteger
from jmetal.util.observer import ProgressBarObserver, LivePlotObserver, GuardarResultadosFisicosObserver, AntenaGraficasObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.generator import InjectorFromFile
"""

"""

if __name__ == "__main__":
    problem = antenaparcheInteger()
    
    ruta_rescate = r"C:\Users\jmate\Desktop\teleco\TFG\Resultados\intermediate_results_3400\VAR.antenaparcheInteger.400"
    
    generador_rescate = InjectorFromFile(ruta_archivo=ruta_rescate, problem=problem)
    
    max_evaluations = 5000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        population_generator=generador_rescate,
        offspring_population_size=100,
        mutation=IntegerPolynomialMutation(probability=2.0 / problem.number_of_variables(), distribution_index=5),
        crossover=IntegerSBXCrossover(probability=1.0, distribution_index=10),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    
    saver = GuardarResultadosFisicosObserver(
        frequency=100, 
        steps_del_problema=problem.steps, # Le pasamos los steps desde la clase del problema
        output_directory=r"C:\Users\jmate\Desktop\teleco\TFG\Resultados\intermediate_results_3800"
    )
    algorithm.observable.register(observer=saver)
    
    plotter_antena = AntenaGraficasObserver(frequency=100, output_directory=r"C:\Users\jmate\Desktop\teleco\TFG\CST plots\Plots cst_3800")
    algorithm.observable.register(observer=plotter_antena)
    
    plotter_pareto = LivePlotObserver(
        frequency=100, 
        max_evaluations=max_evaluations,  
        output_directory=r"C:\Users\jmate\Desktop\teleco\TFG\Frentes pareto\Pareto_Plots_3800"
    )
    algorithm.observable.register(observer=plotter_pareto)
                                  
    algorithm.run()
    front = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
