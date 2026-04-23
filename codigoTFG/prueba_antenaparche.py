from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import IntegerSBXCrossover
from jmetal.operator.mutation import IntegerPolynomialMutation
from jmetal.problem import antenaparcheInteger
from jmetal.util.observer import ProgressBarObserver, LivePlotObserver, GuardarResultadosFisicosObserver, AntenaGraficasObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

"""

"""

if __name__ == "__main__":
    problem = antenaparcheInteger()
    
    max_evaluations = 5000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=IntegerPolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        crossover=IntegerSBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    
    saver = GuardarResultadosFisicosObserver(
        frequency=100, 
        steps_del_problema=problem.steps, # Le pasamos los steps desde la clase del problema
        output_directory=r"C:\Users\jmate\Desktop\teleco\TFG\intermediate_results"
    )
    algorithm.observable.register(observer=saver)
    
    plotter_antena = AntenaGraficasObserver(frequency=100, output_directory=r"C:\Users\jmate\Desktop\teleco\TFG\Plots cst")
    algorithm.observable.register(observer=plotter_antena)
    
    ruta_pareto = r"C:\Users\jmate\Desktop\teleco\TFG\Pareto_Fronts"
    
    plotter_pareto = LivePlotObserver(
        frequency=100, 
        max_evaluations=max_evaluations, 
        input_directory=r"C:\Users\jmate\Desktop\teleco\TFG\intermediate_results", 
        output_directory=r"C:\Users\jmate\Desktop\teleco\TFG\Pareto_Plots"
    )
    algorithm.observable.register(observer=plotter_pareto)
                                  
    algorithm.run()
    front = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")