from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import antenaparche
from jmetal.util.observer import ProgressBarObserver, PlotFrontToFileObserver, SaveResultsObserver, AntenaGraficasObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

"""

"""

if __name__ == "__main__":
    problem = antenaparche()
    
    max_evaluations = 500
    algorithm = NSGAII(
        problem=problem,
        population_size=50,
        offspring_population_size=50,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    
    saver = SaveResultsObserver(frequency=50)
    algorithm.observable.register(observer=saver)
    
    plotter_antena = AntenaGraficasObserver(frequency=50, output_directory=r"C:\Users\jmate\Desktop\teleco\TFG\Plots cst")
    algorithm.observable.register(observer=plotter_antena)
    
    ruta_pareto = r"C:\Users\jmate\Desktop\teleco\TFG\Pareto_Fronts"
    
    plotter_pareto = PlotFrontToFileObserver(
        output_directory=ruta_pareto, 
        step= 50
    )
    algorithm.observable.register(observer=plotter_pareto)
                                  
    algorithm.run()
    front = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")