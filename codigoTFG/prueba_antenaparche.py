from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import antenaparche
from jmetal.util.observer import ProgressBarObserver, SaveResultsObserver, AntenaGraficasObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

"""

"""

if __name__ == "__main__":
    problem = antenaparche()
    
    max_evaluations = 1000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    
    saver = SaveResultsObserver(frequency=500)
    algorithm.observable.register(observer=saver)
    
    plotter_antena = AntenaGraficasObserver(frequency=500, output_directory=r"C:\Users\jmate\Desktop\teleco\TFG\Plots cst")
    algorithm.observable.register(observer=plotter_antena)
                                  
    algorithm.run()
    front = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")