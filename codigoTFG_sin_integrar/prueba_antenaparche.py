from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.lab.visualization import InteractivePlot, Plot
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import antenaparche
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver, SaveResultsObserver, LivePlotObserver
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
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front, display_frequency=100)
    )
    saver = SaveResultsObserver(frequency=500)
    algorithm.observable.register(observer=saver)
    plotter = LivePlotObserver(frequency=500, max_evaluations=max_evaluations)
    algorithm.observable.register(observer=plotter)
                                  
    algorithm.run()
    front = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")