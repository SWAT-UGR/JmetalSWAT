from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SPXCrossover
from jmetal.operator.mutation import BitFlipMutation
from jmetal.problem import ZDT5
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver, SaveResultsObserver, LivePlotObserver

"""  
Program to  configure and run the NSGA-II algorithm configured to solve a binary problem, OneZeroMax, which is 
multiobjective version of the ONE_MAX problem where the numbers of 1s and 0s have to be maximized at the same time.
"""

if __name__ == "__main__":
    problem = ZDT5()

    max_evaluations = 20000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=BitFlipMutation(probability=1.0 / problem.total_number_of_bits()),
        crossover=SPXCrossover(probability=1.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front, display_frequency=100)
    )
    saver = SaveResultsObserver(frequency=1000)
    algorithm.observable.register(observer=saver)
    plotter = LivePlotObserver(frequency=1000, max_evaluations=max_evaluations)
    algorithm.observable.register(observer=plotter)

    algorithm.run()
    front = algorithm.result()

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
