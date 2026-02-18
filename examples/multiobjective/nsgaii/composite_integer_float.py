# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 10:54:53 2026

@author: jmate
"""

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import CompositeCrossover, IntegerSBXCrossover, SBXCrossover
from jmetal.operator.mutation import CompositeMutation, IntegerPolynomialMutation, PolynomialMutation
from jmetal.problem import MixedIntegerFloatProblem
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver, SaveResultsObserver, LivePlotObserver


"""  
Program to  configure and run the NSGA-II algorithm configured to solve a binary problem, OneZeroMax, which is 
multiobjective version of the ONE_MAX problem where the numbers of 1s and 0s have to be maximized at the same time.
"""

if __name__ == "__main__":
    problem = MixedIntegerFloatProblem()

    max_evaluations = 10000
    
    composite_mutation = CompositeMutation([
        IntegerPolynomialMutation(probability=1.0/problem.number_of_variables, distribution_index=20.0),
        PolynomialMutation(probability=1.0/problem.number_of_variables, distribution_index=20.0)
    ])
    
    composite_crossover = CompositeCrossover([
        IntegerSBXCrossover(probability=1.0, distribution_index=20.0),
        SBXCrossover(probability=1.0, distribution_index=20.0)
    ])
    
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=composite_mutation,
        crossover=composite_crossover,
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