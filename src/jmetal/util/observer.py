import logging
import os
from pathlib import Path
from typing import List, TypeVar

import numpy as np
from tqdm import tqdm

from jmetal.core.observer import Observer
from jmetal.core.problem import DynamicProblem
from jmetal.core.quality_indicator import InvertedGenerationalDistance
from jmetal.lab.visualization import Plot, StreamingPlot
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
import matplotlib.pyplot as plt

S = TypeVar("S")

LOGGER = logging.getLogger("jmetal")

"""
.. module:: observer
   :platform: Unix, Windows
   :synopsis: Implementation of algorithm's observers.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class ProgressBarObserver(Observer):
    def __init__(self, max: int) -> None:
        """Show a smart progress meter with the number of evaluations and computing time.

        :param max: Number of expected iterations.
        """
        self.progress_bar = None
        self.progress = 0
        self._max = max

    def update(self, *args, **kwargs):
        if not self.progress_bar:
            self.progress_bar = tqdm(total=self._max, ascii=True, desc="Progress")

        evaluations = kwargs["EVALUATIONS"]

        self.progress_bar.update(evaluations - self.progress)
        self.progress = evaluations

        if self.progress >= self._max:
            self.progress_bar.close()


class BasicObserver(Observer):
    def __init__(self, frequency: int = 1) -> None:
        """Show the number of evaluations, the best fitness and the computing time.
        :param frequency: Display frequency."""

        self.display_frequency = frequency

    def update(self, *args, **kwargs):
        computing_time = kwargs["COMPUTING_TIME"]
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives

            LOGGER.info(
                "Evaluations: {} \n Best fitness: {} \n Computing time: {}".format(evaluations, fitness, computing_time)
            )


class PrintObjectivesObserver(Observer):
    def __init__(self, frequency: int = 1) -> None:
        """Show the number of evaluations, best fitness and computing time.

        :param frequency: Display frequency."""
        self.display_frequency = frequency

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives

            LOGGER.info("Evaluations: {}. fitness: {}".format(evaluations, fitness))


class WriteFrontToFileObserver(Observer):
    def __init__(self, output_directory: str) -> None:
        """Write function values of the front into files.

        :param output_directory: Output directory. Each front will be saved on a file `FUN.x`."""
        self.counter = 0
        self.directory = output_directory

        if Path(self.directory).is_dir():
            LOGGER.warning("Directory {} exists. Removing contents.".format(self.directory))
            for file in os.listdir(self.directory):
                os.remove("{0}/{1}".format(self.directory, file))
        else:
            LOGGER.warning("Directory {} does not exist. Creating it.".format(self.directory))
            Path(self.directory).mkdir(parents=True)

    def update(self, *args, **kwargs):
        problem = kwargs["PROBLEM"]
        solutions = kwargs["SOLUTIONS"]

        if solutions:
            if isinstance(problem, DynamicProblem):
                termination_criterion_is_met = kwargs.get("TERMINATION_CRITERIA_IS_MET", None)

                if termination_criterion_is_met:
                    print_function_values_to_file(solutions, "{}/FUN.{}".format(self.directory, self.counter))
                    self.counter += 1
            else:
                print_function_values_to_file(solutions, "{}/FUN.{}".format(self.directory, self.counter))
                self.counter += 1


class PlotFrontToFileObserver(Observer):
    def __init__(self, output_directory: str, step: int = 100, **kwargs) -> None:
        """Plot and save Pareto front approximations into files.

        :param output_directory: Output directory.
        """
        self.directory = output_directory
        self.plot_front = Plot(title="Pareto front approximation", **kwargs)
        self.last_front = []
        self.fronts = []
        self.counter = 0
        self.step = step

        if Path(self.directory).is_dir():
            LOGGER.warning("Directory {} exists. Removing contents.".format(self.directory))
            for file in os.listdir(self.directory):
                os.remove("{0}/{1}".format(self.directory, file))
        else:
            LOGGER.warning("Directory {} does not exist. Creating it.".format(self.directory))
            Path(self.directory).mkdir(parents=True)

    def update(self, *args, **kwargs):
        problem = kwargs["PROBLEM"]
        solutions = kwargs["SOLUTIONS"]
        evaluations = kwargs["EVALUATIONS"]

        if solutions:
            if (evaluations % self.step) == 0:
                if isinstance(problem, DynamicProblem):
                    termination_criterion_is_met = kwargs.get("TERMINATION_CRITERIA_IS_MET", None)

                    if termination_criterion_is_met:
                        if self.counter > 0:
                            igd = InvertedGenerationalDistance(np.array([s.objectives for s in self.last_front]))
                            igd_value = igd.compute(np.array([s.objectives for s in solutions]))
                        else:
                            igd_value = 1

                        if igd_value > 0.005:
                            self.fronts += solutions
                            self.plot_front.plot(
                                [self.fronts],
                                label=problem.get_name(),
                                filename=f"{self.directory}/front-{evaluations}",
                            )
                        self.counter += 1
                        self.last_front = solutions
                else:
                    self.plot_front.plot(
                        [solutions],
                        label=f"{evaluations} evaluations",
                        filename=f"{self.directory}/front-{evaluations}",
                    )
                    self.counter += 1


class VisualizerObserver(Observer):
    def __init__(
        self, reference_front: List[S] = None, reference_point: list = None, display_frequency: int = 1
    ) -> None:
        self.figure = None
        self.display_frequency = display_frequency

        self.reference_point = reference_point
        self.reference_front = reference_front

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]

        if solutions:
            if self.figure is None:
                self.figure = StreamingPlot(reference_point=self.reference_point, reference_front=self.reference_front)
                self.figure.plot(solutions)

            if (evaluations % self.display_frequency) == 0:
                # check if reference point has changed
                reference_point = kwargs.get("REFERENCE_POINT", None)

                if reference_point:
                    self.reference_point = reference_point
                    self.figure.update(solutions, reference_point)
                else:
                    self.figure.update(solutions)

                self.figure.ax.set_title("Eval: {}".format(evaluations), fontsize=13)

class SaveResultsObserver(Observer):
    
    def __init__(self, frequency: int, output_directory: str = "intermediate_results"):
        """
        :param frequency: Number of evaluations between saves.
        :param output_directory: Folder where the files will be saved.
        """
        self.frequency = frequency
        self.output_directory = output_directory
        
        # Create the directory if it does not exist
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

    def update(self, *args, **kwargs):
        # Retrieve the current evaluation count and solution list
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]
        problem = kwargs["PROBLEM"]
        
        # Check if the current evaluation count is a multiple of the frequency
        if evaluations % self.frequency == 0:
            problem_name = problem.name()
            
            print(f"--> Saving intermediate results for {problem_name} at evaluation {evaluations}...")
            
            # Create filenames including problem name and evaluation number
            fun_filename = f"{self.output_directory}/FUN.{problem_name}.{evaluations}"
            var_filename = f"{self.output_directory}/VAR.{problem_name}.{evaluations}"
            
            # Save the files
            print_function_values_to_file(solutions, fun_filename)
            print_variables_to_file(solutions, var_filename)
            
class LivePlotObserver(Observer):
    
    def __init__(self, frequency: int, max_evaluations: int, input_directory: str = "intermediate_results"):
        """
        :param frequency: Number of evaluations between plot updates.
        :param max_evaluations: Total expected evaluations (to trigger the final summary plot).
        :param input_directory: Folder where the FUN files are located (must match the Saver directory).
        """
        self.frequency = frequency
        self.max_evaluations = max_evaluations
        self.input_directory = input_directory
        self.history = []

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        problem = kwargs["PROBLEM"]
        
        if evaluations % self.frequency == 0:
            problem_name = problem.name()
            
            # Construct the filename of the FUN file we want to load
            fun_filename = f"{self.input_directory}/FUN.{problem_name}.{evaluations}"
            
            # Read data from the file
            x_values = []
            y_values = []
            
            
            if os.path.exists(fun_filename):
                try:
                    with open(fun_filename, 'r') as file:
                        for line in file:
                            # Split the line by whitespace (each column is an objective)
                            values = line.strip().split()
                            if len(values) >= 2:
                                x_values.append(float(values[0]))
                                y_values.append(float(values[1]))
                                
                    # If file was empty, skip
                    if not x_values:
                        return
                    
                    self.history.append({
                        "eval": evaluations,
                        "x": x_values,
                        "y": y_values
                    })
                    
                    
                    # Create a new figure
                    fig, ax = plt.subplots(figsize=(6, 4))
                    
                    # Plot the data read from disk
                    ax.scatter(x_values, y_values, c='blue', alpha=0.6)
                    ax.set_title(f"Pareto Front for evaluation: {evaluations}")
                    ax.set_xlabel("Objective 1")
                    ax.set_ylabel("Objective 2")
                    ax.grid(True)
                    ax.legend()
                    
                    # Show in console
                    plt.show()
                    
                    if evaluations >= self.max_evaluations:
                        self.plot_summary(problem_name)
                    
                except Exception as e:
                    print(f"Error reading file {fun_filename}: {e}")
            else:
                print(f"Warning: File {fun_filename} not found yet.")
        
    def plot_summary(self, problem_name):
        """Generates a combined plot showing the evolution of all stored fronts."""
        print("--> Generating optimization summary plot...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Generate a color gradient (from light to dark) to visualize progress
        total_plots = len(self.history)
        if total_plots > 0:
            colors = plt.cm.viridis(np.linspace(0, 1, total_plots))
            
            for i, data in enumerate(self.history):
                ax.scatter(
                    data["x"], 
                    data["y"], 
                    label=f"Eval {data['eval']}", 
                    color=colors[i], 
                    alpha=0.7,
                    s=20 
                )
                
            ax.set_title(f"Optimization Progress: {problem_name}")
            ax.set_xlabel("Objective 1")
            ax.set_ylabel("Objective 2")
            ax.grid(True)
            ax.legend()
            
            plt.show()
            
            
class AntenaGraficasObserver(Observer):
    def __init__(self, frequency: int, output_directory: str = "Plots_CST"):
        """
        :param frequency: Cada cuántas evaluaciones queremos generar la gráfica.
        :param output_directory: Carpeta donde se guardarán las imágenes .png.
        """
        self.frequency = frequency
        self.output_directory = output_directory
        
        # Crear el directorio si no existe
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

    def update(self, *args, **kwargs):
        evaluaciones = kwargs["EVALUATIONS"]
        soluciones = kwargs["SOLUTIONS"]

        if evaluaciones % self.frequency == 0 and soluciones:
            
            # Cogemos la mejor antena de la generación actual
            mejor_antena = soluciones[0]
            
            
            # Rescatamos los arrays que escondimos en el evaluate()
            frecuencia = mejor_antena.attributes.get('frecuencia')
            s11_imag = mejor_antena.attributes.get('s11_imag')
            s11_real = mejor_antena.attributes.get('s11_real')
            s11_db = mejor_antena.attributes.get('s11_db')
            frec_farfield = mejor_antena.attributes.get('frec_farfield')
            axial_ratios = mejor_antena.attributes.get('axial_ratios')
            
            if frecuencia is None:
                return # Si la antena dio error en CST y no hay datos, saltamos

            print(f"--> [Observer] Guardando 4 gráficas PNG de la evaluación {evaluaciones}...")

            # S11 Imaginario
            plt.figure()
            plt.plot(frecuencia, s11_imag, color='blue')
            plt.title(f'S1,1 Imaginario - Eval {evaluaciones}')
            plt.grid(True)
            plt.savefig(f"{self.output_directory}/S11_Imag_eval_{evaluaciones}.png")
            plt.close() 
            
            # S11 Real
            plt.figure()
            plt.plot(frecuencia, s11_real, color='red')
            plt.title(f'S1,1 Real - Eval {evaluaciones}')
            plt.grid(True)
            plt.savefig(f"{self.output_directory}/S11_Real_eval_{evaluaciones}.png")
            plt.close()

            # S11 dB
            plt.figure()
            plt.plot(frecuencia, s11_db, color='red')
            plt.title(f'S1,1 (dB) - Eval {evaluaciones}')
            plt.grid(True)
            plt.savefig(f"{self.output_directory}/S11_dB_eval_{evaluaciones}.png")
            plt.close()
            
            # Axial Ratio
            plt.figure()
            plt.plot(frec_farfield, axial_ratios, color='green', marker='o')
            plt.title(f'Axial Ratio vs Frec - Eval {evaluaciones}')
            plt.grid(True)
            plt.savefig(f"{self.output_directory}/Axial_Ratio_eval_{evaluaciones}.png")
            plt.close()
            
class GuardarResultadosFisicosObserver(Observer):
    def __init__(self, frequency: int, steps_del_problema: list, output_directory: str = "intermediate_results"):
        """
        Guarda los archivos FUN y VAR, pero traduciendo los enteros a 
        las dimensiones físicas reales usando los steps del problema.
        """
        self.frequency = frequency
        self.steps = steps_del_problema
        self.output_directory = output_directory
        
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]
        problem = kwargs["PROBLEM"]

        if evaluations % self.frequency == 0 and solutions:
            problem_name = problem.name()
            print(f"--> [Observer] Guardando resultados físicos para {problem_name} en evaluación {evaluations}...")
            
            fun_filename = f"{self.output_directory}/FUN.{problem_name}.{evaluations}"
            print_function_values_to_file(solutions, fun_filename)

            # Guardar archivo VAR con las medidas FÍSICAS
            var_filename = f"{self.output_directory}/VAR.{problem_name}.{evaluations}"
            
            with open(var_filename, "w") as f:
                for s in solutions: # Recorremos la población antena por antena
                    # Traducimos de entero a físico para esta antena concreta
                    vars_fisicas = [var * step for var, step in zip(s.variables, self.steps)]
                    
                    # Lo convertimos a texto con 3 decimales, separado por tabulaciones, y escribimos
                    linea = "\t".join([f"{v:.3f}" for v in vars_fisicas])
                    f.write(linea + "\n")
                