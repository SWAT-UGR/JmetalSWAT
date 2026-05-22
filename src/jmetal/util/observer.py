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
from jmetal.util.solution import get_non_dominated_solutions
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
    
    def __init__(self, frequency: int, algorithm, output_directory: str = "intermediate_results"):
        """
        Observador encargado de exportar periódicamente las variables y objetivos a disco.

        :param frequency: Número de evaluaciones entre cada guardado.
        :param algorithm: Instancia del algoritmo en ejecución (para extraer líderes históricos).
        :param output_directory: Directorio destino para los archivos generados.
        """
        self.frequency = frequency
        self.algorithm = algorithm
        self.output_directory = output_directory
        
        # Crear el directorio de salida si no existe
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

    def update(self, *args, **kwargs):
        # Extraer la información empaquetada en las notificaciones del algoritmo
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]
        problem = kwargs["PROBLEM"]
        
        # Comprobar si corresponde guardar los datos según la frecuencia establecida
        if evaluations % self.frequency == 0:
            problem_name = problem.name()
            
            print(f"--> Guardando resultados intermedios para {problem_name} en la evaluación {evaluations}...")
            
            # Hacemos una copia superficial de las soluciones para no alterar la RAM del algoritmo
            solutions_to_save = solutions.copy()
            
            # ESTRATEGIA ANTI-AMNESIA (Soporte dinámico SMPSO / NSGA-II)
            # Si el algoritmo posee un archivo externo de líderes (ej. SMPSO), no podemos perderlos.
            if hasattr(self.algorithm, 'leaders'):
                leaders = self.algorithm.leaders.solution_list
                # Sobrescribimos las primeras posiciones del enjambre con los líderes históricos.
                # Así el Warm-Start arranca exactamente desde el Frente de Pareto real, 
                # manteniendo el resto de partículas del enjambre para no perder diversidad.
                for i in range(min(len(leaders), len(solutions_to_save))):
                    solutions_to_save[i] = leaders[i]
            
            # 3. Generar las rutas de archivo indexadas por nombre y evaluación
            fun_filename = f"{self.output_directory}/FUN.{problem_name}.{evaluations}"
            var_filename = f"{self.output_directory}/VAR.{problem_name}.{evaluations}"
            
            # Volcar los datos matriciales al disco usando la lista corregida
            print_function_values_to_file(solutions_to_save, fun_filename)
            print_variables_to_file(solutions_to_save, var_filename)
            
class LivePlotObserver(Observer):
    
    def __init__(self, frequency: int, max_evaluations: int, output_directory: str = "Pareto_Plots", 
                 objective_labels: list = ["Objetivo 1", "Objetivo 2"], penalty_threshold: float = 1000.0):
        """
        Observador para la renderización dinámica y guardado del frente de Pareto bidimensional.

        :param frequency: Número de evaluaciones entre cada actualización gráfica.
        :param max_evaluations: Número máximo de evaluaciones esperadas (para escalado/historial).
        :param output_directory: Directorio destino para guardar las imágenes PNG.
        :param objective_labels: Lista con los nombres de los ejes X e Y para la gráfica.
        :param penalty_threshold: Umbral máximo para filtrar soluciones con penalizaciones (ej. CST timeout).
        """
        self.frequency = frequency
        self.max_evaluations = max_evaluations
        self.output_directory = output_directory
        self.objective_labels = objective_labels
        self.penalty_threshold = penalty_threshold
        self.history = []
        
        # Crear el directorio para las gráficas si no existe
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        problem = kwargs["PROBLEM"]
        
        # Extraer las soluciones directamente de la memoria RAM
        solutions = kwargs.get("SOLUTIONS")
        
        if not solutions:
            return

        if evaluations % self.frequency == 0:
            problem_name = problem.name()
            
            # Extraer matemáticamente SOLO el conjunto de soluciones no dominadas (Frente de Pareto)
            pareto_front = get_non_dominated_solutions(solutions)
            
            x_values = []
            y_values = []
            
            # Recorrer el frente de Pareto y aislar los objetivos
            for sol in pareto_front:
                val_x = sol.objectives[0]
                val_y = sol.objectives[1]
                
                # Filtro de viabilidad: descartar valores atípicos o penalizaciones del simulador
                if val_x < self.penalty_threshold and val_y < self.penalty_threshold:
                    x_values.append(val_x)
                    y_values.append(val_y)
                    
            if not x_values:
                return
        
            # Almacenar los datos filtrados en el historial para la gráfica acumulada
            self.history.append({
                "eval": evaluations,
                "x": x_values, 
                "y": y_values
            })
            
            # Generar y guardar la gráfica individual de la generación actual
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Renderizar el gráfico de dispersión con el frente actual
            ax.scatter(x_values, y_values, c='red', alpha=0.8, edgecolors='black', label=f'Pareto Eval {evaluations}')
             
            ax.set_title(f"Frente de Pareto Real: Evaluación {evaluations}")
            ax.set_xlabel(self.objective_labels[0])
            ax.set_ylabel(self.objective_labels[1])
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            plot_filename = f"{self.output_directory}/front_{evaluations}.png"
            plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
            plt.close(fig) 
            
            print(f"--> Gráfica del Frente de Pareto guardada: {plot_filename}")
            
            # Invocar la generación de la gráfica resumen acumulada
            self.plot_summary(problem_name, evaluations)
            
    def plot_summary(self, problem_name, current_eval):
        """Genera un gráfico combinado mostrando la evolución de todos los frentes almacenados."""
        print(f"--> Generando gráfica resumen acumulada hasta la evaluación {current_eval}...")
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        total_plots = len(self.history)
        if total_plots > 0:
            colors = plt.cm.viridis(np.linspace(0, 1, total_plots))
            
            for i, data in enumerate(self.history):
                # Destacar visualmente las iteraciones más recientes (mayor opacidad y tamaño)
                alpha_val = 0.4 if i < total_plots - 1 else 1.0
                size_val = 20 if i < total_plots - 1 else 50
                edge_color = 'black' if i == total_plots - 1 else 'none'
                
                ax.scatter(
                    data["x"], 
                    data["y"], 
                    label=f"Eval {data['eval']}", 
                    color=colors[i], 
                    alpha=alpha_val,
                    edgecolors=edge_color,
                    s=size_val 
                )
                
            ax.set_title(f"Evolución del Frente de Pareto (Hasta Eval {current_eval}): {problem_name}")
            ax.set_xlabel(self.objective_labels[0])
            ax.set_ylabel(self.objective_labels[1])
            ax.grid(True, linestyle='--', alpha=0.7)
            
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            summary_filename = f"{self.output_directory}/Summary_hasta_eval_{current_eval}.png"
            plt.savefig(summary_filename, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            print(f"--> Gráfica resumen acumulada guardada en: {summary_filename}")
            
            
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
    
    def __init__(self, frequency: int, algorithm, steps_del_problema: list, output_directory: str = "intermediate_results"):
        """
        Guarda los archivos FUN y VAR, traduciendo los enteros a las dimensiones 
        físicas reales e inyectando los líderes históricos si el algoritmo tiene memoria (SMPSO).
        
        :param frequency: Número de evaluaciones entre cada guardado.
        :param algorithm: Instancia del algoritmo en ejecución (para extraer líderes históricos).
        :param steps_del_problema: Factores de escala multiplicativos para el mapeo geométrico.
        :param output_directory: Directorio destino para los archivos generados.
        """
        self.frequency = frequency
        self.algorithm = algorithm  
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
            
            # Crear una copia de la población actual para proteger el flujo de ejecución principal
            solutions_to_save = solutions.copy()
            
            # Estrategia anti-amnesia para SMPSO
            # Si el algoritmo opera con un archivo externo de líderes, los incrustamos 
            # en las primeras posiciones para que no se pierda el frente histórico en el reinicio.
            if hasattr(self.algorithm, 'leaders'):
                leaders = self.algorithm.leaders.solution_list
                for i in range(min(len(leaders), len(solutions_to_save))):
                    solutions_to_save[i] = leaders[i]
            
            # Guardar el archivo de funciones objetivo (FUN) con el frente corregido
            fun_filename = f"{self.output_directory}/FUN.{problem_name}.{evaluations}"
            print_function_values_to_file(solutions_to_save, fun_filename)

            # Guardar el archivo de variables (VAR) traduciendo a medidas FÍSICAS reales
            var_filename = f"{self.output_directory}/VAR.{problem_name}.{evaluations}"
            
            with open(var_filename, "w") as f:
                for s in solutions_to_save:  # Recorremos la lista corregida (sincronizada con FUN)
                    # Traducimos de entero a dimensión física real (mm) para cada variable
                    vars_fisicas = [var * step for var, step in zip(s.variables, self.steps)]
                    
                    # Convertimos a texto con 3 decimales, separado por tabulaciones, y escribimos en disco
                    linea = "\t".join([f"{v:.3f}" for v in vars_fisicas])
                    f.write(linea + "\n")