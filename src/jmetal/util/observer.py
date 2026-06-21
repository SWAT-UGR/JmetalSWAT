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

# --- CONFIGURACIÓN GLOBAL ÓPTIMA PARA LATEX ---
plt.rcParams.update({
    'font.size': 14,          # Letra base bastante grande
    'axes.titlesize': 16,     # Títulos que destaquen claramente
    'axes.labelsize': 16,     # Ejes X e Y legibles
    'xtick.labelsize': 14,    # Números del eje X
    'ytick.labelsize': 14,    # Números del eje Y
    'legend.fontsize': 14,    # Leyenda que no desaparezca al encoger
    'figure.titlesize': 18    # Títulos superiores (si los hay)
})
# ----------------------------------------------

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
    """
    Observador de persistencia intermedia de datos. Implementa el patrón estructural Observer
    para interceptar los vectores de soluciones del algoritmo y exportar de forma periódica
    los archivos de funciones objetivo (FUN) y variables de decisión (VAR) al disco duro.
    """
    def __init__(self, frequency: int, algorithm, output_directory: str = "intermediate_results"):
        """
        :param frequency: Frecuencia de muestreo o número de evaluaciones entre ciclos de guardado.
        :param algorithm: Instancia del motor algorítmico en ejecución (orquestador evolutivo o cinemático).
        :param output_directory: Descriptor de ruta del sistema de archivos destinado al almacenamiento.
        """
        self.frequency = frequency
        self.algorithm = algorithm
        self.output_directory = output_directory
        
        # Validación y creación automatizada del directorio destino en el sistema de archivos local
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

    def update(self, *args, **kwargs):
        # Desempaquetado analítico de los parámetros posicionales y nominales del entorno jMetalPy
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]
        problem = kwargs["PROBLEM"]
        
        # Verificación del criterio de muestreo periódico conforme al intervalo indexado
        if evaluations % self.frequency == 0:
            problem_name = problem.name()
            
            print(f"--> Guardando resultados intermedios para {problem_name} en la evaluación {evaluations}...")
            
            # Clonación superficial (shallow copy) de las soluciones para mitigar colisiones en la RAM del algoritmo
            solutions_to_save = solutions.copy()
            
            # -------------------------------------------------------------------------
            # MECANISMO DE CONTROL PARA COHESIÓN POBLACIONAL (Soporte SMPSO / NSGA-II)
            # -------------------------------------------------------------------------
            # Si la metaheurística almacena un repositorio externo de líderes no dominados (ej. SMPSO),
            # se inyectan de forma prioritaria para asegurar la persistencia en estrategias de re-arranque.
            if hasattr(self.algorithm, 'leaders'):
                leaders = self.algorithm.leaders.solution_list
                
                # Inserción posicional selectiva de los especímenes del Frente de Pareto real,
                # preservando la porción restante del enjambre para salvaguardar la diversidad estocástica.
                for i in range(min(len(leaders), len(solutions_to_save))):
                    solutions_to_save[i] = leaders[i]
            
            # Generación predictiva de la nomenclatura de archivos indexados por metadatos y conteo evolutivo
            fun_filename = f"{self.output_directory}/FUN.{problem_name}.{evaluations}"
            var_filename = f"{self.output_directory}/VAR.{problem_name}.{evaluations}"
            
            # Volcado matricial síncrono al disco utilizando las rutinas nativas de persistencia
            print_function_values_to_file(solutions_to_save, fun_filename)
            print_variables_to_file(solutions_to_save, var_filename)


class LivePlotObserver:
    """
    Observador de caracterización gráfica bidimensional. Gestiona el procesamiento analítico,
    la acumulación histórica y la renderización dinámica de la frontera no dominada de Pareto
    durante el transcurso de la optimización estocástica.
    """
    def __init__(self, frequency: int, max_evaluations: int, algorithm=None, output_directory: str = "Pareto_Plots", 
                 objective_labels: list = ["Objetivo 1", "Objetivo 2"], penalty_threshold: float = 1000.0):
        """
        :param max_evaluations: Cota superior del presupuesto computacional de la campaña.
        :param objective_labels: Descriptores nominales asociados a los ejes coordenados del espacio de objetivos.
        :param penalty_threshold: Umbral métrico de exclusión para mitigar la distorsión de fenotipos penalizados.
        """
        self.frequency = frequency
        self.max_evaluations = max_evaluations
        self.algorithm = algorithm
        self.output_directory = output_directory
        self.objective_labels = objective_labels
        self.penalty_threshold = penalty_threshold
        self.history = []
        
        # Inicialización del entorno jerárquico de carpetas para las representaciones vectoriales
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

    def update(self, *args, **kwargs):
        evaluations = kwargs["EVALUATIONS"]
        problem = kwargs["PROBLEM"]
        
        # Extracción síncrona de los vectores de soluciones residentes en el búfer de memoria
        solutions = kwargs.get("SOLUTIONS")
        
        if not solutions:
            return

        if evaluations % self.frequency == 0:
            problem_name = problem.name()
            
            # Discriminación del origen de datos: archivo externo (SMPSO) frente a filtrado de dominancia local (NSGA-II)
            if self.algorithm and hasattr(self.algorithm, 'leaders'):
                pareto_front = self.algorithm.leaders.solution_list
            else:
                pareto_front = get_non_dominated_solutions(solutions)
            
            x_values = []
            y_values = []
            
            # Descomposición de la frontera no dominada y aislamiento de los vectores objetivo
            for sol in pareto_front:
                val_x = sol.objectives[0]
                val_y = sol.objectives[1]
                
                # Filtro de acotación analítica: exclusión selectiva de puntos aberrantes por penalización destructiva
                if val_x < self.penalty_threshold and val_y < self.penalty_threshold:
                    x_values.append(val_x)
                    y_values.append(val_y)
                    
            if not x_values:
                return
        
            # Persistencia cronológica en el registro histórico de convergencia evolutiva
            self.history.append({
                "eval": evaluations,
                "x": x_values, 
                "y": y_values
            })
            
            # -------------------------------------------------------------------------
            # 1. CONSTRUCCIÓN DE LA REPRESENTACIÓN GRÁFICA ACUMULADA (Summary Plot)
            # -------------------------------------------------------------------------
            # Ejecución prioritaria del gráfico global para derivar las envolventes ortogonales normativas
            xlims, ylims = self.plot_summary(problem_name, evaluations)
            
            # -------------------------------------------------------------------------
            # 2. CONSTRUCCIÓN DE LA GRÁFICA DE DISPERSIÓN INDIVIDUAL PUNTUAL
            # -------------------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Renderizado del diagrama de dispersión aplicando el mapa cromático estandarizado de la campaña
            ax.scatter(x_values, y_values, c='yellow', alpha=0.8, edgecolors='black', label=f'Pareto Eval {evaluations}')
             
            ax.set_title(f"Frente de Pareto: Evaluación {evaluations}")
            ax.set_xlabel(self.objective_labels[0])
            ax.set_ylabel(self.objective_labels[1])
            
            # Sincronización escalar: homogeneización de las vistas individuales respecto al summary acumulado
            if xlims and ylims:
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)
                
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best')
            
            # Exportación vectorial a formato PDF de alta resolución espacial para la memoria
            plot_filename = f"{self.output_directory}/front_{evaluations}.pdf"
            plt.savefig(plot_filename, bbox_inches='tight')
            plt.close(fig) 
            
            print(f"--> Gráfica del Frente de Pareto individual guardada: {plot_filename}\n")
            
            
    def plot_summary(self, problem_name, current_eval):
        """
        Genera una composición gráfica multiespectral combinada, ilustrando la trayectoria espacial
        e histórica de convergencia de todos los frentes mapeados hasta la iteración contemporánea.
        """
        print(f"--> Generando gráfica resumen acumulada hasta la evaluación {current_eval}...")
        
        fig, ax = plt.subplots(figsize=(8, 5.5))
        
        total_plots = len(self.history)
        if total_plots > 0:
            # Inyección de una rampa cromática de tipo Viridis para denotar la evolución temporal
            colors = plt.cm.viridis(np.linspace(0, 1, total_plots))
            
            for i, data in enumerate(self.history):
                # Gradación adaptativa del canal alfa y tamaño para maximizar el contraste de la última población
                alpha_val = 0.4 if i < total_plots - 1 else 1.0
                size_val = 40 if i < total_plots - 1 else 80
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
                
            ax.set_title(f"Evolución del frente de Pareto: {problem_name}")
            ax.set_xlabel(self.objective_labels[0])
            ax.set_ylabel(self.objective_labels[1])
            ax.grid(True, linestyle='--', alpha=0.7)
            
            ax.legend(loc='upper right', framealpha=0.9, edgecolor='black')
            
            # Captura dinámica de las cotas de proyección euclídea computadas de forma automática por Matplotlib
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
            
            # Persistencia en formato vectorial del mapa acumulado de optimización
            summary_filename = f"{self.output_directory}/Summary_hasta_eval_{current_eval}.pdf"
            plt.savefig(summary_filename, bbox_inches='tight')
            plt.close(fig)
            
            print(f"--> Gráfica resumen acumulada guardada en: {summary_filename}")
            
            # Retorno de las tuplas dimensionales de escala para la sincronización inter-método
            return xlims, ylims
            
        return None, None
            

class AntenaGraficasObserver(Observer):
    """
    Observador de validación y caracterización radioeléctrica. Captura los perfiles analíticos 
    de campo cercano y lejano computados en la interfaz de CST y los transforma en diagramas 
    espectrales vectoriales para la evaluación explícita de la ingeniería del dispositivo.
    """
    def __init__(self, frequency: int, output_directory: str = "Plots_CST"):
        """
        :param frequency: Intervalo periódico o cadencia de evaluación para la actualización gráfica.
        :param output_directory: Repositorio físico en disco destinado a los reportes electromagnéticos.
        """
        self.frequency = frequency
        self.output_directory = output_directory
        
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

    def update(self, *args, **kwargs):
        evaluaciones = kwargs["EVALUATIONS"]
        soluciones = kwargs["SOLUTIONS"]

        if evaluaciones % self.frequency == 0 and soluciones:
            
            # Extracción de la solución de máxima aptitud o menor dominancia relativo a la población actual
            mejor_antena = soluciones[0]
            
            # Desencapsulado de matrices numéricas asociadas a los descriptores físicos de la antena
            frecuencia = mejor_antena.attributes.get('frecuencia')
            s11_imag = mejor_antena.attributes.get('s11_imag')
            s11_real = mejor_antena.attributes.get('s11_real')
            s11_db = mejor_antena.attributes.get('s11_db')
            frec_farfield = mejor_antena.attributes.get('frec_farfield')
            axial_ratios = mejor_antena.attributes.get('axial_ratios')
            
            # Interrupción condicional ante colapso numérico del resolvedor tridimensional u omisión de datos
            if frecuencia is None:
                return 

            print(f"--> [Observer] Guardando 4 gráficas PDF de la evaluación {evaluaciones}...")

            # Cuantización y discretización equidistante de las marcas métricas del eje espectral (X)
            f_min, f_max = min(frecuencia), max(frecuencia)
            x_ticks = np.linspace(f_min, f_max, 5)

            # --- S11 Imaginario ---
            plt.figure(figsize=(7, 5))
            plt.plot(frecuencia, s11_imag, color='blue')
            plt.title(f'S1,1 Imaginario - Eval {evaluaciones}')
            plt.xlabel('Frecuencia (GHz)')
            plt.ylabel('Amplitud')
            plt.xlim(f_min, f_max)
            plt.xticks(x_ticks)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"{self.output_directory}/S11_Imag_eval_{evaluaciones}.pdf", bbox_inches='tight')
            plt.close() 
            
            # --- S11 Real ---
            plt.figure(figsize=(7, 5))
            plt.plot(frecuencia, s11_real, color='red')
            plt.title(f'S1,1 Real - Eval {evaluaciones}')
            plt.xlabel('Frecuencia (GHz)')
            plt.ylabel('Amplitud')
            plt.xlim(f_min, f_max)
            plt.xticks(x_ticks)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"{self.output_directory}/S11_Real_eval_{evaluaciones}.pdf", bbox_inches='tight')
            plt.close()

            # --- S11 dB (Pérdidas de retorno / Coeficiente de reflexión) ---
            plt.figure(figsize=(7, 5))
            plt.plot(frecuencia, s11_db, color='red')
            plt.title(f'S1,1 (dB) - Eval {evaluaciones}')
            plt.xlabel('Frecuencia (GHz)')
            plt.ylabel('Magnitud (dB)')
            plt.xlim(f_min, f_max)
            plt.xticks(x_ticks)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"{self.output_directory}/S11_dB_eval_{evaluaciones}.pdf", bbox_inches='tight')
            plt.close()
            
            # --- Axial Ratio (Análisis de pureza de polarización circular) ---
            plt.figure(figsize=(7, 5))
            plt.plot(frec_farfield, axial_ratios, color='green', marker='o')
            plt.title(f'Axial Ratio vs Frec - Eval {evaluaciones}')
            plt.xlabel('Frecuencia (GHz)')
            plt.ylabel('Axial Ratio (dB)')
            plt.xlim(f_min, f_max)
            plt.xticks(x_ticks)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"{self.output_directory}/Axial_Ratio_eval_{evaluaciones}.pdf", bbox_inches='tight')
            plt.close()
            

class GuardarResultadosFisicosObserver(Observer):
    """
    Observador agnóstico de traducción fenotípica. Intercepta los genotipos mapeados 
    por el motor bioinspirado (vectores discretos enteros) y los decodifica linealmente 
    a dimensiones del mundo físico real (mm y grados) para su análisis normativo de fabricación.
    """
    def __init__(self, frequency: int, algorithm, steps_del_problema: list, output_directory: str = "intermediate_results"):
        """
        :param steps_del_problema: Array de granularidad geométrica para el escalado inverso de variables.
        """
        self.frequency = frequency
        self.algorithm = algorithm  
        self.steps = steps_del_problema
        self.output_directory = output_directory
        
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory, exist_ok=True)

    def update(self, *args, **kwargs):
        evaluaciones = kwargs["EVALUATIONS"]
        problem = kwargs["PROBLEM"]

        # Extracción segura de la población cromosómica real mediante introspección de atributos
        poblacion_real = getattr(self.algorithm, 'solutions', None)

        if evaluaciones % self.frequency == 0 and poblacion_real:
            problem_name = problem.name()
            print(f"--> [Observer] Guardando resultados físicos para {problem_name} en evaluación {evaluaciones}...")
            
            # Duplicación superficial de las estructuras para salvaguardar el hilo de ejecución principal de jMetalPy
            solutions_to_save = poblacion_real.copy()
            
            if hasattr(self.algorithm, 'leaders'):
                leaders = self.algorithm.leaders.solution_list
                # Preservación de diversidad: acotación del reemplazo de líderes a la mitad del censo poblacional
                max_reemplazos = min(len(leaders), len(solutions_to_save) // 2)
                for i in range(max_reemplazos):
                    solutions_to_save[i] = leaders[i]
            
            # Sincronización y vuelco del espacio de objetivos (Fichero FUN)
            fun_filename = f"{self.output_directory}/FUN.{problem_name}.{evaluaciones}"
            print_function_values_to_file(solutions_to_save, fun_filename)

            # Transformación lineal y formateo geométrico del espacio de búsqueda (Fichero VAR)
            var_filename = f"{self.output_directory}/VAR.{problem_name}.{evaluaciones}"
            
            with open(var_filename, "w") as f:
                for s in solutions_to_save: 
                    # Mapeo dimensional: decodificación del genotipo discretizado al fenotipo físico continuo
                    vars_fisicas = [var * step for var, step in zip(s.variables, self.steps)]
                    
                    # Formateo de cadenas tabuladas con precisión flotante normalizada a 3 decimales
                    linea = "\t".join([f"{v:.3f}" for v in vars_fisicas])
                    f.write(linea + "\n")