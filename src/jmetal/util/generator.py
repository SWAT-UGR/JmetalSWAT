import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from jmetal.core.problem import Problem
from jmetal.core.solution import Solution, FloatSolution, IntegerSolution, BinarySolution, CompositeSolution

R = TypeVar("R")

"""
.. module:: generator
   :platform: Unix, Windows
   :synopsis: Population generators implementation.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Generator(Generic[R], ABC):
    @abstractmethod
    def new(self, problem: Problem) -> R:
        pass


class RandomGenerator(Generator):
    def new(self, problem: Problem):
        return problem.create_solution()


class InjectorGenerator(Generator):
    def __init__(self, solutions: List[Solution]):
        super(InjectorGenerator, self).__init__()
        self.population = copy.deepcopy(solutions)

    def new(self, problem: Problem):
        if len(self.population) > 0:
            # If we have more solutions to inject, return one from the list
            return self.population.pop()
        else:
            # Otherwise generate a new solution
            solution = problem.create_solution()

        return solution

class InjectorFromFile(InjectorGenerator):
    """
    Generador reutilizable que lee un archivo 
    y lo inyecta como población inicial.
    Soporta problemas continuos, discretos, binarios y 
    adaptaciones personalizadas con factores de escala (steps).
    """
    def __init__(self, ruta_archivo: str, problem):
        # Leer los datos numéricos del archivo
        try:
            datos_viejos = np.loadtxt(ruta_archivo)
        except Exception as e:
            raise FileNotFoundError(f"Error al leer el archivo {ruta_archivo}: {e}")
        
        # Fabricar la lista de soluciones
        soluciones_para_inyectar = []
        
        # Manejo por si el archivo solo tiene 1 línea (1 solución)
        if datos_viejos.ndim == 1:
            datos_viejos = datos_viejos.reshape(1, -1)
            
        for fila in datos_viejos:
            nueva_sol = problem.create_solution()
            
            # --- LÓGICA GENERALIZADA MULTI-TIPO ---
            
            # Caso 1: Problema personalizado con escalado físico (Tu TFG)
            if hasattr(problem, 'steps') and problem.steps is not None:
                nueva_sol.variables = [int(round(v / s)) for v, s in zip(fila, problem.steps)]
                
            # Caso 2: Problema de Enteros Estándar (Integer)
            elif isinstance(nueva_sol, IntegerSolution):
                nueva_sol.variables = [int(round(v)) for v in fila]
                
            # Caso 3: Problema Continuo Estándar (Float)
            elif isinstance(nueva_sol, FloatSolution):
                nueva_sol.variables = [float(v) for v in fila]
                
            # Caso 4: Problema Binario (Convierte 1.0 y 0.0 a True/False)
            elif isinstance(nueva_sol, BinarySolution):
                # jMetalPy guarda los genes binarios como booleanos o listas de bits
                nueva_sol.variables = [bool(int(round(v))) for v in fila]
                
            # Caso 5: Problemas Compuestos (CompositeSolution)
            elif isinstance(nueva_sol, CompositeSolution):
                # Un Composite contiene soluciones anidadas. Reconstruirlo desde un txt 
                # plano requiere conocer el esquema exacto de las sub-soluciones.
                raise NotImplementedError("La inyección de un CompositeSolution desde un archivo plano requiere un mapeo estructural específico.")
                
            # Fallback de seguridad
            else:
                nueva_sol.variables = list(fila)
            # ---------------------------------------
            
            soluciones_para_inyectar.append(nueva_sol)
            
        # Inicializar la clase padre (el inyector de JMetalPy) con la lista
        super().__init__(solutions=soluciones_para_inyectar)
        
        print(f"--> [InjectorDesdeArchivo] {len(soluciones_para_inyectar)} soluciones cargadas con éxito.")