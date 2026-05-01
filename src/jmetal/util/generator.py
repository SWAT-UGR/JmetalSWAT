import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from jmetal.core.problem import Problem
from jmetal.core.solution import Solution

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
            # Convertimos a enteros porque tu problema es Integer
            nueva_sol.variables = [int(round(v / s)) for v, s in zip(fila, problem.steps)]
            
            soluciones_para_inyectar.append(nueva_sol)
            
        # Inicializar la clase padre (el inyector de JMetalPy) con la lista
        super().__init__(solutions=soluciones_para_inyectar)
        
        print(f"--> [InjectorDesdeArchivo] {len(soluciones_para_inyectar)} antenas cargadas con éxito.")
