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
    Generador reutilizable que lee archivos históricos (VAR y FUN) 
    y los inyecta como población inicial ya evaluada.
    Soporta problemas continuos, discretos, binarios y 
    adaptaciones personalizadas con factores de escala (steps).
    """
    def __init__(self, ruta_var: str, problem, ruta_fun: str = None):
        # 1. Leer los datos numéricos de los genotipos (VAR)
        try:
            datos_var = np.loadtxt(ruta_var)
        except Exception as e:
            raise FileNotFoundError(f"Error al leer el archivo VAR {ruta_var}: {e}")
            
        # 2. Leer los fenotipos (FUN) si se proporciona el archivo
        datos_fun = None
        if ruta_fun:
            try:
                datos_fun = np.loadtxt(ruta_fun)
            except Exception as e:
                print(f"Advertencia: No se encontró el archivo FUN {ruta_fun}. Se forzará la re-evaluación en CST.")
        
        soluciones_para_inyectar = []
        
        # Manejo por si el archivo solo tiene 1 línea (1 solución)
        if datos_var.ndim == 1:
            datos_var = datos_var.reshape(1, -1)
        if datos_fun is not None and datos_fun.ndim == 1:
            datos_fun = datos_fun.reshape(1, -1)
            
        for i, fila in enumerate(datos_var):
            nueva_sol = problem.create_solution()
            
            # --- LÓGICA GENERALIZADA MULTI-TIPO ---
            
            # Caso 1: Problema personalizado con escalado físico
            if hasattr(problem, 'steps') and problem.steps is not None:
                nueva_sol.variables = [int(round(v / s)) for v, s in zip(fila, problem.steps)]
                
            # Caso 2: Problema de Enteros Estándar (Integer)
            elif isinstance(nueva_sol, IntegerSolution):
                nueva_sol.variables = [int(round(v)) for v in fila]
                
            # Caso 3: Problema Continuo Estándar (Float)
            elif isinstance(nueva_sol, FloatSolution):
                nueva_sol.variables = [float(v) for v in fila]
                
            # Caso 4: Problema Binario
            elif isinstance(nueva_sol, BinarySolution):
                nueva_sol.variables = [bool(int(round(v))) for v in fila]
                
            # Caso 5: Problemas Compuestos (CompositeSolution)
            elif isinstance(nueva_sol, CompositeSolution):
                raise NotImplementedError("La inyección de un CompositeSolution desde un archivo plano requiere un mapeo estructural específico.")
                
            # Fallback de seguridad
            else:
                nueva_sol.variables = list(fila)
            # ---------------------------------------
            
            # --- INYECCIÓN DEL FITNESS ---
            if datos_fun is not None:
                nueva_sol.objectives = list(datos_fun[i])
                # AÑADIR ESTA LÍNEA CRÍTICA:
                nueva_sol.attributes['viene_de_warm_start'] = True
                
            soluciones_para_inyectar.append(nueva_sol)
            
        # Inicializar la clase padre con la lista definitiva
        super().__init__(solutions=soluciones_para_inyectar)
        
        print(f"--> [InjectorFromFile] {len(soluciones_para_inyectar)} soluciones cargadas e inyectadas con éxito.")