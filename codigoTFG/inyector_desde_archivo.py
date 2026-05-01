# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:41:53 2026

@author: jmate
"""

import numpy as np
from jmetal.util.generator import InjectorGenerator

class InjectorDesdeArchivo(InjectorGenerator):
    """
    Generador reutilizable que lee un archivo .tsv de JMetalPy 
    y lo inyecta como población inicial.
    """
    def __init__(self, ruta_archivo: str, problem):
        # 1. Leer los datos numéricos del archivo
        try:
            datos_viejos = np.loadtxt(ruta_archivo)
        except Exception as e:
            raise FileNotFoundError(f"Error al leer el archivo {ruta_archivo}: {e}")
        
        # 2. Fabricar la lista de soluciones
        soluciones_para_inyectar = []
        
        # Manejo por si el archivo solo tiene 1 línea (1 solución)
        if datos_viejos.ndim == 1:
            datos_viejos = datos_viejos.reshape(1, -1)
            
        for fila in datos_viejos:
            nueva_sol = problem.create_solution()
            # Convertimos a enteros porque tu problema es Integer
            nueva_sol.variables = [int(round(v / s)) for v, s in zip(fila, problem.steps)]
            
            soluciones_para_inyectar.append(nueva_sol)
            soluciones_para_inyectar.append(nueva_sol)
            
        # 3. Inicializar la clase padre (el inyector de JMetalPy) con la lista
        super().__init__(solutions=soluciones_para_inyectar)
        
        print(f"--> [InjectorDesdeArchivo] {len(soluciones_para_inyectar)} antenas cargadas con éxito.")