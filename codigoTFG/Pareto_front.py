# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:13:54 2026

@author: jmate
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# Ruta a la carpeta donde tienes guardados tus archivos FUN
carpeta_resultados = r"C:\Users\jmate\Desktop\teleco\TFG\intermediate_results"

# Buscamos todos los archivos FUN
archivos_fun = glob.glob(os.path.join(carpeta_resultados, "FUN.*"))
def extraer_numero(ruta):
    """Extrae el número del final del archivo (ej: FUN.Problema.100 -> 100)"""
    try:
        return int(ruta.split('.')[-1])
    except ValueError:
        return 0
    
archivos_fun.sort(key=extraer_numero)
    
plt.figure(figsize=(10, 7))
colores = plt.cm.viridis(np.linspace(0, 1, len(archivos_fun)))

puntos_validos_totales = 0

for i, archivo in enumerate(archivos_fun):
    try:
        datos = np.loadtxt(archivo)
        eval_num = extraer_numero(archivo)
        
        # Si solo hay una fila, lo convertimos a matriz 2D para no romper el código
        if datos.ndim == 1:
            datos = datos.reshape(1, -1)
            
        # Filtramos las filas que son "castigos" (valores menores a 1000)
        datos_limpios = datos[(datos[:, 0] < 1000) & (datos[:, 1] < 1000)]
        
        if len(datos_limpios) > 0:
            plt.scatter(
                datos_limpios[:, 0], 
                datos_limpios[:, 1], 
                color=colores[i], 
                alpha=0.7, 
                s=25,
                label=f"Eval {eval_num}"
            )
            puntos_validos_totales += len(datos_limpios)
            
    except Exception as e:
        pass # Ignoramos archivos vacíos o corruptos

plt.title(f"Frente de Pareto Limpio (Puntos válidos: {puntos_validos_totales})")
plt.xlabel("S11 (dB) [Minimizar]")
plt.ylabel("Axial Ratio [Minimizar]")
plt.grid(True, linestyle='--', alpha=0.7)

plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')

plt.tight_layout()
plt.show()