# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 09:53:55 2026

@author: jmate
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(r"C:\Program Files\CST Studio Suite 2026\AMD64\python_cst_libraries")
import cst.interface
import time
import re

def redondear_al_paso(valor, paso):
    """
    Redondea un número al 'Step' más cercano.
    """
    return round(round(valor / paso) * paso, 3)

def generar_macro_cst(variables_jmetal, ruta_macro_original, ruta_macro_salida):
    """
    Lee la macro base, extrae las variables de JMetal, les aplica el 'Step' 
    y genera una nueva macro lista para CST.
    """
    # Indexar variables de JMetal y aplicar el Step del Excel
    # El orden (0 al 21) dependerá de cómo las declares en tu clase JMetalProblem
    valores_redondeados = {
        "L_metal":  redondear_al_paso(variables_jmetal[0], 0.1),
        "W_metal":  redondear_al_paso(variables_jmetal[1], 0.1),
        "L_GND":    redondear_al_paso(variables_jmetal[2], 1.0),
        "W_GND":    redondear_al_paso(variables_jmetal[3], 1.0),
        "Cx":       redondear_al_paso(variables_jmetal[4], 0.1),
        "Cy":       redondear_al_paso(variables_jmetal[5], 0.1),
        "Lslot1":   redondear_al_paso(variables_jmetal[6], 0.1),
        "Lslot2":   redondear_al_paso(variables_jmetal[7], 0.1),
        "Lslot3":   redondear_al_paso(variables_jmetal[8], 0.1),
        "Lslot4":   redondear_al_paso(variables_jmetal[9], 0.1),
        "Wslot1":   redondear_al_paso(variables_jmetal[10], 0.1),
        "Wslot2":   redondear_al_paso(variables_jmetal[11], 0.1),
        "Wslot3":   redondear_al_paso(variables_jmetal[12], 0.1),
        "Wslot4":   redondear_al_paso(variables_jmetal[13], 0.1),
        "alpha_s1": redondear_al_paso(variables_jmetal[14], 1.0),
        "alpha_s2": redondear_al_paso(variables_jmetal[15], 1.0),
        "alpha_s3": redondear_al_paso(variables_jmetal[16], 1.0),
        "alpha_s4": redondear_al_paso(variables_jmetal[17], 1.0),
        "Off_V":    redondear_al_paso(variables_jmetal[18], 0.2),
        "Off_H":    redondear_al_paso(variables_jmetal[19], 0.2),
        "X_feeding":   redondear_al_paso(variables_jmetal[20], 0.05),
        "Y_feeding":   redondear_al_paso(variables_jmetal[21], 0.05)
    }

    # Leer la macro original
    with open(ruta_macro_original, 'r') as file:
        contenido_macro = file.read()

    # Buscar y reemplazar cada variable de forma inteligente
    for var_macro, nuevo_valor in valores_redondeados.items():
        
        # Busca exactamente la línea StoreParameter("Nombre", <cualquier_numero>)
        patron = r'StoreParameter\("' + var_macro + r'",\s*[-+]?[0-9]*\.?[0-9]+\)'
        reemplazo = f'StoreParameter("{var_macro}",{nuevo_valor})'
        
        # Reemplazamos en el texto
        contenido_macro = re.sub(patron, reemplazo, contenido_macro)

    # Guardar la nueva macro
    with open(ruta_macro_salida, 'w') as file:
        file.write(contenido_macro)
        
    # Devolvemos las variables redondeadas por si queremos hacer prints luego
    return valores_redondeados


    
def ejecutar_cst_en_segundo_plano(ruta_macro):
    """
    Abre CST oculto, crea un proyecto en blanco, ejecuta la macro para que dibuje
    y simule la antena, y luego cierra todo para liberar memoria.
    """
    print("Iniciando CST en segundo plano...")
    
    # Modo Oculto/Silencioso
    cstEnv = cst.interface.DesignEnvironment.new(['--hide', '--quiet']) 
    cst.interface.DesignEnvironment.set_quiet_mode(cstEnv, True)
    
    try:
        # CREAR PROYECTO NUEVO EN BLANCO
        cstProject = cst.interface.DesignEnvironment.new_mws(cstEnv)
        
        # Leer la macro generada con los nuevos valores
        with open(ruta_macro, 'r', encoding='utf-8') as file:
            vbaCode = file.read()
            
        # Ejecutar la macro
        print("Ejecutando simulación...")
        cstProject.schematic.execute_vba_code(vbaCode, timeout=None)
        
        print("Simulación completada con éxito.")
        
    except Exception as e:
        print(f"Error durante la simulación de CST: {e}")
        
    finally:
        # CERRAR EL PROYECTO SIN GUARDAR Y CERRAR CST
        # Como los resultados ya se han exportado a .txt por la macro, 
        # cerramos el proyecto directamente para liberar toda la RAM.
        if 'cstProject' in locals():
            cstProject.close()
        cst.interface.DesignEnvironment.close(cstEnv)
        print("CST cerrado. Memoria liberada.")

#def evaluate(self, solution: FloatSolution) -> FloatSolution:
def evaluate(vars_algoritmo):
        # Extraer el array de variables que propone el algoritmo
       # vars_algoritmo = solution.variables
        
        # Rutas de los archivos
        ruta_plantilla_macro = r"C:\Users\jmate\Desktop\teleco\TFG\CST\MacroPatchCP_Multilayer_Cross.txt"
        ruta_macro_temporal  = r"C:\Users\jmate\Desktop\teleco\TFG\CST\Macro_To_Run.bas"
        
        # Generamos la macro con los pasos del Excel
        generar_macro_cst(vars_algoritmo, ruta_plantilla_macro, ruta_macro_temporal)
        
        # CST se abre oculto, simula, exporta los .txt y se cierra
        # ejecutar_cst_en_segundo_plano(ruta_macro_temporal)
       
        # Cargar los datos de S11
        ruta_archivo_imag = r"C:\Users\jmate\Desktop\teleco\TFG\CST\ImagS11.txt"
        datos_imag = np.loadtxt(ruta_archivo_imag, skiprows=2)
        
        ruta_archivo_real = r"C:\Users\jmate\Desktop\teleco\TFG\CST\RealS11.txt"
        datos_real = np.loadtxt(ruta_archivo_real, skiprows=2)
        
        # Extraer las columnas
        frecuencia = datos_imag[:, 0]
        s11_imag = datos_imag[:, 1]
        s11_real = datos_real[:, 1]
        
        # Convertir a decibelios (dB)
        
        s11_complejo = s11_real + 1j * s11_imag
        
        s11_db = 20 * np.log10(np.abs(s11_complejo))
        # objective1= np.sum(s11_db)/len(s11_db)
        # Crear la PRIMERA gráfica
        plt.figure()
        plt.plot(frecuencia, s11_imag, color='blue')
        plt.title('S1,1 Imaginario (Lineal)')
        plt.xlabel('Frecuencia (GHz)')
        plt.ylabel('Magnitud')
        plt.grid(True)
        
        
        plt.figure()
        plt.plot(frecuencia, s11_real, color='red')
        plt.title('S1,1 Real (Lineal)')
        plt.xlabel('Frecuencia (GHz)')
        plt.ylabel('Magnitud')
        plt.grid(True)
        
        # Crear la SEGUNDA gráfica
        plt.figure()
        plt.plot(frecuencia, s11_db, color='red')
        plt.title('S1,1 (dB)')
        plt.xlabel('Frecuencia (GHz)')
        plt.ylabel('Magnitud (dB)')
        plt.grid(True)
        
        rutas_farfield = [
            r"C:\Users\jmate\Desktop\teleco\TFG\CST\Farfield_f4_8.txt",
            r"C:\Users\jmate\Desktop\teleco\TFG\CST\Farfield_f4_9.txt",
            r"C:\Users\jmate\Desktop\teleco\TFG\CST\Farfield_f5.txt",
            r"C:\Users\jmate\Desktop\teleco\TFG\CST\Farfield_f5_1.txt",
            r"C:\Users\jmate\Desktop\teleco\TFG\CST\Farfield_f5_2.txt"
        ]
        
        frecuencias_farfield = []
        axial_ratios = []
        
        # Leer cada archivo y extraer la frecuencia de su nombre
        for ruta in rutas_farfield:
            # --- Extraer Frecuencia del nombre ---
            nombre_archivo = os.path.basename(ruta)
            # Quitamos el texto sobrante
            texto_num = nombre_archivo.replace("Farfield_f", "").replace(".txt", "")
            # Cambiamos "_" por "." y convertimos a número
            frec_val = float(texto_num.replace("_", "."))
            
            # --- Extraer el dato Axial Ratio ---
            datos_ff = np.loadtxt(ruta, skiprows=2)
            primer_ax_ratio = datos_ff[0, 7] # Fila 0, Columna 8 (índice 7)
            
            # Guardamos ambos valores
            frecuencias_farfield.append(frec_val)
            axial_ratios.append(primer_ax_ratio)
        
        # Ordenar las listas por frecuencia de menor a mayor (por si las rutas estuvieran desordenadas)
        # zip empareja los datos, sorted los ordena guiándose por la frecuencia, y zip(*...) los vuelve a separar
        frecuencias_farfield, axial_ratios = zip(*sorted(zip(frecuencias_farfield, axial_ratios)))
        
        #objective2 = np.sum(axial_ratios)/len(axial_ratios)
        # Crear la TERCERA gráfica para el Axial Ratio
        plt.figure()
        plt.plot(frecuencias_farfield, axial_ratios, color='green', marker='o')
        plt.title('Axial Ratio vs Frecuencia')
        plt.xlabel('Frecuencia (GHz)')
        plt.ylabel('Axial Ratio (dB)')
        plt.grid(True)
        # ---------------------------------------------------------
        
        # Mostrar todas las ventanas
        plt.show()
        
        
        
if __name__ == "__main__":
    variables_iniciales = [
        13.5,  # L_metal
        13.5,  # W_metal
        40.0,  # L_GND
        40.0,  # W_GND
        2.5,   # Cx
        2.5,   # Cy
        4.5,   # Lslot1
        4.5,   # Lslot2
        4.5,   # Lslot3
        4.5,   # Lslot4
        1.0,   # Wslot1
        1.0,   # Wslot2
        1.0,   # Wslot3
        1.0,   # Wslot4
        45.0,  # alpha_s1
        45.0,  # alpha_s2
        45.0,  # alpha_s3
        45.0,  # alpha_s4
        0.0,   # Off_V
        0.0,   # Off_H
        2.5,   # x_feed
        0.5    # y_feed
    ]
    tiempo_inicio = time.time()
    # Ejecutamos la prueba
    evaluate(variables_iniciales)
    
    tiempo_fin = time.time()
    tiempo_total_segundos = tiempo_fin - tiempo_inicio
    
    minutos = int(tiempo_total_segundos // 60)
    segundos = int(tiempo_total_segundos % 60)
    
    print(f"\n TIEMPO TOTAL DE LA PRUEBA: {minutos} minutos y {segundos} segundos ({tiempo_total_segundos:.2f} s)")
    
    