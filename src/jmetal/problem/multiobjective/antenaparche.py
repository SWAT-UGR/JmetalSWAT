import sys
import numpy as np
import os
sys.path.append(r"C:\Program Files\CST Studio Suite 2026\AMD64\python_cst_libraries")
import cst.interface
import re

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class antenaparche(FloatProblem):
    """Problem antena.
    
    """
    def __init__(self, number_of_variables: int = 22):
        """:param number_of_variables: Number of decision variables of the problem."""
        
        super(antenaparche, self).__init__()
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["S11", "axial_ratios"]
        
        self.lower_bound = [
            12.0,  # 0: L_metal (min)
            12.0,  # 1: W_metal (min)
            30.0,  # 2: L_GND (min)
            30.0,  # 3: W_GND (min)
            1.0,   # 4: Cx (min)
            1.0,   # 5: Cy (min)
            2.5,   # 6: Lslot1 (min)
            2.5,   # 7: Lslot2 (min)
            2.5,   # 8: Lslot3 (min)
            2.5,   # 9: Lslot4 (min)
            0.5,   # 10: Wslot1 (min)
            0.5,   # 11: Wslot2 (min)
            0.5,   # 12: Wslot3 (min)
            0.5,   # 13: Wslot4 (min)
            25.0,   # 14: alpha_s1 (min)
            25.0,   # 15: alpha_s2 (min)
            25.0,   # 16: alpha_s3 (min)
            25.0,   # 17: alpha_s4 (min)
            -3.0,  # 18: Off_V (min)
            -3.0,  # 19: Off_H (min)
            -5.0,   # 20: X_feeding (min)
            -5.0    # 21: Y_feeding (min)
        ]
        
        self.upper_bound = [
            15.5,  # 0: L_metal (max)
            15.5,  # 1: W_metal (max)
            55.0,  # 2: L_GND (max)
            55.0,  # 3: W_GND (max)
            4.5,   # 4: Cx (max)
            4.5,   # 5: Cy (max)
            6.5,   # 6: Lslot1 (max)
            6.5,   # 7: Lslot2 (max)
            6.5,   # 8: Lslot3 (max)
            6.5,   # 9: Lslot4 (max)
            2.0,   # 10: Wslot1 (max)
            2.0,   # 11: Wslot2 (max)
            2.0,   # 12: Wslot3 (max)
            2.0,   # 13: Wslot4 (max)
            65.0,  # 14: alpha_s1 (max)
            65.0,  # 15: alpha_s2 (max)
            65.0,  # 16: alpha_s3 (max)
            65.0,  # 17: alpha_s4 (max)
            3.0,   # 18: Off_V (max)
            3.0,   # 19: Off_H (max)
            5.0,   # 20: X_feeding (max)
            5.0    # 21: Y_feeding (max)
        ]
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
    
    def number_of_constraints(self) -> int:
        return 0 # Por ahora
    
    def redondear_al_paso(self, valor, paso):
        """
        Redondea un número al 'Step' más cercano.
        """
        return round(round(valor / paso) * paso, 3)

    def generar_macro_cst(self, variables_jmetal, ruta_macro_original, ruta_macro_salida):
        """
        Lee la macro base, extrae las variables de JMetal, les aplica el 'Step' 
        y genera una nueva macro lista para CST.
        """
        # Indexar variables de JMetal y aplicar el Step del Excel
        # El orden (0 al 21) dependerá de cómo las declares en tu clase JMetalProblem
        valores_redondeados = {
            "L_metal":  self.redondear_al_paso(variables_jmetal[0], 0.1),
            "W_metal":  self.redondear_al_paso(variables_jmetal[1], 0.1),
            "L_GND":    self.redondear_al_paso(variables_jmetal[2], 1.0),
            "W_GND":    self.redondear_al_paso(variables_jmetal[3], 1.0),
            "Cx":       self.redondear_al_paso(variables_jmetal[4], 0.1),
            "Cy":       self.redondear_al_paso(variables_jmetal[5], 0.1),
            "Lslot1":   self.redondear_al_paso(variables_jmetal[6], 0.1),
            "Lslot2":   self.redondear_al_paso(variables_jmetal[7], 0.1),
            "Lslot3":   self.redondear_al_paso(variables_jmetal[8], 0.1),
            "Lslot4":   self.redondear_al_paso(variables_jmetal[9], 0.1),
            "Wslot1":   self.redondear_al_paso(variables_jmetal[10], 0.1),
            "Wslot2":   self.redondear_al_paso(variables_jmetal[11], 0.1),
            "Wslot3":   self.redondear_al_paso(variables_jmetal[12], 0.1),
            "Wslot4":   self.redondear_al_paso(variables_jmetal[13], 0.1),
            "alpha_s1": self.redondear_al_paso(variables_jmetal[14], 1.0),
            "alpha_s2": self.redondear_al_paso(variables_jmetal[15], 1.0),
            "alpha_s3": self.redondear_al_paso(variables_jmetal[16], 1.0),
            "alpha_s4": self.redondear_al_paso(variables_jmetal[17], 1.0),
            "Off_V":    self.redondear_al_paso(variables_jmetal[18], 0.2),
            "Off_H":    self.redondear_al_paso(variables_jmetal[19], 0.2),
            "X_feeding":   self.redondear_al_paso(variables_jmetal[20], 0.05),
            "Y_feeding":  self.redondear_al_paso(variables_jmetal[21], 0.05)
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
            
            
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        #vars_algoritmo = solution.variables
        
        # Por ahora para hacer pruebas
        vars_algoritmo = [
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
        # Rutas de los archivos
        ruta_plantilla_macro = r"C:\Users\jmate\Desktop\teleco\TFG\CST\MacroPatchCP_Multilayer_Cross.txt"
        ruta_macro_temporal  = r"C:\Users\jmate\Desktop\teleco\TFG\CST\Macro_To_Run.bas"
        
        # Generamos la macro con los pasos del Excel
        self.generar_macro_cst(vars_algoritmo, ruta_plantilla_macro, ruta_macro_temporal)
        
        # CST se abre oculto, simula, exporta los .txt y se cierra
        # self.ejecutar_cst_en_segundo_plano(ruta_macro_temporal)
       
        # Cargar los datos de S11
        ruta_archivo_imag = r"C:\Users\jmate\Desktop\teleco\TFG\CST\ImagS11.txt"
        datos_imag = np.loadtxt(ruta_archivo_imag, skiprows=2)
        
        ruta_archivo_real = r"C:\Users\jmate\Desktop\teleco\TFG\CST\RealS11.txt"
        datos_real = np.loadtxt(ruta_archivo_real, skiprows=2)
        
        # Extraer las columnas
        s11_imag = datos_imag[:, 1]
        s11_real = datos_real[:, 1]
        
        # Convertir a decibelios (dB)
        
        s11_complejo = s11_real + 1j * s11_imag
        
        s11_db = 20 * np.log10(np.abs(s11_complejo))
        solution.objectives[0]= np.sum(s11_db)/len(s11_db)
        
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
        
        solution.objectives[1] = np.sum(axial_ratios)/len(axial_ratios)
        
        return solution
    
    def name(self):
        return "AntenaParche"