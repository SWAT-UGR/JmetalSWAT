import sys
import numpy as np
import os
import multiprocessing
import logging
# Añadimos la ruta de CST solo si no está ya añadida
ruta_cst = r"C:\Program Files\CST Studio Suite 2026\AMD64\python_cst_libraries"
if ruta_cst not in sys.path:
    sys.path.append(ruta_cst)

# Intentamos importar CST. Si falla (como en GitHub), lo ignoramos silenciosamente
try:
    import cst.interface
    CST_AVAILABLE = True
except ImportError:
    CST_AVAILABLE = False
    logging.warning("Librería de CST no encontrada. Las simulaciones reales no funcionarán en este entorno.")
import re
import time
import math
from jmetal.core.problem import IntegerProblem, FloatProblem
from jmetal.core.solution import IntegerSolution, FloatSolution
from datetime import datetime


class antenaparcheInteger(IntegerProblem):
    """Problem antena.
    
    """
    def __init__(self, rutas_config: dict, number_of_variables: int = 22):
        """:param number_of_variables: Number of decision variables of the problem."""
        
        super(antenaparcheInteger, self).__init__()
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["S11", "axial_ratios"]
        
        self.carpeta_cst = rutas_config['carpeta_cst']
        self.ruta_plantilla_macro = rutas_config['ruta_plantilla_macro']
        
        # Generamos la ruta del CSV de tiempos dinámicamente
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ruta_csv_tiempos = os.path.join(rutas_config['carpeta_tiempos'], f"estadisticas_tiempos_{timestamp}.csv")
        
        # Rutas dinámicas para los archivos que genera CST
        self.ruta_macro_temporal = os.path.join(self.carpeta_cst, "Macro_To_Run.bas")
        self.ruta_archivo_imag = os.path.join(self.carpeta_cst, "ImagS11.txt")
        self.ruta_archivo_real = os.path.join(self.carpeta_cst, "RealS11.txt")
        
        
        self.steps = [0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.05, 0.05]
        
        lower_bound_real = [
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
        
        upper_bound_real = [
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
        
        self.lower_bound = [int(round(val / step)) for val, step in zip(lower_bound_real, self.steps)]
        self.upper_bound = [int(round(val / step)) for val, step in zip(upper_bound_real, self.steps)]
        
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
    
    def number_of_constraints(self) -> int:
        return 19 

    def generar_macro_cst(self, variables_jmetal, ruta_macro_original, ruta_macro_salida):
        """
        Lee la macro base, extrae las variables de JMetal, les aplica el 'Step' 
        y genera una nueva macro lista para CST.
        """
        # Indexar variables de JMetal
        valores_redondeados = {
            "L_metal":  variables_jmetal[0],
            "W_metal":  variables_jmetal[1],
            "L_GND":    variables_jmetal[2],
            "W_GND":    variables_jmetal[3],
            "Cx":       variables_jmetal[4],
            "Cy":       variables_jmetal[5],
            "Lslot1":   variables_jmetal[6],
            "Lslot2":   variables_jmetal[7],
            "Lslot3":   variables_jmetal[8],
            "Lslot4":   variables_jmetal[9],
            "Wslot1":   variables_jmetal[10],
            "Wslot2":   variables_jmetal[11],
            "Wslot3":   variables_jmetal[12],
            "Wslot4":   variables_jmetal[13],
            "alpha_s1": variables_jmetal[14],
            "alpha_s2": variables_jmetal[15],
            "alpha_s3": variables_jmetal[16],
            "alpha_s4": variables_jmetal[17],
            "Off_V":    variables_jmetal[18],
            "Off_H":    variables_jmetal[19],
            "X_feeding":   variables_jmetal[20],
            "Y_feeding":  variables_jmetal[21]
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
    
    
    def ejecutar_cst_en_segundo_plano(self, ruta_macro):
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
            # 1. Soltamos el proyecto actual
            if 'cstProject' in locals():
                cstProject = None 
            
            # 2. Cierre nativo y oficial de CST (la opción de antes)
            if 'cstEnv' in locals() and cstEnv is not None:
                cst.interface.DesignEnvironment.close(cstEnv)
            
            # 3. Pausa de cortesía para que Windows asimile el cierre nativo
            time.sleep(2)
            
            print("CST cerrado correctamente. Memoria liberada.")
    def _worker_cst(self, ruta_macro, queue):
        """Esta función es el 'clon' que vive en la zona de cuarentena"""
        try:
            # Aquí llamas a tu función normal que abre CST y simula
            resultados = self.ejecutar_cst_en_segundo_plano(ruta_macro)
            queue.put(resultados)
        except Exception as e:
            print(f"Error en el worker: {e}")
            queue.put(None)            
            
    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        tiempo_inicio = time.time()
        
        # ESCUDO WARM-START: Comprobamos la etiqueta secreta
        if solution.attributes.get('viene_de_warm_start', False):
            # Le quitamos la etiqueta. Así, si esta solución se clona para 
            # tener hijos en la siguiente generación, los hijos irán a CST.
            solution.attributes['viene_de_warm_start'] = False
            return solution
        
        vars_fisicas = [var * step for var, step in zip(solution.variables, self.steps)]
        
        self.__evaluate_constraints(vars_fisicas, solution)
        
        # La función all() devuelve True solo si todos los números de la lista son >= 0
        es_valida = all(c >= 0 for c in solution.constraints)
        
        if not es_valida:
            
            # Le asignamos notas para descartarla sin llamar a CST.
            solution.objectives[0] = 10e8
            solution.objectives[1] = 10e8
            
            # Anulamos los datos gráficos para que no se dibujen
            solution.attributes['frecuencia'] = None 
            
            tiempo_fin = time.time()
            t_seg = tiempo_fin - tiempo_inicio
            t_min = t_seg / 60.0
            
            archivo_nuevo = not os.path.exists(self.ruta_csv_tiempos)
            with open(self.ruta_csv_tiempos, "a", encoding="utf-8") as f:
                if archivo_nuevo:
                    f.write("Tiempo_Segundos;Tiempo_Minutos\n")
                seg_str = f"{t_seg:.2f}".replace('.', ',')
                min_str = f"{t_min:.2f}".replace('.', ',')
                f.write(f"{seg_str};{min_str}\n")
                
            return solution 
        
        # Rutas de los archivos
        ruta_plantilla_macro = self.ruta_plantilla_macro
        ruta_macro_temporal  = self.ruta_macro_temporal
        
        # Generamos la macro con los pasos del Excel
        self.generar_macro_cst(vars_fisicas, ruta_plantilla_macro, ruta_macro_temporal)
        
       # Preparamos el canal de comunicación con el clon
        queue = multiprocessing.Queue()
        
        # Lanzamos la simulación en un universo paralelo (Sandbox)
        p = multiprocessing.Process(target=self._worker_cst, args=(ruta_macro_temporal, queue))
        p.start()
        
        # EL CORTAFUEGOS: Esperamos un máximo de 600 segundos (10 minutos)
        p.join(timeout=600) 
        
        if p.is_alive():
            print("¡TIMEOUT! La antena se ha atascado. Exterminando proceso...")
            
            # A) Matamos el clon de Python (corta la conexión COM de forma segura)
            p.terminate()
            p.join()
            
            # B) Limpiamos los restos zombis de CST de la RAM
            procesos_zombi = ["CST_DESIGN_ENVIRONMENT.exe", "cst.exe", "CST_MICROWAVE_STUDIO.exe", "CST_MWS.exe"]
            for zombi in procesos_zombi:
                os.system(f"taskkill /F /IM {zombi} /T >nul 2>&1")
                
            time.sleep(3) # Pausa de cortesía para Windows
            
            # C) EL CASTIGO DEFINITIVO (Lo hacemos altísimo para que muera en el torneo)
            solution.objectives[0] = 10e8
            solution.objectives[1] = 10e8
            
            # D) Rellenamos con basura segura para que no peten los Observers
            solution.attributes['vars_fisicas'] = vars_fisicas
            solution.attributes['frecuencia'] = np.array([0])
            solution.attributes['s11_imag'] = np.array([0])
            solution.attributes['s11_real'] = np.array([0])
            solution.attributes['s11_db'] = np.array([0])
            solution.attributes['frec_farfield'] = [0]
            solution.attributes['axial_ratios'] = [0]
            
        else:
            # SI TERMINÓ A TIEMPO, LEEMOS LOS ARCHIVOS
            # (TODO ESTO VA INDENTADO DENTRO DEL ELSE)
            
            # Cargar los datos de S11 usando las variables dinámicas
            datos_imag = np.loadtxt(self.ruta_archivo_imag, skiprows=2)
            datos_real = np.loadtxt(self.ruta_archivo_real, skiprows=2)
            
            frecuencia = datos_imag[:, 0]
            s11_imag = datos_imag[:, 1]
            s11_real = datos_real[:, 1]
            
            # Convertir a decibelios (dB)
            s11_complejo = s11_real + 1j * s11_imag
            s11_db = 20 * np.log10(np.abs(s11_complejo))
            
            frecuencias_evaluar = ["4_8", "4_9", "5", "5_1", "5_2"]
            rutas_farfield = [os.path.join(self.carpeta_cst, f"Farfield_f{f}.txt") for f in frecuencias_evaluar]
            
            frecuencias_farfield = []
            axial_ratios = []
            
            for ruta in rutas_farfield:
                nombre_archivo = os.path.basename(ruta)
                texto_num = nombre_archivo.replace("Farfield_f", "").replace(".txt", "")
                frec_val = float(texto_num.replace("_", "."))
                
                datos_ff = np.loadtxt(ruta, skiprows=2)
                primer_ax_ratio = datos_ff[0, 7] 
                
                frecuencias_farfield.append(frec_val)
                axial_ratios.append(primer_ax_ratio)
            
            # GUARDAMOS LOS OBJETIVOS REALES
            solution.objectives[0] = np.sum(s11_db)/len(s11_db)
            solution.objectives[1] = np.sum(axial_ratios)/len(axial_ratios)
            
            solution.attributes['vars_fisicas'] = vars_fisicas
            solution.attributes['frecuencia'] = frecuencia
            solution.attributes['s11_imag'] = s11_imag
            solution.attributes['s11_real'] = s11_real
            solution.attributes['s11_db'] = s11_db
            solution.attributes['frec_farfield'] = frecuencias_farfield
            solution.attributes['axial_ratios'] = axial_ratios
              
        # Fuera del if/else, calculamos y escribimos el tiempo final
        tiempo_fin = time.time()
        tiempo_total_segundos = tiempo_fin - tiempo_inicio
        tiempo_total_minutos = tiempo_total_segundos / 60.0
        
        archivo_nuevo = not os.path.exists(self.ruta_csv_tiempos)
        with open(self.ruta_csv_tiempos, "a", encoding="utf-8") as f:
            if archivo_nuevo:
                f.write("Tiempo_Segundos;Tiempo_Minutos\n")
            seg_str = f"{tiempo_total_segundos:.2f}".replace('.', ',')
            min_str = f"{tiempo_total_minutos:.2f}".replace('.', ',')
            f.write(f"{seg_str};{min_str}\n")
            
    def __evaluate_constraints(self, vars_fisicas, solution: IntegerSolution) -> bool:
        """
        Evalúa las restricciones físicas de la antena. 
        Devuelve True si la antena es viable, o False si hay colisión.
        """
        x = vars_fisicas
        
        # Mapeo de variables para que las fórmulas sean legibles
        L_metal, W_metal = x[0], x[1]
        L_GND, W_GND = x[2], x[3]
        Cx, Cy = x[4], x[5]
        Lslot1, Lslot2, Lslot3, Lslot4 = x[6], x[7], x[8], x[9]
        Wslot1, Wslot2, Wslot3, Wslot4 = x[10], x[11], x[12], x[13]
        
        # Convertimos los ángulos de grados a radianes para math.sin y math.cos
        a1 = math.radians(x[14])
        a2 = math.radians(x[15])
        a3 = math.radians(x[16])
        a4 = math.radians(x[17])
        
        Off_V, Off_H = x[18], x[19]
        x_feed, y_feed = x[20], x[21]

        # Constantes de seguridad
        R_safe = 1.2
        SM = 0.5
        
        constraints = [0.0] * self.number_of_constraints()

        # 2.1 Límites del parche
        constraints[0] = (W_metal / 2.0 - R_safe) - abs(x_feed)
        constraints[1] = (L_metal / 2.0 - R_safe) - abs(y_feed)
        
        # 2.2 Evitar cortes en las esquinas (Evaluamos los 4 cuadrantes con abs)
        if (abs(x_feed) > W_metal/2.0 - Cx - R_safe) and (abs(y_feed) > L_metal/2.0 - Cy - R_safe):
            constraints[2] = -1.0 # FAIL (Choca con esquina)
        else:
            constraints[2] = 1.0  # OK

        # 2.3 Colisión con los slots (Función de ayuda interna)
        def check_colision_slot(dx, dy, theta_deg, L_slot, W_slot):
            theta_rad = math.radians(theta_deg)
            # Transformación vectorial
            u = dx * math.cos(-theta_rad) - dy * math.sin(-theta_rad)
            v = dx * math.sin(-theta_rad) + dy * math.cos(-theta_rad)
            
            # Se produce colisión si se cumplen TODAS las inecuaciones
            if (u < R_safe) and (u > -(L_slot + R_safe)) and (abs(v) < W_slot/2.0 + R_safe):
                return -1.0 # FAIL
            return 1.0 # OK

        # Slot 1 y 2
        constraints[3] = check_colision_slot(x_feed - Off_V, y_feed - L_metal/2.0, 270.0 + x[14], Lslot1, Wslot1)
        constraints[4] = check_colision_slot(x_feed + Off_V, y_feed + L_metal/2.0, 90.0 + x[15], Lslot2, Wslot2)
        
        # Colisión Slot 3 (Der) y Slot 4 (Izq)
        constraints[5] = check_colision_slot(x_feed - W_metal/2.0, y_feed - Off_H, 180.0 + x[16], Lslot3, Wslot3)
        constraints[6] = check_colision_slot(x_feed + W_metal/2.0, y_feed + Off_H, x[17], Lslot4, Wslot4)
        
        # 2.4 Integridad del núcleo (no cruzar el centro)
        constraints[7] = (L_metal / 2.0 - 2.0) - Lslot1 * math.cos(a1)
        constraints[8] = (L_metal / 2.0 - 2.0) - Lslot2 * math.cos(a2)
        constraints[9] = (W_metal / 2.0 - 2.0) - Lslot3 * math.cos(a3)
        constraints[10]= (W_metal / 2.0 - 2.0) - Lslot4 * math.cos(a4)
        
        # 2.5 Conflictos de borde (crowding)
        # Usamos max(sin, 1e-6) para prevenir errores fatales de división por cero
        constraints[11] = (W_metal / 2.0) - (Cx + abs(Off_H) + Wslot1 / (2.0 * max(math.sin(a1), 1e-6)) + SM)
        constraints[12] = (W_metal / 2.0) - (Cx + abs(Off_H) + Wslot2 / (2.0 * max(math.sin(a2), 1e-6)) + SM)
        constraints[13] = (L_metal / 2.0) - (Cy + abs(Off_V) + Wslot3 / (2.0 * max(math.sin(a3), 1e-6)) + SM)
        constraints[14] = (L_metal / 2.0) - (Cy + abs(Off_V) + Wslot4 / (2.0 * max(math.sin(a4), 1e-6)) + SM)
        # 2.6 Restricciones de consistencia
        constraints[15] = W_GND - (W_metal + 2.0)
        constraints[16] = L_GND - (L_metal + 2.0)
        
        relacion_aspecto = L_metal / W_metal
        constraints[17] = relacion_aspecto - 0.85
        constraints[18] = 1.15 - relacion_aspecto

        # Asignar al array nativo de JMetal
        solution.constraints = constraints
    
    def name(self):
        return "AntenaParcheInteger"
    
class antenaparcheIntegerMono(IntegerProblem):
    """Problem antena.
    
    """
    def __init__(self, rutas_config: dict, number_of_variables: int = 22):
        """:param number_of_variables: Number of decision variables of the problem."""
        
        super(antenaparcheIntegerMono, self).__init__()
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["S11", "axial_ratios"]
        
        self.carpeta_cst = rutas_config['carpeta_cst']
        self.ruta_plantilla_macro = rutas_config['ruta_plantilla_macro']
        
        # Generamos la ruta del CSV de tiempos dinámicamente
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ruta_csv_tiempos = os.path.join(rutas_config['carpeta_tiempos'], f"estadisticas_tiempos_{timestamp}.csv")
        
        # Rutas dinámicas para los archivos que genera CST
        self.ruta_macro_temporal = os.path.join(self.carpeta_cst, "Macro_To_Run.bas")
        self.ruta_archivo_imag = os.path.join(self.carpeta_cst, "ImagS11.txt")
        self.ruta_archivo_real = os.path.join(self.carpeta_cst, "RealS11.txt")
        
        
        self.steps = [0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.05, 0.05]
        
        lower_bound_real = [
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
        
        upper_bound_real = [
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
        
        self.lower_bound = [int(round(val / step)) for val, step in zip(lower_bound_real, self.steps)]
        self.upper_bound = [int(round(val / step)) for val, step in zip(upper_bound_real, self.steps)]
        
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
    
    def number_of_constraints(self) -> int:
        return 19 

    def generar_macro_cst(self, variables_jmetal, ruta_macro_original, ruta_macro_salida):
        """
        Lee la macro base, extrae las variables de JMetal, les aplica el 'Step' 
        y genera una nueva macro lista para CST.
        """
        # Indexar variables de JMetal
        valores_redondeados = {
            "L_metal":  variables_jmetal[0],
            "W_metal":  variables_jmetal[1],
            "L_GND":    variables_jmetal[2],
            "W_GND":    variables_jmetal[3],
            "Cx":       variables_jmetal[4],
            "Cy":       variables_jmetal[5],
            "Lslot1":   variables_jmetal[6],
            "Lslot2":   variables_jmetal[7],
            "Lslot3":   variables_jmetal[8],
            "Lslot4":   variables_jmetal[9],
            "Wslot1":   variables_jmetal[10],
            "Wslot2":   variables_jmetal[11],
            "Wslot3":   variables_jmetal[12],
            "Wslot4":   variables_jmetal[13],
            "alpha_s1": variables_jmetal[14],
            "alpha_s2": variables_jmetal[15],
            "alpha_s3": variables_jmetal[16],
            "alpha_s4": variables_jmetal[17],
            "Off_V":    variables_jmetal[18],
            "Off_H":    variables_jmetal[19],
            "X_feeding":   variables_jmetal[20],
            "Y_feeding":  variables_jmetal[21]
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
    
    
    def ejecutar_cst_en_segundo_plano(self, ruta_macro):
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
            # 1. Soltamos el proyecto actual
            if 'cstProject' in locals():
                cstProject = None 
            
            # 2. Cierre nativo y oficial de CST (la opción de antes)
            if 'cstEnv' in locals() and cstEnv is not None:
                cst.interface.DesignEnvironment.close(cstEnv)
            
            # 3. Pausa de cortesía para que Windows asimile el cierre nativo
            time.sleep(2)
            
            print("CST cerrado correctamente. Memoria liberada.")
    def _worker_cst(self, ruta_macro, queue):
        """Esta función es el 'clon' que vive en la zona de cuarentena"""
        try:
            # Aquí llamas a tu función normal que abre CST y simula
            resultados = self.ejecutar_cst_en_segundo_plano(ruta_macro)
            queue.put(resultados)
        except Exception as e:
            print(f"Error en el worker: {e}")
            queue.put(None)            
            
    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        tiempo_inicio = time.time()
        
        # ESCUDO WARM-START: Comprobamos la etiqueta secreta
        if solution.attributes.get('viene_de_warm_start', False):
            # Le quitamos la etiqueta. Así, si esta solución se clona para 
            # tener hijos en la siguiente generación, los hijos irán a CST.
            solution.attributes['viene_de_warm_start'] = False
            return solution
        
        vars_fisicas = [var * step for var, step in zip(solution.variables, self.steps)]
        
        self.__evaluate_constraints(vars_fisicas, solution)
        
        # La función all() devuelve True solo si todos los números de la lista son >= 0
        es_valida = all(c >= 0 for c in solution.constraints)
        
        if not es_valida:
            
            # Le asignamos notas para descartarla sin llamar a CST.
            solution.objectives[0] = 10e8
            solution.objectives[1] = 10e8
            
            # Anulamos los datos gráficos para que no se dibujen
            solution.attributes['frecuencia'] = None 
            
            tiempo_fin = time.time()
            t_seg = tiempo_fin - tiempo_inicio
            t_min = t_seg / 60.0
            
            archivo_nuevo = not os.path.exists(self.ruta_csv_tiempos)
            with open(self.ruta_csv_tiempos, "a", encoding="utf-8") as f:
                if archivo_nuevo:
                    f.write("Tiempo_Segundos;Tiempo_Minutos\n")
                seg_str = f"{t_seg:.2f}".replace('.', ',')
                min_str = f"{t_min:.2f}".replace('.', ',')
                f.write(f"{seg_str};{min_str}\n")
                
            return solution 
        
        # Rutas de los archivos
        ruta_plantilla_macro = self.ruta_plantilla_macro
        ruta_macro_temporal  = self.ruta_macro_temporal
        
        # Generamos la macro con los pasos del Excel
        self.generar_macro_cst(vars_fisicas, ruta_plantilla_macro, ruta_macro_temporal)
        
       # Preparamos el canal de comunicación con el clon
        queue = multiprocessing.Queue()
        
        # Lanzamos la simulación en un universo paralelo (Sandbox)
        p = multiprocessing.Process(target=self._worker_cst, args=(ruta_macro_temporal, queue))
        p.start()
        
        # EL CORTAFUEGOS: Esperamos un máximo de 600 segundos (10 minutos)
        p.join(timeout=600) 
        
        if p.is_alive():
            print("¡TIMEOUT! La antena se ha atascado. Exterminando proceso...")
            
            # A) Matamos el clon de Python (corta la conexión COM de forma segura)
            p.terminate()
            p.join()
            
            # B) Limpiamos los restos zombis de CST de la RAM
            procesos_zombi = ["CST_DESIGN_ENVIRONMENT.exe", "cst.exe", "CST_MICROWAVE_STUDIO.exe", "CST_MWS.exe"]
            for zombi in procesos_zombi:
                os.system(f"taskkill /F /IM {zombi} /T >nul 2>&1")
                
            time.sleep(3) # Pausa de cortesía para Windows
            
            # C) EL CASTIGO DEFINITIVO (Lo hacemos altísimo para que muera en el torneo)
            solution.objectives[0] = 10e8
            solution.objectives[1] = 10e8
            
            # D) Rellenamos con basura segura para que no peten los Observers
            solution.attributes['vars_fisicas'] = vars_fisicas
            solution.attributes['frecuencia'] = np.array([0])
            solution.attributes['s11_imag'] = np.array([0])
            solution.attributes['s11_real'] = np.array([0])
            solution.attributes['s11_db'] = np.array([0])
            solution.attributes['frec_farfield'] = [0]
            solution.attributes['axial_ratios'] = [0]
            
        else:
            # SI TERMINÓ A TIEMPO, LEEMOS LOS ARCHIVOS
            
            # 1. Cargar los datos de S11
            datos_imag = np.loadtxt(self.ruta_archivo_imag, skiprows=2)
            datos_real = np.loadtxt(self.ruta_archivo_real, skiprows=2)
            
            frecuencia = datos_imag[:, 0]
            s11_imag = datos_imag[:, 1]
            s11_real = datos_real[:, 1]
            
            # Convertir a decibelios (dB)
            s11_complejo = s11_real + 1j * s11_imag
            s11_db = 20 * np.log10(np.abs(s11_complejo))
            
            # --- CORRECCIÓN OBJETIVO 1: S11 SOLO A 5 GHz ---
            # Buscamos en qué índice del array está la frecuencia más cercana a 5.0
            indice_5ghz = np.argmin(np.abs(frecuencia - 5.0))
            s11_objetivo = s11_db[indice_5ghz] 
            
            # --- CORRECCIÓN OBJETIVO 2: AXIAL RATIO SOLO A 5 GHz ---
            ruta_farfield_5 = os.path.join(self.carpeta_cst, "Farfield_f5.txt")
            datos_ff = np.loadtxt(ruta_farfield_5, skiprows=2)
            axial_ratio_objetivo = datos_ff[0, 7] 
            
            # GUARDAMOS LOS OBJETIVOS REALES (Monofrecuencia)
            solution.objectives[0] = s11_objetivo
            solution.objectives[1] = axial_ratio_objetivo
            
            # Guardamos los atributos para el Observer (las gráficas)
            solution.attributes['vars_fisicas'] = vars_fisicas
            solution.attributes['frecuencia'] = frecuencia
            solution.attributes['s11_imag'] = s11_imag
            solution.attributes['s11_real'] = s11_real
            solution.attributes['s11_db'] = s11_db
            
            # Para mantener la compatibilidad con tus Observers que esperan listas
            solution.attributes['frec_farfield'] = [5.0]
            solution.attributes['axial_ratios'] = [axial_ratio_objetivo]
            
            
        # Fuera del if/else, calculamos y escribimos el tiempo final
        tiempo_fin = time.time()
        tiempo_total_segundos = tiempo_fin - tiempo_inicio
        tiempo_total_minutos = tiempo_total_segundos / 60.0
        
        archivo_nuevo = not os.path.exists(self.ruta_csv_tiempos)
        with open(self.ruta_csv_tiempos, "a", encoding="utf-8") as f:
            if archivo_nuevo:
                f.write("Tiempo_Segundos;Tiempo_Minutos\n")
            seg_str = f"{tiempo_total_segundos:.2f}".replace('.', ',')
            min_str = f"{tiempo_total_minutos:.2f}".replace('.', ',')
            f.write(f"{seg_str};{min_str}\n")
            
    def __evaluate_constraints(self, vars_fisicas, solution: IntegerSolution) -> bool:
        """
        Evalúa las restricciones físicas de la antena. 
        Devuelve True si la antena es viable, o False si hay colisión.
        """
        x = vars_fisicas
        
        # Mapeo de variables para que las fórmulas sean legibles
        L_metal, W_metal = x[0], x[1]
        L_GND, W_GND = x[2], x[3]
        Cx, Cy = x[4], x[5]
        Lslot1, Lslot2, Lslot3, Lslot4 = x[6], x[7], x[8], x[9]
        Wslot1, Wslot2, Wslot3, Wslot4 = x[10], x[11], x[12], x[13]
        
        # Convertimos los ángulos de grados a radianes para math.sin y math.cos
        a1 = math.radians(x[14])
        a2 = math.radians(x[15])
        a3 = math.radians(x[16])
        a4 = math.radians(x[17])
        
        Off_V, Off_H = x[18], x[19]
        x_feed, y_feed = x[20], x[21]

        # Constantes de seguridad
        R_safe = 1.2
        SM = 0.5
        
        constraints = [0.0] * self.number_of_constraints()

        # 2.1 Límites del parche
        constraints[0] = (W_metal / 2.0 - R_safe) - abs(x_feed)
        constraints[1] = (L_metal / 2.0 - R_safe) - abs(y_feed)
        
        # 2.2 Evitar cortes en las esquinas (Evaluamos los 4 cuadrantes con abs)
        if (abs(x_feed) > W_metal/2.0 - Cx - R_safe) and (abs(y_feed) > L_metal/2.0 - Cy - R_safe):
            constraints[2] = -1.0 # FAIL (Choca con esquina)
        else:
            constraints[2] = 1.0  # OK

        # 2.3 Colisión con los slots (Función de ayuda interna)
        def check_colision_slot(dx, dy, theta_deg, L_slot, W_slot):
            theta_rad = math.radians(theta_deg)
            # Transformación vectorial
            u = dx * math.cos(-theta_rad) - dy * math.sin(-theta_rad)
            v = dx * math.sin(-theta_rad) + dy * math.cos(-theta_rad)
            
            # Se produce colisión si se cumplen TODAS las inecuaciones
            if (u < R_safe) and (u > -(L_slot + R_safe)) and (abs(v) < W_slot/2.0 + R_safe):
                return -1.0 # FAIL
            return 1.0 # OK

        # Slot 1 y 2
        constraints[3] = check_colision_slot(x_feed - Off_V, y_feed - L_metal/2.0, 270.0 + x[14], Lslot1, Wslot1)
        constraints[4] = check_colision_slot(x_feed + Off_V, y_feed + L_metal/2.0, 90.0 + x[15], Lslot2, Wslot2)
        
        # Colisión Slot 3 (Der) y Slot 4 (Izq)
        constraints[5] = check_colision_slot(x_feed - W_metal/2.0, y_feed - Off_H, 180.0 + x[16], Lslot3, Wslot3)
        constraints[6] = check_colision_slot(x_feed + W_metal/2.0, y_feed + Off_H, x[17], Lslot4, Wslot4)
        
        # 2.4 Integridad del núcleo (no cruzar el centro)
        constraints[7] = (L_metal / 2.0 - 2.0) - Lslot1 * math.cos(a1)
        constraints[8] = (L_metal / 2.0 - 2.0) - Lslot2 * math.cos(a2)
        constraints[9] = (W_metal / 2.0 - 2.0) - Lslot3 * math.cos(a3)
        constraints[10]= (W_metal / 2.0 - 2.0) - Lslot4 * math.cos(a4)
        
        # 2.5 Conflictos de borde (crowding)
        # Usamos max(sin, 1e-6) para prevenir errores fatales de división por cero
        constraints[11] = (W_metal / 2.0) - (Cx + abs(Off_H) + Wslot1 / (2.0 * max(math.sin(a1), 1e-6)) + SM)
        constraints[12] = (W_metal / 2.0) - (Cx + abs(Off_H) + Wslot2 / (2.0 * max(math.sin(a2), 1e-6)) + SM)
        constraints[13] = (L_metal / 2.0) - (Cy + abs(Off_V) + Wslot3 / (2.0 * max(math.sin(a3), 1e-6)) + SM)
        constraints[14] = (L_metal / 2.0) - (Cy + abs(Off_V) + Wslot4 / (2.0 * max(math.sin(a4), 1e-6)) + SM)
        # 2.6 Restricciones de consistencia
        constraints[15] = W_GND - (W_metal + 2.0)
        constraints[16] = L_GND - (L_metal + 2.0)
        
        relacion_aspecto = L_metal / W_metal
        constraints[17] = relacion_aspecto - 0.85
        constraints[18] = 1.15 - relacion_aspecto

        # Asignar al array nativo de JMetal
        solution.constraints = constraints
    
    def name(self):
        return "AntenaParcheInteger"
    
class antenaparcheFloat(FloatProblem):
    """Problem antena.
    
    """
    def __init__(self, number_of_variables: int = 22):
        """:param number_of_variables: Number of decision variables of the problem."""
        
        super(antenaparcheFloat, self).__init__()
        
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
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ruta_csv_tiempos = rf"C:\Users\jmate\Desktop\teleco\TFG\Tiempos_prueba\estadisticas_tiempos_{timestamp}.csv"
        
    def number_of_objectives(self) -> int:
        return len(self.obj_directions)
    
    def number_of_variables(self) -> int:
        return len(self.lower_bound)
    
    def number_of_constraints(self) -> int:
        return 19 

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
            "L_GND":   self.redondear_al_paso(variables_jmetal[2], 1.0),
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
            "Y_feeding":   self.redondear_al_paso(variables_jmetal[21], 0.05)
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
    
    
    def ejecutar_cst_en_segundo_plano(self, ruta_macro):
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
        tiempo_inicio = time.time()
        
        vars_algoritmo = solution.variables
        
        self.__evaluate_constraints(vars_algoritmo, solution)
        
        # La función all() devuelve True solo si todos los números de la lista son >= 0
        es_valida = all(c >= 0 for c in solution.constraints)
        
        if not es_valida:
            
            # Le asignamos notas para descartarla sin llamar a CST.
            solution.objectives[0] = 10e8
            solution.objectives[1] = 10e8
            
            # Anulamos los datos gráficos para que no se dibujen
            solution.attributes['frecuencia'] = None 
            
            tiempo_fin = time.time()
            t_seg = tiempo_fin - tiempo_inicio
            t_min = t_seg / 60.0
            
            archivo_nuevo = not os.path.exists(self.ruta_csv_tiempos)
            with open(self.ruta_csv_tiempos, "a", encoding="utf-8") as f:
                if archivo_nuevo:
                    f.write("Tiempo_Segundos;Tiempo_Minutos\n")
                seg_str = f"{t_seg:.2f}".replace('.', ',')
                min_str = f"{t_min:.2f}".replace('.', ',')
                f.write(f"{seg_str};{min_str}\n")
                
            return solution 
        
        # Rutas de los archivos
        ruta_plantilla_macro = r"C:\Users\jmate\Desktop\teleco\TFG\CST\MacroPatchCP_Multilayer_Cross.txt"
        ruta_macro_temporal  = r"C:\Users\jmate\Desktop\teleco\TFG\CST\Macro_To_Run.bas"
        
        # Generamos la macro con los pasos del Excel
        self.generar_macro_cst(vars_algoritmo, ruta_plantilla_macro, ruta_macro_temporal)
        
        # CST se abre oculto, simula, exporta los .txt y se cierra
        self.ejecutar_cst_en_segundo_plano(ruta_macro_temporal)
       
        # Cargar los datos de S11
        ruta_archivo_imag = r"C:\Users\jmate\Desktop\teleco\TFG\CST\ImagS11.txt"
        datos_imag = np.loadtxt(ruta_archivo_imag, skiprows=2)
        
        ruta_archivo_real = r"C:\Users\jmate\Desktop\teleco\TFG\CST\RealS11.txt"
        datos_real = np.loadtxt(ruta_archivo_real, skiprows=2)
        frecuencia = datos_imag[:, 0]
        
        # Extraer las columnas
        s11_imag = datos_imag[:, 1]
        s11_real = datos_real[:, 1]
        
        # Convertir a decibelios (dB)
        
        s11_complejo = s11_real + 1j * s11_imag
        
        s11_db = 20 * np.log10(np.abs(s11_complejo))
        
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
        
        # # Ordenar las listas por frecuencia de menor a mayor (por si las rutas estuvieran desordenadas)
        # # zip empareja los datos, sorted los ordena guiándose por la frecuencia, y zip(*...) los vuelve a separar
        # frecuencias_farfield, axial_ratios = zip(*sorted(zip(frecuencias_farfield, axial_ratios)))
        
        solution.objectives[0]= np.sum(s11_db)/len(s11_db)
        solution.objectives[1] = np.sum(axial_ratios)/len(axial_ratios)
        
        solution.attributes['frecuencia'] = frecuencia
        solution.attributes['s11_imag'] = s11_imag
        solution.attributes['s11_real'] = s11_real
        solution.attributes['s11_db'] = s11_db
        solution.attributes['frec_farfield'] = frecuencias_farfield
        solution.attributes['axial_ratios'] = axial_ratios
        
        tiempo_fin = time.time()
        tiempo_total_segundos = tiempo_fin - tiempo_inicio
        tiempo_total_minutos = tiempo_total_segundos / 60.0
        
        archivo_nuevo = not os.path.exists(self.ruta_csv_tiempos)
        
        with open(self.ruta_csv_tiempos, "a", encoding="utf-8") as f:
            if archivo_nuevo:
                
                f.write("Tiempo_Segundos;Tiempo_Minutos\n")
            
            seg_str = f"{tiempo_total_segundos:.2f}".replace('.', ',')
            min_str = f"{tiempo_total_minutos:.2f}".replace('.', ',')
            
            f.write(f"{seg_str};{min_str}\n")
            
    def __evaluate_constraints(self, vars_fisicas, solution: FloatSolution) -> bool:
        """
        Evalúa las restricciones físicas de la antena. 
        Devuelve True si la antena es viable, o False si hay colisión.
        """
        x = vars_fisicas
        
        # Mapeo de variables para que las fórmulas sean legibles
        L_metal, W_metal = x[0], x[1]
        L_GND, W_GND = x[2], x[3]
        Cx, Cy = x[4], x[5]
        Lslot1, Lslot2, Lslot3, Lslot4 = x[6], x[7], x[8], x[9]
        Wslot1, Wslot2, Wslot3, Wslot4 = x[10], x[11], x[12], x[13]
        
        # Convertimos los ángulos de grados a radianes para math.sin y math.cos
        a1 = math.radians(x[14])
        a2 = math.radians(x[15])
        a3 = math.radians(x[16])
        a4 = math.radians(x[17])
        
        Off_V, Off_H = x[18], x[19]
        x_feed, y_feed = x[20], x[21]

        # Constantes de seguridad
        R_safe = 1.2
        SM = 0.5
        
        constraints = [0.0] * self.number_of_constraints()

        # 2.1 Límites del parche
        constraints[0] = (W_metal / 2.0 - R_safe) - abs(x_feed)
        constraints[1] = (L_metal / 2.0 - R_safe) - abs(y_feed)
        
        # 2.2 Evitar cortes en las esquinas (Evaluamos los 4 cuadrantes con abs)
        if (abs(x_feed) > W_metal/2.0 - Cx - R_safe) and (abs(y_feed) > L_metal/2.0 - Cy - R_safe):
            constraints[2] = -1.0 # FAIL (Choca con esquina)
        else:
            constraints[2] = 1.0  # OK

        # 2.3 Colisión con los slots (Función de ayuda interna)
        def check_colision_slot(dx, dy, theta_deg, L_slot, W_slot):
            theta_rad = math.radians(theta_deg)
            # Transformación vectorial
            u = dx * math.cos(-theta_rad) - dy * math.sin(-theta_rad)
            v = dx * math.sin(-theta_rad) + dy * math.cos(-theta_rad)
            
            # Se produce colisión si se cumplen TODAS las inecuaciones
            if (u < R_safe) and (u > -(L_slot + R_safe)) and (abs(v) < W_slot/2.0 + R_safe):
                return -1.0 # FAIL
            return 1.0 # OK

        # Slot 1 y 2
        constraints[3] = check_colision_slot(x_feed - Off_V, y_feed - L_metal/2.0, 270.0 + x[14], Lslot1, Wslot1)
        constraints[4] = check_colision_slot(x_feed + Off_V, y_feed + L_metal/2.0, 90.0 + x[15], Lslot2, Wslot2)
        
        # Colisión Slot 3 (Der) y Slot 4 (Izq)
        constraints[5] = check_colision_slot(x_feed - W_metal/2.0, y_feed - Off_H, 180.0 + x[16], Lslot3, Wslot3)
        constraints[6] = check_colision_slot(x_feed + W_metal/2.0, y_feed + Off_H, x[17], Lslot4, Wslot4)
        
        # 2.4 Integridad del núcleo (no cruzar el centro)
        constraints[7] = (L_metal / 2.0 - 2.0) - Lslot1 * math.cos(a1)
        constraints[8] = (L_metal / 2.0 - 2.0) - Lslot2 * math.cos(a2)
        constraints[9] = (W_metal / 2.0 - 2.0) - Lslot3 * math.cos(a3)
        constraints[10]= (W_metal / 2.0 - 2.0) - Lslot4 * math.cos(a4)
        
        # 2.5 Conflictos de borde (crowding)
        # Usamos max(sin, 1e-6) para prevenir errores fatales de división por cero
        constraints[11] = (W_metal / 2.0) - (Cx + abs(Off_H) + Wslot1 / (2.0 * max(math.sin(a1), 1e-6)) + SM)
        constraints[12] = (W_metal / 2.0) - (Cx + abs(Off_H) + Wslot2 / (2.0 * max(math.sin(a2), 1e-6)) + SM)
        constraints[13] = (L_metal / 2.0) - (Cy + abs(Off_V) + Wslot3 / (2.0 * max(math.sin(a3), 1e-6)) + SM)
        constraints[14] = (L_metal / 2.0) - (Cy + abs(Off_V) + Wslot4 / (2.0 * max(math.sin(a4), 1e-6)) + SM)
        # 2.6 Restricciones de consistencia
        constraints[15] = W_GND - (W_metal + 2.0)
        constraints[16] = L_GND - (L_metal + 2.0)
        
        relacion_aspecto = L_metal / W_metal
        constraints[17] = relacion_aspecto - 0.85
        constraints[18] = 1.15 - relacion_aspecto

        # Asignar al array nativo de JMetal
        solution.constraints = constraints
    
    def name(self):
        return "AntenaParcheFloat"