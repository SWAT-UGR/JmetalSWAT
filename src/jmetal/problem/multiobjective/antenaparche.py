import sys
import numpy as np
import os
import multiprocessing
import logging

# =============================================================================
# CONFIGURACIÓN DEL ENTORNO Y CARGA DE DEPENDENCIAS PROPIETARIAS
# =============================================================================

# Inyección dinámica en el PATH del sistema de las librerías de automatización de CST
ruta_cst = r"C:\Program Files\CST Studio Suite 2026\AMD64\python_cst_libraries"
if ruta_cst not in sys.path:
    sys.path.append(ruta_cst)

# Control de acoplamiento de la API de CST: permite la ejecución en local o el bypass en entornos CI/CD (GitHub)
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


# =============================================================================
# CLASE 1: OPTIMIZACIÓN MULTIOBJETIVO DISCRETA (MULTIFRECUENCIA)
# =============================================================================
class antenaparcheInteger(IntegerProblem):
    """
    Clase que formaliza el problema de optimización electromagnética multiobjetivo
    empleando un dominio de variables discretizadas (espacio entero cuantizado).
    Modela una antena monopolo planar con ranuras asimétricas y optimiza de forma
    simultánea el coeficiente de reflexión (S11) y la relación de compromiso axial.
    """
    def __init__(self, rutas_config: dict, number_of_variables: int = 22):
        """:param number_of_variables: Número de variables de decisión geométricas del problema."""
        
        super(antenaparcheInteger, self).__init__()
        
        # Configuración del vector de objetivos: optimización biobjetivo de minimización
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["S11", "axial_ratios"]
        
        # Persistencia de rutas de configuración del sistema de archivos
        self.carpeta_cst = rutas_config['carpeta_cst']
        self.ruta_plantilla_macro = rutas_config['ruta_plantilla_macro']
        
        # Inicialización dinámica del registro de tiempos de cómputo para el análisis de rendimiento
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ruta_csv_tiempos = os.path.join(rutas_config['carpeta_tiempos'], f"estadisticas_tiempos_{timestamp}.csv")
        
        # Definición del entorno local de intercambio de archivos con el servidor COM de CST
        self.ruta_macro_temporal = os.path.join(self.carpeta_cst, "Macro_To_Run.bas")
        self.ruta_archivo_imag = os.path.join(self.carpeta_cst, "ImagS11.txt")
        self.ruta_archivo_real = os.path.join(self.carpeta_cst, "RealS11.txt")
        
        # Vector de granularidad (pasos métricos de cuantización para tolerancias de fabricación)
        self.steps = [0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.05, 0.05]
        
        # Delimitación del hiperespacio de búsqueda en cotas físicas reales (milímetros y grados)
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
            25.0,  # 14: alpha_s1 (min)
            25.0,  # 15: alpha_s2 (min)
            25.0,  # 16: alpha_s3 (min)
            25.0,  # 17: alpha_s4 (min)
            -3.0,  # 18: Off_V (min)
            -3.0,  # 19: Off_H (min)
            -5.0,  # 20: X_feeding (min)
            -5.0   # 21: Y_feeding (min)
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
        
        # Mapeo directo y cuantización al espacio discreto (Genotipo transformado para jMetalPy)
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
        Lee la macro base de Visual Basic (VBA), mapea las variables discretas del 
        algoritmo a sus dimensiones físicas reales e inyecta los nuevos parámetros 
        geométricos mediante expresiones regulares para actualizar el modelo CAD.
        """
        # Estructuración de pares clave-valor alineados con las variables globales de CST
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

        # Lectura del script estructurado en formato script .bas
        with open(ruta_macro_original, 'r') as file:
            contenido_macro = file.read()

        # Análisis sintáctico y sustitución paramétrica mediante regex
        for var_macro, nuevo_valor in valores_redondeados.items():
            
            # Localización selectiva del comando nativo StoreParameter de CST
            patron = r'StoreParameter\("' + var_macro + r'",\s*[-+]?[0-9]*\.?[0-9]+\)'
            reemplazo = f'StoreParameter("{var_macro}",{nuevo_valor})'
            
            # Ejecución de la sustitución en la cadena de texto en memoria
            contenido_macro = re.sub(patron, reemplazo, contenido_macro)

        # Volcado al disco del script ejecutable dinámico resultante
        with open(ruta_macro_salida, 'w') as file:
            file.write(contenido_macro)
            
        return valores_redondeados
    
    
    def ejecutar_cst_en_segundo_plano(self, ruta_macro):
        """
        Orquesta el lazo de automatización COM OLE para instanciar CST Studio Suite 
        en modo oculto, compilar las primitivas geométricas 3D del diseño, lanzar 
        el motor numérico de simulación electromagnética y liberar la RAM del sistema.
        """
        print("Iniciando CST en segundo plano...")
        
        # Inicialización del servidor de automatización en modo headless y silencioso
        cstEnv = cst.interface.DesignEnvironment.new(['--hide', '--quiet']) 
        cst.interface.DesignEnvironment.set_quiet_mode(cstEnv, True)
        
        try:
            # Creación de una instancia de Microwave Studio limpia
            cstProject = cst.interface.DesignEnvironment.new_mws(cstEnv)
            
            # Carga del bloque de instrucciones macro funcionales modificadas
            with open(ruta_macro, 'r', encoding='utf-8') as file:
                vbaCode = file.read()
                
            # Envío y control remoto de la simulación de onda completa en 3D
            print("Ejecutando simulación...")
            cstProject.schematic.execute_vba_code(vbaCode, timeout=None)
            
            print("Simulación completada con éxito.")
            
        except Exception as e:
            print(f"Error durante la simulación de CST: {e}")
            
        finally:
            # Recolección de basura forzada y cierre del entorno OLE para evitar fugas de memoria
            if 'cstProject' in locals():
                cstProject = None 
            
            if 'cstEnv' in locals() and cstEnv is not None:
                cst.interface.DesignEnvironment.close(cstEnv)
            
            # Retardo operativo de cortesía para la sincronización de hilos en el kernel de Windows
            time.sleep(2)
            
            print("CST cerrado correctamente. Memoria liberada.")

    def _worker_cst(self, ruta_macro, queue):
        """
        Función envoltorio secundaria (Worker) diseñada para ejecutarse de forma aislada
        dentro de la zona de cuarentena multiprocessing, enviando los resultados al canal síncrono.
        """
        try:
            resultados = self.ejecutar_cst_en_segundo_plano(ruta_macro)
            queue.put(resultados)
        except Exception as e:
            print(f"Error en el worker: {e}")
            queue.put(None)            
            
    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        """
        Método central de evaluación heurística de jMetalPy. Decodifica los individuos,
        aplica las restricciones, gestiona la ejecución distribuida con tolerancia a fallos
        mediante cortafuegos computacionales y calcula la función de aptitud multifrecuencia.
        """
        tiempo_inicio = time.time()
        
        # Filtro de optimización Warm-Start: bypass computacional para soluciones precalculadas
        if solution.attributes.get('viene_de_warm_start', False):
            solution.attributes['viene_de_warm_start'] = False
            return solution
        
        # Decodificación matemática: paso del genotipo discreto al fenotipo real continuo
        vars_fisicas = [var * step for var, step in zip(solution.variables, self.steps)]
        
        # Evaluación preliminar del vector de restricciones geométricas
        self.__evaluate_constraints(vars_fisicas, solution)
        
        # Comprobación de viabilidad topológica antes de iniciar la malla 3D
        es_valida = all(c >= 0 for c in solution.constraints)
        
        if not es_valida:
            # Estrategia de penalización letal inmediata: evita sobrecostes en simulaciones imposibles
            solution.objectives[0] = 10e8
            solution.objectives[1] = 10e8
            
            solution.attributes['frecuencia'] = None 
            
            # Registro temporal del coste de la evaluación analítica descartada
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
        
        # Preparación operativa de ficheros temporales para el paso de variables
        ruta_plantilla_macro = self.ruta_plantilla_macro
        ruta_macro_temporal  = self.ruta_macro_temporal
        
        self.generar_macro_cst(vars_fisicas, ruta_plantilla_macro, ruta_macro_temporal)
        
        # Arquitectura de Sandbox: instanciación de un canal IPC seguro mediante cola síncrona
        queue = multiprocessing.Queue()
        
        # Lanzamiento aislado del proceso secundario clonado (evita corrupciones del servidor COM)
        p = multiprocessing.Process(target=self._worker_cst, args=(ruta_macro_temporal, queue))
        p.start()
        
        # Mecanismo Cortafuegos (Watchdog): watchdog temporal estricto fijado en 10 minutos
        p.join(timeout=600) 
        
        if p.is_alive():
            print("¡TIMEOUT! La antena se ha atascado. Exterminando proceso...")
            
            # Aborto seguro del subproceso mutado para mitigar fugas de subhilos
            p.terminate()
            p.join()
            
            # Purga drástica del árbol de procesos del sistema para liberar núcleos hardware
            procesos_zombi = ["CST_DESIGN_ENVIRONMENT.exe", "cst.exe", "CST_MICROWAVE_STUDIO.exe", "CST_MWS.exe"]
            for zombi in procesos_zombi:
                os.system(f"taskkill /F /IM {zombi} /T >nul 2>&1")
                
            time.sleep(3) 
            
            # Penalización masiva por colapso de convergencia geométrica o inestabilidad
            solution.objectives[0] = 10e8
            solution.objectives[1] = 10e8
            
            # Inyección de vectores nulos estructurados para la consistencia de los observadores
            solution.attributes['vars_fisicas'] = vars_fisicas
            solution.attributes['frecuencia'] = np.array([0])
            solution.attributes['s11_imag'] = np.array([0])
            solution.attributes['s11_real'] = np.array([0])
            solution.attributes['s11_db'] = np.array([0])
            solution.attributes['frec_farfield'] = [0]
            solution.attributes['axial_ratios'] = [0]
            
        else:
            # PROCESAMIENTO EXITOSO: Extracción analítica de los datos electromagnéticos exportados
            datos_imag = np.loadtxt(self.ruta_archivo_imag, skiprows=2)
            datos_real = np.loadtxt(self.ruta_archivo_real, skiprows=2)
            
            frecuencia = datos_imag[:, 0]
            s11_imag = datos_imag[:, 1]
            s11_real = datos_real[:, 1]
            
            # Formulación del módulo en decibelios a partir del espectro complejo
            s11_complejo = s11_real + 1j * s11_imag
            s11_db = 20 * np.log10(np.abs(s11_complejo))
            
            # Parseo de archivos de campo lejano (Farfield) en la banda de análisis multifrecuencia
            frecuencias_evaluar = ["4_8", "4_9", "5", "5_1", "5_2"]
            rutas_farfield = [os.path.join(self.carpeta_cst, f"Farfield_f{f}.txt") for f in frecuencias_evaluar]
            
            frecuencias_farfield = []
            axial_ratios = []
            
            for ruta in rutas_farfield:
                nombre_archivo = os.path.basename(ruta)
                texto_num = nombre_archivo.replace("Farfield_f", "").replace(".txt", "")
                frec_val = float(texto_num.replace("_", "."))
                
                # Extracción directa del Axial Ratio en la dirección del lóbulo principal
                datos_ff = np.loadtxt(ruta, skiprows=2)
                primer_ax_ratio = datos_ff[0, 7] 
                
                frecuencias_farfield.append(frec_val)
                axial_ratios.append(primer_ax_ratio)
            
            # Asignación del fitness multiobjetivo: promedio espectral integrado de las pérdidas de retorno y pureza circular
            solution.objectives[0] = np.sum(s11_db)/len(s11_db)
            solution.objectives[1] = np.sum(axial_ratios)/len(axial_ratios)
            
            # Almacenamiento de metadatos estáticos para el motor gráfico y análisis de Pareto
            solution.attributes['vars_fisicas'] = vars_fisicas
            solution.attributes['frecuencia'] = frecuencia
            solution.attributes['s11_imag'] = s11_imag
            solution.attributes['s11_real'] = s11_real
            solution.attributes['s11_db'] = s11_db
            solution.attributes['frec_farfield'] = frecuencias_farfield
            solution.attributes['axial_ratios'] = axial_ratios
              
        # Volcado de métricas de temporización global del bucle iterativo al CSV
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
        Filtro analítico preventivo de restricciones topológicas. Implementa matrices de 
        rototranslación euclídea de 19 inecuaciones geométricas en el plano para blindar la 
        malla frente a colisiones físicas, solapamientos e incoherencias métricas estructurales.
        """
        x = vars_fisicas
        
        # Mapeo de variables de decisión geométricas
        L_metal, W_metal = x[0], x[1]
        L_GND, W_GND = x[2], x[3]
        Cx, Cy = x[4], x[5]
        Lslot1, Lslot2, Lslot3, Lslot4 = x[6], x[7], x[8], x[9]
        Wslot1, Wslot2, Wslot3, Wslot4 = x[10], x[11], x[12], x[13]
        
        # Transformación a radianes para evaluación analítica trigonométrica
        a1 = math.radians(x[14])
        a2 = math.radians(x[15])
        a3 = math.radians(x[16])
        a4 = math.radians(x[17])
        
        Off_V, Off_H = x[18], x[19]
        x_feed, y_feed = x[20], x[21]

        # Márgenes de guarda técnicos y tolerancias físicas de seguridad
        R_safe = 1.2
        SM = 0.5
        
        constraints = [0.0] * self.number_of_constraints()

        # 2.1 Restricciones perimetrales del pin de alimentación sobre el parche
        constraints[0] = (W_metal / 2.0 - R_safe) - abs(x_feed)
        constraints[1] = (L_metal / 2.0 - R_safe) - abs(y_feed)
        
        # 2.2 Validación analítica tridimensional de los chaflanes en los cuatro cuadrantes
        if (abs(x_feed) > W_metal/2.0 - Cx - R_safe) and (abs(y_feed) > L_metal/2.0 - Cy - R_safe):
            constraints[2] = -1.0 # Estado no factible (colisión con corte de esquina)
        else:
            constraints[2] = 1.0  # Estado factible

        # 2.3 Subfunción vectorial para la detección de colisiones e intersección de ranuras inclinadas
        def check_colision_slot(dx, dy, theta_deg, L_slot, W_slot):
            theta_rad = math.radians(theta_deg)
            # Aplicación de matriz de rotación euclídea bidimensional
            u = dx * math.cos(-theta_rad) - dy * math.sin(-theta_rad)
            v = dx * math.sin(-theta_rad) + dy * math.cos(-theta_rad)
            
            # Evaluación del solapamiento en el espacio transformado del plano local de la ranura
            if (u < R_safe) and (u > -(L_slot + R_safe)) and (abs(v) < W_slot/2.0 + R_safe):
                return -1.0 
            return 1.0 

        # Evaluación de colisión para el grupo simétrico de Slots 1 y 2
        constraints[3] = check_colision_slot(x_feed - Off_V, y_feed - L_metal/2.0, 270.0 + x[14], Lslot1, Wslot1)
        constraints[4] = check_colision_slot(x_feed + Off_V, y_feed + L_metal/2.0, 90.0 + x[15], Lslot2, Wslot2)
        
        # Evaluación de colisión para el grupo ortogonal de Slots 3 (Derecha) y 4 (Izquierda)
        constraints[5] = check_colision_slot(x_feed - W_metal/2.0, y_feed - Off_H, 180.0 + x[16], Lslot3, Wslot3)
        constraints[6] = check_colision_slot(x_feed + W_metal/2.0, y_feed + Off_H, x[17], Lslot4, Wslot4)
        
        # 2.4 Restricciones de integridad nuclear: protección del baricentro geométrico del parche
        constraints[7] = (L_metal / 2.0 - 2.0) - Lslot1 * math.cos(a1)
        constraints[8] = (L_metal / 2.0 - 2.0) - Lslot2 * math.cos(a2)
        constraints[9] = (W_metal / 2.0 - 2.0) - Lslot3 * math.cos(a3)
        constraints[10]= (W_metal / 2.0 - 2.0) - Lslot4 * math.cos(a4)
        
        # 2.5 Conflictos de acoplamiento de bordes y desbordamiento topológico (Crowding)
        # Inyección de un épsilon de seguridad (1e-6) para blindar el lazo contra divisiones por cero
        constraints[11] = (W_metal / 2.0) - (Cx + abs(Off_H) + Wslot1 / (2.0 * max(math.sin(a1), 1e-6)) + SM)
        constraints[12] = (W_metal / 2.0) - (Cx + abs(Off_H) + Wslot2 / (2.0 * max(math.sin(a2), 1e-6)) + SM)
        constraints[13] = (L_metal / 2.0) - (Cy + abs(Off_V) + Wslot3 / (2.0 * max(math.sin(a3), 1e-6)) + SM)
        constraints[14] = (L_metal / 2.0) - (Cy + abs(Off_V) + Wslot4 / (2.0 * max(math.sin(a4), 1e-6)) + SM)
        
        # 2.6 Restricciones estructurales de consistencia respecto al sustrato dieléctrico y masa (GND)
        constraints[15] = W_GND - (W_metal + 2.0)
        constraints[16] = L_GND - (L_metal + 2.0)
        
        # Control del factor de forma de la antena monopolo
        relacion_aspecto = L_metal / W_metal
        constraints[17] = relacion_aspecto - 0.85
        constraints[18] = 1.15 - relacion_aspecto

        solution.constraints = constraints
    
    def name(self):
        return "AntenaParcheInteger"
    

# =============================================================================
# CLASE 2: CASO DE ESTUDIO CRÍTICO MONOFRECUENCIA (5 GHz)
# =============================================================================
class antenaparcheIntegerMono(IntegerProblem):
    """
    Variante especializada del problema discreto enfocada exclusivamente en el 
    escenario de sintonización monofrecuencia a 5 GHz. Modifica la función de evaluación 
    para extraer y optimizar el rendimiento puntual, consolidando la máxima relación axial.
    """
    def __init__(self, rutas_config: dict, number_of_variables: int = 22):
        super(antenaparcheIntegerMono, self).__init__()
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["S11", "axial_ratios"]
        
        self.carpeta_cst = rutas_config['carpeta_cst']
        self.ruta_plantilla_macro = rutas_config['ruta_plantilla_macro']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ruta_csv_tiempos = os.path.join(rutas_config['carpeta_tiempos'], f"estadisticas_tiempos_{timestamp}.csv")
        
        self.ruta_macro_temporal = os.path.join(self.carpeta_cst, "Macro_To_Run.bas")
        self.ruta_archivo_imag = os.path.join(self.carpeta_cst, "ImagS11.txt")
        self.ruta_archivo_real = os.path.join(self.carpeta_cst, "RealS11.txt")
        
        self.steps = [0.1, 0.1, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.05, 0.05]
        
        lower_bound_real = [
            12.0, 12.0, 30.0, 30.0, 1.0, 1.0, 2.5, 2.5, 2.5, 2.5, 0.5, 0.5, 0.5, 0.5, 25.0, 25.0, 25.0, 25.0, -3.0, -3.0, -5.0, -5.0
        ]
        
        upper_bound_real = [
            15.5, 15.5, 55.0, 55.0, 4.5, 4.5, 6.5, 6.5, 6.5, 6.5, 2.0, 2.0, 2.0, 2.0, 65.0, 65.0, 65.0, 65.0, 3.0, 3.0, 5.0, 5.0
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
        valores_redondeados = {
            "L_metal":  variables_jmetal[0], "W_metal":  variables_jmetal[1], "L_GND":    variables_jmetal[2], "W_GND":    variables_jmetal[3],
            "Cx":       variables_jmetal[4], "Cy":       variables_jmetal[5], "Lslot1":   variables_jmetal[6], "Lslot2":   variables_jmetal[7],
            "Lslot3":   variables_jmetal[8], "Lslot4":   variables_jmetal[9], "Wslot1":   variables_jmetal[10],"Wslot2":   variables_jmetal[11],
            "Wslot3":   variables_jmetal[12],"Wslot4":   variables_jmetal[13],"alpha_s1": variables_jmetal[14],"alpha_s2": variables_jmetal[15],
            "alpha_s3": variables_jmetal[16],"alpha_s4": variables_jmetal[17],"Off_V":    variables_jmetal[18],"Off_H":    variables_jmetal[19],
            "X_feeding":   variables_jmetal[20],"Y_feeding":  variables_jmetal[21]
        }

        with open(ruta_macro_original, 'r') as file:
            contenido_macro = file.read()

        for var_macro, nuevo_valor in valores_redondeados.items():
            patron = r'StoreParameter\("' + var_macro + r'",\s*[-+]?[0-9]*\.?[0-9]+\)'
            reemplazo = f'StoreParameter("{var_macro}",{nuevo_valor})'
            contenido_macro = re.sub(patron, reemplazo, contenido_macro)

        with open(ruta_macro_salida, 'w') as file:
            file.write(contenido_macro)
            
        return valores_redondeados
    
    
    def ejecutar_cst_en_segundo_plano(self, ruta_macro):
        print("Iniciando CST en segundo plano...")
        cstEnv = cst.interface.DesignEnvironment.new(['--hide', '--quiet']) 
        cst.interface.DesignEnvironment.set_quiet_mode(cstEnv, True)
        
        try:
            cstProject = cst.interface.DesignEnvironment.new_mws(cstEnv)
            with open(ruta_macro, 'r', encoding='utf-8') as file:
                vbaCode = file.read()
                
            print("Ejecutando simulación...")
            cstProject.schematic.execute_vba_code(vbaCode, timeout=None)
            print("Simulación completada con éxito.")
            
        except Exception as e:
            print(f"Error durante la simulación de CST: {e}")
            
        finally:
            if 'cstProject' in locals():
                cstProject = None 
            if 'cstEnv' in locals() and cstEnv is not None:
                cst.interface.DesignEnvironment.close(cstEnv)
            time.sleep(2)
            print("CST cerrado correctamente. Memoria liberada.")

    def _worker_cst(self, ruta_macro, queue):
        try:
            resultados = self.ejecutar_cst_en_segundo_plano(ruta_macro)
            queue.put(resultados)
        except Exception as e:
            print(f"Error en el worker: {e}")
            queue.put(None)            
            
    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        """
        Método de evaluación adaptado para el escenario monofrecuencia. Filtra el índice 
        espectral exacto de la componente de 5 GHz en régimen permanente y asigna la 
        relación de compromiso axial óptima.
        """
        tiempo_inicio = time.time()
        
        if solution.attributes.get('viene_de_warm_start', False):
            solution.attributes['viene_de_warm_start'] = False
            return solution
        
        vars_fisicas = [var * step for var, step in zip(solution.variables, self.steps)]
        self.__evaluate_constraints(vars_fisicas, solution)
        es_valida = all(c >= 0 for c in solution.constraints)
        
        if not es_valida:
            solution.objectives[0] = 10e8
            solution.objectives[1] = 10e8
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
        
        ruta_plantilla_macro = self.ruta_plantilla_macro
        ruta_macro_temporal  = self.ruta_macro_temporal
        self.generar_macro_cst(vars_fisicas, ruta_plantilla_macro, ruta_macro_temporal)
        
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=self._worker_cst, args=(ruta_macro_temporal, queue))
        p.start()
        p.join(timeout=600) 
        
        if p.is_alive():
            print("¡TIMEOUT! La antena se ha atascado. Exterminando proceso...")
            p.terminate()
            p.join()
            
            procesos_zombi = ["CST_DESIGN_ENVIRONMENT.exe", "cst.exe", "CST_MICROWAVE_STUDIO.exe", "CST_MWS.exe"]
            for zombi in procesos_zombi:
                os.system(f"taskkill /F /IM {zombi} /T >nul 2>&1")
                
            time.sleep(3)
            solution.objectives[0] = 10e8
            solution.objectives[1] = 10e8
            
            solution.attributes['vars_fisicas'] = vars_fisicas
            solution.attributes['frecuencia'] = np.array([0])
            solution.attributes['s11_imag'] = np.array([0])
            solution.attributes['s11_real'] = np.array([0])
            solution.attributes['s11_db'] = np.array([0])
            solution.attributes['frec_farfield'] = [0]
            solution.attributes['axial_ratios'] = [0]
            
        else:
            datos_imag = np.loadtxt(self.ruta_archivo_imag, skiprows=2)
            datos_real = np.loadtxt(self.ruta_archivo_real, skiprows=2)
            
            frecuencia = datos_imag[:, 0]
            s11_imag = datos_imag[:, 1]
            s11_real = datos_real[:, 1]
            
            s11_complejo = s11_real + 1j * s11_imag
            s11_db = 20 * np.log10(np.abs(s11_complejo))
            
            # --- CORRECCIÓN CRÍTICA MONOFRECUENCIA: Extracción espectral a 5.0 GHz ---
            # Búsqueda del índice euclídeo mínimo respecto a la frecuencia de diseño central
            indice_5ghz = np.argmin(np.abs(frecuencia - 5.0))
            s11_objetivo = s11_db[indice_5ghz] 
            
            # Carga del patrón electromagnético bidimensional puntual de campo lejano
            ruta_farfield_5 = os.path.join(self.carpeta_cst, "Farfield_f5.txt")
            datos_ff = np.loadtxt(ruta_farfield_5, skiprows=2)
            axial_ratio_objetivo = datos_ff[0, 7] 
            
            # Inyección de las respuestas físicas puntuales en el vector objetivo de jMetalPy
            solution.objectives[0] = s11_objetivo
            solution.objectives[1] = axial_ratio_objetivo
            
            solution.attributes['vars_fisicas'] = vars_fisicas
            solution.attributes['frecuencia'] = frecuencia
            solution.attributes['s11_imag'] = s11_imag
            solution.attributes['s11_real'] = s11_real
            solution.attributes['s11_db'] = s11_db
            
            solution.attributes['frec_farfield'] = [5.0]
            solution.attributes['axial_ratios'] = [axial_ratio_objetivo]
            
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
        x = vars_fisicas
        L_metal, W_metal = x[0], x[1]
        L_GND, W_GND = x[2], x[3]
        Cx, Cy = x[4], x[5]
        Lslot1, Lslot2, Lslot3, Lslot4 = x[6], x[7], x[8], x[9]
        Wslot1, Wslot2, Wslot3, Wslot4 = x[10], x[11], x[12], x[13]
        
        a1 = math.radians(x[14])
        a2 = math.radians(x[15])
        a3 = math.radians(x[16])
        a4 = math.radians(x[17])
        
        Off_V, Off_H = x[18], x[19]
        x_feed, y_feed = x[20], x[21]

        R_safe = 1.2
        SM = 0.5
        
        constraints = [0.0] * self.number_of_constraints()

        constraints[0] = (W_metal / 2.0 - R_safe) - abs(x_feed)
        constraints[1] = (L_metal / 2.0 - R_safe) - abs(y_feed)
        
        if (abs(x_feed) > W_metal/2.0 - Cx - R_safe) and (abs(y_feed) > L_metal/2.0 - Cy - R_safe):
            constraints[2] = -1.0 
        else:
            constraints[2] = 1.0  

        def check_colision_slot(dx, dy, theta_deg, L_slot, W_slot):
            theta_rad = math.radians(theta_deg)
            u = dx * math.cos(-theta_rad) - dy * math.sin(-theta_rad)
            v = dx * math.sin(-theta_rad) + dy * math.cos(-theta_rad)
            if (u < R_safe) and (u > -(L_slot + R_safe)) and (abs(v) < W_slot/2.0 + R_safe):
                return -1.0 
            return 1.0 

        constraints[3] = check_colision_slot(x_feed - Off_V, y_feed - L_metal/2.0, 270.0 + x[14], Lslot1, Wslot1)
        constraints[4] = check_colision_slot(x_feed + Off_V, y_feed + L_metal/2.0, 90.0 + x[15], Lslot2, Wslot2)
        
        constraints[5] = check_colision_slot(x_feed - W_metal/2.0, y_feed - Off_H, 180.0 + x[16], Lslot3, Wslot3)
        constraints[6] = check_colision_slot(x_feed + W_metal/2.0, y_feed + Off_H, x[17], Lslot4, Wslot4)
        
        constraints[7] = (L_metal / 2.0 - 2.0) - Lslot1 * math.cos(a1)
        constraints[8] = (L_metal / 2.0 - 2.0) - Lslot2 * math.cos(a2)
        constraints[9] = (W_metal / 2.0 - 2.0) - Lslot3 * math.cos(a3)
        constraints[10]= (W_metal / 2.0 - 2.0) - Lslot4 * math.cos(a4)
        
        constraints[11] = (W_metal / 2.0) - (Cx + abs(Off_H) + Wslot1 / (2.0 * max(math.sin(a1), 1e-6)) + SM)
        constraints[12] = (W_metal / 2.0) - (Cx + abs(Off_H) + Wslot2 / (2.0 * max(math.sin(a2), 1e-6)) + SM)
        constraints[13] = (L_metal / 2.0) - (Cy + abs(Off_V) + Wslot3 / (2.0 * max(math.sin(a3), 1e-6)) + SM)
        constraints[14] = (L_metal / 2.0) - (Cy + abs(Off_V) + Wslot4 / (2.0 * max(math.sin(a4), 1e-6)) + SM)
        
        constraints[15] = W_GND - (W_metal + 2.0)
        constraints[16] = L_GND - (L_metal + 2.0)
        
        relacion_aspecto = L_metal / W_metal
        constraints[17] = relacion_aspecto - 0.85
        constraints[18] = 1.15 - relacion_aspecto

        solution.constraints = constraints
        
    def name(self):
        return "AntenaParcheInteger"
    

# =============================================================================
# CLASE 3: OPTIMIZACIÓN MULTIOBJETIVO CONTINUA (VARIANTE DE CONTROL FLOAT)
# =============================================================================
class antenaparcheFloat(FloatProblem):
    """
    Clase que implementa el problema de optimización electromagnética operando
    en un espacio analítico estrictamente continuo (Float). Sirve como base de control
    para contrastar la efectividad del mapeo por granularidad (*steps*).
    """
    def __init__(self, number_of_variables: int = 22):
        """:param number_of_variables: Espacio continuo ℝ^22 para optimización puramente flotante."""
        
        super(antenaparcheFloat, self).__init__()
        
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["S11", "axial_ratios"]
        
        # Asignación nativa de cotas métricas continuas sin paso adaptativo previo
        self.lower_bound = [
            12.0, 12.0, 30.0, 30.0, 1.0, 1.0, 2.5, 2.5, 2.5, 2.5, 0.5, 0.5, 0.5, 0.5, 25.0, 25.0, 25.0, 25.0, -3.0, -3.0, -5.0, -5.0
        ]
        
        self.upper_bound = [
            15.5, 15.5, 55.0, 55.0, 4.5, 4.5, 6.5, 6.5, 6.5, 6.5, 2.0, 2.0, 2.0, 2.0, 65.0, 65.0, 65.0, 65.0, 3.0, 3.0, 5.0, 5.0
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
        Ajusta una variable flotante continua de entrada al incremento discreto normalizado 
        más cercano con una precisión fija de tres decimales.
        """
        return round(round(valor / paso) * paso, 3)

    def generar_macro_cst(self, variables_jmetal, ruta_macro_original, ruta_macro_salida):
        """
        Sobrecarga del generador de macros: aplica la cuantización puntual sobre el espacio continuo
        de jMetalPy de manera explícita antes de inyectar las variables en el script .bas.
        """
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
            "Y_feeding":   self.redondear_al_paso(variables_jmetal[21], 0.05)
        }
        
        with open(ruta_macro_original, 'r') as file:
            contenido_macro = file.read()

        for var_macro, nuevo_valor in valores_redondeados.items():
            patron = r'StoreParameter\("' + var_macro + r'",\s*[-+]?[0-9]*\.?[0-9]+\)'
            reemplazo = f'StoreParameter("{var_macro}",{nuevo_valor})'
            contenido_macro = re.sub(patron, reemplazo, contenido_macro)

        with open(ruta_macro_salida, 'w') as file:
            file.write(contenido_macro)
            
        return valores_redondeados
    
    
    def ejecutar_cst_en_segundo_plano(self, ruta_macro):
        print("Iniciando CST en segundo plano...")
        
        cstEnv = cst.interface.DesignEnvironment.new(['--hide', '--quiet']) 
        cst.interface.DesignEnvironment.set_quiet_mode(cstEnv, True)
        
        try:
            cstProject = cst.interface.DesignEnvironment.new_mws(cstEnv)
            with open(ruta_macro, 'r', encoding='utf-8') as file:
                vbaCode = file.read()
                
            print("Ejecutando simulación...")
            cstProject.schematic.execute_vba_code(vbaCode, timeout=None)
            print("Simulación completada con éxito.")
            
        except Exception as e:
            print(f"Error durante la simulación de CST: {e}")
            
        finally:
            if 'cstProject' in locals():
                cstProject.close()
            cst.interface.DesignEnvironment.close(cstEnv)
            print("CST cerrado. Memoria liberada.")
            
            
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        """
        Método de evaluación continua. Orquesta de forma síncrona local la 
        inyección de los parámetros reales continuos y calcula la aptitud multifrecuencia.
        """
        tiempo_inicio = time.time()
        vars_algoritmo = solution.variables
        
        self.__evaluate_constraints(vars_algoritmo, solution)
        es_valida = all(c >= 0 for c in solution.constraints)
        
        if not es_valida:
            solution.objectives[0] = 10e8
            solution.objectives[1] = 10e8
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
        
        ruta_plantilla_macro = r"C:\Users\jmate\Desktop\teleco\TFG\CST\MacroPatchCP_Multilayer_Cross.txt"
        ruta_macro_temporal  = r"C:\Users\jmate\Desktop\teleco\TFG\CST\Macro_To_Run.bas"
        
        self.generar_macro_cst(vars_algoritmo, ruta_plantilla_macro, ruta_macro_temporal)
        self.ejecutar_cst_en_segundo_plano(ruta_macro_temporal)
       
        ruta_archivo_imag = r"C:\Users\jmate\Desktop\teleco\TFG\CST\ImagS11.txt"
        datos_imag = np.loadtxt(ruta_archivo_imag, skiprows=2)
        
        ruta_archivo_real = r"C:\Users\jmate\Desktop\teleco\TFG\CST\RealS11.txt"
        datos_real = np.loadtxt(ruta_archivo_real, skiprows=2)
        frecuencia = datos_imag[:, 0]
        
        s11_imag = datos_imag[:, 1]
        s11_real = datos_real[:, 1]
        
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
        
        for ruta in rutas_farfield:
            nombre_archivo = os.path.basename(ruta)
            texto_num = nombre_archivo.replace("Farfield_f", "").replace(".txt", "")
            frec_val = float(texto_num.replace("_", "."))
            
            datos_ff = np.loadtxt(ruta, skiprows=2)
            primer_ax_ratio = datos_ff[0, 7] 
            
            frecuencias_farfield.append(frec_val)
            axial_ratios.append(primer_ax_ratio)
        
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
        x = vars_fisicas
        L_metal, W_metal = x[0], x[1]
        L_GND, W_GND = x[2], x[3]
        Cx, Cy = x[4], x[5]
        Lslot1, Lslot2, Lslot3, Lslot4 = x[6], x[7], x[8], x[9]
        Wslot1, Wslot2, Wslot3, Wslot4 = x[10], x[11], x[12], x[13]
        
        a1 = math.radians(x[14])
        a2 = math.radians(x[15])
        a3 = math.radians(x[16])
        a4 = math.radians(x[17])
        
        Off_V, Off_H = x[18], x[19]
        x_feed, y_feed = x[20], x[21]

        R_safe = 1.2
        SM = 0.5
        
        constraints = [0.0] * self.number_of_constraints()

        constraints[0] = (W_metal / 2.0 - R_safe) - abs(x_feed)
        constraints[1] = (L_metal / 2.0 - R_safe) - abs(y_feed)
        
        if (abs(x_feed) > W_metal/2.0 - Cx - R_safe) and (abs(y_feed) > L_metal/2.0 - Cy - R_safe):
            constraints[2] = -1.0 
        else:
            constraints[2] = 1.0  

        def check_colision_slot(dx, dy, theta_deg, L_slot, W_slot):
            theta_rad = math.radians(theta_deg)
            u = dx * math.cos(-theta_rad) - dy * math.sin(-theta_rad)
            v = dx * math.sin(-theta_rad) + dy * math.cos(-theta_rad)
            if (u < R_safe) and (u > -(L_slot + R_safe)) and (abs(v) < W_slot/2.0 + R_safe):
                return -1.0 
            return 1.0 

        constraints[3] = check_colision_slot(x_feed - Off_V, y_feed - L_metal/2.0, 270.0 + x[14], Lslot1, Wslot1)
        constraints[4] = check_colision_slot(x_feed + Off_V, y_feed + L_metal/2.0, 90.0 + x[15], Lslot2, Wslot2)
        
        constraints[5] = check_colision_slot(x_feed - W_metal/2.0, y_feed - Off_H, 180.0 + x[16], Lslot3, Wslot3)
        constraints[6] = check_colision_slot(x_feed + W_metal/2.0, y_feed + Off_H, x[17], Lslot4, Wslot4)
        
        constraints[7] = (L_metal / 2.0 - 2.0) - Lslot1 * math.cos(a1)
        constraints[8] = (L_metal / 2.0 - 2.0) - Lslot2 * math.cos(a2)
        constraints[9] = (W_metal / 2.0 - 2.0) - Lslot3 * math.cos(a3)
        constraints[10]= (W_metal / 2.0 - 2.0) - Lslot4 * math.cos(a4)
        
        constraints[11] = (W_metal / 2.0) - (Cx + abs(Off_H) + Wslot1 / (2.0 * max(math.sin(a1), 1e-6)) + SM)
        constraints[12] = (W_metal / 2.0) - (Cx + abs(Off_H) + Wslot2 / (2.0 * max(math.sin(a2), 1e-6)) + SM)
        constraints[13] = (L_metal / 2.0) - (Cy + abs(Off_V) + Wslot3 / (2.0 * max(math.sin(a3), 1e-6)) + SM)
        constraints[14] = (L_metal / 2.0) - (Cy + abs(Off_V) + Wslot4 / (2.0 * max(math.sin(a4), 1e-6)) + SM)
        
        constraints[15] = W_GND - (W_metal + 2.0)
        constraints[16] = L_GND - (L_metal + 2.0)
        
        relacion_aspecto = L_metal / W_metal
        constraints[17] = relacion_aspecto - 0.85
        constraints[18] = 1.15 - relacion_aspecto

        solution.constraints = constraints
        
    def name(self):
        return "AntenaParcheFloat"