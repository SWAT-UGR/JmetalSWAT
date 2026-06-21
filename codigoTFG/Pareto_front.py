import matplotlib.pyplot as plt
import matplotlib.patches as patches  # Generación de primitivas geométricas para el acotado de regiones factibles
import numpy as np
import os
import glob

# =============================================================================
# CONFIGURACIÓN DE PARÁMETROS GRÁFICOS DE MATPLOTLIB ADAPTADOS PARA LATEX
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',   # Cambia la fuente al estilo clásico de LaTeX
    'font.size': 14,          # Escala base de la fuente para garantizar legibilidad en maquetación A4
    'axes.labelsize': 16,     # Escala de las etiquetas descriptivas de los objetivos métricos
    'xtick.labelsize': 14,    # Dimensión tipográfica de los índices numéricos del eje X
    'ytick.labelsize': 14,    # Dimensión tipográfica de los índices numéricos del eje Y
    'legend.fontsize': 14     # Escala de la leyenda para evitar pérdida de resolución al vectorizar
    # Se han eliminado los tamaños de título porque se delega en el \caption de LaTeX
})
# ---------------------------------------------------------------------

# --- CONFIGURACIÓN DE LOS EJES (Envolventes ortogonales continuas de visualización) ---
X_MIN = -50  # Cota inferior del espacio de objetivos en el eje X (Pérdidas de retorno S11)
X_MAX = 0    # Cota superior del espacio de objetivos en el eje X (Pérdidas de retorno S11)
Y_MIN = -0.5 # Cota inferior del espacio de objetivos en el eje Y (Relación de compromiso axial)
Y_MAX = 20   # Cota superior del espacio de objetivos en el eje Y (Relación de compromiso axial)
# ---------------------------------------------------------------------

def get_non_dominated_solutions(puntos):
    """
    Filtra una matriz bidimensional de objetivos aplicando el criterio de optimalidad de Pareto.
    Aísla las soluciones no dominadas correspondientes a un espacio matemático de minimización dual.
    
    :param puntos: Matriz con las coordenadas [Obj1, Obj2] extraídas del simulador.
    :return: Matriz filtrada conteniendo exclusivamente los vectores pertenecientes al frente de Pareto.
    """
    no_dominados = []
    for i, p1 in enumerate(puntos):
        dominado = False
        for j, p2 in enumerate(puntos):
            if i != j:
                # Criterio matemático de dominancia de Pareto para problemas de minimización pura:
                # p2 domina a p1 si ningún objetivo de p2 es peor y al menos uno es estrictamente mejor.
                if (p2[0] <= p1[0] and p2[1] <= p1[1]) and (p2[0] < p1[0] or p2[1] < p1[1]):
                    dominado = True
                    break
        if not dominado:
            no_dominados.append(p1)
    return np.array(no_dominados)

def extraer_numero(ruta):
    """
    Analiza el string de la ruta del sistema de archivos mediante descomposición por tokens
    para aislar y extraer el índice entero de la evaluación del algoritmo (e.g., FUN.Problema.100 -> 100).
    """
    try:
        return int(ruta.split('.')[-1])
    except ValueError:
        return 0

# --- CONFIGURACIÓN DE DIRECTORIOS Y ENTORNO DE TRABAJO ---
carpeta_resultados = r"C:\Users\jmate\Desktop\teleco\TFG\Experimentacion\Resultados_NSGAII\todo"
output_directory = os.path.join(carpeta_resultados, "todo")

# Verificación y creación del directorio destinado al almacenamiento de las gráficas vectoriales
if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

# Recuperación indexada y ordenación cronológica de los históricos de funciones objetivo (FUN)
archivos_fun = glob.glob(os.path.join(carpeta_resultados, "FUN.*"))
archivos_fun.sort(key=extraer_numero)

history = []
objective_labels = ["S11 (dB) [Minimizar]", "Axial Ratio (dB) [Minimizar]"]
penalty_threshold = 1000.0

print(f"Se han encontrado {len(archivos_fun)} archivos FUN. Procesando...")

# =============================================================================
# BLOQUE 1: PROCESADO Y GENERACIÓN DE GRÁFICAS INDIVIDUALES PUNTUALES
# =============================================================================
for archivo in archivos_fun:
    try:
        datos = np.loadtxt(archivo)
        eval_num = extraer_numero(archivo)
        
        # Corrección de consistencia dimensional: reconfiguración de vectores unarios a estructuras matriciales 2D
        if datos.ndim == 1:
            datos = datos.reshape(1, -1)
            
        # Filtro de factibilidad: remoción selectiva de especímenes asociados a penalizaciones asintóticas (CST Timeout)
        datos_limpios = datos[(datos[:, 0] < penalty_threshold) & (datos[:, 1] < penalty_threshold)]
        
        if len(datos_limpios) == 0:
            continue
            
        # Extracción matemática del conjunto frontera no dominado actual
        pareto_front = get_non_dominated_solutions(datos_limpios)
        
        x_values = pareto_front[:, 0]
        y_values = pareto_front[:, 1]
        
        # Persistencia en la estructura histórica para el posterior análisis espectral acumulado
        history.append({
            "eval": eval_num,
            "x": x_values, 
            "y": y_values
        })
        
        # Instanciación del entorno de renderizado individual
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # --- ZONA SOMBREADA (VÁLIDA) ---
        # Delimitación geométrica opcional de la ventana de especificaciones técnicas radioeléctricas
        rect = patches.Rectangle((X_MIN, Y_MIN), -10 - X_MIN, 3 - Y_MIN,
                                 linewidth=0, facecolor='green', alpha=0.15, label='Zona de Compromiso')
        ax.add_patch(rect)
        
        # Renderizado del diagrama de dispersión del frente discreto contemporáneo
        ax.scatter(x_values, y_values, c='yellow', alpha=0.8,s=80, edgecolors='black', label=f'Pareto Eval {eval_num}')
        
        #ax.set_title(f"Frente de Pareto: Evaluación_{eval_num}")
        ax.set_xlabel(objective_labels[0])
        ax.set_ylabel(objective_labels[1])
        
        # Homogeneización escalar de los ejes coordenados conforme a los límites maestros definidos
        ax.set_xlim([X_MIN, X_MAX])
        ax.set_ylim([Y_MIN, Y_MAX])
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best') 
        
        # Guardado en formato vectorial nativo PDF (pad_inches=0 elimina el margen blanco)
        plot_filename = os.path.join(output_directory, f"front_{eval_num}.pdf")
        plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0)
        plt.close(fig) 
        
    except Exception as e:
        pass # Control de excepciones: omisión asíncrona de archivos corruptos o en proceso de escritura

# =============================================================================
# BLOQUE 2: GENERACIÓN DE LA GRÁFICA RESUMEN ACUMULADA (Trayectoria del Frente)
# =============================================================================
print("Generando gráfica resumen acumulada...")
total_plots = len(history)

if total_plots > 0:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    # Generación de una rampa cromática continua basada en el mapa lineal perceptual Viridis
    colors = plt.cm.viridis(np.linspace(0, 1, total_plots))
    
    # # --- ZONA SOMBREADA (VÁLIDA) ---
    rect_summary = patches.Rectangle((X_MIN, Y_MIN), -10 - X_MIN, 3 - Y_MIN,
                                      linewidth=0, facecolor='green', alpha=0.15, label='Zona de Compromiso')
    ax.add_patch(rect_summary)
    
    for i, data in enumerate(history):
        # Gradación analítica de la transparencia y magnitud para contrastar visualmente la convergencia de la última generación
        alpha_val = 0.7 if i < total_plots - 1 else 1.0
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
        
    last_eval = history[-1]["eval"]
    
    #ax.set_title(f"Evolución del frente de Pareto: Problema AntenaParcheInteger")
    ax.set_xlabel(objective_labels[0])
    ax.set_ylabel(objective_labels[1])
    
    # Sincronización a escala de las envolventes maestras en la visualización integrada final
    ax.set_xlim([X_MIN, X_MAX])
    ax.set_ylim([Y_MIN, Y_MAX])
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc="best", framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    
    # Exportación final del plano integrado (pad_inches=0 elimina el margen blanco)
    summary_filename = os.path.join(output_directory, f"Summary_hasta_eval_{last_eval}.pdf")
    plt.savefig(summary_filename, bbox_inches='tight', pad_inches=0)
    print(f"--> ¡Proceso terminado! Gráficas guardadas en: {output_directory}")
    
    plt.show()
else:
    print("No se encontraron datos válidos para generar las gráficas.")