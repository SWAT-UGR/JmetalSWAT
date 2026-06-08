import matplotlib.pyplot as plt
import matplotlib.patches as patches  # <-- NUEVO: Para dibujar la zona sombreada
import numpy as np
import os
import glob

# --- CONFIGURACIÓN GLOBAL ÓPTIMA PARA LATEX ---
plt.rcParams.update({
    'font.size': 14,          # Letra base bastante grande
    'axes.titlesize': 16,     # Títulos que destaquen claramente
    'axes.labelsize': 16,     # Ejes X e Y legibles
    'xtick.labelsize': 14,    # Números del eje X
    'ytick.labelsize': 14,    # Números del eje Y
    'legend.fontsize': 14,    # Leyenda que no desaparezca al encoger
    'figure.titlesize': 18    # Títulos superiores (si los hay)
})
# ----------------------------------------------

# --- CONFIGURACIÓN DE LOS EJES (MODIFICA ESTOS NÚMEROS A TU GUSTO) ---
X_MIN = -50  # Límite mínimo del eje X (S11)
X_MAX = 0    # Límite máximo del eje X (S11)
Y_MIN = -0.5    # Límite mínimo del eje Y (Axial Ratio)
Y_MAX = 20   # Límite máximo del eje Y (Axial Ratio)
# ---------------------------------------------------------------------

def get_non_dominated_solutions(puntos):
    """
    Filtra una matriz 2D para devolver matemáticamente SOLO las soluciones no dominadas (Frente de Pareto).
    Asume un problema de minimización para ambos objetivos (S11 y Axial Ratio).
    """
    no_dominados = []
    for i, p1 in enumerate(puntos):
        dominado = False
        for j, p2 in enumerate(puntos):
            if i != j:
                # p2 domina a p1 si es <= en todos los objetivos y estrictamente < en al menos uno
                if (p2[0] <= p1[0] and p2[1] <= p1[1]) and (p2[0] < p1[0] or p2[1] < p1[1]):
                    dominado = True
                    break
        if not dominado:
            no_dominados.append(p1)
    return np.array(no_dominados)

def extraer_numero(ruta):
    """Extrae el número del final del archivo (ej: FUN.Problema.100 -> 100)"""
    try:
        return int(ruta.split('.')[-1])
    except ValueError:
        return 0

# --- CONFIGURACIÓN DE DIRECTORIOS ---
carpeta_resultados = r"C:\Users\jmate\Desktop\teleco\TFG\Experimentacion\Resultados_NSGAII\todo"
output_directory = os.path.join(carpeta_resultados, "todo2")

# Crear el directorio para las gráficas si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory, exist_ok=True)

# Buscamos y ordenamos todos los archivos FUN
archivos_fun = glob.glob(os.path.join(carpeta_resultados, "FUN.*"))
archivos_fun.sort(key=extraer_numero)

history = []
objective_labels = ["S11 (dB) [Minimizar]", "Axial Ratio (dB) [Minimizar]"]
penalty_threshold = 1000.0

print(f"Se han encontrado {len(archivos_fun)} archivos FUN. Procesando...")

# --- PROCESADO INDIVIDUAL ---
for archivo in archivos_fun:
    try:
        datos = np.loadtxt(archivo)
        eval_num = extraer_numero(archivo)
        
        # Si solo hay una fila, lo convertimos a matriz 2D para no romper el código
        if datos.ndim == 1:
            datos = datos.reshape(1, -1)
            
        # Filtro de viabilidad: descartar valores atípicos o penalizaciones de CST
        datos_limpios = datos[(datos[:, 0] < penalty_threshold) & (datos[:, 1] < penalty_threshold)]
        
        if len(datos_limpios) == 0:
            continue
            
        # Extraer matemáticamente SOLO el conjunto de soluciones no dominadas
        pareto_front = get_non_dominated_solutions(datos_limpios)
        
        x_values = pareto_front[:, 0]
        y_values = pareto_front[:, 1]
        
        # Almacenar en el historial para la gráfica resumen acumulada
        history.append({
            "eval": eval_num,
            "x": x_values, 
            "y": y_values
        })
        
        # Generar y guardar la gráfica INDIVIDUAL
        fig, ax = plt.subplots(figsize=(7, 5))
        
        # --- ZONA SOMBREADA (VÁLIDA) ---
        # Dibuja un rectángulo desde (X_MIN, Y_MIN) hasta (-10, 3)
        rect = patches.Rectangle((X_MIN, Y_MIN), -10 - X_MIN, 3 - Y_MIN,
                                 linewidth=0, facecolor='green', alpha=0.15, label='Zona Válida')
        ax.add_patch(rect)
        
        # CAMBIO: Puntos en 'yellow' con borde negro para que contrasten bien
        ax.scatter(x_values, y_values, c='yellow', alpha=0.8, edgecolors='black', label=f'Pareto Eval {eval_num}')
        
        ax.set_title(f"Frente de Pareto: Evaluación {eval_num}")
        ax.set_xlabel(objective_labels[0])
        ax.set_ylabel(objective_labels[1])
        
        # CAMBIO: Fijar los límites de los ejes a lo que definimos arriba
        ax.set_xlim([X_MIN, X_MAX])
        ax.set_ylim([Y_MIN, Y_MAX])
        
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='best') 
        
        # Extensión PDF y se elimina dpi
        plot_filename = os.path.join(output_directory, f"front_{eval_num}.pdf")
        plt.savefig(plot_filename, bbox_inches='tight')
        plt.close(fig) 
        
    except Exception as e:
        pass # Ignoramos archivos vacíos o corruptos

# --- GRÁFICA RESUMEN ACUMULADA ---
print("Generando gráfica resumen acumulada...")
total_plots = len(history)

if total_plots > 0:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 1, total_plots))
    
    # --- ZONA SOMBREADA (VÁLIDA) ---
    rect_summary = patches.Rectangle((X_MIN, Y_MIN), -10 - X_MIN, 3 - Y_MIN,
                                     linewidth=0, facecolor='green', alpha=0.15, label='Zona Válida')
    ax.add_patch(rect_summary)
    
    for i, data in enumerate(history):
        # Destacar visualmente la iteración más reciente
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
    ax.set_title(f"Evolución del frente de Pareto: Problema AntenaParcheInteger")
    ax.set_xlabel(objective_labels[0])
    ax.set_ylabel(objective_labels[1])
    
    # CAMBIO: Fijar los límites de los ejes también en la gráfica resumen
    ax.set_xlim([X_MIN, X_MAX])
    ax.set_ylim([Y_MIN, Y_MAX])
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc="best", framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    
    # Extensión PDF
    summary_filename = os.path.join(output_directory, f"Summary_hasta_eval_{last_eval}.pdf")
    plt.savefig(summary_filename, bbox_inches='tight')
    print(f"--> ¡Proceso terminado! Gráficas guardadas en: {output_directory}")
    
    plt.show()
else:
    print("No se encontraron datos válidos para generar las gráficas.")