import matplotlib.pyplot as plt
import seaborn as sns

def plot_results(y_test, y_pred):
    """Genera gráficos lineales de valores reales y predicciones con líneas promedio."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    
    # Calcular promedios
    avg_real = y_test.mean()
    avg_pred = y_pred.mean()
    
    # Gráfico para los valores reales
    axes[0].plot(y_test.index, y_test, color="blue", label="Valores Reales", alpha=0.7)
    axes[0].axhline(avg_real, color="red", linestyle="--", label=f"Promedio Real: {avg_real:.2f}")
    axes[0].set_title("Valores Reales")
    axes[0].set_ylabel("Precio")
    axes[0].legend()
    
    # Gráfico para las predicciones
    axes[1].plot(y_test.index, y_pred, color="orange", label="Predicciones", alpha=0.7)
    axes[1].axhline(avg_pred, color="red", linestyle="--", label=f"Promedio Predicción: {avg_pred:.2f}")
    axes[1].set_title("Predicciones")
    axes[1].set_xlabel("Índice de Prueba")
    axes[1].set_ylabel("Precio")
    axes[1].legend()
    
    # Título general de la figura
    fig.suptitle("Comparación de Valores Reales vs Predicciones con Promedios", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Ajusta el layout
    
    return fig
