import streamlit as st
from data_loader import load_data
from preprocessing import preprocess_data
from model import train_model, evaluate_model
from visualization import plot_results
import graphviz

def generate_flowchart():
    """Genera el diagrama de flujo del pipeline de predicción de precios de viviendas."""
    flowchart = graphviz.Digraph(comment="Flujo de Proceso de Predicción de Precios de Viviendas")
    
    # Definir nodos
    flowchart.node("A", "Cargar Datos\ndesde CSV")
    flowchart.node("B", "Preprocesar Datos\n(Filtrar columnas, separar objetivo)")
    flowchart.node("C", "Entrenar Modelo\n(Random Forest)")
    flowchart.node("D", "Evaluar Modelo\n(MSE y R²)")
    flowchart.node("E1", "Visualización de\nValores Reales")
    flowchart.node("E2", "Visualización de\nPredicciones")
    flowchart.node("F", "Comparación en Streamlit")

    # Crear las conexiones
    flowchart.edge("A", "B", label="Datos cargados y ejecutados")
    flowchart.edge("B", "C", label="Datos preprocesados")
    flowchart.edge("C", "D", label="Modelo entrenado")
    flowchart.edge("D", "E1", label="Métricas calculadas")
    flowchart.edge("D", "E2", label="Métricas calculadas")
    flowchart.edge("E1", "F", label="Comparación de\nValores Reales")
    flowchart.edge("E2", "F", label="Comparación de\nPredicciones")
    
    return flowchart

def main():
    st.title('Predicción de Precios de Casas en California')
    
    # Mostrar el diagrama de flujo
    st.subheader("Diagrama de Flujo del Pipeline")
    flowchart = generate_flowchart()
    st.graphviz_chart(flowchart)  # Muestra el diagrama de flujo en Streamlit
    
    # Cargar datos
    df = load_data('housing.csv')
    st.write("Datos cargados:")
    st.write(df.head())
    
    # Preprocesar datos
    X, y = preprocess_data(df)
    
    # Entrenar el modelo
    model, X_test, y_test = train_model(X, y)
    y_pred = model.predict(X_test)
    
    # Evaluar el modelo
    mse, r2 = evaluate_model(y_test, y_pred)
    st.write(f"Error cuadrático medio: {mse:.2f}")
    st.write(f"R²: {r2:.2f}")
    
    # Mostrar gráfico de resultados
    st.subheader('Comparación de Valores Reales vs Predicciones')
    fig = plot_results(y_test, y_pred)
    st.pyplot(fig)  # Muestra el gráfico en Streamlit

if __name__ == "__main__":
    main()
