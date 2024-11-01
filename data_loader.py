import pandas as pd

def load_data(file_path='housing.csv'):
    """Carga el dataset desde un archivo CSV."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo en la ruta especificada: {file_path}")
    return df
