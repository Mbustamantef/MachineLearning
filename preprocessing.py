def preprocess_data(df):
    """Preprocesa el DataFrame para separar características y la variable objetivo."""
    # Seleccionar columnas relevantes y eliminar la columna 'Address'
    X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
            'Avg. Area Number of Bedrooms', 'Area Population']]
    y = df['Price']
    return X, y
