import pandas as pd

# Rutas de archivos
data_path = "data\diabetes_dataset_no_id.csv"
output_path = "data\diabetes_dataset_cleaned.csv"

# Función para cargar y eliminar datos poco realistas
def remove_invalid_data(file_path, output_path):
    """
    Cargar un archivo CSV y eliminar filas con datos poco realistas.
    """
    # Cargar datos
    data = pd.read_csv(file_path)

    # Columnas donde los valores no pueden ser cero
    columns_to_check = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # Eliminar filas donde haya ceros en estas columnas
    for column in columns_to_check:
        before_count = len(data)
        data = data[data[column] > 0]
        after_count = len(data)
        print(f"Filas eliminadas en {column}: {before_count - after_count}")

    # Guardar los datos limpios
    data.to_csv(output_path, index=False)
    print(f"Datos limpios guardados en: {output_path}")

# Ejecutar la función
remove_invalid_data(data_path, output_path)
