import pandas as pd

# Rutas de archivos
data_path = "./data/datasets/diabetes_dataset_no_id.csv"
output_path = "./data/datasets/diabetes_dataset_corrected.csv"

# Función para cargar y corregir datos
def validate_and_fix_data(file_path, output_path):
    """
    Cargar un archivo CSV, validar y corregir datos poco realistas.
    """
    # Cargar datos
    data = pd.read_csv(file_path)

    # Eliminar valores poco realistas para las columnas especificadas
    columns_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for column in columns_to_fix:
        # Agrupar por la etiqueta Outcome para calcular estadísticas separadas
        grouped = data[data[column] > 0].groupby("Outcome")[column]
        
        # Calcular la mediana para cada grupo
        medians = grouped.median()
        
        # Imputar valores basados en la clase Outcome
        def impute_value(row):
            if row[column] == 0:
                return medians[row["Outcome"]]
            return row[column]

        data[column] = data.apply(impute_value, axis=1)

    # Guardar los datos corregidos
    data.to_csv(output_path, index=False)
    print(f"Datos corregidos guardados en: {output_path}")

# Ejecutar la función
validate_and_fix_data(data_path, output_path)
