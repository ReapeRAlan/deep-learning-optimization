import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Cargar el modelo guardado
best_model = joblib.load('diabetes_detection_model.pkl')

# Escalador usado durante el entrenamiento
scaler = StandardScaler()

# Datos extremos y regulares para pruebas manuales
extreme_data = [
    [15, 200, 120, 60, 900, 50.0, 2.5, 80],  # Caso extremo 100% diabetes
    [0, 70, 50, 10, 15, 18.5, 0.1, 18],      # Caso extremo 100% no diabetes
    [3, 120, 70, 25, 80, 28.0, 0.5, 30],     # Caso regular con probabilidad moderada
    [5, 150, 85, 35, 120, 35.5, 1.2, 45],    # Caso regular con probabilidad alta
    [1, 85, 60, 20, 50, 22.0, 0.2, 25]       # Caso regular con probabilidad baja
]

# Convertir a DataFrame
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
manually_created_data = pd.DataFrame(extreme_data, columns=columns)

# Estandarizar los datos manuales
scaled_data = scaler.fit_transform(manually_created_data)

# Realizar predicciones
predictions = best_model.predict(scaled_data)
probabilities = best_model.predict_proba(scaled_data)[:, 1]

# Mostrar resultados
print("\nResultados para datos manuales:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    print(f"Caso {i+1}: Predicción = {pred}, Probabilidad de diabetes = {prob:.4f}")

# Evaluación estimada con datos manuales
manual_labels = [1, 0, 0, 1, 0]  # Etiquetas esperadas para los casos manuales
evaluated_accuracy = accuracy_score(manual_labels, predictions)
print(f"\nPrecisión estimada con datos manuales: {evaluated_accuracy:.4f}")

# Generar informe de clasificación
print("\nInforme de clasificación para datos manuales:")
print(classification_report(manual_labels, predictions))
