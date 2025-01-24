import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib

# Cargar el modelo guardado
best_model = joblib.load('diabetes_detection_model.pkl')

# Cargar el conjunto de datos de prueba
# Nota: Asegúrate de usar un conjunto de datos que no se usó en el entrenamiento
new_data = pd.read_csv('data\diabetes_dataset_no_id.csv') 

# Preprocesar los datos
scaler = StandardScaler()
X_new = scaler.fit_transform(new_data.drop(columns=['Outcome']))  # Ajusta si el CSV no tiene 'Outcome'
y_new = new_data['Outcome']  # Ajusta si el CSV no tiene 'Outcome'

# Realizar predicciones
new_predictions = best_model.predict(X_new)
new_probabilities = best_model.predict_proba(X_new)[:, 1]

# Evaluar el modelo en los nuevos datos
print("\nInforme de clasificación en el nuevo conjunto de datos:")
print(classification_report(y_new, new_predictions))

accuracy = accuracy_score(y_new, new_predictions)
roc_auc = roc_auc_score(y_new, new_probabilities)
print(f"Precisión en el nuevo conjunto de datos: {accuracy:.4f}")
print(f"ROC-AUC en el nuevo conjunto de datos: {roc_auc:.4f}")

# Visualizar las probabilidades
import matplotlib.pyplot as plt

plt.hist(new_probabilities, bins=20, alpha=0.7, label='Probabilidades')
plt.axvline(0.5, color='r', linestyle='--', label='Umbral de 0.5')
plt.xlabel('Probabilidad de Diabetes')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Distribución de probabilidades para detección de diabetes')
plt.show()

# Probar con ejemplos específicos
examples = X_new[:5]  # Selecciona 5 ejemplos de prueba
example_predictions = best_model.predict(examples)
example_probabilities = best_model.predict_proba(examples)

print("\nPredicciones para ejemplos específicos:")
for i, (pred, prob) in enumerate(zip(example_predictions, example_probabilities)):
    print(f"Ejemplo {i+1}: Predicción = {pred}, Probabilidad de clase positiva (diabetes) = {prob[1]:.4f}")
