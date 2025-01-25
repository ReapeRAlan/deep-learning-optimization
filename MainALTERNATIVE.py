import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib
import matplotlib.pyplot as plt

# Cargar los datos del archivo CSV
data = pd.read_csv('data/diabetes_dataset_cleaned.csv')

# Verificar los primeros registros para inspeccionar los datos
print("Primeros registros del conjunto de datos:")
print(data.head())

# Separar características (X) y etiquetas (y)
X = data.drop(columns=['Outcome'])  # 'Outcome' se asume como la etiqueta de diagnóstico
y = data['Outcome']

# Calcular pesos de clase para manejar desbalance de clases
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Estandarizar los datos para mejorar la performance del modelo
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Guardar el escalador
joblib.dump(scaler, 'scaler.pkl')

# Crear el modelo de red neuronal con mayor número de iteraciones y early stopping
mlp = MLPClassifier(max_iter=3000, random_state=42, early_stopping=True, validation_fraction=0.2)

# Definir hiperparámetros a optimizar
param_grid = {
    'hidden_layer_sizes': [(200, 150, 100), (150, 100, 50), (100, 100, 50)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['adaptive']
}

# Optimizar hiperparámetros usando GridSearchCV
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo
best_model = grid_search.best_estimator_
print("\nMejores hiperparámetros:")
print(grid_search.best_params_)

# Evaluar el modelo en el conjunto de prueba
y_pred = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)[:, 1]

print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"Precisión: {accuracy:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Visualizar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusión:")
print(conf_matrix)

# Visualizar el estado del entrenamiento
plt.plot(best_model.loss_curve_)
plt.title('Curva de pérdida durante el entrenamiento')
plt.xlabel('Iteraciones')
plt.ylabel('Pérdida')
plt.show()

# Ajustar el umbral de clasificación y evaluar de nuevo
custom_threshold = 0.4  # Ajuste del umbral
custom_pred = (y_pred_prob >= custom_threshold).astype(int)

print("\nInforme de clasificación con umbral ajustado:")
print(classification_report(y_test, custom_pred))

custom_accuracy = accuracy_score(y_test, custom_pred)
custom_roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"Precisión con umbral ajustado: {custom_accuracy:.4f}")
print(f"ROC-AUC con umbral ajustado: {custom_roc_auc:.4f}")

# Guardar el modelo para uso futuro
joblib.dump(best_model, 'diabetes_detection_model.pkl')

print("\nEl modelo ha sido entrenado, evaluado y guardado exitosamente.")
