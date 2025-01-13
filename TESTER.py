import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from models.nn_model import initialize_nn
from utils.config import CONFIG

def load_model():
    """
    Cargar el modelo entrenado.
    """
    device = CONFIG["device"]
    model = initialize_nn(CONFIG["input_dim"], CONFIG["hidden_dim"], CONFIG["output_dim"]).to(device)
    model.load_state_dict(torch.load(CONFIG["save_model_path"], map_location=device))
    model.eval()
    return model, device

def preprocess_data(file_path):
    """
    Cargar y preprocesar los datos del archivo CSV.
    """
    data = pd.read_csv(file_path)
    X = data.drop(columns=["Outcome"]).values  # Características
    y = data["Outcome"].values  # Etiquetas verdaderas

    # Escalar las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def predict(model, device, X):
    """
    Realizar predicciones con el modelo entrenado.
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32).to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    return predictions.cpu().numpy(), probabilities

def evaluate_and_plot(y_true, y_pred, probabilities):
    """
    Evaluar el modelo y generar gráficas y métricas.
    """
    # Reporte de clasificación
    report = classification_report(y_true, y_pred, target_names=["No Diabetes", "Diabetes"])
    print("\n=== Reporte de Clasificación ===")
    print(report)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print("\n=== Matriz de Confusión ===")
    print(cm)

    # Graficar matriz de confusión
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.title("Matriz de Confusión")
    plt.show()

    # Curva ROC y AUC
    roc_auc = roc_auc_score(y_true, probabilities[:, 1])
    fpr, tpr, _ = roc_curve(y_true, probabilities[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"Curva ROC (Área = {roc_auc:.2f})", color="orange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="blue")
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    plt.show()

    # Precisión global
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nPrecisión del modelo en el conjunto completo: {accuracy:.4f}")
    print(f"Área bajo la curva (AUC): {roc_auc:.4f}")

if __name__ == "__main__":
    # Ruta del archivo CSV
    file_path = CONFIG["data_path"]

    # Cargar el modelo entrenado
    model, device = load_model()

    # Preprocesar los datos del CSV
    X, y_true = preprocess_data(file_path)

    # Realizar predicciones
    predictions, probabilities = predict(model, device, X)

    # Mostrar resultados para las primeras 10 filas
    print("\n=== Resultados de Predicción ===")
    for i in range(10):  # Muestra solo las primeras 10 predicciones como ejemplo
        print(f"Ejemplo {i+1}: Predicción = {'Diabetes' if predictions[i] == 1 else 'No Diabetes'}, "
              f"Probabilidad = {probabilities[i][1]:.2f}, Etiqueta Real = {'Diabetes' if y_true[i] == 1 else 'No Diabetes'}")

    # Evaluar el modelo y generar métricas y gráficas
    evaluate_and_plot(y_true, predictions, probabilities)
'''
Número de embarazos: 0
Nivel de glucosa (mg/dL): 85
Presión arterial (mmHg): 115
Espesor del pliegue cutáneo (mm): 15
Nivel de insulina (µU/mL): 5
Índice de Masa Corporal (BMI): 22
Función de pedigrí de diabetes: 0.1
Edad (años): 25



'''