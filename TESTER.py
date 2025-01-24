import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.config import CONFIG
from models.nn_model import initialize_nn


def load_model():
    """
    Cargar el modelo entrenado.
    """
    device = CONFIG["device"]
    model = initialize_nn(CONFIG["input_dim"], CONFIG["hidden_dim"], CONFIG["output_dim"]).to(device)
    model.load_state_dict(torch.load(CONFIG["save_model_path"], map_location=device))
    model.eval()
    return model, device


def preprocess_input(data):
    """
    Escalar los datos de entrada para que sean compatibles con el modelo.
    """
    # Usar el mismo escalador estándar que en el entrenamiento
    scaler = StandardScaler()
    data = np.array(data).reshape(1, -1)
    return scaler.fit_transform(data)  # Escalar manualmente


def predict(model, device, data, threshold):
    """
    Realizar predicciones con el modelo entrenado.
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(data, dtype=torch.float32).to(device)
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        prediction = int(probabilities[0][1] >= threshold)  # Predicción basada en el umbral
    return prediction, probabilities[0]


def main():
    print("\n=== Prueba de Predicción Manual ===")

    # Capturar datos del paciente
    pregnancies = float(input("Número de embarazos: "))
    glucose = float(input("Nivel de glucosa (mg/dL): "))
    blood_pressure = float(input("Presión arterial (mmHg): "))
    skin_thickness = float(input("Espesor del pliegue cutáneo (mm): "))
    insulin = float(input("Nivel de insulina (µU/mL): "))
    bmi = float(input("Índice de Masa Corporal (BMI): "))
    diabetes_pedigree = float(input("Función de pedigrí de diabetes: "))
    age = float(input("Edad (años): "))

    # Preparar los datos
    patient_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    print(f"\nDatos ingresados: {patient_data}")

    processed_data = preprocess_input(patient_data)

    # Cargar el modelo
    model, device = load_model()

    # Ajustar el umbral
    threshold = float(input("Ingrese el umbral para predicción (por defecto 0.5): ") or 0.5)

    # Realizar predicción
    prediction, probabilities = predict(model, device, processed_data, threshold)

    # Mostrar resultados
    print("\n=== Resultados de Predicción ===")
    print(f"Predicción: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
    print(f"Probabilidad de tener diabetes: {probabilities[1]:.2f}")
    print(f"Probabilidad de no tener diabetes: {probabilities[0]:.2f}")

    # Detectar posibles problemas de sesgo
    if prediction == 1 and probabilities[1] > 0.9:
        print("Advertencia: El modelo tiene una alta confianza en predecir diabetes. Verifica los datos y el umbral.")
    if prediction == 0 and probabilities[1] < 0.1:
        print("Advertencia: El modelo tiene una alta confianza en predecir no diabetes.")


if __name__ == "__main__":
    main()



'''
Número de embarazos: 0
Nivel de glucosa (mg/dL): 85
Presión arterial (mmHg): 115
Espesor del pliegue cutáneo (mm): 15
Nivel de insulina (µU/mL): 5
Índice de Masa Corporal (BMI): 22
Función de pedigrí de diabetes: 0.1
Edad (años): 25

=== Prueba de Predicción Manual ===
Número de embarazos: 8
Nivel de glucosa (mg/dL): 200
Presión arterial (mmHg): 90
Espesor del pliegue cutáneo (mm): 45
Nivel de insulina (µU/mL): 300
Índice de Masa Corporal (BMI): 40
Función de pedigrí de diabetes: 1.5
Edad (años): 50


'''