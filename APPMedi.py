import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
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


def preprocess_input(data):
    """
    Escalar los datos de entrada del paciente para que sean compatibles con el modelo.
    """
    scaler = StandardScaler()
    data = np.array(data).reshape(1, -1)
    return scaler.fit_transform(data)


def predict(model, device, data, threshold=0.5):
    """
    Realizar predicciones con el modelo entrenado utilizando un umbral.
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(data, dtype=torch.float32).to(device)
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

        # Clasificación basada en el umbral
        prediction = 1 if probabilities[0][1] >= threshold else 0

    return prediction, probabilities[0]


def main():
    print("=== Cuestionario para Predicción de Diabetes ===")

    # Capturar datos del paciente
    pregnancies = float(input("Número de embarazos: "))
    glucose = float(input("Nivel de glucosa (mg/dL): "))
    blood_pressure = float(input("Presión arterial (mmHg): "))
    skin_thickness = float(input("Espesor del pliegue cutáneo (mm): "))
    insulin = float(input("Nivel de insulina (µU/mL): "))
    bmi = float(input("Índice de Masa Corporal (BMI): "))
    diabetes_pedigree = float(input("Función de pedigrí de diabetes: "))
    age = float(input("Edad (años): "))

    # Preparar los datos del paciente
    patient_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]
    processed_data = preprocess_input(patient_data)

    # Cargar el modelo entrenado
    model, device = load_model()

    # Definir el umbral para la predicción
    threshold = float(input("Ingrese el umbral para predicción (por defecto 0.5): ") or 0.5)

    # Realizar predicción
    prediction, probabilities = predict(model, device, processed_data, threshold)

    # Mostrar resultados
    print("\n=== Resultados de Predicción ===")
    print(f"Predicción: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
    print(f"Probabilidad de tener diabetes: {probabilities[1]:.2f}")
    print(f"Probabilidad de no tener diabetes: {probabilities[0]:.2f}")
    print(f"Umbral usado: {threshold:.2f}")


if __name__ == "__main__":
    main()
