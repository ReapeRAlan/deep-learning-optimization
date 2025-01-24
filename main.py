import torch
from torch import nn
from utils.config import CONFIG
from utils.metrics import calculate_accuracy, calculate_confusion_matrix
from utils.plot_utils import plot_loss, plot_confusion_matrix
from data.data_loader import load_and_preprocess_data, create_dataloaders
from models.nn_model import initialize_nn
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

def validate_extreme_data(data):
    """Validar si los datos están dentro de rangos realistas."""
    ranges = {
        "Pregnancies": (0, 20),
        "Glucose": (50, 300),
        "BloodPressure": (30, 180),
        "SkinThickness": (0, 99),
        "Insulin": (0, 900),
        "BMI": (10, 70),
        "DiabetesPedigreeFunction": (0.0, 2.5),
        "Age": (0, 120)
    }
    for i, (key, (low, high)) in enumerate(ranges.items()):
        if not (low <= data[i] <= high):
            raise ValueError(f"El valor para {key} ({data[i]}) está fuera de rango ({low}-{high}).")

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)

def evaluate_model(model, test_loader, device, threshold=0.5):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            preds = (probabilities[:, 1] >= threshold).long()

            # Agregar a las listas
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probabilities[:, 1].cpu().numpy())

    # Calcular precisión, matriz de confusión y métricas adicionales
    accuracy = calculate_accuracy(y_true, y_pred)
    cm = calculate_confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    classification_rep = classification_report(y_true, y_pred, target_names=["No Diabetes", "Diabetes"])

    return accuracy, cm, y_true, y_pred, y_probs, roc_auc, classification_rep

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

def validate_and_retrain(model, optimizer, criterion, train_loader, test_loader, device):
    """Validar con múltiples datos extremos y reentrenar si hay falsos positivos o negativos."""
    extreme_data_samples = [
        ([0, 85, 115, 15, 5, 22, 0.1, 25], 0),  # No Diabetes
        ([8, 200, 90, 45, 300, 40.0, 1.5, 50], 1),  # Diabetes
        ([3, 110, 70, 25, 120, 25.0, 0.5, 30], 0),  # No Diabetes
        ([10, 250, 100, 60, 500, 50.0, 2.0, 60], 1),  # Diabetes
        ([1, 95, 60, 20, 0, 18.5, 0.2, 22], 0),  # No Diabetes
        ([12, 240, 95, 55, 450, 45.0, 1.8, 55], 1),  # Diabetes
        ([0, 80, 55, 0, 0, 20.0, 0.1, 20], 0),  # No Diabetes
        ([15, 300, 110, 70, 600, 60.0, 2.2, 70], 1),  # Diabetes
        ([2, 100, 65, 20, 85, 24.0, 0.4, 35], 0),  # No Diabetes
        ([9, 210, 85, 40, 320, 38.0, 1.3, 45], 1)  # Diabetes
    ]

    for data, expected_label in extreme_data_samples:
        validate_extreme_data(data)
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(inputs)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            pred = (probabilities[0, 1] >= CONFIG.get("threshold", 0.5)).item()

        if pred != expected_label:  # Si hay error en predicción
            print(f"\n*** Error detectado con datos extremos {data}. Reentrenando modelo... ***")
            for epoch in range(CONFIG["num_epochs"] * 2):  # Reentrenar con la mitad de las épocas
                train_loss = train_model(model, train_loader, criterion, optimizer, device)
                if (epoch + 1) % 10 == 0:
                    print(f"Reentrenamiento - Época {epoch+1}: Pérdida: {train_loss:.4f}")

def main():
    torch.manual_seed(CONFIG["random_seed"])
    device = CONFIG["device"]
    print(f"Usando dispositivo: {device}")

    # Cargar y preprocesar datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(CONFIG["data_path"], CONFIG["test_size"])
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, CONFIG["batch_size"])

    # Verificar distribución de clases
    print("Distribución de clases en el conjunto de entrenamiento:")
    print(Counter(y_train))
    print("Distribución de clases en el conjunto de prueba:")
    print(Counter(y_test))

    # Inicializar modelo, criterio y optimizador
    model = initialize_nn(CONFIG["input_dim"], CONFIG["hidden_dim"], CONFIG["output_dim"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=0.01)

    # Entrenamiento
    train_losses = []
    for epoch in range(CONFIG["num_epochs"]):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Época {epoch+1}/{CONFIG['num_epochs']} - Pérdida: {train_loss:.4f}")

    # Evaluación inicial
    accuracy, cm, y_true, y_pred, y_probs, roc_auc, classification_rep = evaluate_model(model, test_loader, device, CONFIG.get("threshold", 0.5))
    print(f"\nPrecisión en el conjunto de prueba: {accuracy:.4f}")
    print(f"\n=== Reporte de Clasificación ===\n{classification_rep}")
    print(f"Área bajo la curva (AUC): {roc_auc:.4f}")
    print(f"\n=== Matriz de Confusión ===\n{cm}")

    # Validar y reentrenar si es necesario
    validate_and_retrain(model, optimizer, criterion, train_loader, test_loader, device)

    # Evaluación final después de posible reentrenamiento
    accuracy, cm, y_true, y_pred, y_probs, roc_auc, classification_rep = evaluate_model(model, test_loader, device, CONFIG.get("threshold", 0.5))
    print(f"\n*** Resultados finales después de validación y posible reentrenamiento ***")
    print(f"\nPrecisión en el conjunto de prueba: {accuracy:.4f}")
    print(f"\n=== Reporte de Clasificación ===\n{classification_rep}")
    print(f"Área bajo la curva (AUC): {roc_auc:.4f}")
    print(f"\n=== Matriz de Confusión ===\n{cm}")

    # Graficar pérdida, matriz de confusión y curva ROC
    plot_loss(train_losses, train_losses, CONFIG["plot_save_path"])
    plot_confusion_matrix(y_true, y_pred, save_path=CONFIG["plot_save_path"])
    plot_roc_curve(y_true, y_probs)

    # Guardar modelo
    torch.save(model.state_dict(), CONFIG["save_model_path"])
    print(f"Modelo guardado en: {CONFIG['save_model_path']}")

if __name__ == "__main__":
    main()
