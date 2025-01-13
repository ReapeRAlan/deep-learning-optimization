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
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

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


def evaluate_model(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Agregar a las listas
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Calcular precisión y matriz de confusión
    accuracy = calculate_accuracy(y_true, y_pred)
    cm = calculate_confusion_matrix(y_true, y_pred)

    return accuracy, cm, y_true, y_pred


def test_model_with_custom_data(model, device):
    # Datos de prueba personalizados
    test_data = np.array([
        [2, 120, 70, 30, 80, 25.5, 0.672, 33],  # Ejemplo 1
        [4, 85, 60, 25, 100, 28.2, 0.122, 40], # Ejemplo 2
        [1, 140, 80, 35, 120, 22.0, 0.452, 28] # Ejemplo 3
    ])
    
    # Convertir a tensores
    inputs = torch.tensor(test_data, dtype=torch.float32).to(device)

    # Realizar predicciones
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    
    # Mostrar resultados
    print("\n=== Resultados de Predicción con Datos Personalizados ===")
    for i, (prediction, prob) in enumerate(zip(predictions.cpu().numpy(), probabilities)):
        prob_diabetes = prob[1]
        print(f"Ejemplo {i+1}: Predicción = {'Diabetes' if prediction == 1 else 'No Diabetes'}, "
              f"Probabilidad = {prob_diabetes:.2f}")

    return predictions.cpu().numpy(), probabilities


if __name__ == "__main__":
    # Configuración inicial
    torch.manual_seed(CONFIG["random_seed"])
    device = CONFIG["device"]
    print(f"Usando dispositivo: {device}")

    # Cargar y preprocesar datos
    data_path = CONFIG["data_path"]
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path, CONFIG["test_size"])
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, CONFIG["batch_size"])

    # Verificar la distribución de clases
    print("Distribución de clases en el conjunto de entrenamiento:")
    print(Counter(y_train))

    print("Distribución de clases en el conjunto de prueba:")
    print(Counter(y_test))
    print(f"Tamaño total de test_loader: {len(test_loader.dataset)}")

    # Inicializar modelo, criterio y optimizador
    model = initialize_nn(CONFIG["input_dim"], CONFIG["hidden_dim"], CONFIG["output_dim"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # Entrenamiento
    train_losses = []
    for epoch in range(CONFIG["num_epochs"]):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        print(f"Época {epoch+1}/{CONFIG['num_epochs']} - Pérdida: {train_loss:.4f}")

    # Evaluación
    accuracy, cm, y_true, y_pred = evaluate_model(model, test_loader, device)
    print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")

    # Graficar pérdida y matriz de confusión
    plot_loss(train_losses, train_losses, CONFIG["plot_save_path"])
    plot_confusion_matrix(y_true, y_pred, save_path=CONFIG["plot_save_path"])

    # Guardar modelo
    torch.save(model.state_dict(), CONFIG["save_model_path"])
    print(f"Modelo guardado en: {CONFIG['save_model_path']}")

    # Cargar modelo guardado para pruebas
    model.load_state_dict(torch.load(CONFIG["save_model_path"], map_location=device))

    # Pruebas con datos personalizados
    test_model_with_custom_data(model, device)
    torch.save(model.state_dict(), CONFIG["save_model_path"])

