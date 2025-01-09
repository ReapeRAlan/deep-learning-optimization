import torch
from torch import nn
from utils.config import CONFIG
from utils.metrics import calculate_accuracy, calculate_confusion_matrix
from utils.plot_utils import plot_loss, plot_confusion_matrix
from data.data_loader import load_data, create_dataloaders
from models.nn_model import initialize_nn

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

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Calcular precisión y matriz de confusión
    accuracy = calculate_accuracy(y_true, y_pred)
    cm = calculate_confusion_matrix(y_true, y_pred)

    return accuracy, cm


if __name__ == "__main__":
    # Configuración inicial
    torch.manual_seed(CONFIG["random_seed"])
    device = CONFIG["device"]
    print(f"Usando dispositivo: {device}")

    # Cargar datos
    (X_train, y_train), (X_test, y_test) = load_data(CONFIG["data_path"], CONFIG["test_size"])
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, CONFIG["batch_size"])

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
    accuracy, cm = evaluate_model(model, test_loader, device)
    print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")

    # Graficar pérdida y matriz de confusión
    plot_loss(train_losses, train_losses, CONFIG["plot_save_path"])  # Aquí se usa el mismo gráfico para ilustrar
    class_names = [f"Clase {i}" for i in range(CONFIG["output_dim"])]
    plot_confusion_matrix(cm, class_names, CONFIG["plot_save_path"])

    # Guardar modelo
    torch.save(model.state_dict(), CONFIG["save_model_path"])
    print(f"Modelo guardado en: {CONFIG['save_model_path']}")
