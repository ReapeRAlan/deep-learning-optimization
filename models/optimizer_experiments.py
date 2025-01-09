import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import CNNModel
from data.data_loader import load_data
from utils.metrics import calculate_accuracy
from utils.config import CONFIG

# Función para probar diferentes optimizadores
def optimizer_experiment(optimizer_name, model, train_loader, criterion, learning_rate=0.001):
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSProp":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not recognized")

    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(CONFIG["device"]), labels.to(CONFIG["device"])
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(train_loader)

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar datos
    train_data, test_data = load_data(CONFIG["data_path"])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=CONFIG["batch_size"], shuffle=True)

    # Inicializar modelo
    model = CNNModel().to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss()

    # Probar optimizadores
    optimizers = ["SGD", "Adam", "RMSProp"]
    for opt in optimizers:
        avg_loss = optimizer_experiment(opt, model, train_loader, criterion)
        print(f"Optimizer: {opt}, Average Loss: {avg_loss:.4f}")
