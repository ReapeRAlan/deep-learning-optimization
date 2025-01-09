import torch
from models.cnn_model import CNNModel
from data.data_loader import get_data_loader
from utils.metrics import accuracy

# Configuración
batch_size = 64
epochs = 10
learning_rate = 0.001

# Datos
train_loader = get_data_loader(batch_size)

# Modelo
model = CNNModel()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# Entrenamiento
for epoch in range(epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
