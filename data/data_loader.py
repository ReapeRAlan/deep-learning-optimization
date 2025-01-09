import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

# Función para cargar datos desde un archivo CSV
def load_data(file_path, test_size=0.2, random_state=42):
    # Cargar datos en un DataFrame
    data = pd.read_csv(file_path)
    
    # Separar características (X) y etiquetas (y)
    X = data.iloc[:, :-1].values  # Todas las columnas excepto la última
    y = data.iloc[:, -1].values   # Última columna

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return (X_train, y_train), (X_test, y_test)

# Convertir datos a tensores y cargadores de datos (DataLoader)
def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    # Convertir datos a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Crear TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Crear DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Ejemplo de uso
if __name__ == "__main__":
    # Ruta del archivo de datos (ajustar según tu proyecto)
    data_path = "./data/datasets/sample_data.csv"

    # Cargar datos
    (X_train, y_train), (X_test, y_test) = load_data(data_path)

    # Crear DataLoaders
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)

    print(f"Tamaño del conjunto de entrenamiento: {len(train_loader.dataset)} muestras")
    print(f"Tamaño del conjunto de prueba: {len(test_loader.dataset)} muestras")
