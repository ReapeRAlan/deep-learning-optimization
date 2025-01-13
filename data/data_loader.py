import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_and_preprocess_data(file_path, test_size=0.2):
    """
    Cargar y preprocesar el dataset para predicción de diabetes.
    """
    # Cargar datos
    data = pd.read_csv(file_path)

    # Separar características y etiquetas
    X = data.drop(columns=["Outcome"])  # Características
    y = data["Outcome"]                # Etiqueta (1 o 0)

    # Escalar características numéricas
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f"Dimensiones de X después de procesar: {X.shape}")

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """
    Convertir datos a tensores y crear DataLoaders de PyTorch.
    """
    # Convertir datos a tensores de PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Crear TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Crear DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Ejemplo de uso
if __name__ == "__main__":
    # Ruta del archivo de datos
    data_path = "./data/datasets/diabetes_dataset_corrected.csv"

    # Cargar y preprocesar datos
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path, test_size=0.2)

    # Crear DataLoaders
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)

    print(f"Tamaño del conjunto de entrenamiento: {len(train_loader.dataset)} muestras")
    print(f"Tamaño del conjunto de prueba: {len(test_loader.dataset)} muestras")
