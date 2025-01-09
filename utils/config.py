import torch

CONFIG = {
    # General
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Usar GPU si está disponible
    "random_seed": 42,  # Semilla para reproducibilidad

    # Datos
    "data_path": "./data/datasets/sample_data.csv",  # Ruta al conjunto de datos
    "batch_size": 32,  # Tamaño del lote para DataLoaders
    "test_size": 0.2,  # Porcentaje de datos para pruebas

    # Modelo
    "input_dim": 784,  # Ejemplo: 28x28 imágenes (ajustar según los datos)
    "hidden_dim": 128,  # Neuronas en capas ocultas
    "output_dim": 10,  # Ejemplo: 10 clases para clasificación

    # Entrenamiento
    "learning_rate": 0.001,  # Tasa de aprendizaje
    "num_epochs": 50,  # Número de épocas de entrenamiento

    # Resultados
    "save_model_path": "./models/saved_model.pth",  # Ruta para guardar el modelo entrenado
    "plot_save_path": "./results/loss_accuracy_plot.png",  # Ruta para guardar gráficos
}
