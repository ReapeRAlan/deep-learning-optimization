import torch

CONFIG = {
    "data_path": "./data/diabetes_dataset_cleaned.csv",
    "input_dim": 8,  # Número de características
    "hidden_dim": 128,
    "output_dim": 2,
    "batch_size": 32,
    "test_size": 0.2,
    "learning_rate": 0.001,
    "num_epochs": 100,
    "random_seed": 42,
    "threshold": 0.5,
    "weight_decay": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_model_path": "./models/diabetes_model.pth",  # Ruta donde se guarda el modelo
    "plot_save_path": "./results/diabetes_loss_accuracy.png",
}

