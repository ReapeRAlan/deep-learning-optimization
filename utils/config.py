import torch

CONFIG = {
    "data_path": "/home/ghost/Escritorio/IA/deep-learning-optimization/data/datasets/diabetes_dataset.csv",
    "input_dim": 9,  # Número de características en el dataset
    "hidden_dim": 128,  # Neuronas en la capa oculta
    "output_dim": 2,  # Clases: 0 (sin diabetes) y 1 (con diabetes)
    "batch_size": 32,
    "test_size": 0.2,  # Porcentaje del conjunto de prueba
    "learning_rate": 0.001,
    "num_epochs": 200,
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_model_path": "./models/diabetes_model.pth",
    "plot_save_path": "./results/diabetes_loss_accuracy.png",
}





