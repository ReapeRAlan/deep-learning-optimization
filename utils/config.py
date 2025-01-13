import torch

CONFIG = {
    "data_path": "/home/ghost/Escritorio/IA/deep-learning-optimization/data/datasets/diabetes_dataset_corrected.csv",
    "input_dim": 8,  # Número de características después de eliminar 'Id'
    "hidden_dim": 128,
    "output_dim": 2,
    "batch_size": 32,
    "test_size": 0.2,
    "learning_rate": 0.001,
    "num_epochs": 200,
    "random_seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_model_path": "./models/diabetes_model.pth",
    "plot_save_path": "./results/diabetes_loss_accuracy.png",
}






