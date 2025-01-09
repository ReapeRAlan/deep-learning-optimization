import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Función para graficar pérdida durante el entrenamiento y validación
def plot_loss(train_loss, val_loss, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label="Pérdida de Entrenamiento")
    plt.plot(val_loss, label="Pérdida de Validación")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Evolución de la Pérdida")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Función para graficar precisión durante el entrenamiento y validación
def plot_accuracy(train_acc, val_acc, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label="Precisión de Entrenamiento")
    plt.plot(val_acc, label="Precisión de Validación")
    plt.xlabel("Época")
    plt.ylabel("Precisión")
    plt.title("Evolución de la Precisión")
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Función para graficar una matriz de confusión
def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.title("Matriz de Confusión")
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de datos de pérdida
    train_loss = np.random.rand(10)
    val_loss = np.random.rand(10)

    # Graficar pérdida
    plot_loss(train_loss, val_loss)

    # Ejemplo de matriz de confusión
    cm = np.array([[10, 2, 1], [3, 8, 0], [0, 1, 11]])
    class_names = ["Clase 0", "Clase 1", "Clase 2"]
    plot_confusion_matrix(cm, class_names)
