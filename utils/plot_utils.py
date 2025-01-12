import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Asegúrate de que y_true e y_pred sean unidimensionales
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    print("y_true final:", y_true)
    print("y_pred final:", y_pred)
    print("Tamaños de y_true y y_pred:", len(y_true), len(y_pred))

    # Forzar consistencia en las longitudes
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]

    # Obtener las clases únicas automáticamente
    unique_classes = sorted(set(y_true) | set(y_pred))  # Clases únicas combinadas
    if class_names is None:
        class_names = [f"Clase {cls}" for cls in unique_classes]

    # Generar la matriz de confusión
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    print("Matriz de confusión generada:", cm)

    # Graficar la matriz de confusión
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
