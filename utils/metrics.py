from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Métrica de precisión
def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# Métrica de precisión para cada clase
def calculate_precision(y_true, y_pred, average="weighted"):
    return precision_score(y_true, y_pred, average=average)

# Métrica de recall (sensibilidad)
def calculate_recall(y_true, y_pred, average="weighted"):
    return recall_score(y_true, y_pred, average=average)

# Métrica F1-score
def calculate_f1_score(y_true, y_pred, average="weighted"):
    return f1_score(y_true, y_pred, average=average)

# Matriz de confusión
def calculate_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de datos reales y predicciones
    y_true = [0, 1, 1, 0, 1, 2, 2, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 2, 1, 0, 1]

    print("Precisión:", calculate_accuracy(y_true, y_pred))
    print("Precisión por clase:", calculate_precision(y_true, y_pred))
    print("Recall:", calculate_recall(y_true, y_pred))
    print("F1-Score:", calculate_f1_score(y_true, y_pred))
    print("Matriz de confusión:\n", calculate_confusion_matrix(y_true, y_pred))
