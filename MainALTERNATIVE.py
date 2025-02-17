import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (classification_report, roc_auc_score, 
                             confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, average_precision_score)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from xgboost import XGBClassifier
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

import joblib

# Configuración de estilo para gráficos
plt.style.use('ggplot')
sns.set_palette("Set2")

# =========================================================
# 1. Carga de datos, filtrado de outliers y feature engineering
# =========================================================
data = pd.read_csv('data/diabetes_dataset_cleaned.csv')

# Mostrar distribución inicial y estadísticas
print("Distribución de clases (original):\n", data['Outcome'].value_counts())
print("\nResumen estadístico (original):\n", data.describe())

# Filtrar outliers (ejemplo: Glucose > 600 o BMI > 70)
data = data[(data['Glucose'] <= 600) & (data['BMI'] <= 70)]

# Ejemplo de feature engineering: crear dos nuevas columnas
data['Glucose2'] = data['Glucose'] ** 2
data['AgeBMI'] = data['Age'] * data['BMI']

print("\nResumen estadístico (después de filtrado y feature engineering):\n", data.describe())

# Separar variable objetivo y características
y = data['Outcome']
X = data.drop(columns=['Outcome'])

# =========================================================
# 2. División estratificada de datos
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================================================
# 3. Pipeline de preprocesamiento y modelo (con SMOTE)
# =========================================================
pipeline = ImbPipeline([
    ('scaler', RobustScaler()),
    ('smote', BorderlineSMOTE(random_state=42)),
    ('model', XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    ))
])

# =========================================================
# 4. Definición del espacio de hiperparámetros
# =========================================================
param_dist = {
    'model__n_estimators': [100, 200, 300, 400, 600, 800],
    'model__max_depth': [3, 4, 5, 6, 8],
    'model__learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
    'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'model__gamma': [0, 0.1, 0.2, 0.3],
    'model__scale_pos_weight': [1, 2, 3, 5, 10],
    'model__min_child_weight': [1, 3, 5, 7],
    'model__reg_alpha': [0, 0.01, 0.1, 1],
    'model__reg_lambda': [1, 1.5, 2, 3]
}

# =========================================================
# 5. Búsqueda de hiperparámetros con validación cruzada repetida
# =========================================================
cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=50,
    scoring=['roc_auc', 'f1', 'recall'],
    refit='recall',  # Optimizar para maximizar el recall
    cv=cv_strategy,
    verbose=3,
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f"\nMejores parámetros (según recall): {search.best_params_}")

# =========================================================
# 6. Calibración del modelo
# =========================================================
# Extraer el modelo XGB entrenado del pipeline
xgb_best = best_model.named_steps['model']

# Preparamos X_train con el escalador entrenado (omitimos SMOTE para calibración)
preprocessing_pipeline = ImbPipeline([
    ('scaler', best_model.named_steps['scaler'])
])
X_train_prepared = preprocessing_pipeline.transform(X_train)

# Calibramos usando CalibratedClassifierCV con cv='prefit'
from sklearn.calibration import CalibratedClassifierCV

calibrator = CalibratedClassifierCV(
    estimator=xgb_best,
    method='sigmoid',
    cv='prefit'
)
calibrator.fit(X_train_prepared, y_train)

# Construir un pipeline final que incluya el escalador y la calibración
final_pipeline = Pipeline([
    ('scaler', best_model.named_steps['scaler']),
    ('calibrator', calibrator)
])

# =========================================================
# 7. Evaluación en el set de prueba con el modelo calibrado
# =========================================================
X_test_scaled = final_pipeline.named_steps['scaler'].transform(X_test)
y_proba_calibrated = final_pipeline.named_steps['calibrator'].predict_proba(X_test_scaled)[:, 1]
y_pred_calibrated_05 = (y_proba_calibrated >= 0.5).astype(int)

print("\nMétricas de evaluación (umbral 0.5) - modelo calibrado:")
print(classification_report(y_test, y_pred_calibrated_05))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_calibrated):.4f}")
print(f"Precisión-Recall AUC: {average_precision_score(y_test, y_proba_calibrated):.4f}")

# Ajuste del umbral de decisión (índice de Youden)
fpr, tpr, thresholds = roc_curve(y_test, y_proba_calibrated)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"\nUmbral óptimo (Youden, modelo calibrado): {optimal_threshold:.4f}")

y_pred_calibrated_opt = (y_proba_calibrated >= optimal_threshold).astype(int)
print("\nMétricas (umbral óptimo) - modelo calibrado:")
print(classification_report(y_test, y_pred_calibrated_opt))

# Curva de calibración
prob_true, prob_pred = calibration_curve(y_test, y_proba_calibrated, n_bins=10)
plt.figure(figsize=(6,6))
plt.plot(prob_pred, prob_true, marker='o', label='Calibración (modelo calibrado)')
plt.plot([0,1],[0,1], 'k--')
plt.xlabel('Probabilidad promedio predicha')
plt.ylabel('Proporción real de positivos')
plt.title('Curva de Calibración (modelo calibrado)')
plt.legend()
plt.show()

# Curvas ROC y PR
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fpr_c, tpr_c, _ = roc_curve(y_test, y_proba_calibrated)
ax1.plot(fpr_c, tpr_c, color='darkorange', lw=2, label=f'ROC (AUC = {auc(fpr_c, tpr_c):.2f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=2)
ax1.set_xlabel('Tasa de Falsos Positivos')
ax1.set_ylabel('Tasa de Verdaderos Positivos')
ax1.set_title('Curva ROC (calibrado)')
ax1.legend()

precision_c, recall_c, _ = precision_recall_curve(y_test, y_proba_calibrated)
ax2.plot(recall_c, precision_c, color='blue', lw=2, label=f'PR (AP = {average_precision_score(y_test, y_proba_calibrated):.2f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precisión')
ax2.set_title('Curva Precisión-Recall (calibrado)')
ax2.legend()

plt.tight_layout()
plt.show()

# Matriz de confusión con umbral óptimo
plt.figure(figsize=(6, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_calibrated_opt), annot=True, fmt='d', cmap='Blues',
            cbar=False, annot_kws={'size': 14})
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión (modelo calibrado, umbral óptimo)')
plt.show()

# =========================================================
# 8. Importancia de características (del XGB subyacente)
# =========================================================
feature_importances = xgb_best.feature_importances_
all_features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=all_features, palette='viridis')
plt.title('Importancia de Características (XGBoost - tras feature engineering)')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.tight_layout()
plt.show()

# =========================================================
# 9. Guardar el pipeline final calibrado
# =========================================================
joblib.dump(final_pipeline, 'diabetes_model_pipeline_calibrated.pkl')
print("\n¡Proceso completado exitosamente con calibración y mejoras!")

# =========================================================
# 10. Prueba con datos externos (extreme_data)
# =========================================================
# Asegúrate de que el orden de las columnas coincida con X.columns
# NOTA: Ahora tenemos 10 columnas (8 originales + 2 features extras: Glucose2 y AgeBMI)
extreme_data_array = np.array([
    [15, 200, 120, 60, 900, 50.0, 2.5, 80, 200**2, 80*50.0],
    [0, 70, 50, 10, 15, 18.5, 0.1, 18, 70**2, 18*18.5],
    [3, 120, 70, 25, 80, 28.0, 0.5, 30, 120**2, 30*28.0],
    [5, 150, 85, 35, 120, 35.5, 1.2, 45, 150**2, 45*35.5],
    [1, 85, 60, 20, 50, 22.0, 0.2, 25, 85**2, 25*22.0]
])
# Creamos un DataFrame con el mismo orden de columnas que X
extreme_cols = list(X.columns)
extreme_df = pd.DataFrame(extreme_data_array, columns=extreme_cols)

# Predecir con el pipeline final calibrado
extreme_proba_cal = final_pipeline.predict_proba(extreme_df)[:, 1]
extreme_pred_cal_05 = (extreme_proba_cal >= 0.5).astype(int)

print("\nProbabilidades para datos externos (modelo calibrado):", extreme_proba_cal)
print("Predicciones (umbral 0.5):", extreme_pred_cal_05)
for i, p in enumerate(extreme_pred_cal_05):
    clase = "Diabetes" if p == 1 else "No diabetes"
    print(f"Fila {i} (umbral 0.5): {clase} - Prob = {extreme_proba_cal[i]:.4f}")

# También, usando el umbral óptimo calculado (Youden)
extreme_pred_cal_opt = (extreme_proba_cal >= optimal_threshold).astype(int)
for i, p in enumerate(extreme_pred_cal_opt):
    clase = "Diabetes" if p == 1 else "No diabetes"
    print(f"Fila {i} (umbral óptimo={optimal_threshold:.4f}): {clase} - Prob = {extreme_proba_cal[i]:.4f}")
