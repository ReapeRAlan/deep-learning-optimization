import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Umbral fijo encontrado (Youden ~ 0.4911)
FIXED_THRESHOLD = 0.4911

# Cargar el pipeline calibrado (modelo + escalado, etc.)
model = joblib.load('diabetes_model_pipeline_calibrated.pkl')

# Inicializar o cargar el historial en session_state
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        'BMI', 'DiabetesPedigreeFunction', 'Age', 
        'Glucose2', 'AgeBMI',   # columnas nuevas
        'Probability', 'Prediction', 'Threshold'
    ])

# Título de la aplicación
st.title("Predicción de Diabetes (Umbral Fijo)")

st.write(f"Umbral de decisión fijado en: **{FIXED_THRESHOLD:.4f}**")
st.write("Ingrese los datos del paciente o cargue un archivo CSV para analizar múltiples casos.")
st.write("**Nota**: Se replican las mismas transformaciones de features usadas en el entrenamiento (Glucose2, AgeBMI).")

# Subir un archivo CSV opcional
uploaded_file = st.file_uploader("Cargar archivo CSV con los datos del paciente", type=["csv"])

if uploaded_file is not None:
    # Cargar el CSV original (8 columnas)
    csv_data = pd.read_csv(uploaded_file)
    st.write("### Datos cargados del archivo CSV (original):")
    st.dataframe(csv_data)

    # Validar que las columnas originales estén presentes
    required_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    if all(col in csv_data.columns for col in required_columns):
        # =========== Feature Engineering en la app ===========
        # Crear Glucose2 y AgeBMI
        csv_data['Glucose2'] = csv_data['Glucose'] ** 2
        csv_data['AgeBMI'] = csv_data['Age'] * csv_data['BMI']

        # Calcular probabilidades con el pipeline calibrado
        # (ahora tenemos 10 columnas)
        probabilities = model.predict_proba(csv_data[
            required_columns + ['Glucose2', 'AgeBMI']
        ])[:, 1]
        
        # Convertir probabilidad a predicción según el umbral fijo
        pred_threshold = (probabilities >= FIXED_THRESHOLD).astype(int)
        predictions_label = ['Diabetes' if p == 1 else 'No diabetes' for p in pred_threshold]

        # Agregar las predicciones y probabilidades al DataFrame
        csv_data['Probability'] = probabilities
        csv_data['Prediction'] = predictions_label
        csv_data['Threshold'] = FIXED_THRESHOLD  # Guardamos el umbral usado

        # Mostrar resultados
        st.write("### Datos con las nuevas columnas + Resultados de predicción:")
        st.dataframe(csv_data)

        # Agregar los resultados al historial
        st.session_state['history'] = pd.concat([st.session_state['history'], csv_data], ignore_index=True)

        # Gráfico de distribución de probabilidades
        st.write("### Distribución de Probabilidades de Diabetes")
        fig, ax = plt.subplots()
        sns.histplot(probabilities, kde=True, ax=ax)
        ax.set_xlabel('Probabilidad de Diabetes')
        ax.set_ylabel('Frecuencia')
        st.pyplot(fig)

        # ============================
        # Cálculo de FP y FN (opcional)
        # ============================
        if 'Outcome' in csv_data.columns:
            # Se asume que Outcome es 0 o 1
            y_true = csv_data['Outcome'].values
            # Convertir Prediction (texto) a 0/1
            y_pred = [1 if lbl == 'Diabetes' else 0 for lbl in csv_data['Prediction']]
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            st.write("### Métricas de la Matriz de Confusión (con umbral fijo)")
            st.write(f"**Falsos Positivos (FP):** {fp}")
            st.write(f"**Falsos Negativos (FN):** {fn}")
            st.write(f"**Verdaderos Positivos (TP):** {tp}")
            st.write(f"**Verdaderos Negativos (TN):** {tn}")
        else:
            st.warning("No se encontró la columna 'Outcome' en el CSV, no se pueden calcular FP y FN.")
    else:
        st.error(f"El archivo CSV debe contener las columnas: {', '.join(required_columns)}")

# Entradas manuales del usuario
st.write("### Ingresar datos manualmente:")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Número de embarazos:", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Nivel de glucosa (mg/dL):", min_value=0, max_value=600, value=120)
    blood_pressure = st.number_input("Presión arterial (mmHg):", min_value=0, max_value=200, value=80)
    skin_thickness = st.number_input("Espesor del pliegue cutáneo (mm):", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Nivel de insulina (mu U/ml):", min_value=0, max_value=1000, value=85)
    bmi = st.number_input("Índice de Masa Corporal (BMI):", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    diabetes_pedigree = st.number_input("Función de pedigrí de diabetes:", min_value=0.0, max_value=5.0, value=0.5, format="%.2f")
    age = st.number_input("Edad (años):", min_value=0, max_value=120, value=30)

# Botón para realizar la predicción manual
if st.button("Predecir"):
    # Crear un DataFrame con los datos ingresados (8 columnas)
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })

    # Replicamos el feature engineering
    input_data['Glucose2'] = input_data['Glucose'] ** 2
    input_data['AgeBMI'] = input_data['Age'] * input_data['BMI']

    # Calcular la probabilidad con el pipeline
    probability = model.predict_proba(input_data)[0, 1]
    
    # Asignar la clase según el umbral fijo
    pred_class = 1 if probability >= FIXED_THRESHOLD else 0
    pred_label = 'Diabetes' if pred_class == 1 else 'No diabetes'

    # Agregar los resultados al historial
    new_entry = input_data.copy()
    new_entry['Probability'] = probability
    new_entry['Prediction'] = pred_label
    new_entry['Threshold'] = FIXED_THRESHOLD
    st.session_state['history'] = pd.concat([st.session_state['history'], new_entry], ignore_index=True)

    # Mostrar resultados
    st.write("### Resultados de la Predicción:")
    st.write(f"**Umbral fijo:** {FIXED_THRESHOLD:.2f}")
    st.write(f"**Probabilidad de diabetes:** {probability:.2%}")
    st.write(f"**Diagnóstico:** {pred_label}")

    # Gráfico de probabilidad
    st.write("### Probabilidad de Diabetes")
    fig, ax = plt.subplots()
    ax.bar(['No Diabetes', 'Diabetes'], [1 - probability, probability], color=['green', 'red'])
    ax.set_ylabel('Probabilidad')
    st.pyplot(fig)

# Mostrar historial
st.write("### Historial de Predicciones:")
st.dataframe(st.session_state['history'])

# Opción para descargar el historial
if st.button("Descargar Historial como CSV"):
    csv = st.session_state['history'].to_csv(index=False)
    st.download_button(
        label="Descargar CSV",
        data=csv,
        file_name='historial_predicciones.csv',
        mime='text/csv',
    )

# Gráfico de correlación entre variables (opcional)
st.write("### Correlación entre Variables")
if not st.session_state['history'].empty:
    numeric_columns = st.session_state['history'].select_dtypes(include=['float64', 'int64']).columns
    numeric_data = st.session_state['history'][numeric_columns]
    
    corr = numeric_data.corr()
    
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
