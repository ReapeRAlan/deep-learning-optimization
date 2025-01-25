import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Cargar el modelo y el escalador preentrenado
model = joblib.load('diabetes_detection_model.pkl')
scaler = joblib.load('scaler.pkl')  # Se debe guardar el escalador durante el entrenamiento del modelo

# Inicializar o cargar el historial
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Probability', 'Prediction'
    ])

# Título de la aplicación
st.title("Predicción de Diabetes")
st.write("Ingrese los datos del paciente o cargue un archivo CSV para analizar múltiples casos.")

# Subir un archivo CSV opcional
uploaded_file = st.file_uploader("Cargar archivo CSV con los datos del paciente", type=["csv"])

if uploaded_file is not None:
    # Cargar y procesar el archivo CSV
    csv_data = pd.read_csv(uploaded_file)
    st.write("### Datos cargados del archivo CSV:")
    st.dataframe(csv_data)

    # Validar que las columnas necesarias estén presentes
    required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    if all(col in csv_data.columns for col in required_columns):
        # Escalar los datos
        scaled_csv_data = scaler.transform(csv_data[required_columns])

        # Realizar predicciones
        probabilities = model.predict_proba(scaled_csv_data)[:, 1]
        predictions = model.predict(scaled_csv_data)

        # Agregar las predicciones y probabilidades al DataFrame
        csv_data['Probability'] = probabilities
        csv_data['Prediction'] = ['Diabetes' if pred == 1 else 'No diabetes' for pred in predictions]

        # Mostrar resultados
        st.write("### Resultados de las predicciones:")
        st.dataframe(csv_data)

        # Agregar los resultados al historial
        st.session_state['history'] = pd.concat([st.session_state['history'], csv_data], ignore_index=True)
    else:
        st.error(f"El archivo CSV debe contener las siguientes columnas: {', '.join(required_columns)}")

# Entradas manuales del usuario
st.write("### Ingresar datos manualmente:")
pregnancies = st.number_input("Número de embarazos:", min_value=0, max_value=20, value=0)
glucose = st.number_input("Nivel de glucosa (mg/dL):", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Presión arterial (mmHg):", min_value=0, max_value=200, value=80)
skin_thickness = st.number_input("Espesor del pliegue cutáneo (mm):", min_value=0, max_value=100, value=20)
insulin = st.number_input("Nivel de insulina (mu U/ml):", min_value=0, max_value=1000, value=85)
bmi = st.number_input("Índice de Masa Corporal (BMI):", min_value=0.0, max_value=100.0, value=25.0, format="%.1f")
diabetes_pedigree = st.number_input("Función de pedigrí de diabetes:", min_value=0.0, max_value=5.0, value=0.5, format="%.2f")
age = st.number_input("Edad (años):", min_value=0, max_value=120, value=30)

# Botón para realizar la predicción manual
if st.button("Predecir"):
    # Crear un DataFrame con los datos ingresados
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

    # Escalar los datos con el escalador preentrenado
    scaled_data = scaler.transform(input_data)

    # Realizar la predicción
    probability = model.predict_proba(scaled_data)[:, 1][0]
    prediction = model.predict(scaled_data)[0]

    # Agregar los resultados al historial
    new_entry = input_data.copy()
    new_entry['Probability'] = probability
    new_entry['Prediction'] = 'Diabetes' if prediction == 1 else 'No diabetes'
    st.session_state['history'] = pd.concat([st.session_state['history'], new_entry], ignore_index=True)

    # Mostrar resultados
    st.write("### Resultados de la Predicción:")
    st.write(f"**Probabilidad de diabetes:** {probability:.2%}")
    st.write(f"**Diagnóstico:** {'Diabetes' if prediction == 1 else 'No diabetes'}")

# Mostrar historial
st.write("### Historial de Predicciones:")
st.dataframe(st.session_state['history'])
