import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from datetime import datetime

# Configuración de la API
API_BASE_URL = "http://localhost:8000"  # Cambiar según despliegue
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"

# Configurar tema de visualización
plt.style.use('ggplot')
sns.set_palette("viridis")

# Inicializar estados de sesión
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
        'Probability', 'Diagnóstico', 'Threshold', 'Timestamp'
    ])

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Diseño de la interfaz
st.set_page_config(page_title="DiabeDoc", page_icon="🩺", layout="wide")
st.title("DiabeDoc - Sistema Integrado de Diabetes y Nutrición")

# Sidebar para configuración
with st.sidebar:
    st.header("Configuración")
    api_url = st.text_input("URL de la API", API_BASE_URL)
    st.info("Cambiar solo si se despliega en otro servidor")
    st.divider()
    st.write("Estado del servicio:")
    try:
        health = requests.get(f"{api_url}/health/", timeout=3)
        if health.status_code == 200:
            st.success("✅ API conectada correctamente")
            model_info = health.json()
            st.write(f"Modelo: {model_info['model_type'].split('.')[-1][:-2]}")
            st.write(f"Versión: {model_info['model_version']}")
        else:
            st.error("❌ Error en la API")
    except requests.exceptions.RequestException:
        st.error("🔴 API no disponible")

# Pestañas principales
tab1, tab2, tab3 = st.tabs(["Predicción", "Análisis Batch", "Asistente Virtual"])

with tab1:
    st.header("Predicción Individual")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Embarazos:", min_value=0, max_value=20, value=0)
            glucose = st.number_input("Glucosa (mg/dL):", min_value=0, max_value=600, value=120)
            blood_pressure = st.number_input("Presión Arterial (mmHg):", min_value=0, max_value=200, value=80)
            skin_thickness = st.number_input("Pliegue Cutáneo (mm):", min_value=0, max_value=100, value=20)

        with col2:
            insulin = st.number_input("Insulina (mu U/ml):", min_value=0, max_value=1000, value=85)
            bmi = st.number_input("Índice de Masa Corporal:", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
            diabetes_pedigree = st.number_input("Pedigrí Diabetes:", min_value=0.0, max_value=5.0, value=0.5, format="%.2f")
            age = st.number_input("Edad (años):", min_value=0, max_value=120, value=30)

        if st.form_submit_button("Predecir Diabetes"):
            input_data = {
                "pregnancies": pregnancies,
                "glucose": glucose,
                "blood_pressure": blood_pressure,
                "skin_thickness": skin_thickness,
                "insulin": insulin,
                "bmi": bmi,
                "diabetes_pedigree": diabetes_pedigree,
                "age": age
            }
            
            try:
                response = requests.post(PREDICT_ENDPOINT, json=input_data, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    
                    # Registrar en historial
                    new_entry = {
                        **input_data,
                        "Probability": result["probabilidad_diabetes"],
                        "Diagnóstico": result["diagnóstico"],
                        "Threshold": result["threshold_utilizado"],
                        "Timestamp": datetime.now().isoformat()
                    }
                    
                    st.session_state.history = pd.concat([
                        st.session_state.history,
                        pd.DataFrame([new_entry])
                    ], ignore_index=True)
                    
                    # Mostrar resultados
                    st.subheader("Resultados")
                    prob_percent = result["probabilidad_diabetes"] * 100
                    diagnosis_color = "#ff4b4b" if result["diagnóstico"] == "Diabetes" else "#4CAF50"
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.metric("Probabilidad de Diabetes", f"{prob_percent:.1f}%")
                        st.metric("Diagnóstico", result["diagnóstico"], delta_color="off")
                        
                    with col_res2:
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.pie(
                            [result["probabilidad_diabetes"], 1 - result["probabilidad_diabetes"]],
                            labels=["Diabetes", "No Diabetes"],
                            colors=[diagnosis_color, "#f0f2f6"],
                            startangle=90,
                            autopct='%1.1f%%'
                        )
                        st.pyplot(fig)
                        
                else:
                    st.error(f"Error en la API: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"Error de conexión: {str(e)}")

with tab2:
    st.header("Análisis Batch")
    
    uploaded_file = st.file_uploader("Subir CSV para análisis múltiple", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_cols = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ]
            
            if all(col in df.columns for col in required_cols):
                progress_bar = st.progress(0)
                results = []
                
                for i, row in df.iterrows():
                    data = row[required_cols].to_dict()
                    response = requests.post(PREDICT_ENDPOINT, json=data)
                    if response.status_code == 200:
                        results.append(response.json())
                    progress_bar.progress((i + 1) / len(df))
                
                # Procesar resultados
                result_df = pd.DataFrame([{
                    **r,
                    "probabilidad_diabetes": r["probabilidad_diabetes"],
                    "diagnóstico": 1 if r["diagnóstico"] == "Diabetes" else 0
                } for r in results])
                
                # Mostrar análisis
                st.subheader("Resumen Estadístico")
                st.dataframe(result_df.describe())
                
                # Gráficos
                col_hist, col_corr = st.columns(2)
                
                with col_hist:
                    st.write("Distribución de Probabilidades")
                    fig, ax = plt.subplots()
                    sns.histplot(result_df["probabilidad_diabetes"], kde=True, ax=ax)
                    st.pyplot(fig)
                
                with col_corr:
                    st.write("Correlación de Variables")
                    corr_matrix = df[required_cols].corr()
                    fig, ax = plt.subplots()
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", ax=ax)
                    st.pyplot(fig)
                
            else:
                st.error(f"CSV debe contener las columnas: {', '.join(required_cols)}")
                
        except Exception as e:
            st.error(f"Error procesando archivo: {str(e)}")

with tab3:
    st.header("Asistente Virtual de Nutrición")
    
    user_query = st.chat_input("Escribe tu pregunta sobre diabetes y nutrición...")
    
    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        try:
            response = requests.post(
                CHAT_ENDPOINT,
                json={"query": user_query, "max_length": 1000}
            )
            
            if response.status_code == 200:
                bot_response = response.json()["response"]
                st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "❌ Error al obtener respuesta. Intente nuevamente."
                })
                
        except requests.exceptions.RequestException:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "🔴 Servicio no disponible. Intente más tarde."
            })
    
    # Mostrar historial de chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Panel de historial general
st.sidebar.divider()
st.sidebar.header("Historial Clínico")

if not st.session_state.history.empty:
    st.sidebar.dataframe(
        st.session_state.history[[
            'Glucose', 'BMI', 'Age', 'Diagnóstico', 'Timestamp'
        ]].tail(5),
        use_container_width=True
    )
    if st.sidebar.button("Limpiar Historial"):
        st.session_state.history = pd.DataFrame(columns=st.session_state.history.columns)
        st.experimental_rerun()
else:
    st.sidebar.info("No hay registros en el historial")

# Descarga de datos
st.sidebar.divider()
if not st.session_state.history.empty:
    csv = st.session_state.history.to_csv(index=False)
    st.sidebar.download_button(
        "Descargar Historial Completo",
        data=csv,
        file_name='diabedoc_historial.csv',
        mime='text/csv'
    )