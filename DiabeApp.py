import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import re
import logging
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DiabeDoc")

# -----------------------------------------------------------------------------
# Configuración de la API
API_BASE_URL = "http://localhost:8000"  # Cambia si tu API está en otro lugar
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
PREDICT_BATCH_ENDPOINT = f"{API_BASE_URL}/predict_batch"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"

# -----------------------------------------------------------------------------
# Configuración de visualización
plt.style.use('ggplot')
sns.set_palette("viridis")

# -----------------------------------------------------------------------------
# Inicializar estados de sesión
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
        'Probability', 'Diagnóstico', 'Threshold', 'Timestamp'
    ])

# Historial del chat
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Acciones rápidas
if 'quick_action' not in st.session_state:
    st.session_state.quick_action = ""

# -----------------------------------------------------------------------------
# Configuración de la página
st.set_page_config(page_title="DiabeDoc", page_icon="🩺", layout="wide")
st.title("DiabeDoc - Sistema Integrado de Diabetes y Nutrición")

# -----------------------------------------------------------------------------
# Sidebar: Configuración y estado de la API
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

# -----------------------------------------------------------------------------
# Funciones auxiliares para el chat
def create_chat_report(history):
    """
    Genera un reporte (string) a partir del historial de chat.
    """
    report = "Historial de chat:\n"
    for msg in history:
        role = msg['role'].upper()
        text = msg['content']
        report += f"[{role}] {text}\n"
    return report

def generate_pdf_report():
    """
    Genera un PDF (placeholder).
    """
    return "PDF content (placeholder)"

def clean_bot_response(text: str) -> str:
    """
    Limpia la respuesta del modelo:
      - Elimina etiquetas <think> y </think>.
      - Opcionalmente elimina <div> o las reemplaza por algo simple.
      - Convierte secciones y viñetas a Markdown básico.
    """
    # 1) Quitar <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2) Quitar <div> y </div> si aparecieran
    text = text.replace("<div>", "").replace("</div>", "")

    # 3) Reemplazar algunas secciones especiales (si quieres)
    # Ejemplo: "📌 Resumen Rápido" -> "## 📌 Resumen Rápido"
    sections = {
        "📌 Resumen Rápido": "## 📌 Resumen Rápido",
        "🔍 Análisis Detallado": "## 🔍 Análisis Detallado",
        "💊 Manejo Médico": "## 💊 Manejo Médico",
        "🥗 Plan Nutricional": "## 🥗 Plan Nutricional",
        "🏋️ Actividad Física": "## 🏋️ Actividad Física",
        "⚠️ Precauciones": "## ⚠️ Precauciones"
    }
    for old, new in sections.items():
        text = text.replace(old, new)

    # 4) Convertir viñetas "• " en "-" y saltos de línea en line breaks
    text = text.replace("• ", "- ")

    # 5) Quitar exceso de espacios
    text = text.strip()

    return text

# -----------------------------------------------------------------------------
def run_chat_ui():
    """
    Renderiza la pestaña 'Asistente Virtual de Nutrición' usando st.chat_message.
    """
    st.subheader("🧠 Asistente Virtual de Diabetes ExpertIA")

    # Mensaje de bienvenida si no hay historial
    if len(st.session_state.chat_history) == 0:
        welcome_msg = (
            "¡Hola! Soy **Dr. Gluco**, tu especialista virtual en diabetes. "
            "¿En qué puedo ayudarte hoy?\n\n"
            "**Ejemplos de preguntas:**\n"
            "- ¿Cómo ajustar mi insulina después de ejercicio?\n"
            "- Necesito un plan de comidas para 1500 kcal\n"
            "- ¿Qué hacer si mi glucosa está en 350 mg/dL?\n"
        )
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": welcome_msg,
            "timestamp": datetime.now().isoformat()
        })

    # Acciones rápidas
    with st.expander("🚀 Acciones Rápidas"):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📋 Generar diario de glucosa"):
                st.session_state.quick_action = "Genera una tabla para registrar mis niveles de glucosa 5 veces al día"
        with col2:
            if st.button("🍽 Analizar alimento"):
                st.session_state.quick_action = "Acabo de comer: "
        with col3:
            if st.button("💊 Revisar medicación"):
                st.session_state.quick_action = "Estoy tomando estos medicamentos: "

    # Mostrar historial con st.chat_message
    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("user"):
                st.markdown(msg["content"])

    # Input del usuario (pinned abajo)
    user_input = st.chat_input("Escribe tu pregunta o 'ayuda' para opciones...")
    if not user_input and st.session_state.quick_action:
        # Si no se ingresa nada y hay una acción rápida
        user_input = st.session_state.quick_action
        st.session_state.quick_action = ""

    if user_input:
        # Mostrar inmediatamente lo que el usuario escribió
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        # Llamar a la API
        try:
            with st.spinner("Consultando al modelo..."):
                payload = {"query": user_input, "max_length": 1500}
                resp = requests.post(CHAT_ENDPOINT, json=payload, timeout=120)
            if resp.status_code == 200:
                raw_bot_resp = resp.json().get("response", "")
                clean_resp = clean_bot_response(raw_bot_resp)
                # Agregar a historial
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": clean_resp,
                    "timestamp": datetime.now().isoformat()
                })
                with st.chat_message("assistant"):
                    st.markdown(clean_resp)
            else:
                error_msg = f"❌ Error al obtener respuesta. Código: {resp.status_code}"
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            error_txt = f"🔴 Error: {str(e)}"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_txt,
                "timestamp": datetime.now().isoformat()
            })
            with st.chat_message("assistant"):
                st.markdown(error_txt)

    # Opciones de descarga en el sidebar
    with st.sidebar.expander("🔧 Herramientas de Conversación"):
        if st.button("📥 Descargar historial médico"):
            report = create_chat_report(st.session_state.chat_history)
            st.download_button("Descargar .txt", data=report, file_name="historial_chat.txt")
        if st.button("🧹 Limpiar conversación"):
            st.session_state.chat_history = []
        pdf_data = generate_pdf_report()
        st.download_button("💾 Exportar como PDF", data=pdf_data, file_name="historial_diabetes.pdf")

# -----------------------------------------------------------------------------
# Pestañas principales
tab1, tab2, tab3 = st.tabs(["Predicción", "Análisis Batch", "Asistente Virtual"])

# -----------------------------------------------------------------------------
# Pestaña 1: Predicción Individual
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
            new_entry = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": diabetes_pedigree,
                "Age": age
            }
            try:
                with st.spinner("Calculando predicción..."):
                    response = requests.post(PREDICT_ENDPOINT, json={
                        "pregnancies": pregnancies,
                        "glucose": glucose,
                        "blood_pressure": blood_pressure,
                        "skin_thickness": skin_thickness,
                        "insulin": insulin,
                        "bmi": bmi,
                        "diabetes_pedigree": diabetes_pedigree,
                        "age": age
                    }, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    new_entry["Probability"] = result["probabilidad_diabetes"]
                    new_entry["Diagnóstico"] = result["diagnóstico"]
                    new_entry["Threshold"] = result["threshold_utilizado"]
                    new_entry["Timestamp"] = datetime.now().isoformat()
                    st.session_state.history = pd.concat([
                        st.session_state.history,
                        pd.DataFrame([new_entry])
                    ], ignore_index=True)
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

# -----------------------------------------------------------------------------
# Pestaña 2: Análisis Batch
with tab2:
    st.header("Análisis Batch")
    uploaded_file = st.file_uploader("Subir CSV para análisis múltiple", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            csv_required = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ]
            if all(col in df.columns for col in csv_required):
                csv_mapping = {
                    "Pregnancies": "pregnancies",
                    "Glucose": "glucose",
                    "BloodPressure": "blood_pressure",
                    "SkinThickness": "skin_thickness",
                    "Insulin": "insulin",
                    "BMI": "bmi",
                    "DiabetesPedigreeFunction": "diabetes_pedigree",
                    "Age": "age"
                }
                data_list = []
                for _, row in df.iterrows():
                    row_data = row[csv_required].to_dict()
                    data_list.append({csv_mapping[k]: v for k, v in row_data.items()})
                
                with st.spinner("Procesando predicciones en lote..."):
                    response = requests.post(PREDICT_BATCH_ENDPOINT, json={"data": data_list})
                
                if response.status_code == 200:
                    results = response.json()["results"]
                    result_df = pd.DataFrame(results)
                    
                    df_out = df.copy()
                    df_out["probabilidad_diabetes"] = result_df["probabilidad_diabetes"]
                    df_out["diagnóstico"] = result_df["diagnóstico"]
                    
                    st.subheader("Resumen Estadístico de la Predicción")
                    st.dataframe(result_df.describe())
                    
                    col_hist, col_corr = st.columns(2)
                    with col_hist:
                        st.write("Distribución de Probabilidades")
                        fig, ax = plt.subplots()
                        sns.histplot(result_df["probabilidad_diabetes"], kde=True, ax=ax)
                        st.pyplot(fig)
                    
                    with col_corr:
                        st.write("Histograma de Diagnósticos")
                        fig, ax = plt.subplots()
                        ax.hist(result_df["diagnóstico"], bins=2, color="#4CAF50", edgecolor="black")
                        ax.set_xticks([0, 1])
                        ax.set_xticklabels(["No diabetes", "Diabetes"])
                        st.pyplot(fig)
                    
                    st.subheader("Datos Originales con Predicción")
                    st.dataframe(df_out, use_container_width=True)
                    st.success("Predicciones batch procesadas exitosamente.")
                else:
                    st.error(f"Error en la API batch: {response.text}")
            else:
                st.error(f"CSV debe contener las columnas: {', '.join(csv_required)}")
        except Exception as e:
            st.error(f"Error procesando archivo: {str(e)}")

# -----------------------------------------------------------------------------
# Pestaña 3: Asistente Virtual
with tab3:
    run_chat_ui()

# -----------------------------------------------------------------------------
# Sidebar: Historial Clínico
with st.sidebar:
    st.divider()
    st.header("Historial Clínico Completo")
    if not st.session_state.history.empty:
        with st.expander("Mostrar historial completo"):
            st.dataframe(st.session_state.history, use_container_width=True)
        if st.button("Limpiar Historial"):
            st.session_state.history = pd.DataFrame(columns=st.session_state.history.columns)
    else:
        st.info("No hay registros en el historial")
    
    st.divider()
    st.header("Descarga de Datos")
    if not st.session_state.history.empty:
        csv_data = st.session_state.history.to_csv(index=False)
        st.download_button(
            "Descargar Historial Completo",
            data=csv_data,
            file_name='diabedoc_historial.csv',
            mime='text/csv'
        )
