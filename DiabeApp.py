import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.figure_factory as ff
import re
import logging
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuración del logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DiabeDoc++")

# -----------------------------------------------------------------------------
# Configuración de la API
API_BASE_URL = "http://localhost:8000"
ENDPOINTS = {
    "predict": f"{API_BASE_URL}/predict",
    "batch": f"{API_BASE_URL}/predict_batch",
    "chat": f"{API_BASE_URL}/chat",
    "health": f"{API_BASE_URL}/health/"
}

# -----------------------------------------------------------------------------
# Constantes de diseño
THEME = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "success": "#3D9970",
    "warning": "#FF851B",
    "danger": "#FF4136"
}

# -----------------------------------------------------------------------------
# Inicialización de estado
def init_session_state():
    if 'history' not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=[
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
            'Probability', 'Diagnóstico', 'Threshold', 'Timestamp'
        ])
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None

init_session_state()

# -----------------------------------------------------------------------------
# Configuración de página
st.set_page_config(
    page_title="DiabeDoc Pro",
    page_icon="🩺",
    layout="wide",
    menu_items={
        'Get Help': 'https://diabedoc.com/help',
        'Report a bug': "https://diabedoc.com/bug",
        'About': "### Sistema experto en diabetes v2.1"
    }
)

# -----------------------------------------------------------------------------
# Componentes personalizados
def service_status():
    with st.sidebar.expander("🛠 Estado del Servicio", expanded=True):
        try:
            with st.spinner("🔍 Verificando salud del servicio..."):
                health = requests.get(ENDPOINTS["health"], timeout=3)
            
            if health.status_code == 200:
                model_info = health.json()
                st.success("✅ Servicio operativo")
                cols = st.columns(2)
                cols[0].metric("Modelo", model_info['model_type'].split('.')[-1][:-2])
                cols[1].metric("Versión", model_info['model_version'])
                st.progress(model_info.get('model_accuracy', 0.85))
                st.caption(f"Última actualización: {model_info.get('last_trained', 'N/A')}")
            else:
                st.error("⚠️ Servicio con problemas")
                st.code(f"Detalles: {health.text}", language="json")
        except Exception as e:
            st.error(f"🔌 Error de conexión: {str(e)}")

def prediction_form():
    with st.form("smart_prediction_form"):
        cols = st.columns(2)
        inputs = {}
        
        with cols[0]:
            inputs['pregnancies'] = st.number_input(
                "🤰 Embarazos", 
                min_value=0, max_value=20,
                help="Número total de embarazos"
            )
            inputs['glucose'] = st.number_input(
                "🩸 Glucosa (mg/dL)",
                min_value=0, max_value=600,
                help="Nivel de glucosa en plasma a 2 horas"
            )
            inputs['blood_pressure'] = st.number_input(
                "💓 Presión Arterial (mmHg)",
                min_value=0, max_value=200,
                help="Presión arterial diastólica"
            )
            inputs['skin_thickness'] = st.number_input(
                "📏 Pliegue Cutáneo (mm)",
                min_value=0, max_value=100,
                help="Espesor del pliegue cutáneo del tríceps"
            )
        
        with cols[1]:
            inputs['insulin'] = st.number_input(
                "💉 Insulina (μU/ml)",
                min_value=0, max_value=1000,
                help="Nivel de insulina en suero a 2 horas"
            )
            inputs['bmi'] = st.number_input(
                "⚖️ Índice de Masa Corporal",
                min_value=0.0, max_value=70.0,
                format="%.1f",
                help="Índice de masa corporal (peso en kg/(altura en m)^2"
            )
            inputs['diabetes_pedigree'] = st.number_input(
                "🧬 Pedigrí Diabetes",
                min_value=0.0, max_value=5.0,
                format="%.2f",
                help="Función del pedigrí de la diabetes"
            )
            inputs['age'] = st.number_input(
                "🎂 Edad (años)",
                min_value=0, max_value=120,
                help="Edad del paciente"
            )
        
        if st.form_submit_button("🚀 Ejecutar Análisis", help="Realizar predicción de diabetes"):
            with st.spinner("🔮 Analizando datos biométricos..."):
                try:
                    response = requests.post(ENDPOINTS["predict"], json=inputs, timeout=15)
                    if response.status_code == 200:
                        handle_prediction_response(response.json(), inputs)
                    else:
                        st.error(f"⚠️ Error en el análisis: {response.text}")
                except Exception as e:
                    st.error(f"🔌 Error de conexión: {str(e)}")

def handle_prediction_response(result, inputs):
    new_entry = {
        **inputs,
        "Probability": result["probabilidad_diabetes"],
        "Diagnóstico": result["diagnóstico"],
        "Threshold": result["threshold_utilizado"],
        "Timestamp": datetime.now().isoformat()
    }
    
    st.session_state.history = pd.concat([
        st.session_state.history,
        pd.DataFrame([new_entry])
    ], ignore_index=True)
    
    with st.container(border=True):
        st.subheader("📊 Resultados del Análisis")
        cols = st.columns([1, 2])
        
        with cols[0]:
            st.metric("Probabilidad", 
                     f"{result['probabilidad_diabetes']*100:.1f}%",
                     help="Probabilidad calculada por el modelo")
            st.metric("Diagnóstico", 
                     result["diagnóstico"], 
                     delta="ALERTA" if result["diagnóstico"] == "Diabetes" else "Normal",
                     delta_color="off" if result["diagnóstico"] == "Diabetes" else "normal")
        
        with cols[1]:
            fig = px.pie(
                values=[result["probabilidad_diabetes"], 1 - result["probabilidad_diabetes"]],
                names=["Diabetes", "No Diabetes"],
                hole=0.5,
                color_discrete_sequence=[THEME['danger'], THEME['success']]
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📝 Recomendaciones Médicas"):
        st.markdown("""
        - Realizar seguimiento glucémico diario
        - Consultar con especialista en endocrinología
        - Mantener dieta balanceada y ejercicio regular
        """)

# -----------------------------------------------------------------------------
# Chat Inteligente
def chat_interface():
    st.header("💬 Asistente Virtual de Diabetes")
    
    # Historial de conversación
    for msg in st.session_state.chat_history:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(f"**{role.capitalize()}:** {msg['content']}")
            st.caption(msg.get("timestamp", ""))
    
    # Acciones rápidas
    with st.expander("⚡ Acciones Instantáneas"):
        cols = st.columns(3)
        actions = {
            "📋 Diario Glucémico": "Generar plantilla para registro de glucosa",
            "🍽 Análisis Nutricional": "Analizar composición de comida",
            "💊 Revisión Medicación": "Evaluar régimen medicamentoso"
        }
        for col, (icon, prompt) in zip(cols, actions.items()):
            with col:
                if st.button(icon, use_container_width=True):
                    handle_chat_action(prompt)
    
    # Entrada de usuario
    if prompt := st.chat_input("Escribe tu consulta médica..."):
        handle_chat_action(prompt)

def handle_chat_action(prompt):
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    try:
        with st.spinner("🧠 Procesando consulta..."):
            response = requests.post(
                ENDPOINTS["chat"],
                json={"query": prompt, "max_length": 1500},
                timeout=120
            )
        
        if response.status_code == 200:
            bot_response = clean_bot_response(response.json().get("response", ""))
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": bot_response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.rerun()
        else:
            raise Exception(f"API Error: {response.status_code}")
    except Exception as e:
        error_msg = f"⚠️ Error en la consulta: {str(e)}"
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": error_msg,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.rerun()

def clean_bot_response(text: str) -> str:
    # (Mantener función original con mejoras adicionales)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[.*?\]", "", text)  # Eliminar notas internas
    text = text.replace("• ", "- ").replace("**", "")
    sections = {
        "Resumen Rápido": "## 📌 Resumen",
        "Análisis Detallado": "## 🔍 Detalles",
        "Plan Nutricional": "## 🥗 Nutrición"
    }
    for old, new in sections.items():
        text = text.replace(old, new)
    return text.strip()

# -----------------------------------------------------------------------------
# Análisis Batch
def batch_analysis():
    st.header("📁 Análisis de Datos Masivos")
    
    uploaded_file = st.file_uploader(
        "Subir conjunto de datos (CSV)",
        type=["csv"],
        help="El archivo debe contener las columnas requeridas"
    )
    
    if uploaded_file:
        with st.spinner("🔍 Procesando dataset..."):
            try:
                df = pd.read_csv(uploaded_file)

                # Primero renombrar
                df = df.rename(columns={
                    'Pregnancies': 'pregnancies',
                    'Glucose': 'glucose',
                    'BloodPressure': 'blood_pressure',
                    'SkinThickness': 'skin_thickness',
                    'Insulin': 'insulin',
                    'BMI': 'bmi',
                    'DiabetesPedigreeFunction': 'diabetes_pedigree',
                    'Age': 'age'
                })

                required_cols = ['pregnancies', 'glucose', 'blood_pressure', 
                                'skin_thickness', 'insulin', 'bmi', 
                                'diabetes_pedigree', 'age']
                
                if not all(col in df.columns for col in required_cols):
                    st.error(f"❌ Dataset inválido. Columnas requeridas: {', '.join(required_cols)}")
                    return
                
                st.session_state.uploaded_data = df
                st.success(f"✅ Dataset cargado: {len(df)} registros")
                
                with st.expander("📋 Vista previa de datos"):
                    st.dataframe(df.head(), use_container_width=True)
                
                if st.button("🧠 Ejecutar Predicciones", type="primary"):
                    process_batch_predictions(df)
            
            except Exception as e:
                st.error(f"⚠️ Error procesando archivo: {str(e)}")

def process_batch_predictions(df):
    with st.spinner("🔮 Ejecutando predicciones batch..."):
        try:
            # Mapeo completo de columnas
            data = df.rename(columns={
                'Pregnancies': 'pregnancies',
                'Glucose': 'glucose',
                'BloodPressure': 'blood_pressure',
                'SkinThickness': 'skin_thickness',
                'Insulin': 'insulin',
                'BMI': 'bmi',
                'DiabetesPedigreeFunction': 'diabetes_pedigree',
                'Age': 'age'
            }).to_dict(orient='records')
            
            response = requests.post(ENDPOINTS["batch"], json={"data": data})
            
            if response.status_code == 200:
                results = response.json()["results"]
                result_df = pd.DataFrame(results)
                
                st.session_state.uploaded_data = pd.concat([
                    df,
                    result_df[['probabilidad_diabetes', 'diagnóstico']]
                ], axis=1)
                
            else:
                raise Exception(f"API Error: {response.text}")
        
        except Exception as e:
            st.error(str(e))
    show_batch_results(st.session_state.uploaded_data)

def show_batch_results(df):
    st.subheader("📈 Análisis de Resultados")
    
    # Mover el expander fuera de los tabs
    with st.expander("📊 Estadísticas Clave", expanded=True):
        cols = st.columns(3)
        cols[0].metric("Total Casos", len(df))
        cols[1].metric("Diabetes Detectadas", df['diagnóstico'].value_counts().get("Diabetes", 0))
        cols[2].metric("Probabilidad Promedio", f"{df['probabilidad_diabetes'].mean()*100:.1f}%")
    
    # Usar tabs sin expanders anidados
    tabs = st.tabs(["Distribución", "Correlaciones", "Datos Completos"])
    
    with tabs[0]:
        fig = px.histogram(
            df, x='probabilidad_diabetes',
            nbins=20, title="Distribución de Riesgo",
            color_discrete_sequence=[THEME['primary']]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        corr_matrix = df.corr(numeric_only=True)
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale='Blues',
            showscale=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "📥 Exportar Resultados",
            data=df.to_csv(index=False),
            file_name="diabedoc_resultados.csv",
            mime="text/csv"
        )

# -----------------------------------------------------------------------------
# Sidebar Avanzado
def advanced_sidebar():
    with st.sidebar:
        st.title("🛠 Panel de Control")
        service_status()
        
        st.divider()
        st.subheader("📚 Historial Clínico")
        
        if not st.session_state.history.empty:
            with st.expander("📜 Ver Registros"):
                st.dataframe(
                    st.session_state.history.sort_values('Timestamp', ascending=False),
                    use_container_width=True
                )
            
            cols = st.columns(2)
            cols[0].download_button(
                "💾 Exportar CSV",
                data=st.session_state.history.to_csv(index=False),
                file_name="historial_clinico.csv"
            )
            if cols[1].button("🧹 Limpiar", type="secondary"):
                st.session_state.history = pd.DataFrame(columns=st.session_state.history.columns)
                st.rerun()
        else:
            st.info("📭 No hay registros disponibles")

# -----------------------------------------------------------------------------
# Main App
def main():
    advanced_sidebar()
    
    tabs = st.tabs([
        "🧪 Predicción Individual", 
        "📊 Análisis Masivo", 
        "💬 Asistente IA"
    ])
    
    with tabs[0]:
        prediction_form()
    
    with tabs[1]:
        batch_analysis()
    
    with tabs[2]:
        chat_interface()

if __name__ == "__main__":
    main()