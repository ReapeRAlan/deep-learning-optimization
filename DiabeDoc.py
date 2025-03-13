from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, confloat, conint
import joblib
import pandas as pd
import ollama
import logging
from typing import Optional, List
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Cargar configuración desde .env
load_dotenv()

# Configuración de logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Variables de entorno
MODEL_PATH = os.getenv("MODEL_PATH", "diabetes_model_pipeline_calibrated.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-r1:7b")

print("Iniciando DiabeDoc API...")

# Cargar el modelo entrenado
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Modelo cargado exitosamente")
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("El modelo cargado no tiene el método predict_proba")
except Exception as e:
    logger.error(f"Error crítico al cargar el modelo: {e}")
    raise

# Validar features del modelo
try:
    expected_features = model.feature_names_in_
except AttributeError:
    expected_features = None
    logger.warning("No se pudieron validar las features del modelo")

# Inicializar FastAPI
app = FastAPI(
    title="DiabeDoc API",
    description=(
        "API para diagnóstico y asesoría especializada en diabetes. "
        "Provee predicciones basadas en un modelo clínico y recomendaciones "
        "para el manejo de la enfermedad."
    ),
    version="1.1.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------
# Modelos Pydantic

class DiabetesFeatures(BaseModel):
    pregnancies: conint(ge=0) = Field(..., example=2)
    glucose: conint(ge=0) = Field(..., example=148)
    blood_pressure: conint(ge=0) = Field(..., example=72)
    skin_thickness: conint(ge=0) = Field(..., example=35)
    insulin: conint(ge=0) = Field(..., example=0)
    bmi: confloat(ge=0) = Field(..., example=33.6)
    diabetes_pedigree: confloat(ge=0) = Field(..., example=0.627)
    age: conint(ge=21, le=120) = Field(..., example=50)

class BatchDiabetesFeatures(BaseModel):
    data: List[DiabetesFeatures]

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, example="¿Qué alimentos debo evitar si tengo diabetes?")
    max_length: Optional[int] = Field(500, ge=100, le=2000)
    # Se añade el campo context para almacenar contexto adicional de la conversación
    context: Optional[str] = Field(None, example="Historial previo de la conversación")

class PredictionResponse(BaseModel):
    probabilidad_diabetes: float
    diagnóstico: str
    threshold_utilizado: float
    features_recibidas: List[str]

class HealthCheck(BaseModel):
    status: str
    model_version: Optional[str]
    model_type: Optional[str]

# ---------------------------
# Funciones Auxiliares de Post-procesamiento

def add_safety_warnings(text: str) -> str:
    """
    Añade advertencias médicas automáticas basadas en contenido detectado.
    """
    keywords = {
        'hipoglucemia': '⚠️ Si presenta síntomas como temblores, sudoración o confusión, consuma 15g de carbohidratos rápidos inmediatamente.',
        'hiperglucemia': '⚠️ Niveles persistentes sobre 250 mg/dL requieren atención médica urgente.',
        'cetonas': '⚠️ Si tiene niveles altos de cetonas (>1.5 mmol/L), busque atención médica inmediata.'
    }
    lower_text = text.lower()
    for k, warning in keywords.items():
        if k in lower_text:
            text += f"\n\n{warning}"
    return text

def extract_entities(text: str, entity_types: List[str]) -> List[str]:
    """
    Extrae entidades de tipo 'medicamento', 'número', 'acción', etc.
    (Versión placeholder que no hace NLP real.)
    """
    # Por ahora devolvemos lista vacía o algo muy básico
    return []

def extract_contextual_data(text: str) -> dict:
    """
    Extrae información relevante para seguimiento continuo.
    """
    return {
        "mentioned_medications": extract_entities(text, ['medicamento']),
        "glucose_values": extract_entities(text, ['número']),
        "next_recommendations": extract_entities(text, ['acción'])
    }

# ---------------------------
# Endpoints

@app.get("/", include_in_schema=False)
def read_root():
    return {"message": "Bienvenido a DiabeDoc API - Consulte /docs para documentación"}

@app.get("/health/", response_model=HealthCheck)
def health_check():
    return {
        "status": "OK",
        "model_version": getattr(model, "version", "desconocido"),
        "model_type": str(type(model))
    }

@app.post("/predict/", response_model=PredictionResponse)
async def predict_diabetes(data: DiabetesFeatures):
    """
    Endpoint para predicción individual.
    """
    try:
        input_data = data.model_dump()
        mapping = {
            "pregnancies": "Pregnancies",
            "glucose": "Glucose",
            "blood_pressure": "BloodPressure",
            "skin_thickness": "SkinThickness",
            "insulin": "Insulin",
            "bmi": "BMI",
            "diabetes_pedigree": "DiabetesPedigreeFunction",
            "age": "Age"
        }
        renamed_data = {mapping[k]: v for k, v in input_data.items() if k in mapping}
        
        # Calcular features adicionales
        renamed_data["AgeBMI"] = renamed_data["Age"] * renamed_data["BMI"]
        renamed_data["Glucose2"] = renamed_data["Glucose"] ** 2

        if expected_features is not None:
            df = pd.DataFrame([[renamed_data[col] for col in expected_features]], columns=expected_features)
        else:
            df = pd.DataFrame([renamed_data])
        
        if expected_features is not None:
            if set(df.columns) != set(expected_features):
                missing = set(expected_features) - set(df.columns)
                extra = set(df.columns) - set(expected_features)
                raise HTTPException(
                    status_code=400,
                    detail=f"Error en features: Faltantes {missing}, Extras {extra}"
                )
                
        proba = model.predict_proba(df)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        
        return {
            "probabilidad_diabetes": round(proba[0], 4),
            "diagnóstico": "Diabetes" if pred[0] == 1 else "No diabetes",
            "threshold_utilizado": THRESHOLD,
            "features_recibidas": list(df.columns)
        }
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno al procesar la predicción"
        )

@app.post("/predict_batch")
async def predict_batch(batch: BatchDiabetesFeatures):
    """
    Endpoint para procesar un batch de registros en una sola petición.
    """
    try:
        mapping = {
            "pregnancies": "Pregnancies",
            "glucose": "Glucose",
            "blood_pressure": "BloodPressure",
            "skin_thickness": "SkinThickness",
            "insulin": "Insulin",
            "bmi": "BMI",
            "diabetes_pedigree": "DiabetesPedigreeFunction",
            "age": "Age"
        }
        transformed = []
        for entry in batch.data:
            data_dict = entry.model_dump()
            renamed = {mapping[k]: v for k, v in data_dict.items() if k in mapping}
            renamed["AgeBMI"] = renamed["Age"] * renamed["BMI"]
            renamed["Glucose2"] = renamed["Glucose"] ** 2
            transformed.append(renamed)
        
        if expected_features is not None:
            df = pd.DataFrame(transformed, columns=expected_features)
        else:
            df = pd.DataFrame(transformed)
        
        proba = model.predict_proba(df)[:, 1]
        predictions = (proba >= THRESHOLD).astype(int)
        
        results = []
        for prob, pred in zip(proba, predictions):
            results.append({
                "probabilidad_diabetes": round(prob, 4),
                "diagnóstico": "Diabetes" if pred == 1 else "No diabetes",
                "threshold_utilizado": THRESHOLD
            })
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Error en batch: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno en procesamiento batch"
        )

@app.post("/chat/")
async def chat_with_doctor(data: ChatRequest):
    """
    Endpoint para consultas especializadas en diabetes con enfoque médico personalizado.
    """
    try:
        # Configuración del rol (Endocrinólogo, 20 años de experiencia, etc.)
        expert_config = {
            "role": "system",
            "content": (
                "ACTÚA COMO: Endocrinólogo especialista en diabetes con 20 años de experiencia. "
                "REQUISITOS:\n"
                "1. Analiza detalladamente historial médico, glucosa en sangre, HbA1c, medicación actual.\n"
                "2. Considera edad, peso, tipo de diabetes, complicaciones existentes.\n"
                "3. Proporciona planes de tratamiento personalizados incluyendo:\n"
                "   - Ajuste de medicación (insulina, metformina, etc.)\n"
                "   - Plan nutricional detallado con ejemplos de menús\n"
                "   - Rutina de ejercicio adaptada\n"
                "   - Manejo de hipo/hiperglucemias\n"
                "4. Evalúa y sugiere ajustes basados en últimos valores glucémicos.\n"
                "5. Proporciona interpretación de análisis clínicos.\n"
                "6. Ofrece apoyo emocional y manejo del estrés.\n"
                "\nPROTOCOLO DE SEGURIDAD:\n"
                "- Verificar si se necesita atención de emergencia\n"
                "- En casos de valores peligrosos (glucosa <70 o >300 mg/dL) indicar acción inmediata\n"
                "- Siempre recomendar consultar con médico tratante\n"
                "\nFORMATO RESPUESTAS:\n"
                "📌 [Resumen rápido]\n"
                "🔍 [Análisis detallado]\n"
                "💊 [Manejo médico]\n"
                "🥗 [Plan nutricional]\n"
                "🏋️ [Actividad física]\n"
                "📉 [Monitorización]\n"
                "⚠️ [Precauciones]"
            )
        }

        messages = [expert_config, {"role": "user", "content": data.query}]
        
        # Si hay contexto previo, lo insertamos en la conversación
        if data.context:
            messages.insert(1, {"role": "assistant", "content": data.context})

        response = ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            options={
                'num_predict': data.max_length,
                'temperature': 0.3
            }
        )
        
        raw_text = response['message']['content']
        # Añadir advertencias de seguridad
        processed_text = add_safety_warnings(raw_text)
        # Extraer contexto
        context_info = extract_contextual_data(raw_text)
        
        return {
            "response": processed_text,
            "medical_context": context_info
        }
    except Exception as e:
        logger.error(f"Error en consulta médica: {str(e)}", extra={"query": data.query})
        raise HTTPException(
            status_code=503,
            detail="Servicio médico no disponible. Por favor intente nuevamente."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true"
    )
