from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, confloat, conint
import joblib
import pandas as pd
import ollama
import logging
from typing import Optional
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
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")

print("Iniciando DiabeDoc API...")

# Cargar el modelo entrenado
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Modelo cargado exitosamente")

    # Verificar que el modelo tiene el método necesario
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
    description="API para diagnóstico y asesoría en diabetes",
    version="1.1.0"
)

# Configurar CORS (más seguro)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),  
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

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

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, example="¿Qué alimentos debo evitar si tengo diabetes?")
    max_length: Optional[int] = Field(500, ge=100, le=2000)

class PredictionResponse(BaseModel):
    probabilidad_diabetes: float
    diagnóstico: str
    threshold_utilizado: float
    features_recibidas: list[str]

class HealthCheck(BaseModel):
    status: str
    model_version: Optional[str]
    model_type: Optional[str]

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
    try:
        input_data = data.model_dump()
        df = pd.DataFrame([input_data])
        
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
            "features_recibidas": list(input_data.keys())
        }
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno al procesar la predicción"
        )

@app.post("/chat/")
async def chat_with_doctor(data: ChatRequest):
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": data.query}],
            options={'num_predict': data.max_length}
        )
        return {"response": response['message']['content']}
    
    except Exception as e:
        logger.error(f"Error en chat: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail="Error al conectar con el servicio de chat"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true"
    )