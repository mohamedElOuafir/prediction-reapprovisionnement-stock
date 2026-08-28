
from fastapi import FastAPI
import datetime
import math
from mangum import Mangum
from src.api.schema import PredictionRequest,PredictionResponse
from src.predictor.predict import get_cached_artifact, predict_next_month


app = FastAPI(title="API for stock consumption prediction")

model_artifact = None

def get_model():
    global model_artifact
    if model_artifact is None:
        model_artifact = get_cached_artifact()
    return model_artifact


@app.get("/")
async def get_rac():
    return {
        "server": "ok"
    }

# Route pour vérifier le status du serveur
@app.get("/health_check")
async def check_api_health():
    return {
        "status": 200,
        "api_healthy": True
    }


# Route pour faire une prédiction sur la quantité de consommation d'un article sur un site spécifique
@app.post("/prediction", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    model_artifact = get_model()
    
    article = request.article
    site_article = request.site_article

    date = datetime.datetime.now()
    date_prediction = date.strftime("%Y-%m-%d")

    result = predict_next_month(
        article,
        site_article,
        date_prediction,
        model_artifact
    )

    return PredictionResponse(
        date_prediction=result["date_prediction"],
        prediction_quantite=math.floor(result["prediction_quantite"]),
    )


handler = Mangum(app)
