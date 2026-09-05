
from fastapi import FastAPI, HTTPException, Header, Depends
import datetime
import os
import math
from src.api.schema import PredictionRequest,PredictionResponse
from src.predictor.predict_local import predict_next_month


app = FastAPI(title="API for stock consumption prediction")

API_ML_SECRET_KEY = os.getenv("API_ML_SECRET_KEY", "WWqYKuzyjAmYUND2WzGXJbeqqEcmXXFzTmaZc0xd")

async def authentication_verification(authorization:str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer ") or authorization.split(" ")[1] != API_ML_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalide API key")
    return authorization

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
@app.post("/prediction", dependencies=[Depends(authentication_verification)], response_model=PredictionResponse)
async def predict(request: PredictionRequest):

    
    article = request.article
    site_article = request.site_article

    date = datetime.datetime.now()
    date_prediction = date.strftime("%Y-%m-%d")

    result = predict_next_month(
        article,
        site_article,
        date_prediction
    )

    return PredictionResponse(
        date_prediction=result["date_prediction"],
        prediction_quantite=math.floor(result["prediction_quantite"]),
    )


