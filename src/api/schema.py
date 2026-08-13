from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    article: str
    site_article: str


class PredictionResponse(BaseModel):
    date_prediction: str
    prediction_quantite: float