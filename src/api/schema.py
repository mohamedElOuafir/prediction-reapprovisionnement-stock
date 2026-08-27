from pydantic import BaseModel


class PredictionRequest(BaseModel):
    article: str
    site_article: str


class PredictionResponse(BaseModel):
    date_prediction: str
    prediction_quantite: float


class ResTest(BaseModel):
    status: int
    api_healthy: bool