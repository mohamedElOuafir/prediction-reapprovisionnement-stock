import datetime
import numpy as np
import boto3
import os
import json
import joblib
import pandas as pd
from fastapi import HTTPException
from src.features.feature_engineering import apply_one_hot_encoding, apply_target_encoding, build_features




s3_client = boto3.client("s3")
S3_BUCKET_DATA = os.getenv("S3_BUCKET_DATA")
S3_BUCKET_MODELS = os.getenv("S3_BUCKET_MODELS")



def get_cached_artifact():
    
    model_key = os.getenv("MODEL_KEY") 
    if not model_key:
        # lecture du manifeste si la variable n'est pas définie
        obj = s3_client.get_object(Bucket=S3_BUCKET_MODELS, Key="production_manifest.json")
        manifest = json.loads(obj["Body"].read().decode("utf-8"))
        model_key = f"versions/{manifest['active_version']}/model_artifact.joblib"

    local_path = f"/tmp/{os.path.basename(model_key)}"
    s3_client.download_file(S3_BUCKET_MODELS, model_key, local_path)
    artifact = joblib.load(local_path)
        
    return artifact


def load_latest_historical_data():
    local_path = "/tmp/latest_conso_brut.csv"
    
    date_mois = datetime.datetime.today().strftime("%Y-%m")
    fichier_data = f"data_brut_{date_mois}.csv"
    target_path = f"raw/{date_mois}/{fichier_data}"

    s3_client.download_file(S3_BUCKET_DATA, target_path, local_path)
    return pd.read_csv(local_path)



def predict_next_month(article: str, site_article: str, date_prediction: str, artifact):

    # Récupération du modèle en cache
    model = artifact["model"]
    one_hot_encoder = artifact["one_hot_encoder"]
    target_encoder = artifact["target_encoder"]
    one_hot_columns = artifact["one_hot_columns"]
    target_columns = artifact["target_columns"]
    feature_columns = artifact.get("feature_columns", None)

    # Chargement des données historiques
    df_historique_brut = load_latest_historical_data()

    # Construction et enrichissement des features
    ligne_future = build_future_row(df_historique_brut, article, site_article, date_prediction)
    df_avec_future = pd.concat([df_historique_brut, ligne_future], ignore_index=True)
    df_feat = build_features(df_avec_future)

    # Extraction de la ligne cible
    df_feat['date_mois'] = pd.to_datetime(df_feat['date_mois'])
    match_rows = df_feat[
        (df_feat['article'] == article) & 
        (df_feat['site_article'] == site_article)
    ]

    if match_rows.empty:
        raise HTTPException(status_code=404, detail="Combinaison Article / Site introuvable dans l'historique.")

    ligne = pd.DataFrame([match_rows.iloc[-1]])

    # Encodage & Alignement des colonnes
    x_new = ligne.drop(columns=['conso_ce_mois', 'quantite', 'date_mois'], errors='ignore')
    x_new = apply_one_hot_encoding(x_new, one_hot_encoder, one_hot_columns)
    x_new = apply_target_encoding(x_new, target_encoder, target_columns)

    if feature_columns:
        for col in feature_columns:
            if col not in x_new.columns:
                x_new[col] = 0
        x_new = x_new.reindex(columns=feature_columns, fill_value=0)

    # Inférence
    prediction = model.predict(x_new)[0]
    return {
        "article": article,
        "site_article": site_article,
        "date_prediction": date_prediction,
        "prediction_quantite": float(prediction)
    }




# Fonction pour construire une dataframe correspond au format des données d'ntraînement
def build_future_row(df_historique_brut: pd.DataFrame, article: str, site_article: str, date_prediction: str) -> pd.DataFrame:
    
    historique_article = df_historique_brut[
        (df_historique_brut['article'] == article) &
        (df_historique_brut['site_article'] == site_article)
    ].copy()
 
    if historique_article.empty:
        raise ValueError(f"Aucun historique trouvé pour {article}/{site_article}.")
 
    historique_article['date_mois'] = pd.to_datetime(historique_article['date_mois'])
    derniere_ligne = historique_article.sort_values('date_mois').iloc[-1]
 
    date_future = pd.to_datetime(date_prediction) + pd.DateOffset(months=1)
 
    nouvelle_ligne = derniere_ligne.copy()
    nouvelle_ligne['date_mois'] = date_future
    nouvelle_ligne['quantite'] = np.nan
 
    return pd.DataFrame([nouvelle_ligne])