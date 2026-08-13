import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from features.feature_engineering import apply_one_hot_encoding, apply_target_encoding, build_features
from models.presistance import load_artifact

script_folder = os.path.dirname(os.path.abspath(__file__))
target_folder = os.path.normpath(os.path.join(script_folder, '../..', 'data/raw'))

    
# Fonction d'inférence pour prédire la quantité de consommation du stock sur:
# * Article
# * Site - article
# * date prédiction
def predict_next_month( 
        article: str, 
        site_article: str,
        date_prediction: str,
        artifact_name: str = "regression"
    ):

    df_historique_brut = pd.read_csv(os.path.join(target_folder, 'conso_mensuelle_sim.csv'))
    
    # Récuppération de l'artifact du modèle
    artifact = load_artifact()

    # Récuppération de tous les élèments de l'artifact
    # * modèle
    # * one hot encoder
    # * target encoder
    # * tous les variables du modèle
    # * colonnes pour le one hot encoding
    # * colonnes pour le target encoding
    model = artifact["model"]
    one_hot_encoder = artifact["one_hot_encoder"]
    target_encoder = artifact["target_encoder"]
    feature_columns = artifact["feature_columns"]
    one_hot_columns = artifact["one_hot_columns"]
    target_columns = artifact["target_columns"]

    # Construire la ligne future 
    ligne_future, date_future = build_future_row(df_historique_brut, article, site_article, date_prediction)
 
    # L'ajouter à l'historique brut,
    df_avec_future = pd.concat([df_historique_brut, ligne_future], ignore_index=True)
    df_feat = build_features(df_avec_future)
 
    # Récupérer la ligne du mois futur, maintenant enrichie de tous ses lags
    df_feat['date_mois'] = pd.to_datetime(df_feat['date_mois'])
    ligne = pd.DataFrame([df_feat[
        (df_feat['article'] == article) &
        (df_feat['site_article'] == site_article)
    ].iloc[-1]])
 
    if ligne.empty:
        raise ValueError("La ligne future n'a pas été retrouvée après build_features() — "
                          "vérifiez le tri/groupby de build_features().")
 
    # Encodage avec les encodeurs déjà entraînés
    x_new = ligne.drop(columns=['conso_ce_mois', 'quantite', 'date_mois'])
    x_new = apply_one_hot_encoding(x_new, one_hot_encoder, one_hot_columns)
    x_new = apply_target_encoding(x_new, target_encoder, target_columns)
 
    for col in feature_columns:
        if col not in x_new.columns:
            x_new[col] = 0
    x_new = x_new.reindex(columns=feature_columns, fill_value=0)

    print(f"features in model artifact:\n\n{model.feature_names_}\n\n")
    print(f"features provient:\n\n{list(x_new.columns)}")
 
    # Prédiction
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
 
    return pd.DataFrame([nouvelle_ligne]), date_future

