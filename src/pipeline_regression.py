import pandas as pd
import os

from sklearn.metrics import r2_score
from src.models.registry import get_regression_models
from src.models.hyper_tuning import tune_regression_model
from src.models.model_selection import select_best_regression_model
from src.models.training import train_regression_models
from src.models.evaluate import bias_normalized_score, evaluate_regression_models, wape
from src.models.presistance import save_artifact
from src.features.feature_engineering import (
    build_features,
    encode_split,
    split_data_regression,
    fit_one_hot_encoder,
    apply_one_hot_encoding,
    fit_target_encoder
)


script_folder = os.path.dirname(os.path.abspath(__file__))
target_folder = os.path.normpath(os.path.join(script_folder, '..', 'data/raw'))



one_hot_categorical_columns = [
    'categorie',
    'stock_unit',
    'politique_reapprovisionnement'
]

target_categorical_columns = [
    'article',
    'site_article'
]

numerical_columns = [
    'prix_unitaire',
    'stock_securite',
    'delai_reapprovisionnement',
    'seuil_reapprovisionnement',
    'mode_reapprovisionnement',
    'month_num',
    'conso_M_precedent_1',
    'conso_M_precedent_2',
    'conso_M_precedent_3',
    'conso_meme_mois_annee_dernier',
    'year',
    'moyenne_conso_3mois',
    'ecart_type_6mois',
    'mois_depuis_dernier_conso'
]




def run_pipeline():

    # === Chargement du dataset toute entier ===
    df = pd.read_csv(os.path.join(target_folder, 'conso_mensuelle_brut.csv'))
    # === Néttoyage et préparation des données ===
    df = build_features(df)
 
    # === Dévision des données du dataset en ===
    # * données d'entraînement
    # * données de validation
    # * données de test
    (x_train_list, y_train_list, x_val_list, y_val_list,
     x_train_final, y_train_final, x_test_final, y_test_final) = split_data_regression(df)
 
    # === Encodage des données catégoriale en utilisant ces methodes ===

    # * one-hot-encoding: pour les variables à faible dimension
    # * target-encoding: pour les variables à haute dimension
    x_train_enc, x_val_enc = [], []
    for i in range(len(x_train_list)):
        xt, xv, _, _ = encode_split(
            x_train_list[i], 
            x_val_list[i], 
            y_train_list[i],
            one_hot_categorical_columns, 
            target_categorical_columns, 
            task="regression"
        )
        x_train_enc.append(xt)
        x_val_enc.append(xv)
 
    # === La phase d'entraînement des models ===
    trained_models = train_regression_models(x_train_enc, y_train_list)
 
    # === la phase d'évaluation des models entraîné en se basant sur ces métriques ===
    # * WAPE
    # * R2
    # * Bias
    results = evaluate_regression_models(x_val_enc, y_val_list, trained_models)

    # === la phase de séléction du meilleur modéle ===
    best_model_name, summary, scores = select_best_regression_model(results)


    # === La phase d'hyperparameter tuning du modèle final, sur toute le dataset ===
    # encodage final
    x_train_final_enc, x_test_final_enc, _, _ = encode_split(
        x_train_final, 
        x_test_final, 
        y_train_final,
        one_hot_categorical_columns, 
        target_categorical_columns, 
        task="regression"
    )
 
    tuned_model, best_params = tune_regression_model(
        best_model_name, 
        x_train_final_enc, 
        y_train_final
    )


    # === Évaluation finale sur les données de test ===
    y_pred_test = tuned_model.predict(x_test_final_enc)
    print("\n=== Evaluation finale sur le test ===")
    print(f"Modele retenu   : {best_model_name}")
    print(f"Hyperparamètres : {best_params}")
    print(f"WAPE  : {wape(y_test_final, y_pred_test):.2f}%")
    print(f"R2    : {r2_score(y_test_final, y_pred_test):.4f}")
    print(f"Bias  : {bias_normalized_score(y_test_final, y_pred_test):.4f}")


    # === Entraînement du modèle choisi sur la totalité du dataset ===
    # * Préparation du dataset:
    x_all = pd.concat([x_train_final_enc, x_test_final_enc], axis=0)
    y_all = pd.concat([y_train_final, y_test_final], axis=0)
    
    # * Préparation du modèle final avec les meilleurs paramétres:
    final_model = get_regression_models()[best_model_name]
    final_model.set_params(**best_params)

    # * Entraînement du modéle final:
    final_model.fit(x_all, y_all)

    # * La récupperation des encodeurs utilisé sur le dataset:
    one_hot_encoder_final = fit_one_hot_encoder(x_train_final, one_hot_categorical_columns)
    x_train_ohe_final = apply_one_hot_encoding(x_train_final, one_hot_encoder_final, one_hot_categorical_columns)
    target_encoder_final = fit_target_encoder(x_train_ohe_final, y_train_final,target_categorical_columns)

    # * Enregistrement du modèle final:
    save_artifact(
        model=final_model,
        one_hot_encoder=one_hot_encoder_final,
        target_encoder=target_encoder_final,
        feature_columns=final_model.feature_names_,
        one_hot_columns=one_hot_categorical_columns,
        target_columns=target_categorical_columns
    )


if __name__ == "__main__":
    run_pipeline()