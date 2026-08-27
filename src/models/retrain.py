import os
import json
import joblib
import pandas as pd
import boto3
from datetime import datetime, timezone
from sklearn.metrics import r2_score

# Importer vos fonctions personnalisées
from src.features.feature_engineering import build_features
from src.models.evaluate import wape, bias_normalized_score
from src.features.feature_engineering import (
    split_data_regression, 
    encode_split, 
    fit_one_hot_encoder, 
    apply_one_hot_encoding, 
    fit_target_encoder
)
from models.training import train_regression_models
from models.evaluate import evaluate_regression_models
from models.model_selection import select_best_regression_model, compare_two_models
from models.hyper_tuning import tune_regression_model
from models.registry import get_regression_models


s3_service = boto3.client("s3")
lambda_service = boto3.client("lambda")

S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET")
S3_BUCKET_MODELS = os.getenv("S3_BUCKET_MODELS")



one_hot_categorical_columns = [
    'categorie',
    'stock_unit',
    'politique_reapprovisionnement'
]

target_categorical_columns = [
    'article',
    'site_article'
]



def get_current_model_manifest():
    try:
        obj = s3_service.get_object(Bucket=S3_BUCKET_MODELS, Key="production_manifest.json")
        return json.loads(obj["Body"].read().decode("utf-8"))
    except s3_service.exceptions.NoSuchKey:
        return None




def get_next_version_id():
    response = s3_service.list_objects_v2(Bucket=S3_BUCKET_MODELS, Prefix="versions/")
    existing_versions = []
    for content in response.get("Contents", []):
        key = content["Key"]
        if "/model_artifact.joblib" in key:
            v_num = int(key.split("/")[1].replace("v", ""))
            existing_versions.append(v_num)
    return max(existing_versions, default=0) + 1




def run_retraining_pipeline():
    os.makedirs("tmp", exist_ok=True)
    
    date_load = datetime.datetime.today().strftime("%Y-%m-%d")
    date_mois = datetime.datetime.today().strftime("%Y/%m")
    
    nom_fichier_data_brut = f"data_brut_{date_load}.csv"
    chemin_data_brut_local = "tmp/conso_mensuelle_brut.csv"
    chemin_data_processed_local = "tmp/conso_mensuelle_processed.parquet"

    # 1. Extraction & Feature Engineering
    fichier_cible = f"raw/{date_mois}/{nom_fichier_data_brut}"
    s3_service.download_file(S3_DATA_BUCKET, fichier_cible, chemin_data_brut_local)

    df = pd.read_csv(chemin_data_brut_local)
    df_processed = build_features(df)
    df_processed.to_parquet(chemin_data_processed_local, index=False)

    nom_fichier_data_processed = f"data_processed_{date_load}.parquet"
    s3_service.upload_file(
        chemin_data_processed_local,
        S3_DATA_BUCKET,
        f"processed/{date_mois}/{nom_fichier_data_processed}"
    )

    # 2. Split des données transformées
    (x_train_list, y_train_list, x_val_list, y_val_list,
     x_train_final, y_train_final, x_test_final, y_test_final) = split_data_regression(df_processed)

    # 3. Encodage et Sélection du meilleur modèle
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

    trained_models = train_regression_models(x_train_enc, y_train_list)
    results = evaluate_regression_models(x_val_enc, y_val_list, trained_models)
    best_model_name, summary, scores = select_best_regression_model(results)

    # 4. Hyperparameter Tuning
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

    # 5. Évaluation finale du Challenger sur le jeu de Test
    y_pred_test = tuned_model.predict(x_test_final_enc)
    
    wape_score = wape(y_test_final, y_pred_test)
    r2 = r2_score(y_test_final, y_pred_test)
    bias_score = bias_normalized_score(y_test_final, y_pred_test)

    print("\n=== Évaluation finale du Challenger ===")
    print(f"Modèle retenu   : {best_model_name}")
    print(f"Hyperparamètres : {best_params}")
    print(f"WAPE : {wape_score:.2f}% | R2 : {r2:.4f} | Bias : {bias_score:.4f}")

    # 6. Entraînement final sur tout le dataset
    x_all = pd.concat([x_train_final_enc, x_test_final_enc], axis=0)
    y_all = pd.concat([y_train_final, y_test_final], axis=0)
    
    final_model = get_regression_models()[best_model_name]
    final_model.set_params(**best_params)
    final_model.fit(x_all, y_all)

    # Génération des encodeurs
    one_hot_encoder_final = fit_one_hot_encoder(x_train_final, one_hot_categorical_columns)
    x_train_ohe_final = apply_one_hot_encoding(x_train_final, one_hot_encoder_final, one_hot_categorical_columns)
    target_encoder_final = fit_target_encoder(x_train_ohe_final, y_train_final, target_categorical_columns)

    # Comparaison Champion vs Challenger
    challenger_metrics = {
        "r2": r2, 
        "wape": wape_score, 
        "bias": bias_score
    }
    current_model_manifest = get_current_model_manifest()

    if current_model_manifest:
        champion_metrics = current_model_manifest['metrics']
        should_promote, _ = compare_two_models(champion_metrics, challenger_metrics)
    else:
        # Si aucun modèle n'est en prod
        should_promote = True

    # 8. Sauvegarde & Promotion
    version_num = get_next_version_id()
    version_id = f"v{version_num}"
    
    # Étape essentielle manquante : Sauvegarde physique et Upload de l'artéfact
    local_artifact_path = f"tmp/model_artifact_{version_id}.joblib"
    joblib.dump({
        "model": final_model,
        "one_hot_encoder": one_hot_encoder_final,
        "target_encoder": target_encoder_final,
        "one_hot_columns": one_hot_categorical_columns,
        "target_columns": target_categorical_columns
    }, local_artifact_path)

    s3_version_key = f"versions/{version_id}/model_artifact.joblib"
    s3_service.upload_file(local_artifact_path, S3_BUCKET_MODELS, s3_version_key)

    if should_promote:
        print(f"\n Promotion : Modèle {version_id} promu en production.")

        new_manifest = {
            "active_version": version_id,
            "promoted_at": datetime.datetime.now(timezone.utc).isoformat(),
            "metrics": challenger_metrics,
            "previous_version": current_model_manifest.get("active_version") if current_model_manifest else None
        }

        # Mise à jour du manifeste prod sur S3
        s3_service.put_object(
            Bucket=S3_BUCKET_MODELS,
            Key="production_manifest.json",
            Body=json.dumps(new_manifest, indent=4)
        )

        # Bascule de l'API AWS Lambda
        lambda_service.update_function_configuration(
            FunctionName="reapprovisionnement-api",
            Environment={"Variables": {
                "MODEL_BUCKET": S3_BUCKET_MODELS,
                "MODEL_KEY": s3_version_key
            }}
        )
    else:
        active_version = current_model_manifest.get('active_version') if current_model_manifest else None
        print(f"\n Conservation : Le Champion {active_version} reste en production.")


if __name__ == "__main__":
    run_retraining_pipeline()