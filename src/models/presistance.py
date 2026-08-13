import joblib
import os

script_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.normpath(os.path.join(script_folder, '../..'))
artifact_folder = os.path.join(parent_folder, 'artifacts')
os.makedirs(artifact_folder, exist_ok=True)


# Fonction pour sauvegarder le modèle final
def save_artifact(
        model,
        one_hot_encoder,
        target_encoder,
        feature_columns,
        one_hot_columns,
        target_columns
):

    model_path = os.path.join(artifact_folder, "model_artifact.joblib")
    joblib.dump({
        "model": model,
        "one_hot_encoder": one_hot_encoder,
        "target_encoder": target_encoder,
        "feature_columns": feature_columns,
        "one_hot_columns": one_hot_columns,
        "target_columns": target_columns
    }, model_path)



# Fonction pour récuppérer le modéle sauvegardé
def load_artifact():
    artifact_path = os.path.join(artifact_folder, "model_artifact.joblib")
    return joblib.load(artifact_path)