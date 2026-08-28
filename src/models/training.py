import numpy as np
from src.models.registry import get_classification_models, get_regression_models



# Fonction pour lancer l'entraînement des models de classification:
def train_classification_models(x_train_list, y_train_list):
    trained_models_all_splits = {}
 
    for i in range(len(x_train_list)):
        x_train, y_train = x_train_list[i], y_train_list[i]
 
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        scale_pos_weight = float(n_neg) / n_pos if n_pos > 0 else 1.0
 
        print(f"************ Split {i} — n_pos={n_pos}, n_neg={n_neg} *************")
 
        models = get_classification_models(scale_pos_weight=scale_pos_weight)
        trained_split = {}
        for name, model in models.items():
            model.fit(x_train, y_train)
            trained_split[name] = model
 
        trained_models_all_splits[f"split_{i}"] = trained_split
 
    return trained_models_all_splits



# Fonction pour lancer l'entraînement des modèles de regréssion:
def train_regression_models(x_train_list, y_train_list):
    trained_models_all_splits = {}
 
    for i in range(len(x_train_list)):
        x_train, y_train = x_train_list[i], y_train_list[i]
        print(f"************ Split {i + 1} *************")
 
        models = get_regression_models()
        trained_split = {}
        for name, model in models.items():
            model.fit(x_train, y_train)
            trained_split[name] = model
 
        trained_models_all_splits[f"split_{i}"] = trained_split
 
    return trained_models_all_splits

