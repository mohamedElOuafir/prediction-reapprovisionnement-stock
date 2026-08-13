from collections import defaultdict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,

)




# Fonction pour parcourir les models de classification entraîné et calculer les metriques suivantes pour l'évaluation:
# * accuracy
# * precision
# * recall
# * f1 score
def evaluate_classification_models(
        x_val_list, 
        y_val_list, 
        trained_models_all_splits,
        threshold=0.5, 
        verbose=True
    ):

    results = defaultdict(lambda: defaultdict(list))
 
    for split_name, models in trained_models_all_splits.items():
        i = int(split_name.split('_')[1])
        x_val, y_val = x_val_list[i], y_val_list[i]
 
        for name, model in models.items():
            y_prob = model.predict_proba(x_val)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
 
            results[name]["accuracy"].append(accuracy_score(y_val, y_pred))
            results[name]["precision"].append(precision_score(y_val, y_pred, zero_division=0))
            results[name]["recall"].append(recall_score(y_val, y_pred, zero_division=0))
            results[name]["f1"].append(f1_score(y_val, y_pred, zero_division=0))
 
    if verbose:
        for name, metrics in results.items():
            print(f"===== {name} =====")
            for metric_name, values in metrics.items():
                vals_str = [f"{v:.3f}" for v in values]
                print(f"{metric_name} par Split: {vals_str}")
                print(f"{metric_name}: mean = {np.mean(values):.4f}, std = {np.std(values):.4f}")
            print()
 
    return results

    


# Fonction pour parcourir les models de regression entraîné et calculer les metriques suivants pour l'évaluation:
# * WAPE
# * R2
# * Bias
def evaluate_regression_models(x_val_list, y_val_list, trained_models_all_splits, verbose=True):

    results = defaultdict(lambda: defaultdict(list))
 
    for split_name, models in trained_models_all_splits.items():
        i = int(split_name.split('_')[1])
        x_val, y_val = x_val_list[i], y_val_list[i]
 
        for name, model in models.items():
            y_pred = model.predict(x_val)
            results[name]["wape"].append(wape(y_val, y_pred))
            results[name]["bias"].append(bias_normalized_score(y_val, y_pred))
            results[name]["r2"].append(r2_score(y_val, y_pred))
 
    if verbose:
        for name, metrics in results.items():
            print(f"===== {name} =====")
            for metric_name, values in metrics.items():
                vals_str = [f"{v:.3f}" for v in values]
                print(f"{metric_name} par fold: {vals_str}")
                print(f"{metric_name}: mean = {np.mean(values):.4f}, std = {np.std(values):.4f}")
            print()
 
    return results



"""def evaluate_regression_models(x_test, y_test, models):

    print(f"============== Evaluation phase ==============\n\n")
    wape_list = []
    r2_list = []
    bias_list = []


    for name, model in models.items():

        print(f"============== Evaluation du model {name} ==============\n\n")
        y_pred = model.predict(x_test)

        wape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        bias = bias_normalized_score(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        wape_list.append(wape)
        r2_list.append(r2)
        bias_list.append(bias)

        print(f"wape: {wape:.4f}\nr2 score: {r2:.4f}\nbias: {bias:.4f}\n\n")

    print(f"============== Evaluation phase finished ==============\n\n")

    return wape_list, r2_list, bias_list"""




def evaluate_classification_models_2(x_test, y_test, model):

    y_prob = model.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    print(
        f"accuracy: {accuracy:.4f}, recall: {recall:.4f}, precision: {precision:.4f}, f1: {f1:.4f}\n"
    )


# fonction pour calculer le WAPE:
def wape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

# fonction pour calculer le bias normalizé:
def bias_normalized_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(y_pred - y_true) / np.sum(y_true) * 100