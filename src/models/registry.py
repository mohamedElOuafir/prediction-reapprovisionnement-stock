from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier


# Fonction pour récuperrer les modéles de regréssion
def get_regression_models():
    return {
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Light GBM": LGBMRegressor(random_state=42, verbosity=-1),
        "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
    }


# Fonction pour récuperrer les modéles de classification
def get_classification_models(scale_pos_weight: float = 1.0):
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "Light GBM": LGBMClassifier(random_state=42, class_weight='balanced', verbosity=-1),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42, auto_class_weights='Balanced'),
        "XGBoost": XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight),
    }


# list des hyperparamétres pour le tuning du modèle séléctionné
hyperparametres_grid = {
    "Decision Tree": {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "min_samples_split": [2, 10, 20],
    },
    "Linear Regression": {},   # rien à tuner
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 400],
        "max_depth": [5, 10, 15, None],
        "min_samples_leaf": [1, 5, 10],
        "max_features": ["sqrt", "log2", 0.5],
    },
    "Light GBM": {
        "n_estimators": [100, 200, 400],
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_child_samples": [5, 10, 20],
    },
    "CatBoost": {
        "iterations": [200, 400, 600],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_leaf_reg": [1, 3, 5, 10],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 400],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
    },
}