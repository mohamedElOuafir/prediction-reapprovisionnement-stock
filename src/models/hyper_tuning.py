from sklearn.metrics import make_scorer
from sklearn.model_selection import  TimeSeriesSplit, GridSearchCV
from models.evaluate import wape
from models.registry import get_regression_models, hyperparametres_grid


def _wape_scorer(y_true, y_pred):
    return -wape(y_true, y_pred)


def tune_regression_model(
        model_name: str, 
        x_train_final, 
        y_train_final,
        n_iter: int = 20, 
        n_splits: int = 5, 
        random_state: int = 42
    ):

    base_model = get_regression_models()[model_name]
    param_grid = hyperparametres_grid.get(model_name, {})
 
    if not param_grid:
        print(f"[{model_name}] Pas d'hyperparametre à optimiser — entraînement direct")
        base_model.fit(x_train_final, y_train_final)
        return base_model, {}
 
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scorer = make_scorer(_wape_scorer, greater_is_better=True)
 
    search = GridSearchCV(
        base_model, 
        param_grid=param_grid, 
        cv=tscv, 
        scoring=scorer, 
        n_jobs=-1
    )
    search.fit(x_train_final, y_train_final)
 
    print(f"[{model_name}] Meilleurs hyperparametres : {search.best_params_}")
    return search.best_estimator_, search.best_params_