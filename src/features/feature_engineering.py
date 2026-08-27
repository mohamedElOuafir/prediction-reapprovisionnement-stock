import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder, OrdinalEncoder, Normalizer
from sklearn.model_selection import TimeSeriesSplit
import os


script_folder = os.path.dirname(os.path.abspath(__file__))
target_folder = os.path.normpath(os.path.join(script_folder, '../..', 'data/processed'))


# Fonction pour la création des nouvelles variables 
def build_features(df: pd.DataFrame):

    # Triage des lignes des données:
    # * Tri par article , site et la date
    # * Groupement des quantite consommés par article et site d'article
    df = df.sort_values(['article', 'site_article', 'date_mois'])
    group = df.groupby(['article', 'site_article'])['quantite']

    # ---- Création des variables ----:
    # * variable pour donner le nombre de mois depuis la dernier consommation positive de l'article sur un site
    df['quantite_conso_passe'] = group.shift(1)
    df['indice_mois'] = group.cumcount()
    df['dernier_conso_indice'] = np.where(df['quantite_conso_passe'] > 0, df['indice_mois'] - 1, np.nan)
    df['dernier_conso_indice'] = df.groupby(['article', 'site_article'])['dernier_conso_indice'].ffill()
    df['mois_depuis_dernier_conso'] = df['indice_mois'] - df['dernier_conso_indice']
    df = df.drop(columns=['indice_mois', 'dernier_conso_indice', 'quantite_conso_passe'])

    # * variable pour calculer la frequence de consommation sur 12 mois
    df['frequence_12M'] = group.transform(lambda s: (s > 0).astype(int).shift(1).rolling(window=12).sum()) / 12

    # * variable cible pour la classification 
    df['conso_ce_mois'] = (df['quantite'] > 0).astype('Int64')

    # * quantité consommé de l'article sur un site il y a 1 mois
    # * quantité consommé de l'article sur un site il y a 2 mois
    # * quantité consommé de l'article sur un site il y a 3 mois
    # * quantité consommé de l'article sur un site l'année dérnière sur le même mois
    df['conso_M_precedent_1'] = group.shift(1)
    df['conso_M_precedent_2'] = group.shift(2)
    df['conso_M_precedent_3'] = group.shift(3)
    df['conso_meme_mois_annee_dernier'] = group.shift(12)

    # * extraction de l'année et le mois
    df['date_mois'] = pd.to_datetime(df['date_mois'])
    df['year'] = df['date_mois'].dt.year
    df['month'] = df['date_mois'].dt.month

    # * moyenne de la consommation sur les 3 mois précedents
    # * l'écart type de la quantité consommé sur les 6 mois précedents
    df['moyenne_conso_3mois'] = df[['conso_M_precedent_1', 'conso_M_precedent_2', 'conso_M_precedent_3']].mean(axis=1)
    df['ecart_type_6mois'] = group.transform(lambda s: s.shift(1).rolling(window=6).std())

    df = df.sort_values(['date_mois', 'article', 'site_article'])

    # Elimination des variables inutiles: date_mois, ABC_class et article_statut
    df.drop(columns=['ABC_class', 'article_statut'], inplace=True)

    # Gestion des valeurs nulles crée par les lags (conso_M_precedent_1,2,3)
    df = df.fillna(-1)

    return df



# Fonction pour faire l'encodage pour les variables catégoriales en utilisant One-Hot-Encoding
def fit_one_hot_encoder(x_train: pd.DataFrame, columns: list):
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(x_train[columns])
    return encoder
 
def apply_one_hot_encoding(x: pd.DataFrame, encoder: OneHotEncoder, columns: list):
    encoded = encoder.transform(x[columns])
    x_enc = pd.DataFrame(
        encoded.toarray(),
        columns=encoder.get_feature_names_out(columns),
        index=x.index
    )
    return pd.concat([x.drop(columns=columns), x_enc], axis=1)




# Fonction pour faire l'encodage pour les variables catégoriales en utilisant Target-Encoding
def fit_target_encoder(x_train: pd.DataFrame, y_train: pd.DataFrame, columns: list, task: str = "regression"):

    if task == "regression":
        target_type = "continuous"
    else:
        target_type = "binary"

    encoder = TargetEncoder(target_type=target_type)
    encoder.fit(x_train[columns], y_train)
    return encoder


def apply_target_encoding(x: pd.DataFrame, encoder: TargetEncoder, columns: list):
    x = x.copy()
    x[columns] = encoder.transform(x[columns])
    return x



# Fonction pour faire l'encodage de la totalité du dataset en utilisant:
# * one-hot-encoding: pour les variables catégoriales à faible dimensionalité
# * target-encoding: pour les variables catégoriales à haute dimensionalité
def encode_split(
        x_train: pd.DataFrame, 
        x_val: pd.DataFrame, 
        y_train: pd.DataFrame, 
        one_hot_columns: list, 
        target_columns: list, 
        task: str ="regression"
    ):
    
    one_hot_encoder = fit_one_hot_encoder(x_train, one_hot_columns)
    x_train_enc = apply_one_hot_encoding(x_train, one_hot_encoder, one_hot_columns)
    x_val_enc = apply_one_hot_encoding(x_val, one_hot_encoder, one_hot_columns)
 
    target_encoder = fit_target_encoder(x_train_enc, y_train, target_columns, task=task)
    x_train_enc = apply_target_encoding(x_train_enc, target_encoder, target_columns)
    x_val_enc = apply_target_encoding(x_val_enc, target_encoder, target_columns)
 
    return x_train_enc, x_val_enc, one_hot_encoder, target_encoder




def ordinal_encoding_categorical_features(
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        columns: list
):
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    for col in columns:
        x_train[[col]] = ordinal_encoder.fit_transform(x_train[[col]])
        x_test[[col]] = ordinal_encoder.transform(x_test[[col]])

    return x_train, x_test



def standrize_numerical_columns(x_train, x_val, columns):
    scaler = StandardScaler()
    x_train[columns] = scaler.fit_transform(x_train[columns])
    x_val[columns] = scaler.transform(x_val[columns])

    return x_train, x_val

def normalize_numerical_columns(x_train, x_val, columns):
    normlizer = Normalizer()
    x_train[columns] = normlizer.fit_transform(x_train[columns])
    x_val[columns] = normlizer.transform(x_val[columns])

    return x_train, x_val



# Fonction pour faire une séparation des données avec une séparation multiple en utilisant la fenêtre extensive technique:
# * données d'entraînement
# * données de validation
# * données de test
def split_data(
        df: pd.DataFrame, 
        target_col: str, 
        drop_cols: list,
        test_ratio: float = 0.15, 
        n_splits: int = 5
    ):

    df = df.sort_values(['date_mois', 'article', 'site_article'])
 
    dates = sorted(df['date_mois'].unique())
    train_edge = dates[int(len(dates) * (1 - test_ratio))]
 
    train_mask = df['date_mois'] <= train_edge
    test_mask = df['date_mois'] > train_edge
 
    x_all = df.drop(columns=drop_cols)
    y_all = df[target_col]
 
    x_train_final, y_train_final = x_all[train_mask], y_all[train_mask]
    x_test_final, y_test_final = x_all[test_mask], y_all[test_mask]
 
    tscv = TimeSeriesSplit(n_splits=n_splits)
    x_train_list, y_train_list, x_val_list, y_val_list = [], [], [], []
 
    for train_idx, val_idx in tscv.split(x_train_final):
        x_train_list.append(x_train_final.iloc[train_idx])
        y_train_list.append(y_train_final.iloc[train_idx])
        x_val_list.append(x_train_final.iloc[val_idx])
        y_val_list.append(y_train_final.iloc[val_idx])
 
    return (x_train_list, y_train_list, x_val_list, y_val_list,
            x_train_final, y_train_final, x_test_final, y_test_final)



# Fonction pour faire la séparation des données pour le cas de la regression
def split_data_regression(df: pd.DataFrame, only_positive: bool = True, **kwargs):
    
    if only_positive:
        df = df[df['quantite'] > 0].copy()
    return split_data(
        df, 
        target_col='quantite',
        drop_cols=['conso_ce_mois', 'quantite', 'date_mois'], 
        **kwargs
    )


# Fonction pour faire la séparation des données pour le cas de la classification
def split_data_classification(df: pd.DataFrame, **kwargs):
    return split_data(
        df, 
        target_col='conso_ce_mois',
        drop_cols=['conso_ce_mois', 'quantite', 'date_mois'], 
        **kwargs
    )



"""def split_data_classification(df: pd.DataFrame):

    # triage par date pour faire construire une serie temporelle
    df = df.sort_values(['date_mois', 'article', 'site_article'])

    # Séparation entre des données utilisés pour l'entraînement et le test
    dates = sorted(df['date_mois'].unique())
    dates_length = len(dates)

    train_edge = dates[int(dates_length * 0.85)]

    train_mask = df['date_mois'] <= train_edge
    test_mask = df['date_mois'] > train_edge

    # Séparation des variables caractéristique et la variable cible
    x_temp = df.drop(columns=['conso_ce_mois', 'quantite', 'date_mois'])
    y_temp = df['conso_ce_mois']

    # Séparation entre les donnée d'entraînement et validation pour la phase d'entraînement et évaluation du model
    # Séparation des données de test final
    x_train_final, y_train_final = x_temp[train_mask], y_temp[train_mask]
    x_test_final, y_test_final = x_temp[test_mask], y_temp[test_mask]

    # Séparation des données d'entraînement et de test par le time split séries
    time_split = TimeSeriesSplit(n_splits=5)

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []

    for train_index, val_index in time_split.split(x_train_final):
        x_train, y_train = x_train_final.iloc[train_index], y_train_final.iloc[train_index]
        x_val, y_val = x_train_final.iloc[val_index], y_train_final.iloc[val_index]

        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_val_list.append(x_val)
        y_val_list.append(y_val)


    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_final, y_test_final


def split_data_regression(df: pd.DataFrame):
    # triage par date pour faire construire une serie temporelle
    df = df.sort_values(['date_mois', 'article', 'site_article'])

    # Séparation entre des données utilisés pour l'entraînement et le test
    dates = sorted(df['date_mois'].unique())
    dates_length = len(dates)

    train_edge = dates[int(dates_length * 0.85)]

    train_mask = df['date_mois'] <= train_edge
    test_mask = df['date_mois'] > train_edge

    # Séparation des variables caractéristique et la variable cible
    x_temp = df.drop(columns=['conso_ce_mois', 'quantite', 'date_mois'])
    y_temp = df['quantite']

    # Séparation entre les donnée d'entraînement et validation pour la phase d'entraînement et évaluation du model
    # Séparation des données de test final
    x_train_final, y_train_final = x_temp[train_mask], y_temp[train_mask]
    x_test_final, y_test_final = x_temp[test_mask], y_temp[test_mask]

    # Séparation des données d'entraînement et de test par le time split séries
    time_split = TimeSeriesSplit(n_splits=5)

    x_train_list = []
    y_train_list = []
    x_val_list = []
    y_val_list = []

    for train_index, val_index in time_split.split(x_train_final):
        x_train, y_train = x_train_final.iloc[train_index], y_train_final.iloc[train_index]
        x_val, y_val = x_train_final.iloc[val_index], y_train_final.iloc[val_index]

        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_val_list.append(x_val)
        y_val_list.append(y_val)


    return x_train_list, y_train_list, x_val_list, y_val_list, x_test_final, y_test_final"""    
        

"""def split_data_regression(df: pd.DataFrame):

    dates = sorted(df['date_mois'].unique())
    dates_length = len(dates)

    train_edge = dates[int(dates_length * 0.7)]
    val_edge = dates[int(dates_length * 0.85)]

    train_mask = df['date_mois'] <= train_edge
    val_mask = (df['date_mois'] > train_edge) & (df['date_mois'] <= val_edge)
    test_mask = df['date_mois'] > val_edge

    x = df.drop(columns=['conso_ce_mois', 'quantite', 'date_mois'])
    y = df['quantite']


    return x[train_mask], x[val_mask], x[test_mask], y[train_mask], y[val_mask], y[test_mask]"""