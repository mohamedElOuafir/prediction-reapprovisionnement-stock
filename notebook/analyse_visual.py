import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay


script_folder = os.path.dirname(os.path.abspath(__file__))
target_folder = os.path.normpath(os.path.join(script_folder, '..', 'data/processed'))

df = pd.read_csv(os.path.join(target_folder, 'conso_mensuelle_clean.csv'))

categorical_columns = [
    'article',
    'site_article',
    'categorie',
    'stock_unit',
    'politique_reapprovisionnement',
    'conso_ce_mois'
]

numerical_columns = [
    'prix_unitaire',
    'stock_securite',
    'delai_reapprovisionnement',
    'seuil_reapprovisionnement',
    'mode_reapprovisionnement',
    'month',
    'conso_M_precedent_1',
    'conso_M_precedent_2',
    'conso_M_precedent_3',
    'conso_meme_mois_annee_dernier',
    'year',
    'moyenne_conso_3mois',
    'ecart_type_6mois',
    'mois_depuis_dernier_conso'
]

def stats():
    print(f"categorical columns info: \n\n")
    for col in categorical_columns:
        print(f"value counts for {col}:\n{df[col].value_counts()}\n\n")

    print(f"numerical columns info: \n\n")
    for col in numerical_columns:
        print(f"value counts for {col}:\n{df[col].value_counts()}\n\n")



def plot_confusion_matrix(cms, models,classes):

    fig, axes = plt.subplots(3, 2)
    axes = axes.flatten()
    i = 0
    for name, model in models.items():
        disp = ConfusionMatrixDisplay(confusion_matrix=cms[i], display_labels=classes)
        disp.plot(cmap="Blues", colorbar=False, ax=axes[i])
        axes[i].set_title(f"Model {name}")
        i += 1

    plt.tight_layout()
    plt.show()


def plot_corr_matrix(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 4)
    axes = axes.flatten()

    x = df[(df['article'] == "COM002") & (df['site_article'] == "AO012")] 
    for i, col in enumerate(numerical_columns):
        axes[i].boxplot(x[col])
        axes[i].set_title(col)

    plt.show()

    


    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    df = df.drop(columns=['date_mois'])
    df_class = df.copy()
    df = df.drop(columns=['conso_ce_mois'])
    df_class = df_class.drop(columns=['quantite'])


    corr_reg = df.corr()
    corr_cls = df_class.corr()

    plt.figure()
    sns.heatmap(corr_reg, annot=True, fmt=".2f", cmap="coolwarm")
    plt.show()

    sns.heatmap(corr_reg['quantite'].to_frame(), annot=True, cmap="coolwarm")
    plt.show()

    sns.heatmap(corr_cls, annot=True, fmt=".2f", cmap="coolwarm")
    plt.show()

    sns.heatmap(corr_cls['conso_ce_mois'].to_frame(), annot=True, cmap="coolwarm")
    plt.show()


if __name__ == "__main__":
    plot_corr_matrix(df)





