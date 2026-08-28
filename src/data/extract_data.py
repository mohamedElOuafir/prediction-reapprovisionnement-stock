import datetime
import boto3
import os
from dotenv import load_dotenv
from src.data.db_connector import getConnection
import pandas as pd

load_dotenv()

# récuperration de la connexion au base de données
db = getConnection()


# requête SQL pour la récuperration des données brute pour le dataset
query_sql = """
WITH CalendrierMois AS (
    SELECT DISTINCT 
        DATEFROMPARTS(YEAR(IPTDAT_0), MONTH(IPTDAT_0), 1) AS date_mois
    FROM SEED.STOJOU
),
conso_mensuelle AS (
    SELECT
        stj.ITMREF_0,
        stj.STOFCY_0,
        DATEFROMPARTS(YEAR(stj.IPTDAT_0), MONTH(stj.IPTDAT_0), 1) AS date_mois,
        ABS(SUM(stj.QTYSTU_0)) AS qty
    FROM SEED.STOJOU stj
    WHERE stj.TRSTYP_0 = 4 -- Livraisons clients uniquement
    GROUP BY 
        stj.ITMREF_0, 
        stj.STOFCY_0, 
        DATEFROMPARTS(YEAR(stj.IPTDAT_0), MONTH(stj.IPTDAT_0), 1)
),
articles_sites AS (
    SELECT DISTINCT
        itf.ITMREF_0, 
        itf.STOFCY_0, 
        itm.TCLCOD_0, 
        itm.STU_0,
        its.BASPRI_0, 
        itf.ABCCLS_0, 
        itm.ITMSTA_0, 
        itf.SAFSTO_0,
        itm.OFS_0,
        itf.REOTSD_0, 
        itf.REOPOL_0, 
        itf.REOMGTCOD_0,
        DATEFROMPARTS(YEAR(itf.CREDAT_0), MONTH(itf.CREDAT_0), 1) AS date_creation_article
    FROM SEED.ITMFACILIT itf -- Partir du catalogue Article-Site
    JOIN SEED.ITMMASTER itm 
      ON itf.ITMREF_0 = itm.ITMREF_0
    LEFT JOIN SEED.ITMSALES its -- LEFT JOIN pour ne pas perdre d'articles
      ON itf.ITMREF_0 = its.ITMREF_0
    WHERE
        (
		itf.ITMREF_0 IN (
			SELECT DISTINCT ITMREF_0 
			FROM SEED.STOJOU 
			WHERE TRSTYP_0 = 4
		)
		AND(
			(itm.PURFLG_0 = 2 AND itm.DLVFLG_0 = 2 AND itm.MFGFLG_0 = 1) -- Négoce / Acheté
			OR 
			(itm.PURFLG_0 = 1 AND itm.DLVFLG_0 = 2 AND itm.MFGFLG_0 = 2) -- Produit Fini / Fabriqué
        )
	)
),
grille_complete AS (
    SELECT 
        ars.*, 
        cmo.date_mois
    FROM articles_sites ars
    CROSS JOIN CalendrierMois cmo
)
SELECT
    g.ITMREF_0 AS article, 
    g.STOFCY_0 AS site_article, 
    g.TCLCOD_0 AS categorie, 
    g.STU_0 AS stock_unit, 
    COALESCE(g.BASPRI_0, 0) AS prix_unitaire,
    g.ABCCLS_0 AS ABC_class, 
    g.ITMSTA_0 AS article_statut, 
    g.SAFSTO_0 AS stock_securite, 
    g.OFS_0 AS delai_reapprovisionnement,
    g.REOTSD_0 AS seuil_reapprovisionnement, 
    g.REOPOL_0 AS politique_reapprovisionnement, 
    g.REOMGTCOD_0 AS mode_reapprovisionnement,
    g.date_mois,
    COALESCE(cm.qty, 0) AS quantite
FROM grille_complete g
LEFT JOIN conso_mensuelle cm 
  ON cm.ITMREF_0 = g.ITMREF_0 
 AND cm.STOFCY_0 = g.STOFCY_0 
 AND cm.date_mois = g.date_mois
WHERE g.date_mois >= g.date_creation_article 
ORDER BY g.ITMREF_0, g.STOFCY_0, g.date_mois;"""


S3_bucket = os.getenv("S3_DATA_BUCKET")

date_extraction = datetime.datetime.today().strftime("%Y-%m")
date_mois = datetime.datetime.today().strftime("%Y/%m")
fichier_data = f"data_brut_{date_extraction}.csv"


# Fonction pour l'extraction des données depuis la base de données et l'uploader vers amazon S3
def extract_and_upload():
    df = pd.read_sql(
        sql=query_sql,
        con=db,
    )

    df.to_csv(fichier_data, mode='w', encoding='utf-8', index=False)

    s3_service = boto3.client("s3")
    s3_service.upload_file(
        fichier_data,
        S3_bucket,
        f"raw/{date_mois}/{fichier_data}"
    )

    print(f"Fichier de données <<{fichier_data}>> uploadé à S3 avec succés!")



if __name__ == "__main__":
    extract_and_upload()