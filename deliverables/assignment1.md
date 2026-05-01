# Assignment 1 - NBA Points Prediction Project

## 1. Informations sur le Dataset
* **Nom du dataset :** NBA League Game Logs (Multi-Seasons)
* **Source :** API officielle de la NBA via la bibliothèque Python `nba_api`.
* **Lien :** [https://github.com/swar/nba_api](https://github.com/swar/nba_api)
* **Type de données :** Données tabulaires structurées (58 459 lignes de statistiques de performance par match).

## 2. Définition du Problème ML
* **Variable cible (Target) :** `PTS` (Nombre de points marqués par un joueur lors d'un match).
* **Type de ML :** Apprentissage supervisé - **Régression**.
* **Modèle utilisé :** Random Forest Regressor (avec 100 estimateurs).
* **Features principales (Variables prédictives) :**
    * **Volume :** `MIN` (Minutes), `FGA` (Tentatives de tirs), `FG3A` (Tentatives à 3 pts), `FTA` (Tentatives de lancers francs).
    * **Activité :** `AST` (Passes), `REB` (Rebonds), `STL` (Interceptions), `BLK` (Contres).
    * **Impact & Discipline :** `PLUS_MINUS` (Impact score), `TOV` (Pertes de balles), `PF` (Fautes).
    * *Note : Les variables de réussite directe (FGM, FTM) ont été exclues pour éviter toute triche du modèle (Data Leakage).*

## 3. Qualité des Données et Checks
* **Données manquantes :** **0%** de valeurs manquantes (nettoyage effectué via `src/data.py`).
* **Prétraitement :** Application d'un **StandardScaler** pour normaliser les features avant l'entraînement du modèle Random Forest.
* **Détection des Outliers :** * Méthode : **Interquartile Range (IQR)**.
    * Décision : Les performances extrêmes (scores > 50 pts) sont conservées car elles sont représentatives du talent réel en NBA et essentielles à la robustesse du modèle.
* **Feature Drift & Distribution :** Vérification par histogrammes. Les variables de volume (`MIN`, `FGA`) suivent des distributions cohérentes avec la réalité physique des matchs de 48 minutes.

## 4. Limites éventuelles
* Le dataset se concentre sur les statistiques "box-score". Il ne prend pas en compte les facteurs externes comme les blessures, le repos (back-to-back), ou les schémas tactiques défensifs spécifiques qui influencent fortement le score final.