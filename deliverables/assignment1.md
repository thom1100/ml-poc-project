# Assignment 1 - NBA Points Prediction Project (V2 - Optimized)

## 1. Informations sur le Dataset
- **Nom du dataset :** NBA League Game Logs (Saisons 2022-23 à 2025-26, incluant Playoffs).
- **Source :** API officielle de la NBA via la bibliothèque Python `nba_api` (endpoint `playergamelogs`).
- **Volume :** Environ 126 700 lignes de statistiques individuelles. L’historique sur plusieurs saisons a été choisi pour stabiliser les moyennes mobiles en début de saison et capturer une diversité de profils de joueurs.
- **Type de données :** Séries temporelles transformées en données tabulaires (Lag Features) pour permettre une prédiction "avant-match".

## 2. Définition du Problème ML
- **Variable cible (Target) :** `PTS` (Nombre de points marqués par le joueur lors du match à prédire).
- **Type de ML :** Apprentissage supervisé – **Régression**.
- **Modèles comparés :**
  - Baseline Model (Moving Average)
  - XGBoost Regressor
  - Random Forest optimisé par **Optuna** (recherche bayésienne d’hyperparamètres)
- **Features principales (Variables prédictives) :**
  - **Moyennes mobiles (Lag 5) :** Calcul des performances récentes (`AVG_MIN`, `AVG_FGA`, `AVG_REB`, `AVG_PTS`, etc.) sur les 5 derniers matchs. L’utilisation de `.shift(1)` garantit qu’aucune donnée du match actuel n’est utilisée pour la prédiction (**absence de data leakage**).
  - **Forme récente (Lag 3) :** `SHORT_FORM_PTS` – moyenne mobile sur 3 matchs, capturant les "hot streaks".
  - **Indicateurs de fatigue :** `DAYS_REST` (jours exacts depuis le dernier match) et `IS_B2B` (back‑to‑back, repos <= 1 jour).
  - **Retour d’absence :** `RETURNING_FROM_ABSENCE` (repos > 7 jours).
  - **Difficulté défensive adverse :** `OPP_AVG_PTS_ALLOWED` – moyenne des points encaissés par l’équipe adverse sur la saison (target encoding).
  - **Variables catégorielles :** `TEAM_ABBREVIATION`, `OPPONENT`, `LOCATION` (Home/Away), `DAY_OF_WEEK`.

## 3. Qualité des Données et Checks
- **Données manquantes :** Traitées via `dropna()` après le calcul des moyennes mobiles (les premières lignes de chaque joueur sont exclues si l’historique est insuffisant).
- **Prétraitement & Feature Engineering :**
  - **Encodage :** Utilisation de `TableVectorizer` (bibliothèque `skrub`) pour un target‑encoding automatique des variables catégorielles et un encodage cyclique des dates.
  - **Normalisation :** Application d’un `StandardScaler` pour harmoniser l’échelle des features numériques (indispensable pour la régression linéaire et la stabilité du gradient boosting).
  - **Réduction de dimension (optionnelle) :** Une PCA peut être activée (`use_pca=True`) pour réduire le nombre de variables, mais elle n’est pas utilisée par défaut (les modèles d’arbres n’en ont pas besoin).
- **Optimisation :** Recherche d’hyperparamètres via **Optuna** pour le Random Forest (nombre d’arbres, profondeur maximale, échantillonnage des features). XGBoost utilise ses paramètres par défaut, déjà performants sur ce type de données.

## 4. Résultats et Performance
Les métriques (MAE, R², RMSE) sont calculées sur un jeu de test (20% des données) et sauvegardées dans `results/model_metrics.csv`.  
Exemple de performances (après optimisation) :

| Modèle                         | MAE    | R²     | RMSE   |
|--------------------------------|--------|--------|--------|
| Baseline (Moving Average)      | 4.95   | 0.484  | 6.36   |
| XGBoost Regressor              | 4.75   | 0.526  | 6.10   |
| Random Forest (Optuna)         | 4.82   | 0.514  | 6.18   |

Le **XGBoost** offre le meilleur compromis erreur / interprétabilité et a été retenu pour la production.

## 5. Limites et Évolutions
- **Points forts :** Le modèle capture la forme récente, la fatigue (back‑to‑back), l’avantage du terrain et la force défensive adverse. Il est déployé dans une application interactive Streamlit.
- **Limites actuelles :** 
  - Ne prend pas en compte les rapports de blessures de dernière minute (injury reports) ni les changements tactiques soudains (coaching).
  - Les prédictions en production nécessitent l’application du même scaler que celui utilisé à l’entraînement (sinon biais possible).
- **Perspectives :** 
  - Intégrer des données plus fines (usage rate, statistiques de matchup individuel).
  - Ajouter un mécanisme de mise à jour en temps réel des moyennes mobiles après chaque match.