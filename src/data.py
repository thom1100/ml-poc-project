import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from sklearn.model_selection import train_test_split
import time

def load_dataset_split():
    seasons = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
    all_games = []

    print("🚀 Récupération massive des données de la ligue...")
    for season in seasons:
        try:
            # LeagueGameLog récupère TOUS les matchs de TOUS les joueurs d'une saison d'un coup
            log = leaguegamelog.LeagueGameLog(
                season=season, 
                player_or_team_abbreviation='P' # 'P' pour Players
            )
            df = log.get_data_frames()[0]
            all_games.append(df)
            print(f"✅ Saison {season} récupérée.")
            time.sleep(1) # Sécurité pour l'API
        except Exception as e:
            print(f"❌ Erreur saison {season}: {e}")

    full_df = pd.concat(all_games, ignore_index=True)

    # --- COMPLEXIFICATION DES FEATURES ---
    # On ajoute plein de métriques pour donner du "grain" au modèle
    features = [
        'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 
        'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS'
    ]
    target = 'PTS'

    # --- DATA CLEANING ---
    # 1. Conversion des minutes en float
    full_df['MIN'] = full_df['MIN'].astype(float)
    
    # 2. On ne garde que les joueurs qui ont vraiment joué (évite le bruit)
    full_df = full_df[full_df['MIN'] >= 5]
    
    # 3. Suppression des lignes avec des valeurs manquantes
    full_df = full_df.dropna(subset=features + [target])

    X = full_df[features]
    y = full_df[target]

    print(f"📊 Dataset final : {len(full_df)} matchs et {len(features)} colonnes.")
    return train_test_split(X, y, test_size=0.2, random_state=42)