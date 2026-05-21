import pandas as pd
import numpy as np
import os
import time
from nba_api.stats.endpoints import playergamelogs
from sklearn.model_selection import train_test_split

def load_dataset_split():
    csv_path = 'data/nba_raw_data.csv'
    os.makedirs('data', exist_ok=True)

    # 1. RÉCUPÉRATION FILTRÉE (Historique + Playoffs Actuels)
    print("🚀 Récupération des données NBA (Historique + Playoffs 2026)...")
    
    # Configuration des saisons
    # Pour les saisons passées, on prend la Regular Season. 
    # Pour 2025-26, on prend Regular Season + Playoffs pour avoir les matchs d'hier.
    seasons_config = [
        ('2022-23', 'Regular Season'),
        ('2023-24', 'Regular Season'),
        ('2024-25', 'Regular Season'),
        ('2025-26', 'Regular Season'),
        ('2025-26', 'Playoffs') # <--- Ajout crucial pour les matchs actuels
    ]
    
    all_games = []
    for s_year, s_type in seasons_config:
        try:
            log = playergamelogs.PlayerGameLogs(
                season_nullable=s_year, 
                season_type_nullable=s_type
            )
            data = log.get_data_frames()[0]
            if not data.empty:
                all_games.append(data)
                print(f"✅ {s_year} ({s_type}) récupérée.")
            time.sleep(1.5) 
        except Exception as e:
            print(f"❌ Erreur sur {s_year} {s_type}: {e}")
    
    if not all_games:
        print("⚠️ Aucune donnée récupérée. Vérifiez votre connexion ou l'API NBA.")
        return None

    full_df = pd.concat(all_games, ignore_index=True)

    # 2. NETTOYAGE PRÉLIMINAIRE
    full_df['GAME_DATE'] = pd.to_datetime(full_df['GAME_DATE'])
    # On trie par date pour que les calculs de moyenne mobile (rolling) soient chronologiques
    full_df = full_df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    # 3. CALCUL DES FEATURES
    print("📈 Calcul des indicateurs de performance...")
    
    # Moyennes mobiles (5 matchs) - Reflète le rythme du joueur
    stats_to_avg = ['PTS', 'MIN', 'FGA', 'FG3A', 'FTA', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']
    for stat in stats_to_avg:
        full_df[f'AVG_{stat}'] = full_df.groupby('PLAYER_ID')[stat].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )

    # Forme récente (3 matchs) - Très important en Playoffs
    full_df['SHORT_FORM_PTS'] = full_df.groupby('PLAYER_ID')['PTS'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
    )

    # Fatigue & Repos (S'adapte aux séries de Playoffs)
    full_df['DAYS_REST'] = full_df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days.fillna(4)
    full_df['IS_B2B'] = (full_df['DAYS_REST'] <= 1).astype(int)
    full_df['RETURNING_FROM_ABSENCE'] = (full_df['DAYS_REST'] > 7).astype(int)
    full_df['IS_STARTER'] = (full_df['AVG_MIN'] >= 25).astype(int)

    # 4. OPPOSITION & CONTEXTE
    full_df['OPPONENT'] = full_df['MATCHUP'].apply(lambda x: str(x).split(' ')[-1] if pd.notna(x) else "UNK")
    
    # Force défensive de l'adversaire (Points encaissés en moyenne)
    def_stats = full_df.groupby(['TEAM_ABBREVIATION', 'SEASON_YEAR'])['PTS'].mean().reset_index()
    def_stats.columns = ['OPPONENT', 'SEASON_YEAR', 'OPP_AVG_PTS_ALLOWED']
    full_df = full_df.merge(def_stats, on=['OPPONENT', 'SEASON_YEAR'], how='left')
    full_df['OPP_AVG_PTS_ALLOWED'] = full_df['OPP_AVG_PTS_ALLOWED'].fillna(112.0)

    full_df['LOCATION'] = full_df['MATCHUP'].apply(lambda x: 'Away' if '@' in str(x) else 'Home')
    full_df['DAY_OF_WEEK'] = full_df['GAME_DATE'].dt.day_name()

    # 5. SAUVEGARDE DU CSV PROPRE
    full_df.to_csv(csv_path, index=False)
    
    # 6. FILTRAGE FINAL POUR L'ENTRAÎNEMENT DU MODÈLE
    features = [
        'AVG_PTS', 'SHORT_FORM_PTS', 'AVG_MIN', 'AVG_FGA', 'AVG_FG3A', 'AVG_FTA', 
        'AVG_REB', 'AVG_AST', 'AVG_STL', 'AVG_BLK', 'AVG_TOV', 'AVG_PF', 
        'AVG_PLUS_MINUS', 'DAYS_REST', 'IS_B2B', 'RETURNING_FROM_ABSENCE', 
        'IS_STARTER', 'OPP_AVG_PTS_ALLOWED', 'TEAM_ABBREVIATION', 'OPPONENT', 
        'LOCATION', 'DAY_OF_WEEK'
    ]
    
    # Filtrage des joueurs avec un temps de jeu significatif
    df_final = full_df[(full_df['AVG_MIN'] >= 10) & (full_df['AVG_PTS'] > 2)].dropna(subset=features + ['PTS'])
    
    print(f"✅ Dataset finalisé : {len(df_final)} lignes prêtes pour l'entraînement.")
    return train_test_split(df_final[features], df_final['PTS'], test_size=0.2, random_state=42)