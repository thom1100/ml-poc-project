import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skrub import TableVectorizer

def enrich_nba_features(df):
    """
    Crée de nouvelles variables pour aider le modèle à mieux comprendre le contexte.
    S'applique AVANT le split Train/Test.
    """
    df = df.copy()
    
    # 1. Moyennes Mobiles Pondérées (Donne plus de poids au match précédent)
    # Si ton dataset a déjà une colonne 'AVG_PTS', on peut ajouter la 'FORME_RECENTE'
    if 'PTS' in df.columns:
        # On calcule la moyenne des 3 derniers matchs pour capter les "hot streaks"
        df['FORME_RECENTE'] = df.groupby('PLAYER_ID')['PTS'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean().shift(1)
        )
    
    # 2. Fatigue : Indicateur de Back-to-Back (B2B)
    # On vérifie si le match précédent était il y a moins de 2 jours
    if 'GAME_DATE' in df.columns:
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        df['DAYS_SINCE_LAST_GAME'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days
        df['IS_B2B'] = (df['DAYS_SINCE_LAST_GAME'] <= 1).astype(int)
    
    # 3. Ratio de points à domicile vs extérieur
    # Souvent, certains joueurs surperforment chez eux
    df['HOME_AWAY_FACTOR'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)

    # On remplace les NaN créés par le shift(1) par la moyenne globale
    df = df.fillna(0)
    return df

def preprocess_features(X_train, X_test, use_pca=False): 
    """
    Transforme les données encodées.
    Note: On passe use_pca=False par défaut pour la stabilité.
    """
    
    # 1. ENCODAGE (Skrub)
    # TableVectorizer est excellent car il gère les dates et les catégories automatiquement
    encoder = TableVectorizer()
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    
    # 2. STANDARD SCALING (Indispensable pour la PCA et la Régression Linéaire)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)
    
    # 3. PCA (Conditionnelle)
    if use_pca:
        # On garde 95% de la variance
        pca = PCA(n_components=0.95)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
        print(f"📉 PCA effectuée : {X_train_final.shape[1]} variables.")
    else:
        # Pour les modèles d'arbres, on garde les noms de colonnes (plus interprétable)
        X_train_final = X_train_encoded
        X_test_final = X_test_encoded
        pca = None
        print(f"✅ Features brutes : {X_train_final.shape[1]} variables conservées.")
    
    return X_train_final, X_test_final, pca, encoder, scaler