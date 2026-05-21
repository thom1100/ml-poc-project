import sys
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ajout du chemin pour trouver le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import load_dataset_split
from src.features import preprocess_features

def train():
    print("⏳ Chargement des données brutes (incluant Repos, Défense, Titulaire et AVG_PTS)...")
    X_train, X_test, y_train, y_test = load_dataset_split()

    print("⚙️ Application du Feature Engineering (Skrub sans PCA pour la forêt)...")
    # On met use_pca=False car les arbres de décision sont plus performants sur les données brutes
    X_train_processed, X_test_processed, _, encoder, _ = preprocess_features(X_train, X_test, use_pca=False)

    # --- ÉTAPE GRID SEARCH ---
    print("🔍 Recherche des meilleurs hyperparamètres (Grid Search)...")
    
    param_grid = {
        'n_estimators': [100, 150], 
        'max_depth': [10, 15], 
        'min_samples_split': [5, 10],
        'bootstrap': [True]
    }

    rf = RandomForestRegressor(random_state=42)

    # cv=2 pour la rapidité pendant tes tests, tu peux mettre cv=3 pour plus de robustesse
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=2, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1, 
        verbose=1
    )

    grid_search.fit(X_train_processed, y_train)
    
    model = grid_search.best_estimator_
    print(f"🏆 Meilleurs paramètres trouvés : {grid_search.best_params_}")

    # --- CALCUL DES MÉTRIQUES ---
    print("📊 Calcul des performances détaillées...")
    y_pred = model.predict(X_test_processed)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Sauvegarde
    print("💾 Sauvegarde du modèle et de l'encodeur...")
    os.makedirs("models", exist_ok=True)
    
    # On sauvegarde le modèle et l'encodeur (indispensable pour Streamlit sans PCA)
    with open("models/model_rf_classic.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/encoder_rf.pkl", "wb") as f:
        pickle.dump(encoder, f)

    # --- AFFICHAGE FINAL ---
    print("\n" + "="*40)
    print("✨ ENTRAÎNEMENT RF CLASSIQUE TERMINÉ ✨")
    print("="*40)
    print(f"📈 R² Score (Précision) : {r2:.4f}")
    print(f"🎯 MAE (Erreur moyenne) : {mae:.2f} points")
    print(f"📉 RMSE (Écart-type)    : {rmse:.2f} points")
    print("="*40)

if __name__ == "__main__":
    train()