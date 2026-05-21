import os
import pickle
import optuna
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Ajout du chemin vers le dossier racine pour trouver 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset_split
from src.features import preprocess_features
from src.metrics import evaluate_model

# 1. Chargement des données
X_train, X_test, y_train, y_test = load_dataset_split()

print("🚀 Préparation features pour RF (Optuna)...")
# Pour le Random Forest, on n'utilise pas la PCA (use_pca=False) 
# afin de garder l'interprétabilité des colonnes
X_train_proc, X_test_proc, _, encoder, _ = preprocess_features(X_train, X_test, use_pca=False)

# 2. Définition de la fonction objectif pour l'optimisation
def objective(trial):
    # Hyperparamètres à tester
    n_estimators = trial.suggest_int('n_estimators', 50, 100)
    max_depth = trial.suggest_int('max_depth', 5, 12)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    # Création et entraînement du modèle temporaire
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split,
        n_jobs=-1, 
        random_state=42
    )
    model.fit(X_train_proc, y_train)
    
    # Calcul de l'erreur (on veut minimiser la MAE)
    preds = model.predict(X_test_proc)
    mae = mean_absolute_error(y_test, preds)
    
    return mae

# 3. Lancement de l'étude Optuna
print("🚀 Optimisation flash (3 essais en cours)...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=3) 

# 4. Entraînement du modèle final avec les meilleurs paramètres
print(f"🏆 Meilleurs paramètres : {study.best_params}")
final_model = RandomForestRegressor(
    **study.best_params, 
    n_jobs=-1, 
    random_state=42
)
final_model.fit(X_train_proc, y_train)

# 5. Évaluation finale
y_pred = final_model.predict(X_test_proc)
evaluate_model(y_test, y_pred, 'Random Forest (Optuna)')

# 6. Sauvegarde du modèle et de l'encodeur synchronisé
os.makedirs('models', exist_ok=True)

with open('models/model_optuna.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# On écrase l'encodeur principal pour garantir la cohérence dans main.py
with open('models/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("✅ Optuna synchronisé et modèle sauvegardé !")