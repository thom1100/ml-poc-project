import sys
from pathlib import Path
# Ajoute le dossier racine du projet au chemin de recherche de Python
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset_split
import os
import pickle
import xgboost as xgb
from src.data import load_dataset_split
from src.features import preprocess_features  # <-- On importe ton préprocesseur
from src.metrics import evaluate_model

# 1. Charger les données brutes
X_train, X_test, y_train, y_test = load_dataset_split()

print(f"🚀 Préparation des features (Sans PCA pour XGBoost)...")
# 2. Prétraitement intelligent
# On utilise use_pca=False pour garder les colonnes AVG_PTS, IS_STARTER, etc. lisibles
X_train_proc, X_test_proc, _, encoder, _ = preprocess_features(X_train, X_test, use_pca=False)

print(f"🚀 Entraînement de l'artillerie lourde : XGBoost...")

# 3. Configuration du modèle
model = xgb.XGBRegressor(
    n_estimators=300,        # On augmente un peu car XGBoost est robuste
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)

# 4. Apprentissage sur les données encodées
model.fit(X_train_proc, y_train)

# 5. Évaluation
y_pred = model.predict(X_test_proc)
evaluate_model(y_test, y_pred, 'XGBoost + Features Complètes')

# 6. Sauvegarde du modèle ET de l'encodeur
# C'est CRUCIAL pour le Streamlit : on doit sauvegarder l'encodeur pour 
# transformer les noms d'équipes de la même façon plus tard.
os.makedirs('models', exist_ok=True)

with open('models/model_xgboost.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("\n✅ Modèle et Encodeur sauvegardés !")