import os
import pickle
import sys
from pathlib import Path
from sklearn.linear_model import LinearRegression

# Ajout du path pour src
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.data import load_dataset_split
from src.features import preprocess_features

X_train, X_test, y_train, y_test = load_dataset_split()

print("📊 Entraînement Baseline (SANS PCA pour cohérence)...")
# FORCE use_pca=False ici
X_train_proc, X_test_proc, _, encoder, _ = preprocess_features(X_train, X_test, use_pca=False)

model = LinearRegression()
model.fit(X_train_proc, y_train)

# Sauvegarde
os.makedirs('models', exist_ok=True)
with open('models/model_baseline.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("✅ Baseline ré-entraînée sur les 93 features !")