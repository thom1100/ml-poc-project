from pathlib import Path

# --- CHEMINS DE BASE ---
# On remonte d'un niveau depuis src/ pour arriver à la racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dossiers principaux
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

# --- FICHIERS ---
ENV_FILE = PROJECT_ROOT / ".env"
# La variable qui manquait :
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"

# --- CONFIGURATION DES MODÈLES ---
# Utilisé par scripts/main.py pour l'évaluation
MODELS = {
    "baseline": {
        "name": "Baseline Model (Moving Average)",
        "path": "models/model_baseline.pkl"
    },
    "xgboost": {
        "name": "XGBoost Regressor",
        "path": "models/model_xgboost.pkl"
    },
    "optuna": {
        "name": "Random Forest (Optuna Optimized)",
        "path": "models/model_optuna.pkl"
    }
}

# --- PARAMÈTRES INTERFACE (STREAMLIT) ---
STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501
APP_ENTRYPOINT = SRC_DIR / "app.py"