from __future__ import annotations
import importlib.util
import os
import subprocess
import sys
import pickle
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# --- CHARGEMENT DES MODULES DYNAMIQUE ---
def _load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module `{module_name}` from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Chargement de la config projet
config = _load_module("project_config", PROJECT_ROOT / "src" / "config.py")
sys.modules["config"] = config
load_dotenv(PROJECT_ROOT / ".env")

# Import des fonctions métier depuis src/
data_module = _load_module("project_data", PROJECT_ROOT / "src" / "data.py")
metrics_module = _load_module("project_metrics", PROJECT_ROOT / "src" / "metrics.py")
results_module = _load_module("project_results", PROJECT_ROOT / "src" / "results.py")
model_io_module = _load_module("project_model_io", PROJECT_ROOT / "src" / "model_io.py")

load_dataset_split = data_module.load_dataset_split
compute_metrics = metrics_module.compute_metrics
write_metrics = results_module.write_metrics
load_model = model_io_module.load_model

def _evaluate_selected_models(X_test: Any, y_test: Any) -> list[dict[str, object]]:
    rows = []
    models_dir = PROJECT_ROOT / "models"
    
    # On charge l'encoder commun
    encoder_path = models_dir / "encoder.pkl"
    if not encoder_path.exists():
        raise FileNotFoundError(f"L'encodeur est introuvable à {encoder_path}")
        
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    # Modèles définis dans config.py
    target_models = ["baseline", "xgboost", "optuna"]

    for model_key in target_models:
        if model_key not in config.MODELS:
            continue
            
        model_config = config.MODELS[model_key]
        m_path = PROJECT_ROOT / model_config["path"]
        
        if not m_path.exists():
            print(f"⏩ Fichier {m_path.name} introuvable, skip.")
            continue
        
        print(f"🧐 Évaluation en cours : {model_key}...")
        try:
            # Chargement du modèle
            model = load_model(m_path)
            
            # Transformation unifiée pour TOUS les modèles (plus de PCA spécifique)
            X_test_input = encoder.transform(X_test)
            
            # Prédiction
            y_pred = model.predict(X_test_input)
            metrics = compute_metrics(y_test, y_pred)

            # Stockage des résultats
            row = {
                "model_key": model_key,
                "model_name": model_config.get("name", model_key),
                "model_path": str(model_config["path"]),
            }
            row.update({k: float(v) for k, v in metrics.items()})
            rows.append(row)
            print(f"✅ {model_key} évalué avec succès.")

        except Exception as e:
            print(f"❌ Erreur sur {model_key} : {e}")
            
    return rows

def _launch_streamlit() -> None:
    app_path = PROJECT_ROOT / "src" / "app.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    print(f"\n🚀 Lancement du Dashboard sur http://{config.STREAMLIT_HOST}:{config.STREAMLIT_PORT}")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.address", config.STREAMLIT_HOST,
            "--server.port", str(config.STREAMLIT_PORT),
        ], check=True, cwd=PROJECT_ROOT, env=env)
    except KeyboardInterrupt:
        print("\n👋 Arrêt du dashboard.")

def main() -> None:
    print("--- Pipeline NBA Predictor ---")
    
    # 1. Chargement Data & Évaluation
    print("📊 Chargement des données et calcul des métriques...")
    try:
        # Récupération du split de test
        _, X_test, _, y_test = load_dataset_split()
        
        # Évaluation des modèles
        metrics_rows = _evaluate_selected_models(X_test, y_test)
        
        if metrics_rows:
            # Sauvegarde dans results/model_metrics.csv
            metrics_df = write_metrics(metrics_rows)
            print("\n📊 TABLEAU RÉCAPITULATIF DES SCORES :")
            print("--------------------------------------")
            print(metrics_df.to_string(index=False))
            print("--------------------------------------")
        else:
            print("⚠️ Aucun modèle n'a pu être évalué.")
            
    except Exception as e:
        print(f"❌ Erreur lors de la phase d'évaluation : {e}")

    # 2. Lancement Streamlit
    _launch_streamlit()

if __name__ == "__main__":
    main()