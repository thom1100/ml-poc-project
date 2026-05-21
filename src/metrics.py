from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np

def evaluate_model(y_true, y_pred, model_name='Modèle'):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # On garde tes prints pour le débug console
    print(f"\n--- 📊 Scores : {model_name} ---")
    print(f"MAE  : {mae:.2f}")
    print(f"R²   : {r2:.4f}")
    print(f"RMSE : {rmse:.2f}")
    
    # ATTENTION : Ton main.py attend probablement des clés en MAJUSCULES (MAE, R2, RMSE)
    # On retourne un dictionnaire compatible avec ce que le reste du projet attend
    return {'MAE': mae, 'R2': r2, 'RMSE': rmse}

# --- ALIAS POUR LE MAIN.PY ---
# On définit compute_metrics comme étant la même chose qu'evaluate_model
def compute_metrics(y_true, y_pred):
    return evaluate_model(y_true, y_pred)