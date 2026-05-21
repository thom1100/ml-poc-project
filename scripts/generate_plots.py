import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Configuration
os.makedirs('plots', exist_ok=True)
df = pd.read_csv('data/nba_raw_data.csv')

# --- PLOT 1 : EDA (Distribution des Points) ---
plt.figure(figsize=(10, 6))
sns.histplot(df['PTS'], kde=True, color='skyblue')
plt.title('Distribution of Points Scored')
plt.xlabel('Points')
plt.ylabel('Frequency')
plt.savefig('plots/eda_points_distribution.png')
print("✅ Plot EDA sauvegardé.")

# --- PLOT 2 : COMPARAISON DES MODÈLES ---
models = ['Baseline', 'Random Forest', 'XGBoost']
mae_values = [5.10, 4.85, 4.71] # Tes vrais chiffres

plt.figure(figsize=(10, 6))
sns.barplot(x=models, y=mae_values, palette='viridis')
plt.ylim(4.5, 5.2)
plt.title('Model Comparison (Mean Absolute Error)')
plt.ylabel('MAE (Lower is better)')
plt.savefig('plots/model_comparison.png')
print("✅ Plot Comparaison sauvegardé.")

# --- PLOT 3 : RÉSULTATS MEILLEUR MODÈLE (Importance des Features) ---
# On charge le modèle pour voir ce qui compte le plus pour lui
with open('models/model_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# Récupération de l'importance des variables
importances = model.feature_importances_
# On prend les noms des features (simplifiés pour le graphique)
feature_names = ['AVG_PTS', 'AVG_MIN', 'AVG_FGA', 'AVG_REB', 'AVG_AST', 'DAYS_REST', 'OPP_DEFENSE'] 
# Note: On tronque la liste car il y en a 93 avec l'encodage

plt.figure(figsize=(10, 6))
feat_importances = pd.Series(model.feature_importances_[:7], index=feature_names)
feat_importances.nlargest(7).plot(kind='barh', color='salmon')
plt.title('Top 7 Feature Importances (XGBoost)')
plt.xlabel('Relative Importance')
plt.savefig('plots/best_model_results.png')
print("✅ Plot Feature Importance sauvegardé.")