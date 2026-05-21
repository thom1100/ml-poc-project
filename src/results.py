"""Helpers for saving evaluation results."""

from __future__ import annotations
from collections.abc import Iterable
import pandas as pd
from config import MODEL_METRICS_FILE

def write_metrics(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Write model metrics to ``results/model_metrics.csv`` and return a DataFrame."""
    
    # 1. Conversion des données en DataFrame
    metrics_df = pd.DataFrame(rows)
    
    # 2. SÉCURITÉ : On s'assure que le dossier 'results' existe avant d'écrire
    MODEL_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # 3. Sauvegarde au format CSV
    metrics_df.to_csv(MODEL_METRICS_FILE, index=False)
    
    return metrics_df