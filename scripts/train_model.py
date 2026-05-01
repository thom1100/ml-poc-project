from nba_api.stats.endpoints import playercareerstats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from pathlib import Path


def main():
    player_id = "2544"

    career = playercareerstats.PlayerCareerStats(player_id=player_id)
    df = career.get_data_frames()[0]

    df = df[["PTS", "REB", "AST", "FG_PCT", "MIN"]].dropna()

    X = df.drop(columns=["PTS"])
    y = df["PTS"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # ✅ FIX IMPORTANT : pickle au lieu de joblib
    with open(models_dir / "model_a.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved successfully!")


if __name__ == "__main__":
    main()
