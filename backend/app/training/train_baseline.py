import joblib
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

from backend.app.dataset.config import (
    SPLITS_PATH,
)

MODEL_PATH = Path("backend/checkpoints")
MODEL_PATH.mkdir(parents=True, exist_ok=True)


def train():

    train_df = pd.read_csv(SPLITS_PATH / "train.csv")
    val_df = pd.read_csv(SPLITS_PATH / "validation.csv")
    test_df = pd.read_csv(SPLITS_PATH / "test.csv")

    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    feature_columns = [
        col
        for col in train_df.columns
        if col.startswith("mfcc")
        or col.startswith("spectral")
        or col in ["rms", "zcr", "chroma_mean"]
    ]

    X_train = train_df[feature_columns]
    X_val = val_df[feature_columns]
    X_test = test_df[feature_columns]

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(train_df["label"])
    y_val = encoder.transform(val_df["label"])
    y_test = encoder.transform(test_df["label"])

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    print("\nValidation Accuracy")
    print(accuracy_score(y_val, val_pred))

    print("\nTest Accuracy")
    print(accuracy_score(y_test, test_pred))

    print("\nClassification Report")
    print(classification_report(y_test, test_pred))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, test_pred))

    joblib.dump(model, MODEL_PATH / "baseline_random_forest.pkl")

    print("\nModel saved successfully.")


if __name__ == "__main__":
    train()