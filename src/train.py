import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from data_preprocessing import load_data, clean_data, build_preprocessor


def main():
    # Load and clean data
    df = load_data()
    df = clean_data(df)

    # Split features and target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build preprocessor
    preprocessor = build_preprocessor(df)

    # Create full pipeline
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    # Train model
    model_pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = model_pipeline.predict(X_test)

    # Evaluation
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = Path("models/full_pipeline.pkl")
    joblib.dump(model_pipeline, model_path)

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
