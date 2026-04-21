import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def load_data():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "customer_churn.csv"

    df = pd.read_csv(data_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Fix TotalCharges (string -> numeric)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 2. Handle missing values created during conversion
    df = df.dropna(subset=["TotalCharges"])

    # 3. Drop customerID (not useful for prediction)
    df = df.drop(columns=["customerID"])

    # 4. Convert target to numeric
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


def build_preprocessor(df: pd.DataFrame):
    X = df.drop("Churn", axis=1)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print("\nNumeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def main():
    df = load_data()

    print("Before cleaning:")
    print(df.dtypes)

    df_clean = clean_data(df)

    print("\nAfter cleaning:")
    print(df_clean.dtypes)

    print("\nShape after cleaning:", df_clean.shape)

    output_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "processed"
        / "cleaned_data.csv"
    )
    df_clean.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")

    preprocessor = build_preprocessor(df_clean)

    X = df_clean.drop("Churn", axis=1)
    X_transformed = preprocessor.fit_transform(X)

    print("\nTransformed data shape:", X_transformed.shape)


if __name__ == "__main__":
    main()
