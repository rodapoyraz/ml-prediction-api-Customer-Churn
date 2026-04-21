import pandas as pd
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "raw" / "customer_churn.csv"

    if not data_path.exists():
        print(f"Dataset not found at: {data_path}")
        return

    df = pd.read_csv(data_path)

    print("=" * 50)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 50)

    print("\nShape of dataset:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nNumber of duplicate rows:")
    print(df.duplicated().sum())


if __name__ == "__main__":
    main()
