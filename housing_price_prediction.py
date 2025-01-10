import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(filepath):
    """
    Load the dataset from the provided file path.
    :param filepath: str, path to the CSV file
    :return: pd.DataFrame, loaded dataset
    """
    try:
        df = pd.read_csv(filepath)
        print("Dataset loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None


def preprocess_data(df):
    """
    Preprocess the dataset:
    - Drop unnecessary columns.
    - Encode categorical features.
    - Normalize numerical features.
    :param df: pd.DataFrame, input dataset
    :return: pd.DataFrame, preprocessed dataset
    """
    print("Preprocessing dataset...")

    df = df.drop("Home", axis=1)

    categorical_features = ["Brick", "Neighborhood"]
    # drop first to avoid multicollinearity
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    encoded_cats = pd.DataFrame(
        encoder.fit_transform(df[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features)
    )
    df = pd.concat([df, encoded_cats], axis=1)
    df = df.drop(categorical_features, axis=1)

    numerical_features = ["SqFt", "Bedrooms", "Bathrooms", "Offers"]
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    print("Preprocessing complete.")
    return df


if __name__ == "__main__":
    # Step 1: Load the dataset
    dataset_path = "datasets/house-prices.csv"  # Update with your file path
    data = load_data(dataset_path)

    # Step 2: Preprocess the data
    if data is not None:
        data = preprocess_data(data)
        print("Preprocessed Dataset:")
        print(data.head())
