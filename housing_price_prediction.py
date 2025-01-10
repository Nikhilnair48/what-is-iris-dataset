import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


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


def split_data(df, target_column):
    """
    Split the dataset into features and target, and then into training and testing sets.
    :param df: pd.DataFrame, preprocessed dataset
    :param target_column: str, name of the target column
    :return: X_train, X_test, y_train, y_test
    """
    print("Splitting dataset into training and testing sets...")
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Dataset split complete.")
    return x_train, x_test, y_train, y_test


def train_linear_regression(x_train, y_train):
    """
    Train a linear regression model on the training data.
    :param X_train: pd.DataFrame, training features
    :param y_train: pd.Series, training target
    :return: trained LinearRegression model
    """
    print("Training linear regression model...")
    model = LinearRegression()
    model.fit(x_train, y_train)
    print("Model training complete.")
    return model


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the linear regression model on the testing data.
    :param model: trained LinearRegression model
    :param X_test: pd.DataFrame, testing features
    :param y_test: pd.Series, testing target
    """
    print("Evaluating model performance...")
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    return y_pred


def visualize_predictions(y_test, y_pred):
    """
    Plot actual vs. predicted prices to analyze model performance.
    :param y_test: pd.Series, actual target values
    :param y_pred: np.ndarray, predicted target values
    """
    print("Visualizing predictions...")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs. Predicted Prices")
    plt.show()


def add_polynomial_features(x_train, x_test, degree):
    """
    Add polynomial features to the dataset for non-linear modeling.
    :param x_train: pd.DataFrame, training features
    :param x_test: pd.DataFrame, testing features
    :param degree: int, degree of polynomial features
    :return: transformed X_train, X_test
    """
    print(f"Adding polynomial features (degree={degree})...")

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    print("Polynomial features added.")
    return x_train_poly, x_test_poly


def train_ridge_regression(x_train, y_train, alpha=1.0):
    """
    Train a Ridge Regression model with L2 regularization.
    :param x_train: Training features (can include polynomial features)
    :param y_train: Training target
    :param alpha: Regularization strength
    :return: Trained Ridge model
    """
    print(f"Training Ridge Regression model with alpha={alpha}...")
    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    print("Ridge Regression model training complete.")
    return model


def train_and_evaluate_polynomial_regression(x_train, x_test, y_train, y_test, degree):
    """
    Train and evaluate a polynomial regression model.
    """
    # Step 1: Add polynomial features
    x_train_poly, x_test_poly = add_polynomial_features(x_train, x_test, degree)

    # For degree 2, produced: Mean Squared Error (MSE): 160831795.65 and R² Score: 0.73
    # model = train_linear_regression(x_train_poly, y_train)
    # y_pred = evaluate_model(model, x_test_poly, y_test)
    # visualize_predictions(y_test, y_pred)

    ridge_model = train_ridge_regression(x_train_poly, y_train, alpha=1.0)
    y_pred_ridge = evaluate_model(ridge_model, x_test_poly, y_test)
    visualize_predictions(y_test, y_pred_ridge)


if __name__ == "__main__":
    dataset_path = "datasets/house-prices.csv"
    data = load_data(dataset_path)

    if data is not None:
        data = preprocess_data(data)
        print("Preprocessed Dataset:")
        print(data.head())

        target_column = "Price"
        x_train, x_test, y_train, y_test = split_data(data, target_column)

        model_type = "polynomial"

        if model_type == "linear":
            model = train_linear_regression(x_train, y_train)
            y_pred = evaluate_model(model, x_test, y_test)
            visualize_predictions(y_test, y_pred)

        elif model_type == "polynomial":
            degree = 1
            train_and_evaluate_polynomial_regression(x_train, x_test, y_train, y_test, degree)

        else:
            print(f"Error: Unknown model type '{model_type}'. Choose 'linear' or 'polynomial'.")
