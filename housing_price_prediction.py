import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
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


def train_random_forest(x_train, y_train, n_estimators=100, max_depth=None):
    """
    Train a Random Forest model on the training data.
    :param x_train: Training features
    :param y_train: Training target
    :param n_estimators: Number of trees in the forest
    :param max_depth: Maximum depth of each tree
    :return: Trained Random Forest model
    """
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(x_train, y_train)
    print("Random Forest model training complete.")
    return model


def train_xgboost(x_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=3):
    """
    Train an XGBoost model on the training data.
    :param x_train: Training features
    :param y_train: Training target
    :param n_estimators: Number of boosting rounds
    :param learning_rate: Learning rate (shrinkage factor)
    :param max_depth: Maximum depth of each tree
    :return: Trained XGBoost model
    """
    print("Training XGBoost model...")
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
    model.fit(x_train, y_train)
    print("XGBoost model training complete.")
    return model


def perform_hyperparameter_tuning(model, param_grid, x_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.
    :param model: The model to tune
    :param param_grid: Dictionary of hyperparameters to test
    :param x_train: Training features
    :param y_train: Training target
    :return: Best model after tuning
    """
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best R² score from tuning: {grid_search.best_score_:.2f}")
    return grid_search.best_estimator_


if __name__ == "__main__":
    dataset_path = "datasets/house-prices.csv"
    data = load_data(dataset_path)

    if data is not None:
        data = preprocess_data(data)
        print("Preprocessed Dataset:")
        print(data.head())

        target_column = "Price"
        x_train, x_test, y_train, y_test = split_data(data, target_column)

        model_type = "random_forest"  # Options: "linear", "polynomial", "random_forest", "xgboost", or tuned versions

        if model_type == "linear":
            model = train_linear_regression(x_train, y_train)
            y_pred = evaluate_model(model, x_test, y_test)
            visualize_predictions(y_test, y_pred)

        elif model_type == "polynomial":
            degree = 2
            train_and_evaluate_polynomial_regression(x_train, x_test, y_train, y_test, degree)

        elif model_type == "random_forest":
            model = train_random_forest(x_train, y_train, n_estimators=500, max_depth=10)
            y_pred = evaluate_model(model, x_test, y_test)
            visualize_predictions(y_test, y_pred)

        elif model_type == "random_forest_tuned":
            param_grid_rf = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
            }
            rf_model = RandomForestRegressor(random_state=42)
            best_rf_model = perform_hyperparameter_tuning(rf_model, param_grid_rf, x_train, y_train)
            y_pred = evaluate_model(best_rf_model, x_test, y_test)
            visualize_predictions(y_test, y_pred)

        elif model_type == "xgboost":
            model = train_xgboost(x_train, y_train, n_estimators=200, learning_rate=0.05, max_depth=6)
            y_pred = evaluate_model(model, x_test, y_test)
            visualize_predictions(y_test, y_pred)

        elif model_type == "xgboost_tuned":
            param_grid_xgb = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
            }
            xgb_model = XGBRegressor(random_state=42)
            best_xgb_model = perform_hyperparameter_tuning(xgb_model, param_grid_xgb, x_train, y_train)
            y_pred = evaluate_model(best_xgb_model, x_test, y_test)
            visualize_predictions(y_test, y_pred)

        else:
            print(f"Error: Unknown model type '{model_type}'. Choose 'linear', 'polynomial', 'random_forest', "
                  f"or 'xgboost'.")

