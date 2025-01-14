import matplotlib.pyplot as plt
import shap


def interpret_linear_model(model, feature_names):
    """
    Interpret a linear regression model using coefficients.
    :param model: Trained LinearRegression model
    :param feature_names: List of feature names
    """
    print("\nLinear Regression Coefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"{name}: {coef:.4f}")

    # Optional: Plot coefficients
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, model.coef_, color="blue")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.title("Linear Regression Coefficients")
    plt.show()


def interpret_tree_model(model, feature_names):
    """
    Interpret a tree-based model (e.g., Random Forest or XGBoost) using feature importance.
    :param model: Trained tree-based model
    :param feature_names: List of feature names
    """
    importances = model.feature_importances_
    sorted_indices = importances.argsort()[::-1]

    print("\nFeature Importance (Tree-Based Model):")
    for i in sorted_indices:
        print(f"{feature_names[i]}: {importances[i]:.4f}")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh([feature_names[i] for i in sorted_indices], importances[sorted_indices], color="green")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance (Tree-Based Model)")
    plt.show()


def interpret_with_shap(model, x_train, feature_names):
    """
    Interpret model predictions using SHAP values.
    :param model: Trained model (e.g., RandomForest, XGBoost, or LinearRegression)
    :param x_train: Training dataset (features only)
    :param feature_names: List of feature names
    """

    print("\nGenerating SHAP explanations...")
    explainer = shap.Explainer(model, x_train)
    shap_values = explainer(x_train)

    # Global Interpretability: Feature importance summary
    print("\nSHAP Summary Plot:")
    shap.summary_plot(shap_values, x_train, feature_names=feature_names)

    # Local Interpretability: Single prediction explanation (force plot)
    print("\nSHAP Force Plot (First Prediction):")

    # Handle scalar expected value
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple)):
        expected_value = expected_value[0]

    shap.force_plot(
        expected_value,
        shap_values[0].values,
        x_train.iloc[0],
        matplotlib=True
    )
