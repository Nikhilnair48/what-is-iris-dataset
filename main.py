import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(filepath):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)
        print("Dataset successfully loaded!")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None


def explore_dataset(df, target_column):
    """Print basic information, summary statistics, missing values, and class distribution."""
    print("\n--- Dataset Preview ---")
    print(df.head())

    print("\n--- Dataset Information ---")
    print(df.info())

    print("\n--- Statistical Summary ---")
    print(df.describe())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    if target_column in df.columns:
        print(f"\n--- Class Distribution for '{target_column}' ---")
        print(df[target_column].value_counts())
    else:
        print(f"\nError: Target column '{target_column}' not found in the dataset.")


def visualize_histograms(df):
    """Visualize histograms for each numeric feature."""
    print("\n--- Visualizing Feature Distributions ---")
    df.hist(figsize=(10, 8))
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.show()


def visualize_pairplot(df, target_column):
    """Visualize pairplots to explore feature relationships."""
    print("\n--- Visualizing Pairwise Feature Relationships ---")
    if target_column in df.columns:
        sns.pairplot(df, hue=target_column)
        plt.show()
    else:
        print(f"Error: Target column '{target_column}' not found in the dataset.")


def compute_group_means(df, groupby_column):
    """Compute and display mean values of numeric features grouped by a target column."""
    print(f"\n--- Average Measurements Grouped by {groupby_column} ---")
    if groupby_column in df.columns:
        grouped_means = df.groupby(groupby_column).mean()
        print(grouped_means)
    else:
        print(f"Error: Column '{groupby_column}' not found in the dataset.")


def visualize_correlation_heatmap(df):
    """Visualize the correlation heatmap for numeric features."""
    print("\n--- Visualizing Correlation Heatmap ---")
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.empty:
        print("Error: No numeric columns available for correlation.")
        return

    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()


if __name__ == "__main__":
    # Step 1: Load the dataset
    dataset_path = "iris_dataset.csv"
    df = load_dataset(dataset_path)

    # Proceed only if the dataset is successfully loaded
    if df is not None:
        target_column = "target"
        explore_dataset(df, target_column=target_column)

        visualize_histograms(df)

        visualize_pairplot(df, target_column=target_column)

        groupby_column = "target"
        compute_group_means(df, groupby_column=groupby_column)

        visualize_correlation_heatmap(df)