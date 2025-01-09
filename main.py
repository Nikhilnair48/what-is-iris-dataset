import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(filepath):
    """Load the dataset from a CSV file."""
    return pd.read_csv(filepath)


def explore_dataset(df):
    """Print basic information and summary statistics of the dataset."""
    print("Dataset Preview:")
    print(df.head())

    print("\nDataset Information:")
    print(df.info())

    print("\nStatistical Summary:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())


def visualize_histograms(df):
    """Visualize histograms for each numeric feature."""
    df.hist(figsize=(10, 8))
    plt.suptitle("Feature Distributions", fontsize=16)
    plt.show()


def visualize_pairplot(df, target_column):
    """Visualize pairplots to explore feature relationships."""
    sns.pairplot(df, hue=target_column)
    plt.show()


def compute_group_means(df, groupby_column):
    """Compute and display mean values of numeric features grouped by a target column."""
    grouped_means = df.groupby(groupby_column).mean()
    print("\nAverage Measurements by Group:")
    print(grouped_means)


def visualize_correlation_heatmap(df):
    """Visualize the correlation heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()


if __name__ == "__main__":
    dataset_path = "iris_dataset.csv"
    df = load_dataset(dataset_path)

    explore_dataset(df)

    # visualize_histograms(df)

    # visualize_pairplot(df, target_column="target")

    compute_group_means(df, groupby_column="target")

    visualize_correlation_heatmap(df)
