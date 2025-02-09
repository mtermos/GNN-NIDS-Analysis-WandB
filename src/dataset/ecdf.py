import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def prepare_df(df, timestamp_col):
    """
    Prepare the dataset for analysis.

    Parameters:
    - df: DataFrame.
    - timestamp_col: the string name of the timestamp column

    Returns:
    - df: DataFrame, the prepared dataset.
    """
    # Sort data chronologically
    df.sort_values(timestamp_col, inplace=True)

    # Reset the index and add an "index" column to the sorted dataframe
    df.reset_index(inplace=True, drop=True)
    df.reset_index(inplace=True)

    return df


def ecdf(data):
    """
    Calculate the Empirical Cumulative Distribution Function (ECDF) for a dataset.

    Parameters:
    - data: array-like, a list or array of numerical data points.

    Returns:
    - x: array, the sorted data points.
    - y: array, the ECDF value for each data point.
    """
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y


def plot_ecdf(df, label_col, show_figs=True, save_path=None):
    """
    Plot ECDF for benign and attack records in the dataset.

    Parameters:
    - df: DataFrame, the dataset.
    - label_col: str, the column name indicating the label (0 for benign, 1 for attack).
    - show_figs: bool, whether to display the figures.
    - save_path: str, the directory path to save the figures.
    """
    plt.figure(figsize=(14, 10))
    plt.rcParams['font.size'] = 18

    benign_data = df[df[label_col] == 0]["index"]
    attack_data = df[df[label_col] == 1]["index"]

    # Compute ECDF
    x1, y1 = ecdf(benign_data)
    x2, y2 = ecdf(attack_data)

    # Plot the ECDF
    plt.plot(x1, y1, color="#3062d9", linewidth=4,
             marker='o', markersize=4, label='Benign')
    plt.plot(x2, y2, color="#eb4034", linewidth=4,
             marker='o', markersize=4, label='Attack')

    plt.xlabel('Index')
    plt.ylabel('ECDF')
    plt.legend()
    plt.title('Empirical Cumulative Distribution Function (ECDF)')

    if save_path:
        plt.savefig(f'{save_path}/ecdf_index.png')
    if show_figs:
        plt.show()


def plot_feature_ecdf(df, label_col, network_features, show_figs=True, save_path=None):
    """
    Plot ECDF for each network feature in the dataset.

    Parameters:
    - df: DataFrame, the dataset.
    - label_col: str, the column name indicating the label (0 for benign, 1 for attack).
    - network_features: list, the list of network feature column names.
    - show_figs: bool, whether to display the figures.
    - save_path: str, the directory path to save the figures.
    """
    for feature in network_features:
        print(f"Plotting ECDF for feature: {feature}")

        plt.clf()
        plt.figure(figsize=(14, 10))
        plt.rcParams['font.size'] = 18

        benign_data = df[df[label_col] == 0][feature]
        attack_data = df[df[label_col] == 1][feature]

        # Compute ECDF
        x1, y1 = ecdf(benign_data)
        x2, y2 = ecdf(attack_data)

        # Plot the ECDF
        plt.plot(x1, y1, color="#3062d9", linewidth=4,
                 marker='o', markersize=4, label='Benign')
        plt.plot(x2, y2, color="#eb4034", linewidth=4,
                 marker='o', markersize=4, label='Attack')

        plt.xlabel(feature)
        plt.ylabel('ECDF')
        plt.legend()
        plt.title(f'ECDF of {feature}')

        if save_path:
            plt.savefig(f'{save_path}/ecdf_{feature}.png')
        if show_figs:
            plt.show()


def main(df, timestamp_col, label_col, network_features, show_figs=True, save_path=None):
    """
    Main function to prepare dataset and plot ECDFs.

    Parameters:
    - df: DataFrame, the dataset.
    - timestamp_col: str, the column name of the timestamp.
    - label_col: str, the column name indicating the label (0 for benign, 1 for attack).
    - network_features: list, the list of network feature column names.
    - show_figs: bool, whether to display the figures.
    - save_path: str, the directory path to save the figures.
    """
    # Prepare the dataset
    df = prepare_df(df, timestamp_col)

    # Plot ECDF for index
    plot_ecdf(df, label_col, show_figs, save_path)

    # Plot ECDF for each network feature
    plot_feature_ecdf(df, label_col, network_features, show_figs, save_path)


if __name__ == "__main__":
    data = {
        'timestamp': pd.date_range(start='1/1/2020', periods=10, freq='T'),
        'label': [0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        'feature1': np.random.randn(10),
        'feature2': np.random.randn(10)
    }
    df = pd.DataFrame(data)
    timestamp_col = "timestamp"
    label_col = "label"
    # Replace with actual feature names
    network_features = ["feature1", "feature2"]
    show_figs = False
    save_path = "./plots"

    main(df, timestamp_col, label_col, network_features, show_figs, save_path)
