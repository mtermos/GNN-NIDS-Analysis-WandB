
import numpy as np
import pandas as pd


def gini_coefficient(values):
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumulative_values = np.cumsum(sorted_values)
    gini = (n + 1 - 2 * np.sum(cumulative_values) / np.sum(sorted_values)) / n
    return gini


def calculate_gini_metrics(df, dataset, results):
    """
    Calculate multi-class and binary classification Gini coefficients,
    along with other related statistics, and update the provided results dictionary.

    Parameters:
        df (pandas.DataFrame): The dataframe containing your data.
        dataset: An object with attributes 'class_col' and 'label_col' that specify 
                 the column names for class and label in df.
        results (dict): A dictionary that will be updated with the calculated metrics.

    Returns:
        dict: The updated results dictionary with the following new keys added under results:
            - "Class Counts / Proportions": A dict mapping each class to its count and proportion.
            - "Multi-class Gini Coefficient": The Gini coefficient calculated from class proportions.
            - "Binary Classification Gini Coefficient": The Gini coefficient calculated from label proportions.
            - "length": Total number of records in df.
            - "num_benign": Number of records where the label is 0.
            - "percentage_of_benign_records": Percentage of records with a benign label.
            - "num_attack": Number of records where the label is 1.
            - "percentage_of_attack_records": Percentage of records with an attack label.
            - "attacks": A list of unique class names in the dataset.
    """

    # --- Multi-class Gini Coefficient ---
    class_counts = df[dataset.class_col].value_counts()
    class_proportions = class_counts.values / class_counts.values.sum()
    multi_class_gini = gini_coefficient(class_proportions)

    print("Class Counts:")
    print(class_counts)
    print("Class Proportions:", class_proportions)
    print("Multi-class Gini Coefficient:", multi_class_gini)

    # Update results with class counts and proportions.
    results["Class Counts / Proportions"] = {
        class_name: {
            "count": int(count),
            "proportion": float(proportion)
        }
        for class_name, count, proportion in zip(class_counts.index, class_counts.values, class_proportions)
    }

    # --- Binary Classification Gini Coefficient ---
    label_counts = df[dataset.label_col].value_counts()
    label_proportions = label_counts.values / label_counts.values.sum()
    binary_gini = gini_coefficient(label_proportions)

    print("Label Counts:")
    print(label_counts)
    print("Label Proportions:", label_proportions)
    print("Binary Classification Gini Coefficient:", binary_gini)

    # Store Gini coefficients in results.
    results["Multi-class Gini Coefficient"] = multi_class_gini
    results["Binary Classification Gini Coefficient"] = binary_gini

    # --- Additional Metrics ---
    total_count = len(df)
    results["length"] = total_count

    num_benign = len(df[df[dataset.label_col] == 0])
    num_attack = len(df[df[dataset.label_col] == 1])
    results["num_benign"] = num_benign
    results["percentage_of_benign_records"] = (
        num_benign * 100) / total_count
    results["num_attack"] = num_attack
    results["percentage_of_attack_records"] = (
        num_attack * 100) / total_count
    results["attacks"] = list(df[dataset.class_col].unique())

    # Interpretation:
    # - A Gini coefficient closer to 0 indicates a balanced distribution.
    # - A Gini coefficient closer to 1 indicates an imbalanced distribution.

    return results


if __name__ == '__main__':
    # Example usage (you will need to replace this with your actual data and dataset object):
    # Create a dummy dataset and results dictionary for demonstration.
    data = {
        'class': ['A', 'B', 'A', 'C', 'B', 'A'],
        'label': [0, 1, 0, 1, 1, 0]
    }
    df_example = pd.DataFrame(data)

    class DummyDataset:
        class_col = 'class'
        label_col = 'label'

    dataset_example = DummyDataset()
    results_example = {'example_data': {}}

    # Call the function to update the results.
    updated_results = calculate_gini_metrics(
        df_example, dataset_example, 'example_data', results_example)

    print("\nUpdated Results Dictionary:")
    print(updated_results)
