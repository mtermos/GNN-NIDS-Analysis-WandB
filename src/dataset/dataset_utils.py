import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_dataset(df, timestamp_col=None, flow_id_col=None):
    print(f"==>> original df.shape[0]: {df.shape[0]}")
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with any NaN values
    df.dropna(axis=0, how='any', inplace=True)
    # df.dropna(axis=0, how='any', inplace=True, subset=list(set(
    #     df.columns) - set(id_columns)))
    print(f"==>> after drop na df.shape[0]: {df.shape[0]}")

    # Drop duplicate rows except for the first occurrence, based on all columns except timestamp and flow_id
    id_columns = []
    if timestamp_col:
        id_columns.append(timestamp_col)
    if flow_id_col:
        id_columns.append(flow_id_col)

    if len(id_columns) == 0:
        df.drop_duplicates(keep="first", inplace=True)
    else:
        df.drop_duplicates(subset=list(set(
            df.columns) - set(id_columns)), keep="first", inplace=True)
    print(f"==>> after drop_duplicates df.shape[0]: {df.shape[0]}")

    return df


def convert_file(input_path, output_format):
    # Supported output formats
    supported_formats = ['csv', 'parquet', 'pkl']

    # Validate the output format
    if output_format not in supported_formats:
        raise ValueError(
            f"Unsupported output file format. Supported formats are: {', '.join(supported_formats)}")

    # Determine the input format based on the file extension
    input_extension = os.path.splitext(input_path)[1].lower()

    # Read the input file
    if input_extension == '.csv':
        df = pd.read_csv(input_path)
    elif input_extension == '.parquet':
        df = pd.read_parquet(input_path)
    elif input_extension == '.pkl':
        df = pd.read_pickle(input_path)
    else:
        raise ValueError("Unsupported input file format")

    # Determine the output path
    output_path = os.path.splitext(input_path)[0] + f".{output_format}"

    # Save the file in the desired format
    if output_format == 'csv':
        df.to_csv(output_path, index=False)
    elif output_format == 'parquet':
        df.to_parquet(output_path, index=False)
    elif output_format == 'pkl':
        df.to_pickle(output_path)

    print(f"File saved as {output_path}")


def one_dataset_class_num_col(df, class_num_col, class_col):
    classes = df[class_col].unique()
    label_encoder = LabelEncoder()

    label_encoder.fit(list(classes))
    df[class_num_col] = label_encoder.transform(df[class_col])

    labels_names = dict(zip(label_encoder.transform(
        label_encoder.classes_), label_encoder.classes_))

    print(f"==>> labels_names: {labels_names}")

    return df, labels_names


def two_dataset_class_num_col(df1, df2, class_num_col, class_col, class_num_col2=None, class_col2=None):
    if class_num_col2 == None:
        class_num_col2 = class_num_col
    if class_col2 == None:
        class_col2 = class_col

    classes1 = df1[class_col].unique()
    classes2 = df2[class_col2].unique()

    classes = set(np.concatenate([classes2, classes1]))
    label_encoder = LabelEncoder()
    label_encoder.fit(list(classes))

    df1[class_num_col] = label_encoder.transform(
        df1[class_col])
    df2[class_num_col2] = label_encoder.transform(
        df2[class_col2])
    labels_names = dict(zip(label_encoder.transform(
        label_encoder.classes_), label_encoder.classes_))

    print(f"==>> labels_names: {labels_names}")

    return df1, df2, labels_names


def undersample_classes(df, class_col, n_undersample, fraction=0.5):
    """
    Undersamples the classes with the highest number of records.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        class_col (str): The name of the class column.
        n_undersample (int): The number of classes to undersample.
        fraction (float): The fraction of samples to keep from the undersampled classes.

    Returns:
        pd.DataFrame: The undersampled DataFrame.
    """
    # Group by the class column and get the count of records in each class
    class_counts = df.groupby(class_col).size()

    # Sort the counts in descending order
    class_counts_sorted = class_counts.sort_values(ascending=False)

    # Get the classes with the highest number of records to undersample
    classes_to_undersample = class_counts_sorted.index[:n_undersample]

    # Undersample the classes with the highest number of records
    dfs = []
    for class_label in class_counts_sorted.index:
        print(f"==>> class_label: {class_label}")
        class_df = df[df[class_col] == class_label]
        if class_label in classes_to_undersample:
            # Specify the fraction of samples to keep
            undersampled_df = class_df.sample(frac=fraction)
            dfs.append(undersampled_df)
        else:
            dfs.append(class_df)

    # Concatenate all DataFrames and shuffle the undersampled DataFrame
    df_undersampled = pd.concat(dfs).sample(frac=1).reset_index(drop=True)

    return df_undersampled
