import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold


def prepare_dataset(original_df, drop_columns, label_col):
    cleaned_df = original_df.drop(drop_columns, axis=1)
    cleaned_df = cleaned_df.drop(label_col, axis=1)
    return cleaned_df


def apply_variance_threshold(df, threshold):
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df)
    selected_features = df.columns[selector.get_support(indices=True)]
    dropped_features = [
        col for col in df.columns if col not in selected_features]
    df_filtered = df[selected_features]
    return df_filtered, dropped_features


def apply_correlation_threshold(df, threshold):
    corr_matrix = df.corr()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_features = [
        column for column in upper.columns if any(upper[column] > threshold)]
    features_to_remove = set()
    for feature in correlated_features:
        correlated_with_feature = list(upper.index[upper[feature] > threshold])
        for correlated_feature in correlated_with_feature:
            if correlated_feature not in features_to_remove:
                features_to_remove.add(correlated_feature)
    df_filtered = df.drop(features_to_remove, axis=1)
    return df_filtered, features_to_remove


def feature_analysis_pipeline(df, drop_columns, label_col, var_threshold=0.00, corr_threshold=0.75):
    new_df = prepare_dataset(df, drop_columns, label_col)
    new_df, var_dropped = apply_variance_threshold(new_df, var_threshold)
    new_df, corr_dropped = apply_correlation_threshold(new_df, corr_threshold)
    return new_df, var_dropped, corr_dropped
