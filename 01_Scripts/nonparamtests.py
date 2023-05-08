"""This module holds implementations of nonparametric statistical tests."""

from typing import List

from scipy import stats

import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score


def calculate_kruskal(
    df: pd.DataFrame,
    selection_set: List[str],
    target_feature: pd.Series,
    col_name: str='KRUSKAL',
) -> pd.Series:
    """Perform the Kruskal–Wallis one-way analysis of variance test for numeric features.

    Args:
        df (pd.DataFrame): DataFrame holding the columns for which the test will be ran.
        selection_set (List[str]): A list with numeric features for which the test will be ran.
        target_feature (pd.Series): A series with the target feature values for each observation in `df`.
        col_name (str): The name that will be given to the series (therefore the column) with the p-values. Defaults to "KRUSKAL".

    Returns:
        pd.Series: A series with the p-values after each test.
        If the p-value is less than a user-determined threshold (typically 0.05)
        then the feature has a strong predictive power and can be used
        to distinguish between the class values.
    """

    class_indices = range(target_feature.nunique())

    result = pd.Series(index=selection_set, data=selection_set, name=col_name).map(
        lambda feature: stats.kruskal(
            *(pd.crosstab(
                df[feature],
                target_feature,
                dropna=False,
                margins=True,
                )[class_indices].values
            )).pvalue
    )

    return result


if __name__ == "__main__":
    print(f"Hello from {__file__}")
