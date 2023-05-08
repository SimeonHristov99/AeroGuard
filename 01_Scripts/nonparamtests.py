"""This module holds implementations of nonparametric statistical tests."""

from typing import List

import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_mutual_info_score


def calculate_ami(
    df: pd.DataFrame,
    selection_set: List[str],
    target_feature: pd.Series,
    col_name: str='AMI',
    ) -> pd.Series:
    """Calculate the adjusted mutual information between two categorical features.

    Args:
        df (pd.DataFrame): DataFrame holding the categorical columns.
        selection_set (List[str]): A list with the categorical features for which to calculate the AMI.
        target_feature (pd.Series): A series with the target value values for each observation in `df`.
        col_name (str, optional): The name that will be given to the series (therefore the column) with the AMI values. Defaults to "AMI".

    Returns:
        pd.Series: A series with the AMI values for each variable in `selection_set`.
        The AMI takes a value of 1 when the two partitions are identical
        and 0 when the MI between two partitions equals the value expected due to chance alone.
        Therefore, the value for AMI is directly proportional to the predictive power of the variable.
    """
    
    result = pd.Series(index=selection_set, data=selection_set, name=col_name).map(
        lambda feature:
            adjusted_mutual_info_score(
                pd.factorize(df[feature].values)[0],
                target_feature
            )
    )

    return result
    

def calculate_kruskal(
    df: pd.DataFrame,
    selection_set: List[str],
    target_feature: pd.Series,
    col_name: str='KRUSKAL',
    ) -> pd.Series:
    """Perform the Kruskal–Wallis one-way analysis of variance test between a numeric feature and a categorical feature.

    Args:
        df (pd.DataFrame): DataFrame holding the columns for which the test will be ran.
        selection_set (List[str]): A list with numeric features for which the test will be ran.
        target_feature (pd.Series): A series with the target value for each observation in `df`.
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
