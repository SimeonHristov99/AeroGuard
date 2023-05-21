"""This module holds implementations of functions that generate Excel reports."""

import pandas as pd


def correlation_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a correlation report.

    The correlation report is a matrix with pairwise Pearson and Spearman correlation coefficients
    calculated for all numeric features of `df`.

    Args:
        df (pd.DataFrame): The dataframe holding the variables for which to produce the report.

    Returns:
        pd.DataFrame: The matrix holding the pairwise correlation coefficients.
    """
    corr_types = ["pearson", "spearman"]

    df = df.copy()

    numeric_columns = df.select_dtypes("number").columns

    corrs = []

    for col in numeric_columns:
        corr_col_others_p = df.drop(col, axis=1).corrwith(
            df[col], method=corr_types[0], numeric_only=True
        )
        corr_col_others_s = df.drop(col, axis=1).corrwith(
            df[col], method=corr_types[1], numeric_only=True
        )

        corr_col_others = pd.DataFrame(
            {
                "variable_x": col,
                "variable_y": corr_col_others_p.index,
                "pearson": corr_col_others_p.values,
                "spearman": corr_col_others_s.values,
            }
        )

        corrs.append(corr_col_others)

    result = pd.concat(corrs, ignore_index=True)
    result = result.fillna(0)
    return result
