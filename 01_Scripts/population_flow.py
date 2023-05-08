"""This module holds the implementation of the class that visualizes the distribution of a feature's values."""

import warnings
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def population_flow(
    df: pd.DataFrame,
    target_feature_name: str,
    y_axis: str = "# Total",
    split: str = None,
    class_order: List[str] = None,
) -> None:
    """Create charts with the distribution of the target feature.

    Args:
        df (pd.DataFrame): DataFrame storing the target feature.
        target_feature_name (str): Name of target feature.
        y_axis (str, optional): What to plot on the `y` axis. Has to be either "% Total" or "# Total". Defaults to "# Total".
        split (str, optional): Column from which to get a train-val-test split. If None, no split is made and only `df` is used. Defaults to None.
        class_order (List[str], optional): Order of the value on the `x` axis. If None, the values are sorted. Defaults to None.
    """
    if y_axis not in ("% Total", "# Total"):
        warnings.warn('Parameter `y_axis` has to be either "% Total" or "# Total".')
        return

    if split is not None:
        df_train = df.query(f'{split} == "Train"')
        df_val = df.query(f'{split} == "Validation"')
        df_test = df.query(f'{split} == "Test"')
        dfs = [
            ("Total", df, 0, 0),
            ("Train", df_train, 0, 1),
            ("Validation", df_val, 1, 0),
            ("Test", df_test, 1, 1),
        ]
    else:
        dfs = [
            ("Total", df, 0, 0),
        ]

    len_dfs = len(dfs)

    if len_dfs > 1:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    if y_axis == "# Total":
        y_cap = df[target_feature_name].value_counts().max() * 1.07
    else:
        y_cap = 1

    for dataset, dfi, i, j in dfs:
        flag_frequencies = pd.concat(
            [
                dfi[target_feature_name].value_counts(dropna=False),
                dfi[target_feature_name].value_counts(dropna=False, normalize=True),
            ],
            axis=1,
            keys=["# Total", "% Total"],
        ).sort_index()

        if class_order is not None:
            flag_frequencies = flag_frequencies.reindex(class_order)

        index = flag_frequencies[y_axis].index
        values = [round(val, 2) for val in flag_frequencies[y_axis].values]

        if len_dfs > 1:
            bars = ax[i, j].bar(index, values)
            ax[i, j].tick_params(axis="x", labelrotation=35)
            ax[i, j].bar_label(bars)
            ax[i, j].set_ylim([0, y_cap])
            ax[i, j].set_title(dataset)
        else:
            bars = ax.bar(index, values)
            ax.tick_params(axis="x", labelrotation=35)
            ax.bar_label(bars)
            ax.set_ylim([0, y_cap])
            ax.set_title(dataset)

    plt.tight_layout()
