"""This module holds the implementation of the class that visualizes the distribution of a feature's values."""

import warnings
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def population_flow(
    df: pd.DataFrame,
    target_feature_name: str,
    y_axis: str = "# Total",
    split: str | None = None,
    class_order: List[str | int] | None = None,
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
        dfs = [
            ("Total", df),
            ("Train", df.query(f'{split} == "Train"')),
            ("Validation", df.query(f'{split} == "Validation"')),
            ("Test", df.query(f'{split} == "Test"')),
        ]
    else:
        dfs = [
            ("Total", df),
        ]

    len_dfs = len(dfs)

    _, ax = plt.subplots(len_dfs // 2, 2, figsize=(10, 10))

    if y_axis == "# Total":
        y_cap = df[target_feature_name].value_counts().max() * 1.07
    else:
        y_cap = 1

    for (dataset, dfi), axi in zip(dfs, ax.flatten()):
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

        bars = axi.bar(index, values)
        axi.tick_params(axis="x", labelrotation=35)
        axi.bar_label(bars)
        axi.set_ylim([0, y_cap])
        axi.set_title(dataset)

        if class_order is not None:
            axi.set_xticks(range(len(class_order)))
            axi.set_xticklabels(class_order)

    plt.tight_layout()
