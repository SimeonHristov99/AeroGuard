"""This module holds implementation of the class performing univariate analysis (data audit)."""

from typing import List, Optional

import pandas as pd
from IPython.display import display
from tqdm import tqdm

import warnings


class DataAudit:
    """Perform univariate analysis on a set of features."""

    def __init__(self):
        self.summary_sheet_name: str = "Summary"
        self._feature_name_limit: int = 31
        self.df: pd.DataFrame = None
        self.data_audit_summary: pd.DataFrame = None
        self.feature_dfs: dict(str, pd.DataFrame) = {}
        self.columns: List[str] = [
            "Variable",
            "Label",
            "Type",
            "% Unique",
            "% Missing",
            "mean",
            "std",
            "min",
            "50%",
            "max",
            "Comment",
        ]

    def fit(
        self,
        df: pd.DataFrame,
        target_feature: pd.Series,
        selection_set: List[str] = None,
    ) -> None:
        """Create the table with statisitcs for each variable.

        Args:
            df (pd.DataFrame): A dataframe with the features for which the analysis will be done.
            target_feature (pd.Series): A series with the target values for each observation in `df`.
            selection_set (List[str], optional): A list of variables for which to perform the analysis (has to be a subset of `df.columns`). Defaults to None.
        """
        df = df.copy()

        self.df = df if selection_set is None else df[selection_set]

        # Construct skeleton of data audit dataframe
        self.data_audit_summary = (
            self.df.describe(include="all", percentiles=[0.5])
            .T.convert_dtypes()
            .reset_index(drop=False, names="Variable")
            .drop(["top", "freq"], axis=1)
        )

        # Add label and comment for each column
        self.data_audit_summary["Label"] = "."
        self.data_audit_summary["Comment"] = "."

        # Add the types of each column
        self.data_audit_summary["Type"] = self.df[
            self.data_audit_summary["Variable"]
        ].dtypes.values

        # Calculate the number of entries for all columns
        self.data_audit_summary["count"] = (
            self.df[self.data_audit_summary["Variable"]].count().values
        )

        # Calculate the number of unique entries for all columns
        self.data_audit_summary["unique"] = (
            self.df[self.data_audit_summary["Variable"]].nunique().values
        )

        # Calculate percentage of unique values
        self.data_audit_summary["% Unique"] = (
            self.data_audit_summary["unique"] / self.data_audit_summary["count"]
        ).map(lambda value: round(value, 4))

        # Calculate percentage of missing values
        self.data_audit_summary["% Missing"] = (
            self.df[self.data_audit_summary["Variable"]].isna().mean().values
        )
        self.data_audit_summary["% Missing"] = self.data_audit_summary["% Missing"].map(
            lambda value: round(value, 4)
        )

        # Value counts for all features
        pbar = tqdm(self.df.columns)
        for feature in pbar:
            pbar.set_postfix_str(f"Processing {feature}")

            flag_frequencies = pd.concat(
                [
                    self.df[feature].value_counts(dropna=False),
                    self.df[feature].value_counts(dropna=False, normalize=True),
                ],
                axis=1,
                keys=["# Total", "% Total"],
            ).sort_index()
            flag_frequencies = flag_frequencies.reset_index(drop=False, names="Value")
            flag_frequencies["% Total"] = flag_frequencies["% Total"].map(
                lambda value: round(value, 4)
            )
            flag_frequencies = flag_frequencies.fillna("NA")
            self.feature_dfs[feature[: self._feature_name_limit]] = flag_frequencies

        # Reorder and format the columns
        self.data_audit_summary = self.data_audit_summary[self.columns]
        self.data_audit_summary.columns = self.data_audit_summary.columns.map(str.upper)

    def export(self, filepath: str) -> None:
        """Create an Excel file with the data audit.

        Args:
            filepath (str): Fully qualified path (including name and extension of the file) to file in which to save the audit.
        """
        with pd.ExcelWriter(filepath) as writer:
            self.data_audit_summary.to_excel(
                excel_writer=writer, sheet_name=self.summary_sheet_name
            )

            # Get the openpyxl workbook and worksheet objects
            wb = writer.book
            ws_summary = wb[self.summary_sheet_name]

            # Add hyperlinks to the summary sheet for each feature
            pbar = tqdm(enumerate(self.feature_dfs))
            for i, feature in pbar:
                pbar.set_postfix_str(f"Exporting {feature}")

                # Add a hyperlink from the summary sheet to the feature sheet
                cell = ws_summary.cell(row=i + 2, column=2)
                cell.value = feature
                cell.hyperlink = f"#{feature}!A1"

                # Write the feature data to a new sheet in the output file
                self.feature_dfs[feature].to_excel(writer, sheet_name=feature)

                # Add a hyperlink from the feature sheet to the summary sheet
                ws_feature = wb[feature]
                cell = ws_feature.cell(row=1, column=1)
                cell.value = self.summary_sheet_name
                cell.hyperlink = f"#{self.summary_sheet_name}!A1"

    def extend_summary(self, columns: List[pd.Series]) -> None:
        """Add more columns to the data audit summary table.

        Args:
            columns (List[pd.Series]): A list of `pd.Series` objects
            that will be appended in the order they are passed
            to the end of the data audit summary table.
        """
        for col in columns:
            self.data_audit_summary = pd.merge(
                self.data_audit_summary,
                col,
                how="left",
                left_on="VARIABLE",
                right_index=True,
            )

    def view_summary(self, display_only=True) -> Optional[pd.DataFrame]:
        """View or return a copy of the data audit summary table.

        Args:
            display_only (bool, optional): Whether to only print the data audit summary table. If False, then a copy of the table is returned. Defaults to True.

        Returns:
            Optional[pd.DataFrame]: A copy of the data audit summary table. Only done if `display_only=False`
        """
        if not display_only:
            return self.data_audit_summary.copy()

        display(self.data_audit_summary)

    def view(self, feature, display_only=True) -> Optional[pd.DataFrame]:
        """View or return a copy of the data audit for a specific feature.

        The data audit for a specific feature contains a `pd.value_counts` of the series in the dataframe the data audit object was fitted on.

        Args:
            feature (str): The name of the feature for which to view or return the audit.
            display_only (bool, optional): Whether to only print the data audit for the feature. If False, then a copy of the table is returned. Defaults to True.

        Returns:
            Optional[pd.DataFrame]: A copy of the data audit table for `feature`. Only done if `display_only=False`
        """
        df_result = self.feature_dfs.get(feature, None)

        if df_result is None:
            warnings.warn(f"Feature {feature} not present")
            return None

        if not display_only:
            return df_result.copy()

        display(df_result)
