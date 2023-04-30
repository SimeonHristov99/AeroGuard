"""This module holds the logic for the class dealing with outputs
and the file structure."""

import os
import warnings


class Controller:
    """Defines the logic of the `Controller` class
    """

    def __init__(self, iteration: str) -> None:
        self.iteration = iteration
        self.path_data_original = './00_Data/Original Data/'
        self.path_data_prepared = './00_Data/Prepared Data/'
        self.path_scripts = '01_Scripts/'
        self.path_outputs = '02_Outputs/'

    def create_folder_structure(self) -> None:
        """Creates the file structure used in the project
        """
        os.makedirs(self.path_data_original, exist_ok=True)
        os.makedirs(self.path_data_prepared, exist_ok=True)
        os.makedirs(self.path_scripts, exist_ok=True)
        os.makedirs(self.path_outputs, exist_ok=True)

    def get_path_data_original(self) -> str:
        """Returns the path to the directory holding the original data.

        Returns:
            str: Path to the directory holding the original data.
        """
        if not os.path.isdir(self.path_data_original):
            warnings.warn(
                f'Directory {self.path_data_original} may not exist.')
        return self.path_data_original

    def get_path_data_prepared(self) -> str:
        """Returns the path to the directory holding the original data.

        Returns:
            str: Path to the directory holding the original data.
        """
        if not os.path.isdir(self.path_data_prepared):
            warnings.warn(
                f'Directory {self.path_data_prepared} may not exist.')
        return self.path_data_prepared

    def get_path_iteration(self, iteration=None) -> str:
        """Returns the path to a new folder in which output results
        from the passed iteration can be saved. If no iteration is passed,
        the iteration passed in the constructor is used.

        If such a folder already exists, nothing will happen.

        Returns:
            str: Path to the directory holding the outputs of the passed iteration.
        """
        if iteration is None:
            iteration = self.iteration

        path = fr'./{self.path_outputs}{iteration}'
        os.makedirs(path, exist_ok=True)
        return path


if __name__ == '__main__':
    c = Controller('i01')
    c.create_folder_structure()
