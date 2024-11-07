"""Module for handling tabular machine learning prediction tasks.

This module provides the TabularPredictionTask class which encapsulates data and metadata
for tabular machine learning tasks, including datasets and their associated metadata.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypeVar, cast

import joblib
import pandas as pd
import s3fs
from autogluon.tabular import TabularDataset

# Type aliases for better readability
DatasetType = Union[Path, pd.DataFrame, TabularDataset]
PathLike = Union[str, Path]

# Constants
TRAIN = "train"
TEST = "test"
OUTPUT = "output"

T = TypeVar('T', bound='TabularPredictionTask')

class TabularPredictionTask:
    """A class representing a tabular machine learning prediction task.
    
    This class contains data and metadata for a tabular machine learning task,
    including datasets, problem type, and other relevant metadata.
    
    Attributes:
        metadata (Dict[str, Any]): Task metadata including name, description, and configuration
        filepaths (List[Path]): List of paths to task-related files
        cache_data (bool): Whether to cache loaded data in memory
        dataset_mapping (Dict[str, Union[Path, pd.DataFrame, TabularDataset]]): Mapping of dataset types to their data
    """

    def __init__(
        self,
        filepaths: List[Path],
        metadata: Dict[str, Any],
        name: str = "",
        description: str = "",
        cache_data: bool = True,
    ) -> None:
        """Initialize a new TabularPredictionTask.

        Args:
            filepaths: List of paths to task-related files
            metadata: Dictionary containing task metadata
            name: Name of the task
            description: Description of the task
            cache_data: Whether to cache loaded data in memory
        """
        self.metadata = {
            "name": name,
            "description": description,
            "label_column": None,
            "problem_type": None,
            "eval_metric": None,
            "test_id_column": None,
        }
        self.metadata.update(metadata)

        self.filepaths = filepaths
        self.cache_data = cache_data
        self.dataset_mapping: Dict[str, Optional[DatasetType]] = {
            TRAIN: None,
            TEST: None,
            OUTPUT: None,
        }

    def __repr__(self) -> str:
        """Return a string representation of the task."""
        return (
            f"TabularPredictionTask(name={self.metadata['name']}, "
            f"description={self.metadata['description'][:100]}, "
            f"{len(self.dataset_mapping)} datasets)"
        )

    @staticmethod
    def read_task_file(task_path: Path, filename_pattern: str, default_filename: str = "description.txt") -> str:
        """Read contents of a task file, searching recursively in the task path.

        Args:
            task_path: Base path to search for the file
            filename_pattern: Pattern to match the filename
            default_filename: Fallback filename if pattern isn't found

        Returns:
            Contents of the found file as a string
        """
        try:
            matching_paths = sorted(
                list(task_path.glob(filename_pattern)),
                key=lambda x: len(x.parents),  # top level files take precedence
            )
            if not matching_paths:
                return Path(task_path / default_filename).read_text()
            return matching_paths[0].read_text()
        except (FileNotFoundError, IndexError):
            return ""

    @staticmethod
    def save_artifacts(
        full_save_path: PathLike,
        predictor: Any,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        sample_submission_data: pd.DataFrame,
    ) -> None:
        """Save model artifacts either locally or to S3.

        Args:
            full_save_path: Path where artifacts should be saved
            predictor: AutoGluon TabularPredictor instance
            train_data: Training data
            test_data: Test data
            sample_submission_data: Sample submission data
        """
        artifacts = {
            "trained_model": predictor,
            "train_data": train_data,
            "test_data": test_data,
            "out_data": sample_submission_data,
        }

        ag_model_dir = predictor.predictor.path
        full_save_path_pkl_file = f"{full_save_path}/artifacts.pkl"

        if str(full_save_path).startswith("s3://"):
            fs = s3fs.S3FileSystem()
            with fs.open(full_save_path_pkl_file, "wb") as f:
                joblib.dump(artifacts, f)

            s3_model_dir = f"{full_save_path}/{os.path.dirname(ag_model_dir)}/{os.path.basename(ag_model_dir)}"
            fs.put(ag_model_dir, s3_model_dir, recursive=True)
        else:
            os.makedirs(str(full_save_path), exist_ok=True)
            with open(full_save_path_pkl_file, "wb") as f:
                joblib.dump(artifacts, f)

            local_model_dir = os.path.join(str(full_save_path), ag_model_dir)
            shutil.copytree(ag_model_dir, local_model_dir, dirs_exist_ok=True)

    @classmethod
    def from_path(cls: type[T], task_root_dir: Path, name: Optional[str] = None) -> T:
        """Create a TabularPredictionTask instance from a directory path.

        Args:
            task_root_dir: Root directory containing task files
            name: Optional name for the task

        Returns:
            A new TabularPredictionTask instance
        """
        task_data_filenames = []
        for root, _, files in os.walk(task_root_dir):
            for file in files:
                relative_path = os.path.relpath(os.path.join(root, file), task_root_dir)
                task_data_filenames.append(relative_path)

        return cls(
            filepaths=[task_root_dir / fn for fn in task_data_filenames],
            metadata={"name": name or task_root_dir.name},
        )

    def describe(self) -> Dict[str, Any]:
        """Generate a description of the task including metadata and data statistics.

        Returns:
            Dictionary containing task description and statistics
        """
        description = {
            "name": self.metadata["name"],
            "description": self.metadata["description"],
            "metadata": self.metadata,
            "train_data": self.train_data.describe().to_dict(),
            "test_data": self.test_data.describe().to_dict(),
        }
        
        if self.sample_submission_data is not None:
            description["sample_submission_data"] = self.sample_submission_data.describe().to_dict()

        return description

    def get_filenames(self) -> List[str]:
        """Get all filenames associated with the task.

        Returns:
            List of filenames
        """
        return [f.name for f in self.filepaths]

    def _set_task_files(self, dataset_name_mapping: Dict[str, Optional[Union[str, DatasetType]]]) -> None:
        """Set the task files based on the provided mapping.

        Args:
            dataset_name_mapping: Mapping of dataset names to their sources
        """
        for key, value in dataset_name_mapping.items():
            if value is None:
                self.dataset_mapping[key] = None
                continue

            if isinstance(value, (pd.DataFrame, TabularDataset)):
                self.dataset_mapping[key] = value
            elif isinstance(value, (str, Path)):
                filepath = (
                    value if isinstance(value, Path)
                    else next(
                        (path for path in self.filepaths if path.name == value),
                        self.filepaths[0].parent / value
                    )
                )
                
                if not filepath.is_file():
                    raise ValueError(f"File {value} not found in task {self.metadata['name']}")

                if filepath.suffix in [".xlsx", ".xls"]:
                    self.dataset_mapping[key] = (
                        pd.read_excel(filepath, engine="calamine")
                        if self.cache_data else filepath
                    )
                else:
                    self.dataset_mapping[key] = (
                        TabularDataset(str(filepath))
                        if self.cache_data else filepath
                    )
            else:
                raise TypeError(f"Unsupported type for dataset_mapping: {type(value)}")

    @property
    def train_data(self) -> TabularDataset:
        """Get the training dataset.

        Returns:
            Training dataset as TabularDataset
        """
        return self.load_task_data(TRAIN)

    @train_data.setter
    def train_data(self, data: Union[str, Path, pd.DataFrame, TabularDataset]) -> None:
        """Set the training dataset.

        Args:
            data: Training data to set
        """
        self._set_task_files({TRAIN: data})

    @property
    def test_data(self) -> TabularDataset:
        """Get the test dataset.

        Returns:
            Test dataset as TabularDataset
        """
        return self.load_task_data(TEST)

    @test_data.setter
    def test_data(self, data: Union[str, Path, pd.DataFrame, TabularDataset]) -> None:
        """Set the test dataset.

        Args:
            data: Test data to set
        """
        self._set_task_files({TEST: data})

    @property
    def sample_submission_data(self) -> Optional[TabularDataset]:
        """Get the sample submission dataset.

        Returns:
            Sample submission dataset as TabularDataset if available
        """
        return self.load_task_data(OUTPUT)

    @sample_submission_data.setter
    def sample_submission_data(self, data: Union[str, Path, pd.DataFrame, TabularDataset]) -> None:
        """Set the sample submission dataset.

        Args:
            data: Sample submission data to set

        Raises:
            ValueError: If output data is already set
        """
        if self.sample_submission_data is not None:
            raise ValueError("Output data already set for task")
        self._set_task_files({OUTPUT: data})

    @property
    def output_columns(self) -> Optional[List[str]]:
        """Get the output dataset columns.

        Returns:
            List of column names or None if not available
        """
        if self.sample_submission_data is None:
            return [self.label_column] if self.label_column else None
        return self.sample_submission_data.columns.tolist()

    @property
    def label_column(self) -> Optional[str]:
        """Get the label column name.

        Returns:
            Name of the label column or None if not set
        """
        return self.metadata.get("label_column") or self._infer_label_column_from_sample_submission_data()

    @label_column.setter
    def label_column(self, label_column: str) -> None:
        """Set the label column name.

        Args:
            label_column: Name of the label column
        """
        self.metadata["label_column"] = label_column

    @property
    def columns_in_train_but_not_test(self) -> List[str]:
        """Get columns that exist in training data but not in test data.

        Returns:
            List of column names
        """
        return list(set(self.train_data.columns) - set(self.test_data.columns))

    def _infer_label_column_from_sample_submission_data(self) -> Optional[str]:
        """Infer the label column from sample submission data.

        Returns:
            Inferred label column name or None if cannot be inferred

        Raises:
            ValueError: If unable to infer the label column
        """
        if self.output_columns is None:
            return None

        relevant_output_cols = self.output_columns[1:]  # Ignore first column (assumed to be ID)
        existing_output_cols = [col for col in relevant_output_cols if col in self.train_data.columns]

        if len(existing_output_cols) == 1:
            return existing_output_cols[0]

        output_set = {col.lower() for col in relevant_output_cols}
        for col in self.train_data.columns:
            unique_values = {str(val).lower() for val in self.train_data[col].unique() if pd.notna(val)}
            if output_set == unique_values or output_set.issubset(unique_values):
                return col

        raise ValueError("Unable to infer the label column. Please specify it manually.")

    def load_task_data(self, dataset_key: str) -> Optional[TabularDataset]:
        """Load task data for a specific dataset type.

        Args:
            dataset_key: Key identifying the dataset to load

        Returns:
            Loaded dataset as TabularDataset or None if not available

        Raises:
            ValueError: If dataset type is not found
            TypeError: If file format is not supported
        """
        if dataset_key not in self.dataset_mapping:
            raise ValueError(f"Dataset type {dataset_key} not found for task {self.metadata['name']}")

        dataset = self.dataset_mapping[dataset_key]
        if dataset is None:
            return None

        if isinstance(dataset, pd.DataFrame):
            return TabularDataset(dataset)
        if isinstance(dataset, TabularDataset):
            return dataset
        
        if dataset.suffix == ".json":
            raise TypeError(f"File {dataset.name} has unsupported type: json")

        return TabularDataset(str(dataset))
