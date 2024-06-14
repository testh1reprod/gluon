import copy
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping

import dbinfer_bench as dbb
import numpy as np
import pandas as pd
from autogluon.common.features.feature_metadata import FeatureMetadata
from dbinfer_bench.dataset_meta import DBBColumnSchema, DBBRDBDatasetMeta, DBBTableSchema, DBBTaskMeta
from dbinfer.device import get_device_info
from dbinfer.preprocess.dfs.dfs_preprocess import DFSPreprocess, DFSPreprocessConfig
from dbinfer.preprocess.transform_preprocess import RDBTransformPreprocess, RDBTransformPreprocessConfig
from dbinfer.yaml_utils import load_pyd, save_pyd

from autogluon_assistant.task import TabularPredictionTask

from ..base import TransformTimeoutError
from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)


# From https://github.com/awslabs/multi-table-benchmark/blob/main/configs/transform/pre-dfs.yaml
_PRE_DFS_TRANSFORM_CONFIG = RDBTransformPreprocessConfig.parse_obj(
    {
        "transforms": [
            {"name": "handle_dummy_table"},
            {"name": "key_mapping"},
            {
                "name": "column_transform_chain",
                "config": {
                    "transforms": [
                        {"name": "canonicalize_numeric"},
                        {"name": "canonicalize_datetime"},
                        {"name": "glove_text_embedding"},
                    ]
                },
            },
        ]
    }
)


_DTYPE_MAP = {
    "Categorical": "category",  # NOTE: Limitation, int dtype columns could be just categorical columns but AG won't infer.
    "Numerical_INT": "float",  # NOTE: dbinfer limitation; cast int to float
    "Numerical_FLOAT": "float",
    "Text": "text",
    "DateTime": "datetime",
    "Other": "category",  # NOTE: forced as category; could be text. Fallback to LLM prompting later for tiebreak.
}


# NOTE: Inherits from BaseFeatureTransformer but reimplement methods
#       as DFS has different requirements compared to a regular FeatureTransformer
class DFSTransformer(BaseFeatureTransformer):
    """
    Deep Feature Synthesis (DFS) Feature Transformer for feature engineering.

    The DFSTransformer class is designed to automatically generate new features on single
    table datasets.

    https://ieeexplore.ieee.org/document/7344858
    """

    identifier = "dfs"

    def __init__(self, depth: int = 2, use_cat_vars_as_fks: bool = False, **kwargs) -> None:
        self.depth = depth
        self.use_cat_vars_as_fks = use_cat_vars_as_fks
        self.metadata: Dict[str, Any] = {"transformer": "DFS"}

    def fit(self, task: TabularPredictionTask) -> "BaseFeatureTransformer":
        # No fitting process required for DFS transformer, only perform setup here
        try:
            self.id_column = task.test_id_column
            self.column_type_dict = self._infer_column_types(task.train_data)
            # Add ID Column to column_type_dict map
            self.column_type_dict[self.id_column] = "ID"
            self.time_column = self._get_time_column(self.column_type_dict)
            self.target_column = task.label_column
            self.problem_type = task.problem_type
            self.dataset_description = task.data_description
        except TransformTimeoutError:
            logger.warning(f"FeatureTransformer {self.__class__.__name__} timed out.")
        except Exception:
            logger.warning(f"FeatureTransformer {self.__class__.__name__} failed to fit.")
            logger.warning(traceback.format_exc())
        finally:
            return self

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        try:
            train_data = task.train_data
            test_data = task.test_data

            # create a shallow copy of test_data
            # and add dummy target column to it
            test_data_w_dummy_target = test_data.copy()
            test_data_w_dummy_target[self.target_column] = np.nan

            # concatenate both to create a single dataframe
            all_data = pd.concat([train_data, test_data_w_dummy_target])

            # run dfs transformer
            all_data_transformed = self._run_dfs(
                input_df=all_data,
                df_name=task.name,
                target_column=self.target_column,
                time_column=self.time_column,
                column_type_dict=self.column_type_dict,
                depth=self.depth,
                use_cat_vars_as_fks=self.use_cat_vars_as_fks,
            )

            # postprocess the data back to train and test
            transformed_train_data, transformed_test_data = DFSTransformer._reorder_and_split_data(
                all_data_transformed, train_data
            )
            # drop fake target column from test data
            transformed_test_data = transformed_test_data.drop([self.target_column], axis="columns")

            task = copy.deepcopy(task)
            task.train_data = transformed_train_data
            task.test_data = transformed_test_data
        except:
            logger.warning(f"FeatureTransformer {self.__class__.__name__} failed to transform.")
        finally:
            return task

    def get_metadata(self) -> Mapping:
        return self.metadata

    def _remap_column_dtypes(self, dataframe, column_name_dtypes_map):
        """
        Remaps AG inferred column dtypes to dbinfer specific type group string categories.
        """
        transformed_dict = {}
        for column_name, dtype in column_name_dtypes_map.items():
            datatype_str = str(dtype)
            if "int" in datatype_str:
                # Check the number of unique values in the integer column.
                unique_count = dataframe[column_name].nunique()
                # If the number of unique values is less than 20 (heuristic number assumption),
                # consider it as categorical. This is because a low number of unique values
                # suggests that the column likely represents a category or label rather
                # than a continuous numerical value.
                if unique_count < 5:
                    transformed_dict[column_name] = "Categorical"
                else:
                    transformed_dict[column_name] = "Numerical_INT"
            elif "float" in datatype_str:
                transformed_dict[column_name] = "Numerical_FLOAT"
            elif "text" in datatype_str:
                transformed_dict[column_name] = "Text"
            elif "datetime" in datatype_str:
                transformed_dict[column_name] = "DateTime"
            elif "category" in datatype_str:
                transformed_dict[column_name] = "Categorical"
            else:
                transformed_dict[column_name] = "Other"
        return transformed_dict

    def _infer_column_types(self, dataframe):
        """
        Infers the data types of the columns in the given DataFrame.

        This function uses the FeatureMetadata class to extract metadata from the DataFrame,
        which includes inferred data types for each column. It then remaps these data types
        to the dbinfer metadata.yaml desired format using the `_remap_column_dtypes` method.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input DataFrame for which the column types are inferred.

        Returns:
        --------
        dict : A dictionary mapping column names to their inferred data types.
        """
        feature_metadata = FeatureMetadata.from_df(dataframe)
        column_name_types_map = self._remap_column_dtypes(dataframe, feature_metadata.to_dict())
        return column_name_types_map

    def _get_dtype_with_llm(self):
        raise NotImplementedError

    @staticmethod
    def _get_time_column(column_name_types_map):
        for key, value in column_name_types_map.items():
            if value == "DateTime":
                return key
        return None

    @staticmethod
    def _reorder_columns(df_reference, df_to_reorder):
        """
        Reorder columns in df_to_reorder to match the order of columns in df_reference,
        with any additional columns in df_to_reorder placed after the ordered columns.

        Parameters:
        -----------
        df_reference : pd.DataFrame
            DataFrame with the desired column order
        df_to_reorder : pd.DataFrame
            DataFrame to reorder

        Returns:
        --------
        pd.DataFrame: Reordered DataFrame
        """
        reference_columns = df_reference.columns.tolist()
        all_columns = df_to_reorder.columns.tolist()

        ordered_columns = [col for col in reference_columns if col in all_columns]
        remaining_columns = [col for col in all_columns if col not in reference_columns]
        new_order = ordered_columns + remaining_columns
        df_reordered = df_to_reorder[new_order]
        return df_reordered

    @staticmethod
    def _reorder_and_split_data(all_data, train_data):
        """
        Split the combined DataFrame back into train and test sets based on the original indices.

        Parameters:
        --------
        all_data : pd.DataFrame
            The combined DataFrame after transformations.
        train_data : pd.DataFrame
            The original training DataFrame.

        Returns:
        --------
        pd.DataFrame, pd.DataFrame: Transformed training and testing DataFrames.
        """
        # reorder columns post dfs
        all_data = DFSTransformer._reorder_columns(train_data, all_data)

        # split
        train_indices = len(train_data)
        train_data_transformed = all_data.loc[: train_indices - 1]
        test_data_transformed = all_data.loc[train_indices:]

        return train_data_transformed, test_data_transformed

    def _dataframe_to_rdb_dataset(
        self,
        dataframe: pd.DataFrame,
        name: str,
        target_column: str,
        time_column: str,
        column_type_dict: dict,
        path: Path,
        use_cat_vars_as_fks: bool,
    ):
        """
        Converts a DataFrame to a DBInfer-Bench (DBB) dataset and saves it to the specified path.
        Also generate the necessary schema and metadata required for a DBB dataset,
        including creating dummy validation and test tables. The dataset is then saved in the
        specified directory.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            The input DataFrame to be converted.
        name : str
            The name of the dataset.
        target_column : str
            The name of the target column.
        time_column : str
            The name of the time column.
        column_type_dict : dict
            A dictionary mapping column names to their data types.
        path : Path
            The directory where the dataset will be saved.
        use_cat_vars_as_fks: bool
            If categorical variables should be treated as IDs for FKs.
        """
        column_type_dict = {k: v for k, v in column_type_dict.items() if v != "null"}
        column_config = []
        if use_cat_vars_as_fks:
            fk_list = ["ID", "Categorical"]
        else:
            fk_list = ["ID"]
        for k, v in column_type_dict.items():
            if v in fk_list:
                column_config.append(
                    {
                        "name": k,
                        "dtype": "foreign_key",
                        "link_to": f"{k}.key",
                    }
                )
            else:
                column_config.append(
                    {
                        "name": k,
                        "dtype": _DTYPE_MAP[v],
                    }
                )

        path.mkdir(exist_ok=True, parents=True)
        dataframe.to_parquet(path / "data_train.pqt")
        # The current DFS pipeline requires the DBB dataset to have a training,
        # validation, and test table, so we fake the latter two.
        dummy = dataframe.iloc[:0]
        dummy.to_parquet(path / "data_validation.pqt")
        dummy.to_parquet(path / "data_test.pqt")

        column_schemas = [DBBColumnSchema.parse_obj(c) for c in column_config]
        table_schema = DBBTableSchema(
            name=name,
            source="data_train.pqt",
            format="parquet",
            columns=column_schemas,
            time_column=time_column,
        )
        task_meta = DBBTaskMeta(
            name=target_column,
            source=str("data_{split}.pqt"),
            format="parquet",
            columns=column_schemas,
            time_column=time_column,
            target_column=target_column,
            target_table=name,
            # These two arguments are not relevant for the DFS.
            # However they are required by the existing dbinfer pipeline.
            # Therefore, we provide arbitrary placeholder values for these args.
            evaluation_metric="rmse",
            task_type="regression",
        )

        dataset_meta = DBBRDBDatasetMeta(dataset_name=name, tables=[table_schema], tasks=[task_meta])

        save_pyd(dataset_meta, path / "metadata.yaml")

    def _rdb_dataset_to_dataframe(self, path: Path):
        """
        Extract the dataframe from a DBInfer-Bench dataset generated by DFS.
        """
        meta = load_pyd(DBBRDBDatasetMeta, path / "metadata.yaml")
        assert len(meta.tasks) == 1
        data = np.load(path / meta.tasks[0].source.format(split="train"), allow_pickle=True)
        df_data = {}
        for k, v in data.items():
            if v.ndim > 1:
                v = v.reshape((v.shape[0], -1))
                for i in range(v.shape[1]):
                    df_data[f"{k}_{i}"] = v[:, i]
            else:
                df_data[k] = v
        df = pd.DataFrame(df_data)
        return df

    def _make_dfs_config(self, depth: int):
        """
        Creates and returns the DFS configuration.

        Parameters:
        -----------
        depth : int
            The maximum depth for the DFS transformation.
        """
        return DFSPreprocessConfig.parse_obj(
            {
                "dfs": {
                    "max_depth": depth,
                    "use_cutoff_time": True,
                    # "engine": "dfs2sql",  # dbinfer doesn't offer dfs2sql engine yet; use tab2graph instead
                }
            }
        )

    def _launch_pre_dfs(self, path: Path, output_path: Path):
        """
        Launch the pre-DFS processing including
        dummy table creation, key mapping, etc.
        and output the result at a separate directory.
        """
        device = get_device_info()
        logging.info("Loading rdb data ...")
        dataset = dbb.load_rdb_data(path)

        logging.info("Creating preprocess for pre-DFS transform...")
        preprocess = RDBTransformPreprocess(_PRE_DFS_TRANSFORM_CONFIG)

        preprocess.run(dataset, output_path, device)

    def _launch_dfs(self, path: Path, output_path: Path, depth: int):
        """
        Launch DFS on processed dataset and output
        the result at a separate directory.
        """
        device = get_device_info()
        logging.info("Loading rdb data ...")
        dataset = dbb.load_rdb_data(path)

        logging.info("Running DFS...")
        preprocess = DFSPreprocess(self._make_dfs_config(depth))

        preprocess.run(dataset, output_path, device)

    def _run_dfs(self, input_df, df_name, target_column, time_column, column_type_dict, depth, use_cat_vars_as_fks):
        """
        Runs the Deep Feature Synthesis transformation on the input DataFrame.

        Parameters:
        -----------
        input_df : pd.DataFrame
            The input DataFrame to be transformed.
        df_name : str
            The name of the DataFrame.
        target_column : str
            The name of the target/label column.
        time_column : str
            The name of the time column.
        column_type_dict : dict
            A dictionary mapping column names to their data types.
        depth : int
            The depth of the DFS transformation.
        use_cat_vars_as_fks: bool
            If categorical variables should be treated as IDs for FKs.

        Returns:
        --------
        pd.DataFrame
            The transformed DataFrame.
        """
        self._dataframe_to_rdb_dataset(
            input_df,
            df_name,
            target_column,
            time_column,
            column_type_dict,
            Path("__workspace__/rdb"),
            use_cat_vars_as_fks,
        )
        self._launch_pre_dfs(Path("__workspace__/rdb"), Path("__workspace__/pre_dfs"))
        self._launch_dfs(Path("__workspace__/pre_dfs"), Path("__workspace__/dfs"), depth)
        output_df = self._rdb_dataset_to_dataframe(Path("__workspace__/dfs"))
        return output_df
