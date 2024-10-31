import copy
import logging
import traceback
from typing import Tuple

import pandas as pd

from autogluon_assistant.task import TabularPredictionTask
from autogluon_assistant.transformer.base import BaseTransformer, TransformTimeoutError

logger = logging.getLogger(__name__)


class BaseFeatureTransformer(BaseTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _fit_dataframes(self, train_X: pd.DataFrame, train_y: pd.Series, **kwargs) -> None:
        raise NotImplementedError

    def fit(self, task: TabularPredictionTask) -> "BaseFeatureTransformer":
        try:
            train_x = task.train_data.drop(
                columns=task.columns_in_train_but_not_test + [task.test_id_column],
                errors="ignore",
            )
            train_y = task.train_data[task.label_column]

            self._fit_dataframes(
                train_X=train_x,
                train_y=train_y,
                target_column_name=task.label_column,
                problem_type=task.problem_type,
                dataset_description=task.metadata["description"],
            )
        except TransformTimeoutError:
            logger.warning(f"FeatureTransformer {self.__class__.__name__} timed out.")
        except Exception:
            logger.warning(f"FeatureTransformer {self.__class__.__name__} failed to fit.")
            logger.warning(traceback.format_exc())
        finally:
            return self

    def _transform_dataframes(self, train_X: pd.DataFrame, test_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        try:
            train_x = task.train_data.drop(
                columns=task.columns_in_train_but_not_test + [task.test_id_column],
                errors="ignore",
            )
            train_y = task.train_data[task.label_column]
            test_x = task.test_data.drop(columns=[task.test_id_column])

            train_x, test_x = self._transform_dataframes(train_X=train_x, test_X=test_x)

            # add back id and label columns
            transformed_train_data = pd.concat([train_x, train_y.rename(task.label_column)], axis=1)
            if task.test_id_column in task.train_data.columns:
                transformed_train_data = pd.concat(
                    [transformed_train_data, task.train_data[task.test_id_column]], axis=1
                )
            transformed_test_data = pd.concat([test_x, task.test_data[task.test_id_column]], axis=1)

            task = copy.deepcopy(task)
            task.train_data = transformed_train_data
            task.test_data = transformed_test_data
        except:
            logger.warning(f"FeatureTransformer {self.__class__.__name__} failed to transform.")
        finally:
            return task
