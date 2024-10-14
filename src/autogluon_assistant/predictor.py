"""Predictors solve tabular prediction tasks"""

import logging
from collections import defaultdict
from typing import Any, Dict

import numpy as np
from autogluon.common.features.feature_metadata import FeatureMetadata
from autogluon.core.metrics import make_scorer
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.metrics import mean_squared_log_error

from .task import TabularPredictionTask

logger = logging.getLogger(__name__)

BINARY_PROBA_INDICATORS = [
    "logloss",
    "log loss",
    "logarithmic loss",
    "ROC curve",
    "AUCROC",
    "Gini Coefficient",
]


def rmsle_func(y_true, y_pred, **kwargs):
    return np.sqrt(mean_squared_log_error(y_true, y_pred, **kwargs))


root_mean_square_logarithmic_error = make_scorer(
    "root_mean_square_logarithmic_error",
    rmsle_func,
    optimum=1,
    greater_is_better=False,
)


class Predictor:
    def fit(self, task: TabularPredictionTask) -> "Predictor":
        return self

    def predict(self, task: TabularPredictionTask) -> Any:
        raise NotImplementedError

    def fit_predict(self, task: TabularPredictionTask) -> Any:
        return self.fit(task).predict(task)


class AutogluonTabularPredictor(Predictor):

    def __init__(self, config: Any):
        self.config = config
        self.metadata: Dict[str, Any] = defaultdict(dict)
        self.tabular_predictor: TabularPredictor = None

    def save_dataset_details(self, task: TabularPredictionTask) -> None:
        for key, data in (
            ("train", task.train_data),
            ("test", task.test_data),
        ):
            self.metadata["dataset_summary"][key] = data.describe().to_dict()
            self.metadata["feature_metadata_raw"][key] = FeatureMetadata.from_df(data).to_dict()
            self.metadata["feature_missing_values"][key] = (data.isna().sum() / len(data)).to_dict()

    def describe(self) -> Dict[str, Any]:
        return dict(self.metadata)

    def fit(self, task: TabularPredictionTask) -> "AutogluonTabularPredictor":
        """Trains an AutoGluon TabularPredictor with parsed arguments. Saves trained predictor to
        `self.predictor`.

        Raises
        ------
        Exception
            TabularPredictor fit failures
        """
        eval_metric = task.eval_metric
        if eval_metric == "root_mean_squared_logarithmic_error":
            eval_metric = root_mean_square_logarithmic_error

        predictor_init_kwargs = {
            "learner_kwargs": {"ignored_columns": task.columns_in_train_but_not_test},
            "label": task.label_column,
            "eval_metric": eval_metric,
            **self.config.predictor_init_kwargs,
        }
        predictor_fit_kwargs = self.config.predictor_fit_kwargs.copy()

        logger.info("Fitting AutoGluon TabularPredictor")
        logger.info(f"predictor_init_kwargs: {predictor_init_kwargs}")
        logger.info(f"predictor_fit_kwargs: {predictor_fit_kwargs}")

        if predictor_fit_kwargs.get("dynamic_stacking", False):
            # Use config value for num_stack_levels
            # Default 1 if dynamic_stacking is True
            if predictor_fit_kwargs["num_stack_levels"] == 0:
                predictor_fit_kwargs["num_stack_levels"] = 1
            logger.info(
                f"Dynamic stacking is enabled; setting num_stack_levels={predictor_fit_kwargs['num_stack_levels']}"
            )

        self.metadata |= {
            "predictor_init_kwargs": predictor_init_kwargs,
            "predictor_fit_kwargs": predictor_fit_kwargs,
        }
        self.save_dataset_details(task)
        self.predictor = TabularPredictor(**predictor_init_kwargs).fit(task.train_data, **predictor_fit_kwargs)

        self.metadata["leaderboard"] = self.predictor.leaderboard().to_dict()
        return self

    def predict(self, task: TabularPredictionTask) -> TabularDataset:
        """Calls `TabularPredictor.predict` or `TabularPredictor.predict_proba` on `self.transformed_test_data`.
        Saves predictions to `self.predictions`.

        Raises
        ------
        Exception
            `TabularPredictor.predict` fails
        """
        if self.predictor.problem_type == "binary" and any(
            indicator in task.metadata["description"] for indicator in BINARY_PROBA_INDICATORS
        ):
            # TODO: Turn BINARY_PROBA_INDICATORS into an llm call in the future (add test cases)
            return self.predictor.predict_proba(task.test_data)[1]

        elif self.predictor.problem_type == "multiclass" and len(task.output_columns) != 2:
            # TODO: match prediction columns with submission columns
            predictions = self.predictor.predict_proba(task.test_data)
            if len(predictions.columns) != len(task.output_columns) - 1:
                raise Exception(
                    "Predicted number of multiclass classes does not match number in sample submission file"
                )
            return predictions

        return self.predictor.predict(task.test_data)
