from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
import logging
from .base import BaseFeatureTransformer
import pandas as pd
import torch

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")
import re

from collections import namedtuple
import os
from sentence_transformers import SentenceTransformer

DeviceInfo = namedtuple("DeviceInfo", ["cpu_count", "gpu_devices"])


def get_device_info():
    if torch.cuda.is_available():
        gpu_devices = [f"cuda:{devid}" for devid in range(torch.cuda.device_count())]
    else:
        gpu_devices = []
    cpu_count = int(os.environ.get("NUM_VISIBLE_CPUS", os.cpu_count()))
    return DeviceInfo(cpu_count, gpu_devices)


def _run_one_proc(model, data):
    if all(isinstance(x, str) for x in data) and any(len(x.split(" ")) > 10 for x in data):
        data = np.where(pd.isna(data), "", data)
        return model.encode(data).astype("float32")
    else:
        return np.zeros(len(data))


class PretrainedEmbeddingTransformer(BaseFeatureTransformer):
    def __init__(self, model_name, **kwargs) -> None:
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(self.model_name)
        except:
            logger.warning(f"No model {self.model_name} is found.")

    def _fit_dataframes(self, train_X: pd.DataFrame, train_y: pd.Series, **kwargs) -> None:
        pass

    def _transform_dataframes(self, train_X: pd.DataFrame, test_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        assert (
            train_X.columns.values.tolist() == test_X.columns.values.tolist()
        ), "The columns of the training set does not matach the columns of the test set"
        for series_name in train_X.columns.values.tolist():
            transformed_train_column = _run_one_proc(self.model, np.transpose(train_X[series_name].to_numpy()).T)
            transformed_test_column = _run_one_proc(self.model, np.transpose(test_X[series_name].to_numpy()).T)

            if transformed_train_column.any() and transformed_test_column.any():
                transformed_train_column = pd.DataFrame(transformed_train_column)
                transformed_test_column = pd.DataFrame(transformed_test_column)
                transformed_train_column.columns = transformed_test_column.columns = [
                    series_name + " " + str(i) for i in range(len(transformed_train_column.columns))
                ]
                train_X = pd.concat([train_X.drop([series_name], axis=1), transformed_train_column], axis=1)
                test_X = pd.concat([test_X.drop([series_name], axis=1), transformed_test_column], axis=1)

        return train_X, test_X
