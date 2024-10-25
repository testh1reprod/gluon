import logging
import os
from collections import namedtuple
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
from gensim.utils import tokenize

from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

DeviceInfo = namedtuple("DeviceInfo", ["cpu_count", "gpu_devices"])


def get_device_info():
    if torch.cuda.is_available():
        gpu_devices = [f"cuda:{devid}" for devid in range(torch.cuda.device_count())]
    else:
        gpu_devices = []
    cpu_count = int(os.environ.get("NUM_VISIBLE_CPUS", os.cpu_count()))
    return DeviceInfo(cpu_count, gpu_devices)


def huggingface_run(model, data):
    if all(isinstance(x, str) for x in data) and any(len(x.split(" ")) > 10 for x in data):
        data = np.where(pd.isna(data), "", data)
        return model.encode(data).astype("float32")
    else:
        return np.zeros(len(data))


def glove_run_one_proc(model, data):
    embeddings = []
    if all(isinstance(x, str) for x in data) and any(len(x.split(" ")) > 10 for x in data):
        for text in data:
            token_list = list(tokenize(text))
            embed = model.get_mean_vector(token_list)
            embeddings.append(embed)
    else:
        return np.zeros(len(data))
    return np.stack(embeddings).astype('float32') 


class PretrainedEmbeddingTransformer(BaseFeatureTransformer):
    def __init__(self, model_name, **kwargs) -> None:
        self.model_name = model_name
        if torch.cuda.is_available():
            try:
                self.model = SentenceTransformer(self.model_name)
            except:
                logger.warning(f"No model {self.model_name} is found.")
            
        else:
            logger.warning(f"Cuda is not found. For an optimized user experience, we switched to the glove embeddings")
            self.model_name = "glove-twitter"
            self.dim = 100
            self.max_num_procs = 16
            try:
                self.model = api.load(f"{self.model_name}-{self.dim}")
            except:
                logger.warning(f"No model {self.model_name}-{self.dim} is found.")
            self.cpu_count = int(os.environ.get("NUM_VISIBLE_CPUS", os.cpu_count()))

    def _fit_dataframes(self, train_X: pd.DataFrame, train_y: pd.Series, **kwargs) -> None:
        pass

    def _transform_dataframes(self, train_X: pd.DataFrame, test_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        assert (
                train_X.columns.values.tolist() == test_X.columns.values.tolist()
            ), "The columns of the training set does not matach the columns of the test set"
        
        for series_name in train_X.columns.values.tolist():
            if torch.cuda.is_available():
                transformed_train_column = huggingface_run(self.model, np.transpose(train_X[series_name].to_numpy()).T)
                transformed_test_column = huggingface_run(self.model, np.transpose(test_X[series_name].to_numpy()).T)
            else:
                transformed_train_column = glove_run_one_proc(self.model, np.transpose(train_X[series_name].to_numpy()).T)
                transformed_test_column = glove_run_one_proc(self.model, np.transpose(test_X[series_name].to_numpy()).T)

            if transformed_train_column.any() and transformed_test_column.any():
                transformed_train_column = pd.DataFrame(transformed_train_column)
                transformed_test_column = pd.DataFrame(transformed_test_column)
                transformed_train_column.columns = transformed_test_column.columns = [
                    series_name + " " + str(i) for i in range(len(transformed_train_column.columns))
                ]
                train_X = pd.concat([train_X.drop([series_name], axis=1), transformed_train_column], axis=1)
                test_X = pd.concat([test_X.drop([series_name], axis=1), transformed_test_column], axis=1)
                
        return train_X, test_X