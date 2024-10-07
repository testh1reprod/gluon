from typing import Tuple, Dict, Optional, List
import pydantic
import numpy as np
import logging
from .base import BaseFeatureTransformer
import pandas as pd

from tqdm import tqdm
import torch
import torch.multiprocessing as mp


logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

import gensim.downloader as api
from gensim.utils import tokenize

from collections import namedtuple
import os
from gensim.test.utils import common_texts

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer

DeviceInfo = namedtuple('DeviceInfo', ['cpu_count', 'gpu_devices'])

def get_device_info():
    if torch.cuda.is_available():
        gpu_devices = [f"cuda:{devid}" for devid in range(torch.cuda.device_count())]
    else:
        gpu_devices = []
    cpu_count = int(os.environ.get("NUM_VISIBLE_CPUS", os.cpu_count()))
    return DeviceInfo(cpu_count, gpu_devices)

def _run_one_proc(proc_id, model, dim, data):
    embeddings = []
    if proc_id == 0:
        iterator = tqdm(data)
    else:
        iterator = data
    
    for text in iterator:
        if not isinstance(text, str):
            embed = np.zeros(dim)
        else:
            token_list = list(tokenize(text))
            if len(token_list) == 0:
                embed = np.zeros(dim)
            else:
                embed = model.get_mean_vector(token_list)
        embeddings.append(embed)
    
    #embeddings = model.encode(data)
    return np.stack(embeddings).astype('float32') #embeddings.astype('float32') #

class GloveTextEmbeddingTransformer(BaseFeatureTransformer):
    def __init__(self, **kwargs) -> None:
        self.model_name = "glove-twitter"
        self.dim = 100
        self.max_num_procs = 16
        self.model = api.load(f"{self.model_name}-{self.dim}")
        #self.model = SentenceTransformer("all-MiniLM-L6-v2")
        #assert self.model.vector_size == self.dim, \
        #    "Dimension of the model does not match the config."
        self.cpu_count = int(os.environ.get("NUM_VISIBLE_CPUS", os.cpu_count()))


    def _fit_dataframes(self, train_X: pd.DataFrame, train_y: pd.Series, **kwargs) -> None:
        pass
        

    def _transform_dataframes(self, train_X: pd.DataFrame, test_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        transformed_train_X = _run_one_proc(0, self.model, self.dim, np.transpose(train_X['boilerplate'].to_numpy()).T)
        transformed_test_X = _run_one_proc(0, self.model, self.dim, np.transpose(test_X['boilerplate'].to_numpy()).T)
        return pd.concat([train_X.drop(['boilerplate'], axis=1), pd.DataFrame(transformed_train_X)], axis=1), pd.concat([test_X.drop(['boilerplate'], axis=1), pd.DataFrame(transformed_test_X)], axis=1) 
    