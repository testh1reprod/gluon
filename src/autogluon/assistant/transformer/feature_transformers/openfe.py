import gc
import hashlib
import logging
import random
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Mapping, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from tqdm import tqdm

from .base import BaseFeatureTransformer

warnings.filterwarnings(action="ignore")

logger = logging.getLogger(__name__)

try:
    from openfe import OpenFE, transform
except ImportError:
    raise ImportError(
        "OpenFE required for feature generation but not installed. Please install with: `pip install 'openfe>=0.0.12,<0.1'`"
    )


def _subsample(iterators, min_chunk_size: int = 100):
    data = list(iterators) if not isinstance(iterators, list) else iterators
    np.random.RandomState(42).shuffle(data)
    data_blocks = []
    block_size = min_chunk_size
    index = 0
    while index < len(data):
        next_index = min(index + block_size, len(data))
        data_blocks.append(data[:next_index])
        index = next_index
        block_size = int(block_size * 2)
    return data_blocks


class AssistantOpenFE(OpenFE):
    def fit(
        self,
        data: pd.DataFrame,
        label: pd.DataFrame,
        task: str = None,
        train_index=None,
        val_index=None,
        candidate_features_list=None,
        init_scores=None,
        categorical_features=None,
        metric=None,
        drop_columns=None,
        n_data_blocks=8,
        min_candidate_features=2000,
        feature_boosting=False,
        stage1_metric="predictive",
        stage2_metric="gain_importance",
        stage2_params=None,
        is_stage1=True,
        n_repeats=1,
        tmp_save_path="./openfe_tmp_data_xx.feather",
        n_jobs=1,
        seed=1,
        verbose=True,
        skip_stage2: bool = True,
    ):
        assert stage2_metric in ["gain_importance", "permutation"]
        assert stage1_metric in ["predictive", "corr", "mi"]
        if metric:
            assert metric in ["binary_logloss", "multi_logloss", "auc", "rmse"]
        np.random.seed(seed)
        random.seed(seed)

        self.data = data
        self.label = label
        self.metric = metric
        self.drop_columns = drop_columns
        self.n_data_blocks = n_data_blocks
        self.min_candidate_features = min_candidate_features
        self.stage1_metric = stage1_metric
        self.stage2_metric = stage2_metric
        self.feature_boosting = feature_boosting
        self.stage2_params = stage2_params
        self.is_stage1 = is_stage1
        self.n_repeats = n_repeats
        self.tmp_save_path = tmp_save_path
        self.n_jobs = n_jobs
        self.seed = seed
        self.verbose = verbose

        self.data_to_dataframe()
        self.task = self.get_task(task)
        self.process_label()
        self.process_and_save_data()

        self.metric = self.get_metric(metric)
        self.categorical_features = self.get_categorical_features(categorical_features)
        self.candidate_features_list = self.get_candidate_features(candidate_features_list)
        self.train_index, self.val_index = self.get_index(train_index, val_index)
        self.init_scores = self.get_init_score(init_scores)

        logger.info(f"The number of candidate features is {len(self.candidate_features_list)}")
        logger.info("Start stage I selection.")
        self.candidate_features_list = self.stage1_select()
        self.new_features_list = self.candidate_features_list
        if not skip_stage2:
            logger.info(f"The number of remaining candidate features is {len(self.candidate_features_list)}")
            logger.info("Start stage II selection.")
            self.new_features_scores_list = self.stage2_select()
            self.new_features_list = [feature for feature, _ in self.new_features_scores_list]
            final_nodes = []
            for node, score in self.new_features_scores_list:
                final_nodes.append(node.name)
                node.delete()
            res = self.new_features_list
        else:
            final_nodes = []
            for node in self.candidate_features_list:
                final_nodes.append(node.name)
                node.delete()
            res = self.candidate_features_list

        return res

    def stage1_select(self, ratio=0.5):
        if self.is_stage1 is False:
            train_index = _subsample(self.train_index, self.n_data_blocks)[0]
            val_index = _subsample(self.val_index, self.n_data_blocks)[0]
            self.data = self.data.loc[train_index + val_index]
            self.label = self.label.loc[train_index + val_index]
            self.train_index = train_index
            self.val_index = val_index
            return [[f, 0] for f in self._calculate(self.candidate_features_list, train_index, val_index)]
        train_index_samples = _subsample(self.train_index, self.n_data_blocks)
        val_index_samples = _subsample(self.val_index, self.n_data_blocks)
        idx = 0
        train_idx = train_index_samples[idx]
        val_idx = val_index_samples[idx]
        idx += 1
        results = self._calculate_and_evaluate(self.candidate_features_list, train_idx, val_idx)
        candidate_features_scores = sorted(results, key=lambda x: x[1], reverse=True)
        candidate_features_scores = self.delete_same(candidate_features_scores)

        while idx != len(train_index_samples):
            n_reserved_features = max(
                int(len(candidate_features_scores) * ratio),
                min(len(candidate_features_scores), self.min_candidate_features),
            )
            train_idx = train_index_samples[idx]
            val_idx = val_index_samples[idx]
            idx += 1
            if n_reserved_features <= self.min_candidate_features:
                train_idx = train_index_samples[-1]
                val_idx = val_index_samples[-1]
                idx = len(train_index_samples)
                logger.info("Meet early-stopping in successive feature-wise halving.")
            candidate_features_list = [item[0] for item in candidate_features_scores[:n_reserved_features]]
            del candidate_features_scores[n_reserved_features:]
            gc.collect()

            results = self._calculate_and_evaluate(candidate_features_list, train_idx, val_idx)
            candidate_features_scores = sorted(results, key=lambda x: x[1], reverse=True)

        return_results = [item[0] for item in candidate_features_scores if item[1] > 0]
        if not return_results:
            return_results = [item[0] for item in candidate_features_scores[:100]]
        return return_results

    def _calculate_multiprocess(self, candidate_features, train_idx, val_idx):
        try:
            results = []
            base_features = {"openfe_index"}
            for candidate_feature in candidate_features:
                base_features |= set(candidate_feature.get_fnode())

            data = pd.read_feather(self.tmp_save_path, columns=list(base_features)).set_index("openfe_index")
            data_temp = data.loc[train_idx + val_idx]
            del data
            gc.collect()

            for candidate_feature in candidate_features:
                candidate_feature.calculate(data_temp, is_root=True)
                candidate_feature.f_delete()
                results.append(candidate_feature)
            return results
        except:
            print(traceback.format_exc())
            exit()

    def _calculate(self, candidate_features, train_idx, val_idx):
        results = []
        length = int(np.ceil(len(candidate_features) / self.n_jobs / 4))
        n = int(np.ceil(len(candidate_features) / length))
        random.shuffle(candidate_features)
        with ProcessPoolExecutor(max_workers=self.n_jobs) as ex:
            with tqdm(total=n) as progress:
                for i in range(n):
                    if i == (n - 1):
                        future = ex.submit(
                            self._calculate_multiprocess, candidate_features[i * length :], train_idx, val_idx
                        )
                    else:
                        future = ex.submit(
                            self._calculate_multiprocess,
                            candidate_features[i * length : (i + 1) * length],
                            train_idx,
                            val_idx,
                        )
                    future.add_done_callback(lambda p: progress.update())
                    results.append(future)
                res = []
                for r in results:
                    res.extend(r.result())
        return res

    def _evaluate(self, candidate_feature, train_y, val_y, train_init, val_init, init_metric):
        train_x = pd.DataFrame(candidate_feature.data.loc[train_y.index])
        val_x = pd.DataFrame(candidate_feature.data.loc[val_y.index])
        if self.stage1_metric == "predictive":
            params = {
                "n_estimators": 100,
                "importance_type": "gain",
                "num_leaves": 16,
                "seed": 1,
                "deterministic": True,
                "n_jobs": 1,
                "verbosity": -1,
            }
            if self.metric is not None:
                params.update({"metric": self.metric})
            if self.task == "classification":
                gbm = lgb.LGBMClassifier(**params)
            else:
                gbm = lgb.LGBMRegressor(**params)
            gbm.fit(
                train_x,
                train_y.values.ravel(),
                init_score=train_init,
                eval_init_score=[val_init],
                eval_set=[(val_x, val_y.values.ravel())],
                callbacks=[lgb.early_stopping(3, verbose=False)],
            )
            key = list(gbm.best_score_["valid_0"].keys())[0]
            if self.metric in ["auc"]:
                score = gbm.best_score_["valid_0"][key] - init_metric
            else:
                score = init_metric - gbm.best_score_["valid_0"][key]
        elif self.stage1_metric == "corr":
            score = np.corrcoef(
                pd.concat([train_x, val_x], axis=0).fillna(0).values.ravel(),
                pd.concat([train_y, val_y], axis=0).fillna(0).values.ravel(),
            )[0, 1]
            score = abs(score)
        elif self.stage1_metric == "mi":
            if self.task == "regression":
                r = mutual_info_regression(
                    pd.concat([train_x, val_x], axis=0).replace([np.inf, -np.inf], 0).fillna(0),
                    pd.concat([train_y, val_y], axis=0).values.ravel(),
                )
            else:
                r = mutual_info_classif(
                    pd.concat([train_x, val_x], axis=0).replace([np.inf, -np.inf], 0).fillna(0),
                    pd.concat([train_y, val_y], axis=0).values.ravel(),
                )
            score = r[0]
        else:
            raise NotImplementedError("Cannot recognize filter_metric %s." % self.stage1_metric)
        return score

    def _calculate_and_evaluate_multiprocess(self, candidate_features, train_idx, val_idx):
        results = []
        base_features = {"openfe_index"}
        for candidate_feature in candidate_features:
            base_features |= set(candidate_feature.get_fnode())

        data = pd.read_feather(self.tmp_save_path, columns=list(base_features)).set_index("openfe_index")
        data_temp = data.loc[train_idx + val_idx]
        del data
        gc.collect()

        train_y = self.label.loc[train_idx]
        val_y = self.label.loc[val_idx]
        train_init = self.init_scores.loc[train_idx]
        val_init = self.init_scores.loc[val_idx]
        init_metric = self.get_init_metric(val_init, val_y)
        for candidate_feature in candidate_features:
            candidate_feature.calculate(data_temp, is_root=True)
            score = self._evaluate(candidate_feature, train_y, val_y, train_init, val_init, init_metric)
            candidate_feature.delete()
            results.append([candidate_feature, score])
        return results

    def _calculate_and_evaluate(self, candidate_features, train_idx, val_idx):
        results = []
        length = int(np.ceil(len(candidate_features) / self.n_jobs / 4))
        n = int(np.ceil(len(candidate_features) / length))
        random.shuffle(candidate_features)
        for f in candidate_features:
            f.delete()

        with tqdm(total=n) as progress:
            for i in range(n):
                if i == (n - 1):
                    future = self._calculate_and_evaluate_multiprocess(
                        candidate_features[i * length :], train_idx, val_idx
                    )
                else:
                    future = self._calculate_and_evaluate_multiprocess(
                        candidate_features[i * length : (i + 1) * length], train_idx, val_idx
                    )
                results.append(future)
                progress.update()

        res = []
        for r in results:
            res.extend(r)

        return res


class OpenFETransformer(BaseFeatureTransformer):
    identifier = "openfe"

    def __init__(self, n_jobs: int = 1, num_features_to_keep: int = 10, **kwargs) -> None:
        self.n_jobs = n_jobs
        self.num_features_to_keep = num_features_to_keep
        self.column_name_decodings: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {"transformer": "OpenFE"}

    def _fit_dataframes(self, train_X: pd.DataFrame, train_y: pd.Series, **kwargs) -> None:
        train_y_df = pd.DataFrame(train_y, index=train_X.index)

        problem_type = kwargs.get("problem_type")
        if problem_type is not None:
            problem_type = "regression" if problem_type == "regression" else "classification"

        # Fill columns of type 'object' with nan values with 'unknonwn'. OpenFE doesn't handle this case.
        object_columns = train_X.select_dtypes(include="object").columns
        train_X[object_columns] = train_X[object_columns].fillna("unknown")

        train_X_renamed = self.encode_column_names(train_X)
        self.features = AssistantOpenFE().fit(
            data=train_X_renamed,
            label=train_y_df,
            task=problem_type,
            n_jobs=self.n_jobs,
        )
        train_X = self.decode_column_names(train_X_renamed)

    def _transform_dataframes(self, train_X: pd.DataFrame, test_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        object_columns = train_X.select_dtypes(include="object").columns
        train_X[object_columns] = train_X[object_columns].fillna("unknown")
        test_X[object_columns] = test_X[object_columns].fillna("unknown")

        train_X_renamed = self.encode_column_names(train_X)
        test_X_renamed = self.encode_column_names(test_X)

        transformed_train_X, transformed_test_X = transform(
            train_X_renamed,
            test_X_renamed,
            self.features[: self.num_features_to_keep],
            n_jobs=self.n_jobs,
        )

        transformed_train_X = self.decode_column_names(transformed_train_X)
        transformed_test_X = self.decode_column_names(transformed_test_X)

        return transformed_train_X, transformed_test_X

    def get_metadata(self) -> Mapping:
        return self.metadata

    def hash_feature_name(self, feature_name) -> str:
        return hashlib.md5(feature_name.encode("utf-8")).hexdigest()[:8]

    def encode_column_names(self, X: pd.DataFrame) -> pd.DataFrame:
        new_column_names = []
        for column in X.columns:
            encoded_column_name = self.hash_feature_name(column)
            self.column_name_decodings[encoded_column_name] = column
            new_column_names.append(encoded_column_name)
        X.columns = new_column_names
        return X

    def decode_column_names(self, X: pd.DataFrame) -> pd.DataFrame:
        X.columns = [self.column_name_decodings.get(column, column) for column in X.columns]
        return X
