from typing import Mapping, Tuple, Optional, Dict
import logging
import ast
import re
import os
import traceback

import pandas as pd
import numpy as np
import sklearn
import dspy
import copy
from sklearn.model_selection import train_test_split
from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)
# pd.set_option("future.no_silent_downcasting", True)


def create_mappings(df_train: pd.DataFrame) -> Dict[str, Dict[int, str]]:
    """
    Creates a dictionary of mappings for categorical columns in the given dataframe.

    Parameters:
    df_train (pandas.DataFrame): The dataframe to create mappings for.

    Returns:
    Dict[str, Dict[int, str]]: A dictionary of mappings for categorical columns in the dataframe.
    """
    mappings = {}
    for col in df_train.columns:
        if df_train[col].dtype.name == "category" or df_train[col].dtype.name == "object":
            mappings[col] = dict(enumerate(df_train[col].astype("category").cat.categories))
    return mappings


def make_datasets_numeric(
    df_train: pd.DataFrame,
    df_test: Optional[pd.DataFrame],
    target_column: str,
    return_mappings: Optional[bool] = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[Dict[str, Dict[int, str]]]]:
    """
    Converts the categorical columns in the given training and test dataframes to integer values using mappings created from the training dataframe.

    Parameters:
    df_train (pandas.DataFrame): The training dataframe to convert.
    df_test (pandas.DataFrame, optional): The test dataframe to convert. Defaults to None.
    target_column (str): The name of the target column.
    return_mappings (bool, optional): Whether to return the mappings used for the conversion. Defaults to False.

    Returns:
    Tuple[pandas.DataFrame, Optional[pandas.DataFrame], Optional[Dict[str, Dict[int, str]]]]: The converted training dataframe, the converted test dataframe (if it exists), and the mappings used for the conversion (if `return_mappings` is True).
    """
    df_train = copy.deepcopy(df_train)
    df_train = df_train.infer_objects()
    if df_test is not None:
        df_test = copy.deepcopy(df_test)
        df_test = df_test.infer_objects()

    # Create the mappings using the train and test datasets
    mappings = create_mappings(df_train)

    # Apply the mappings to the train and test datasets
    non_target = [c for c in df_train.columns if c != target_column]
    df_train[non_target] = make_dataset_numeric(df_train[non_target], mappings)

    if df_test is not None:
        df_test[non_target] = make_dataset_numeric(df_test[non_target], mappings)

    if return_mappings:
        return df_train, df_test, mappings

    return df_train, df_test


def get_llm(model_str, kwargs) -> dspy.lm:
    if model_str == "bedrock-claude-3.5-sonnet":  # load credential from envrionment variables
        return dspy.AWSAnthropic(
            aws_provider=dspy.Bedrock(region_name=os.environ.get("AWS_DEFAULT_REGION")),
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        )
    elif model_str == "bedrock-claude-3-sonnet":  # load credential from envrionment variables
        return dspy.AWSAnthropic(
            aws_provider=dspy.Bedrock(region_name=os.environ.get("AWS_DEFAULT_REGION")),
            model="anthropic.claude-3-sonnet-20240229-v1:0",
        )
    elif model_str == "gpt-4-turbo":
        return dspy.OpenAI(model="gpt-4-turbo", api_key=kwargs.get("openai_api_key", os.environ.get("OPENAI_API_KEY")))
    else:
        raise


def extract_python_code(raw_output: str) -> str:
    return re.search(r"```python\s*(.*?)\s*```", raw_output, re.DOTALL).group(1)


def validate_python_code(code_string: str) -> Tuple[bool, str]:
    try:
        ast.parse(code_string)
    except SyntaxError as e:
        return False, f"Syntax error in the code: {e}"
    return True, "Code has no syntax error"


def construct_context(context: list) -> str:
    return "\n".join(
        [f"-----\nGenerated code:\n\n {code} \n\n Observed results: {status}\n-----" for code, status in context]
    )


def exec_python_code(code: str, df: pd.DataFrame):
    access_scope = {"df": df, "pd": pd, "np": np, "sklearn": sklearn}
    try:
        ast.parse(code)
    except Exception as e:
        raise

    locals_dict = {"df": df}
    try:
        exec(code, globals(), locals_dict, access_scope=access_scope)
        assert "df" in locals_dict
        return locals_dict["df"]
    except Exception as e:
        raise


def eval_xgboost(X_train, X_test, y_train, y_test, task_type) -> float:
    from xgboost import XGBClassifier, XGBRegressor
    from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

    if task_type == "binary":
        metric_func = roc_auc_score
        model = XGBClassifier()
    elif task_type == "multiclass":
        metric_func = accuracy_score
        model = XGBClassifier()
    else:
        assert task_type == "regression"
        metric_func = r2_score
        model = XGBRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return metric_func(y_test, y_pred)


class ExtractInternalKnowledge(dspy.Signature):
    """Given task description and sample data, first guess the domain/application of
    the data and then generate at most 3 feature eningeering tricks specific to this
    dataset and domain. Don't do feature selection and dimension reduction."""

    task_desc = dspy.InputField(desc="Task description")
    sample_data = dspy.InputField(desc="Sample rows and columns of the dataset")
    context = dspy.InputField(desc="Observations from previous solutions")
    feature_description = dspy.OutputField(
        desc="A list of at most 3 features engineering tricks specific to this dataset"
    )


class DescriptionToCode(dspy.Signature):
    """Assume a dataframe called 'df' has been loaded in memory, given some sample data of 'df',
    a list of feature engineering descriptions, generate python codeblock for each feature engineering
    description and add these features as new columns to 'df'"""

    sample_data = dspy.InputField(desc="Sample rows and columns of the dataset")
    feature_description = dspy.InputField(desc="A list of descriptions of useful features engineering method")
    context = dspy.InputField(desc="Observations from previous solutions")
    python_code = dspy.OutputField(
        desc="Python code to transform dataset into features. Don't use function in the code."
    )


class MultiStepGen(dspy.Module):

    def __init__(self, max_iter: int):
        super().__init__()
        self.max_iter = max_iter
        self.gen_domain_feature_desc = dspy.Predict(ExtractInternalKnowledge)
        self.gen_code_from_desc = dspy.Predict(DescriptionToCode)

    def forward(self, df, task_desc) -> Optional[str]:
        sample_data_desc = df.head(5).to_string()
        context = []
        for _ in range(self.max_iter):
            context_str = construct_context(context)

            feat_desc = self.gen_domain_feature_desc(
                task_desc=task_desc,
                sample_data=sample_data_desc,
                context=context_str,
            ).feature_description

            code = self.gen_code_from_desc(
                sample_data=sample_data_desc,
                feature_description=feat_desc,
                context=context_str,
            ).python_code

            try:
                code = extract_python_code(code)
            except Exception as e:
                context.append((code, "error extracting python code {e}"))
                continue

            is_valid, message = validate_python_code(code)

            if is_valid:
                try:
                    exec_python_code(code, df)
                    context.append((code, "successfully run"))
                    return code
                except Exception as e:
                    tb = traceback.format_exc()
                    context.append((code, f"Runtime Error: {e}. Trace: {tb}"))
                    continue
            else:
                context.append((code, message))
                continue


class SimpleGenTransformer(BaseFeatureTransformer):

    identifier = "simple-gen"

    def __init__(
        self,
        llm_model: str = "bedrock-claude-3.5-sonnet",
        num_iterations: int = 3,
        max_num_retries: int = 3,
        eval_model: str = "xgboost",
        **kwargs,
    ) -> None:
        assert eval_model == "xgboost", eval_model
        self.iterations = num_iterations
        self.max_num_retries = max_num_retries
        self.eval_model = eval_model
        self.metadata = {"transformer": "dspy-basic"}
        self.llm_model = llm_model
        llm = get_llm(llm_model, kwargs)
        dspy.configure(lm=llm)

    def _fit_dataframes(
        self,
        train_X: pd.DataFrame,
        train_y: pd.Series,
        target_column_name: str,
        problem_type: str,
        dataset_description: str,
        **kwargs,
    ) -> None:

        assert problem_type in ("binary", "multiclass", "regression")
        categorical_target = not pd.api.types.is_numeric_dtype(train_y)
        if categorical_target:
            encoded_y, _ = train_y.factorize()

        train_y = encoded_y if categorical_target else train_y

        # comptue baseline performance without any new features
        best_code = ""
        X_train_internal, X_test_internal, y_train_internal, y_test_internal = train_test_split(
            train_X, train_y, test_size=0.2, random_state=42
        )
        transformed_train_X, transformed_test_X = make_datasets_numeric(
            X_train_internal, X_test_internal, target_column_name
        )
        best_perf = eval_xgboost(
            transformed_train_X,
            transformed_test_X,
            y_train_internal,
            y_test_internal,
            problem_type,
        )

        for _ in range(self.iterations):
            df_train = train_X.copy()
            multi_step_gen = MultiStepGen(max_iter=self.max_num_retries)
            code = multi_step_gen(df_train, dataset_description)

            if code is not None:
                transformed_train_X = exec_python_code(code, X_train_internal)
                transformed_test_X = exec_python_code(code, X_test_internal)
                transformed_train_X, transformed_test_X = make_datasets_numeric(
                    transformed_train_X, transformed_test_X, target_column_name
                )
                perf = eval_xgboost(
                    transformed_train_X,
                    transformed_test_X,
                    y_train_internal,
                    y_test_internal,
                    problem_type,
                )
                if perf > best_perf:
                    best_code = code
                    best_perf = perf

        self.code = best_code

        if self.code is None:
            logger.info("SimpleGen doesn't generate any good features")
            logger.info(self.code)
        else:
            logger.info("SimpleGen generated features:\n")
            logger.info(self.code)

    def _transform_dataframes(self, train_X: pd.DataFrame, test_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.code == "":
            return train_X, test_X
        transformed_train_X = exec_python_code(self.code, train_X)
        transformed_test_X = exec_python_code(self.code, test_X)
        transformed_train_X, transformed_test_X = make_datasets_numeric(transformed_train_X, transformed_test_X, "")
        return transformed_train_X, transformed_test_X

    def get_metadata(self) -> Mapping:
        return self.metadata
