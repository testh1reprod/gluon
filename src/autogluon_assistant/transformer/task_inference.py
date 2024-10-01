import difflib
import logging
from typing import Dict, List, Type, Union

import numpy as np
import pandas as pd
from autogluon.core.utils.utils import infer_problem_type
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from autogluon_assistant.llm import AssistantChatOpenAI
from autogluon_assistant.llm.prompts import (
    basic_intro_prompt,
    basic_system_prompt,
    columns_in_train_not_test_template,
    data_description_template,
    eval_metric_prompt,
    evaluation_description_template,
    format_instructions_template,
    infer_test_id_column_template,
    parse_fields_template,
    task_files_template,
    zip_file_prompt,
)
from autogluon_assistant.task import TabularPredictionTask

from .base import BaseTransformer

logger = logging.getLogger(__name__)


class FilenameTraits(BaseModel):
    train: str = Field(description="train data file name")
    test: str = Field(description="test data file name")
    output: str = Field(description="sample submission file name")


class LabelColumnTrait(BaseModel):
    label_column: str = Field(description="label or target column name")


class IdColumnTrait(BaseModel):
    id_column: str = Field(description="Id column name")


class SubmissionColumnsTraits(BaseModel):
    submission_columns: list[str] = Field(description="column names in valid submission")


class EvalMetricTrait(BaseModel):
    eval_metric: str = Field(description=("Evaluation metric to be used in scoring a submission in the competition."))


class LLMParserTransformer(BaseTransformer):
    """Parses data and metadata of a task with the aid of an instruction-tuned LLM."""

    traits: Type[BaseModel]

    def __init__(self, llm: AssistantChatOpenAI, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm

    @property
    def parser(self):
        return self._get_parser_for_traits(self.traits)

    def _chat_and_parse_prompt_output(
        self,
        prompt: Union[str, List[str]],
        system_prompt: str,
    ) -> Dict[str, str]:
        """Chat with the LLM and parse the output"""
        try:
            if isinstance(prompt, list):
                prompt = "\n\n".join(prompt)

            chat_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=prompt),
                ]
            )
            logger.debug(f"LLM chat_prompt:\n{chat_prompt.format_messages()}")
            output = self.llm(chat_prompt.format_messages())
            logger.debug(f"LLM output:\n{output}")

            return self.parser.parse(output.content)
        except OutputParserException as e:
            logger.error(f"Failed to parse output: {e}")
            logger.error(self.llm.describe())  # noqa
            raise e

    @staticmethod
    def _get_parser_for_traits(traits: BaseModel) -> StructuredOutputParser:
        response_schemas = [
            ResponseSchema(name=field_name, description=field_dict["description"])
            for field_name, field_dict in traits.schema()["properties"].items()
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)


class ProblemTypeInferenceTransformer(BaseTransformer):
    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        task.metadata["problem_type"] = infer_problem_type(task.train_data[task.label_column], silent=True)
        return task


class FilenameInferenceTransformer(LLMParserTransformer):
    """Uses an LLM to locate the filenames of the train, test, and output data,
    and assigns them to the respective properties of the task.
    """

    traits = FilenameTraits

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        composite_prompt = [
            basic_intro_prompt,
            data_description_template.format(task.data_description),
            task_files_template.format(task.get_filenames()),
            zip_file_prompt,
            parse_fields_template.format(" ".join(self.traits.__fields__.keys())),
            format_instructions_template.format(self.parser.get_format_instructions()),
        ]
        parser_output = self._chat_and_parse_prompt_output(composite_prompt, basic_system_prompt)

        task.train_data = parser_output["train"]
        task.test_data = parser_output["test"]
        task.output_data = parser_output["output"]

        return task


class LabelColumnInferenceTransformer(LLMParserTransformer):
    traits = LabelColumnTrait

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        composite_prompt = [
            basic_intro_prompt,
            data_description_template.format(task.data_description),
            evaluation_description_template.format(task.evaluation_description),
            columns_in_train_not_test_template.format("\n".join(task.columns_in_train_but_not_test)),
            parse_fields_template.format(" ".join(self.parser.__fields__.keys())),
            format_instructions_template.format(self.parser.get_format_instructions()),
        ]

        try:
            parser_output = self._chat_and_parse_prompt_output(composite_prompt, basic_system_prompt)
            task.metadata["label_column"] = parser_output["label_column"]
        except OutputParserException as e:
            # Use the fallback method to infer the label column
            task.metadata["label_column"] = task._infer_label_column_from_output_data()
            if not task.metadata["label_column"]:
                # raise the error if the fallback logic returns an Empty/None value
                raise e

        return task


class AbstractIdColumnInferenceTransformer(LLMParserTransformer):
    """Identifies the ID column in the data to be used when generating predictions."""

    traits = IdColumnTrait

    def _parse_id_column_in_data(self, data: pd.DataFrame, output_id_column: str):
        columns = data.columns.to_list()
        if len(columns) > 50:
            columns = columns[:10] + columns[-10:]

        composite_prompt = [
            infer_test_id_column_template.format(
                output_id_column=output_id_column,
                test_columns="\n".join(columns),
            ),
            format_instructions_template.format(self.parser.get_format_instructions()),
        ]
        try:
            parser_output = self._chat_and_parse_prompt_output(composite_prompt, basic_system_prompt)
            parsed_id_column = parser_output["id_column"]
        except OutputParserException:
            parsed_id_column = "NO_ID_COLUMN_IDENTIFIED"
        return parsed_id_column


class TestIdColumnTransformer(AbstractIdColumnInferenceTransformer):
    """Identifies the ID column in the test data to be used when generating predictions.

    Side Effect
    -----------
    If no valid ID column could be identified by the LLM, a new ID column is added to the test data.
    """

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        output_id_column = task.output_id_column

        if output_id_column in task.test_data.columns:
            # if the output ID column, by default the first column of the sample output dataset in the task
            # is in the test column, we assume this is the valid test ID column
            test_id_column = output_id_column
        else:
            # if the output ID column is not in the test column, we try to identify the test ID column by
            # chatting with the LLM
            parsed_id_column = self._parse_id_column_in_data(task.test_data, output_id_column)
            if parsed_id_column == "NO_ID_COLUMN_IDENTIFIED" or parsed_id_column not in task.test_data.columns:
                # if no valid column could be identified by the LLM, we also transform the test data and add a new ID column
                start_val, end_val = task.output_data[output_id_column].iloc[[0, -1]]
                if all(task.output_data[output_id_column] == np.arange(start_val, end_val + 1)):
                    new_test_data = task.test_data.copy()
                    new_test_data[output_id_column] = np.arange(start_val, start_val + len(task.test_data))
                    task.test_data = new_test_data
                    parsed_id_column = output_id_column  # type: ignore
                    logger.warning("No valid ID column identified by LLM, adding a new ID column to the test data")
                else:
                    raise Exception(
                        f"Output Id column: {output_id_column} not in test data, and LLM could not identify a valid Id column"
                    )
            test_id_column = parsed_id_column

        task.metadata["test_id_column"] = test_id_column
        return task


class TrainIdColumnDropTransformer(AbstractIdColumnInferenceTransformer):
    """Identifies the ID column in training data and drops it from the training data set if the identified column is valid."""

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        output_id_column = task.output_id_column
        train_id_column = None

        if output_id_column in task.train_data.columns:
            train_id_column = output_id_column
        else:
            # if the output ID column is not in the test column, we try to identify the test ID column by
            # chatting with the LLM
            parsed_id_column = self._parse_id_column_in_data(task.train_data, output_id_column)

            if (
                parsed_id_column != "NO_ID_COLUMN_IDENTIFIED"
                and parsed_id_column in task.train_data.columns
                and task.train_data[parsed_id_column].nunique() == len(task.train_data)
            ):
                train_id_column = parsed_id_column

        if train_id_column is not None:
            task.train_data = task.train_data.drop(columns=[train_id_column])
            logger.info(f"Dropping ID column {train_id_column} from training data.")
            task.metadata["dropped_train_id_column"] = True

        task.metadata["train_id_column"] = train_id_column
        return task


class EvalMetricInferenceTransformer(LLMParserTransformer):
    traits = EvalMetricTrait

    metrics_by_problem_type = {
        "binary": [
            ("roc_auc", "Area under the receiver operating characteristics (ROC) curve"),
            ("log_loss", "Log loss, also known as logarithmic loss"),
            ("accuracy", "Accuracy"),
            ("f1", "F1 score"),
            ("quadratic_kappa", "Quadratic kappa, i.e., the Cohen kappa metric"),
            ("balanced_accuracy", "Balanced accuracy, i.e., the arithmetic mean of sensitivity and specificity"),
        ],
        "multiclass": [
            ("roc_auc", "Area under the receiver operating characteristics (ROC) curve"),
            ("log_loss", "Log loss, also known as logarithmic loss"),
            ("accuracy", "Accuracy"),
            ("f1", "F1 score"),
            ("quadratic_kappa", "Quadratic kappa, i.e., the Cohen kappa metric"),
            ("balanced_accuracy", "Balanced accuracy, i.e., the arithmetic mean of sensitivity and specificity"),
        ],
        "regression": [
            ("root_mean_squared_error", "Root mean squared error (RMSE)"),
            ("mean_squared_error", "Mean squared error (MSE)"),
            ("mean_absolute_error", "Mean absolute error (MAE)"),
            ("r2", "R-squared"),
            ("root_mean_squared_logarithmic_error", "Root mean squared logarithmic error (RMSLE)"),
        ],
    }

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        all_metrics = []
        for problem_type in self.metrics_by_problem_type:
            all_metrics.extend(self.metrics_by_problem_type[problem_type])

        candidate_metrics = (
            all_metrics if task.problem_type is None else self.metrics_by_problem_type[task.problem_type]
        )

        composite_prompt = [
            basic_intro_prompt,
            data_description_template.format(task.data_description),
            eval_metric_prompt.format(
                evaluation_description=task.evaluation_description,
                metrics="\n".join([name for name, _ in candidate_metrics]),
                metric_descriptions="\n".join([desc for _, desc in candidate_metrics]),
            ),
            parse_fields_template.format(" ".join(self.parser.__fields__.keys())),
            format_instructions_template.format(self.parser.get_format_instructions()),
        ]
        try:
            parser_output = self._chat_and_parse_prompt_output(composite_prompt, basic_system_prompt)
            allowed_eval_metrics = [name for name, _ in all_metrics]
            parsed_eval_metric = parser_output["eval_metric"]

            if parsed_eval_metric not in allowed_eval_metrics:
                close_matches = difflib.get_close_matches(parsed_eval_metric, allowed_eval_metrics)
                if len(close_matches) == 0:
                    raise ValueError(f"Eval metric: {parsed_eval_metric} not recognized")
                parsed_eval_metric = close_matches[0]

            logger.info(f"Using parsed eval metric: {parsed_eval_metric}")
            task.metadata["eval_metric"] = parsed_eval_metric
        except OutputParserException:
            logger.warning("Langchain failed to parse eval metric output. Will use default eval metric")
        except ValueError as e:
            logger.warning(
                "Unrecognized eval metric parsed by the LLM. Will use default eval metric."
                f"The parser output was {parser_output}, and exception was {str(e)}"
            )
        except Exception:
            logger.warning("Unknown exception during eval metric parsing. Will use default eval metric.")

        return task
