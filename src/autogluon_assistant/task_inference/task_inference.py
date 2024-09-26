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
from autogluon_assistant.prompting import (
    EvalMetricPromptGenerator,
    FilenamePromptGenerator,
    IdColumnPromptGenerator,
    LabelColumnPromptGenerator,
)
from autogluon_assistant.task import TabularPredictionTask

from .base import BaseTransformer
from ..constants import METRICS_BY_PROBLEM_TYPE, METRICS_DESCRIPTION

logger = logging.getLogger(__name__)


class TaskInference():
    """Parses data and metadata of a task with the aid of an instruction-tuned LLM."""

    def __init__(self, llm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.prompt_generator = None
        self.valid_values = None

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        parser_output = self._chat_and_parse_prompt_output()
        for key in parser_output:
            setattr(task, key, parser_output[key])
        return task

    def _chat_and_parse_prompt_output(
        self,
    ) -> Dict[str, str]:
        """Chat with the LLM and parse the output"""
        try:
            chat_prompt = self.prompt_generator.generate_chat_prompt()
            logger.debug(f"LLM chat_prompt:\n{chat_prompt.format_messages()}")
            output = self.llm(chat_prompt.format_messages())
            logger.debug(f"LLM output:\n{output}")

            parsed_output = self.parser.parse(output.content)
        except OutputParserException as e:
            logger.error(f"Failed to parse output: {e}")
            logger.error(self.llm.describe())  # noqa
            raise e
        
        if self.valid_values:
            for key, parsed_value in parsed_output.items():
                if parsed_value not in self.valid_values:
                    close_matches = difflib.get_close_matches(parsed_value, self.valid_values)
                    if len(close_matches) == 0:
                        raise ValueError(f"Parsed value: {parsed_eval_metric} for key {key} not recognized")
                    parsed_output[key] = close_matches[0]

        return parsed_output


class FilenameInference(TaskInference):
    """Uses an LLM to locate the filenames of the train, test, and output data,
    and assigns them to the respective properties of the task.
    """
    def __init__(self, llm, data_description: str, filenames: list, *args, **kwargs):
        super().__init__(llm, *args, **kwargs)
        self.valid_values = filenames
        self.prompt_generator = FilenamePromptGenerator(data_description=data_description, filenames=filenames)


# TODO: add inference
class ProblemTypeInference(TaskInference):
    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        task.metadata["problem_type"] = infer_problem_type(task.train_data[task.label_column], silent=True)
        return task


class LabelColumnInference(TaskInference):
    def __init__(self, llm, data_description: str, column_names: list, *args, **kwargs):
        super().__init__(llm, *args, **kwargs)
        self.valid_values = column_names
        self.prompt_generator = LabelColumnPromptGenerator(data_description=data_description, column_names=column_names)


class EvalMetricInference(TaskInference):
    def __init__(self, llm, data_description: str, problem_type: str, *args, **kwargs):
        super().__init__(llm, *args, **kwargs)
        self.metrics = METRICS_DESCRIPTION.keys() if problem_type is None else METRICS_BY_PROBLEM_TYPE[problem_type]
        self.valid_values = self.metrics
        self.prompt_generator = EvalMetricPromptGenerator(data_description=data_description, metrics=self.metrics)

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:

        try:
            parser_output = self._chat_and_parse_prompt_output()
        except OutputParserException:
            logger.warning("Langchain failed to parse eval metric output. Will use default eval metric")
        except ValueError as e:
            logger.warning(
                "Unrecognized eval metric parsed by the LLM. Will use default eval metric."
                f"The parser output was {parser_output}, and exception was {str(e)}"
            )
        except Exception as e:
            # TODO: fix whatever the Unknown exception is
            logger.warning(f"Unknown exception {e} during eval metric parsing. Will use default eval metric.")

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
