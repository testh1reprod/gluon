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
    LabelColumnPromptGenerator,
    OutputIDColumnPromptGenerator,
    ProblemTypePromptGenerator,
    TestIDColumnPromptGenerator,
)
from autogluon_assistant.task import TabularPredictionTask

from ..constants import METRICS_BY_PROBLEM_TYPE, METRICS_DESCRIPTION, NO_ID_COLUMN_IDENTIFIED, PROBLEM_TYPES

logger = logging.getLogger(__name__)


class TaskInference():
    """Parses data and metadata of a task with the aid of an instruction-tuned LLM."""

    def __init__(self, llm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.fallback_value = None

    def initialize_task(self, task):
        self.prompt_generator = None
        self.valid_values = None
        pass

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        self.initialize_task(task)
        parser_output = self._chat_and_parse_prompt_output()
        for key in parser_output:
            setattr(task, key, parser_output[key])
        return task

    def parse_output(self, output):
        assert self.prompt_generator is not None, "prompt_generator is not initialized"
        return self.prompt_generator.parser.parse(output.content)

    def _chat_and_parse_prompt_output(
        self,
    ) -> Dict[str, str]:
        """Chat with the LLM and parse the output"""
        try:
            chat_prompt = self.prompt_generator.generate_chat_prompt()
            logger.info(f"LLM chat_prompt:\n{chat_prompt.format_messages()}")
            output = self.llm(chat_prompt.format_messages())
            logger.info(f"LLM output:\n{output}")

            parsed_output = self.parse_output(output)
        except OutputParserException as e:
            logger.error(f"Failed to parse output: {e}")
            logger.error(self.llm.describe())  # noqa
            raise e
        
        if self.valid_values is not None:
            for key, parsed_value in parsed_output.items():
                if parsed_value not in self.valid_values:
                    close_matches = difflib.get_close_matches(parsed_value, self.valid_values)
                    if len(close_matches) == 0:
                        if self.fallback_value:
                            logger.warning(
                                f"Unrecognized value: {parsed_value} for key {key} parsed by the LLM." 
                                f"Will use default value: {self.fallback_value}."
                            )
                        else:
                            raise ValueError(f"Unrecognized value: {parsed_value} for key {key} parsed by the LLM." )
                    parsed_output[key] = close_matches[0]

        return parsed_output


class FilenameInference(TaskInference):
    """Uses an LLM to locate the filenames of the train, test, and output data,
    and assigns them to the respective properties of the task.
    """
    def initialize_task(self, task):
        filenames = [str(path) for path in task.filepaths]
        self.valid_values = filenames
        self.prompt_generator = FilenamePromptGenerator(data_description=task.metadata["description"], filenames=filenames)


class LabelColumnInference(TaskInference):
    def initialize_task(self, task):
        column_names = list(task.train_data.columns)
        self.valid_values = column_names
        self.prompt_generator = LabelColumnPromptGenerator(data_description=task.metadata["description"], column_names=column_names)


class ProblemTypeInference(TaskInference):
    def initialize_task(self, task):
        self.valid_values = PROBLEM_TYPES
        self.fallback_value = infer_problem_type(task.train_data[task.label_column], silent=True)
        self.prompt_generator = ProblemTypePromptGenerator(data_description=task.metadata["description"])


class TestIDColumnInference(TaskInference):
    def initialize_task(self, task):
        column_names = list(task.test_data.columns)
        self.valid_values = column_names + [NO_ID_COLUMN_IDENTIFIED]
        self.fallback_value = NO_ID_COLUMN_IDENTIFIED
        self.prompt_generator = TestIDColumnPromptGenerator(data_description=task.metadata["description"], column_names=column_names)


class OutputIDColumnInference(TaskInference):
    def initialize_task(self, task):
        column_names = list(task.output_data.columns)
        self.valid_values = column_names + [NO_ID_COLUMN_IDENTIFIED]
        self.fallback_value = NO_ID_COLUMN_IDENTIFIED
        self.prompt_generator = OutputIDColumnPromptGenerator(data_description=task.metadata["description"], column_names=column_names)


class EvalMetricInference(TaskInference):
    def initialize_task(self, task):
        problem_type = task.problem_type
        self.metrics = METRICS_DESCRIPTION.keys() if problem_type is None else METRICS_BY_PROBLEM_TYPE[problem_type]
        self.valid_values = self.metrics
        if problem_type:
            self.fallback_value = METRICS_BY_PROBLEM_TYPE[problem_type][0]
        self.prompt_generator = EvalMetricPromptGenerator(data_description=task.metadata["description"], metrics=self.metrics)
