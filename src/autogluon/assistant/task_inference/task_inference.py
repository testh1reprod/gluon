import difflib
import logging
from typing import Any, Dict, List  # Added Union for type hinting

from autogluon.core.utils.utils import infer_problem_type
from langchain_core.exceptions import OutputParserException  # Updated import

from autogluon.assistant.prompting import (
    DataFileNamePromptGenerator,
    DescriptionFileNamePromptGenerator,
    EvalMetricPromptGenerator,
    LabelColumnPromptGenerator,
    OutputIDColumnPromptGenerator,
    ProblemTypePromptGenerator,
    TestIDColumnPromptGenerator,
    TrainIDColumnPromptGenerator,
)
from autogluon.assistant.task import TabularPredictionTask

from ..constants import (
    CLASSIFICATION_PROBLEM_TYPES,
    METRICS_BY_PROBLEM_TYPE,
    METRICS_DESCRIPTION,
    NO_FILE_IDENTIFIED,
    NO_ID_COLUMN_IDENTIFIED,
    PROBLEM_TYPES,
)

logger = logging.getLogger(__name__)


class TaskInference:
    """Parses data and metadata of a task with the aid of an instruction-tuned LLM."""

    def __init__(self, llm, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.fallback_value = None
        self.ignored_value: List[str] = []  # Added type hint

    def initialize_task(self, task):
        self.prompt_generator = None
        self.valid_values = None

    def log_value(self, key: str, value: Any, max_width: int = 1600) -> None:
        """Logs a key-value pair with formatted output."""
        if not value:
            logger.info(f"WARNING: Failed to identify the {key} of the task, it is set to None.")
            return

        prefix = key  # f"Identified the {key} of the task: "
        value_str = str(value).replace("\n", "\\n")

        if len(prefix) + len(value_str) > max_width:
            value_str = value_str[: max_width - len(prefix) - 3] + "..."

        bold_start = "\033[1m"
        bold_end = "\033[0m"

        logger.info(f"{bold_start}{prefix}{bold_end}: {value_str}")

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        self.initialize_task(task)
        parser_output = self._chat_and_parse_prompt_output()
        for k, v in parser_output.items():
            if v in self.ignored_value:
                v = None
            self.log_value(k, v)
            setattr(task, k, self.post_process(task=task, value=v))
        return task

    def post_process(self, task, value):
        return value

    def parse_output(self, output):
        assert self.prompt_generator is not None, "prompt_generator is not initialized"
        return self.prompt_generator.parser.parse(output.content)

    def _chat_and_parse_prompt_output(self) -> Dict[str, str]:
        """Chat with the LLM and parse the output"""
        try:
            chat_prompt = self.prompt_generator.generate_chat_prompt()
            logger.debug(f"LLM chat_prompt:\n{chat_prompt.format_messages()}")
            output = self.llm.invoke(chat_prompt.format_messages())
            logger.debug(f"LLM output:\n{output}")
            parsed_output = self.parse_output(output)
        except OutputParserException as e:
            logger.error(f"Failed to parse output: {e}")
            logger.error(self.llm.describe())
            raise e

        if self.valid_values is not None:
            for key, parsed_value in parsed_output.items():
                if parsed_value not in self.valid_values:
                    # Currently only support single parsed value
                    if isinstance(parsed_value, str):
                        close_matches = difflib.get_close_matches(parsed_value, self.valid_values)
                    elif isinstance(parsed_value, list) and len(parsed_value) == 1:
                        parsed_value = parsed_value[0]
                        close_matches = difflib.get_close_matches(parsed_value, self.valid_values)
                    else:
                        logger.warning(
                            f"Unrecognized parsed value: {parsed_value} for key {key} parsed by the LLM. "
                            f"It has type: {type(parsed_value)}."
                        )
                        close_matches = []

                    if len(close_matches) == 0:
                        if self.fallback_value:
                            logger.warning(
                                f"Unrecognized value: {parsed_value} for key {key} parsed by the LLM. "
                                f"Will use default value: {self.fallback_value}."
                            )
                            parsed_output[key] = self.fallback_value
                        else:
                            raise ValueError(f"Unrecognized value: {parsed_value} for key {key} parsed by the LLM.")
                    else:
                        parsed_output[key] = close_matches[0]

        return parsed_output


class DescriptionFileNameInference(TaskInference):
    """Uses an LLM to locate the filenames of description files.
    TODO: merge the logics with DataFileNameInference and add support for multiple files per field.
    """

    def initialize_task(self, task):
        filenames = [str(path) for path in task.filepaths]
        self.valid_values = filenames + [NO_FILE_IDENTIFIED]
        self.fallback_value = NO_FILE_IDENTIFIED
        self.prompt_generator = DescriptionFileNamePromptGenerator(filenames=filenames)

    def _read_descriptions(self, parser_output: dict) -> str:
        description_parts = []
        for key, file_paths in parser_output.items():
            if isinstance(file_paths, str):
                file_paths = [file_paths]  # Convert single string to list

            for file_path in file_paths:
                if file_path == NO_FILE_IDENTIFIED:
                    continue
                else:
                    try:
                        with open(file_path, "r") as file:
                            content = file.read()
                            description_parts.append(f"{key}: {content}")
                    except FileNotFoundError:
                        continue
                    except IOError:
                        continue
        return "\n\n".join(description_parts)

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        self.initialize_task(task)
        parser_output = self._chat_and_parse_prompt_output()
        descriptions_read = self._read_descriptions(parser_output)
        if descriptions_read:
            task.metadata["description"] = descriptions_read
        self.log_value("description", descriptions_read)
        return task


class DataFileNameInference(TaskInference):
    """Uses an LLM to locate the filenames of the train, test, and output data,
    and assigns them to the respective properties of the task.
    """

    def initialize_task(self, task):
        filenames = [str(path) for path in task.filepaths]
        self.valid_values = filenames + [NO_FILE_IDENTIFIED]
        self.fallback_value = NO_FILE_IDENTIFIED
        self.ignored_value = [NO_FILE_IDENTIFIED]
        self.prompt_generator = DataFileNamePromptGenerator(
            data_description=task.metadata["description"], filenames=filenames
        )


class LabelColumnInference(TaskInference):
    def initialize_task(self, task):
        column_names = list(task.train_data.columns)
        self.valid_values = column_names
        self.prompt_generator = LabelColumnPromptGenerator(
            data_description=task.metadata["description"], column_names=column_names
        )


class ProblemTypeInference(TaskInference):
    def initialize_task(self, task):
        self.valid_values = PROBLEM_TYPES
        self.prompt_generator = ProblemTypePromptGenerator(data_description=task.metadata["description"])

    def post_process(self, task, value):
        # LLM may get confused between BINARY and MULTICLASS as it cannot see the whole label column
        if value in CLASSIFICATION_PROBLEM_TYPES:
            problem_type_infered_by_autogluon = infer_problem_type(task.train_data[task.label_column], silent=True)
            if problem_type_infered_by_autogluon in CLASSIFICATION_PROBLEM_TYPES:
                value = problem_type_infered_by_autogluon
        return value


class BaseIDColumnInference(TaskInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid_values = []
        self.fallback_value = NO_ID_COLUMN_IDENTIFIED
        self.prompt_generator = None

    def initialize_task(self, task, description=None):
        if self.get_data(task) is None:
            return

        column_names = list(self.get_data(task).columns)
        # Assume ID column can only appear in first 3 columns
        if len(column_names) >= 3:
            column_names = column_names[:3]
        self.valid_values = column_names + [NO_ID_COLUMN_IDENTIFIED]
        if not description:
            description = task.metadata["description"]
        self.prompt_generator = self.get_prompt_generator()(
            data_description=description, column_names=column_names, label_column=task.metadata["label_column"]
        )

    def get_data(self, task):
        pass

    def get_prompt_generator(self):
        pass

    def get_id_column_name(self):
        pass

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        if self.get_data(task) is None:
            setattr(task, self.get_id_column_name(), None)
            return task

        self.initialize_task(task)
        parser_output = self._chat_and_parse_prompt_output()
        id_column_name = self.get_id_column_name()

        if parser_output[id_column_name] == NO_ID_COLUMN_IDENTIFIED:
            logger.warning(
                "Failed to infer ID column with data descriptions. " "Retry the inference without data descriptions."
            )
            self.initialize_task(
                task, description="Missing data description. Please infer the ID column based on given column names."
            )
            parser_output = self._chat_and_parse_prompt_output()

        id_column = parser_output[id_column_name]
        id_column = self.process_id_column(task, id_column)
        self.log_value(id_column_name, id_column)
        setattr(task, id_column_name, id_column)
        return task

    def process_id_column(self, task, id_column):
        pass


class TestIDColumnInference(BaseIDColumnInference):
    def get_data(self, task):
        return task.test_data

    def get_prompt_generator(self):
        return TestIDColumnPromptGenerator

    def get_id_column_name(self):
        return "test_id_column"

    def process_id_column(self, task, id_column):
        if task.output_id_column != NO_ID_COLUMN_IDENTIFIED:
            # if output data has id column but test data does not
            if id_column == NO_ID_COLUMN_IDENTIFIED:
                if task.output_id_column not in task.test_data:
                    id_column = task.output_id_column
                else:
                    id_column = "id_column"
                new_test_data = task.test_data.copy()
                new_test_data[id_column] = task.sample_submission_data[task.output_id_column]
                task.test_data = new_test_data

        return id_column


class TrainIDColumnInference(BaseIDColumnInference):
    def get_data(self, task):
        return task.train_data

    def get_prompt_generator(self):
        return TrainIDColumnPromptGenerator

    def get_id_column_name(self):
        return "train_id_column"

    def process_id_column(self, task, id_column):
        if id_column != NO_ID_COLUMN_IDENTIFIED:
            new_train_data = task.train_data.copy()
            new_train_data = new_train_data.drop(columns=[id_column])
            task.train_data = new_train_data
            logger.info(f"Dropping ID column {id_column} from training data.")
            task.metadata["dropped_train_id_column"] = True

        return id_column


class OutputIDColumnInference(BaseIDColumnInference):
    def get_data(self, task):
        return task.sample_submission_data

    def get_prompt_generator(self):
        return OutputIDColumnPromptGenerator

    def get_id_column_name(self):
        return "output_id_column"

    def process_id_column(self, task, id_column):
        return id_column


class EvalMetricInference(TaskInference):
    def initialize_task(self, task):
        problem_type = task.problem_type
        self.metrics = METRICS_DESCRIPTION.keys() if problem_type is None else METRICS_BY_PROBLEM_TYPE[problem_type]
        self.valid_values = self.metrics
        if problem_type:
            self.fallback_value = METRICS_BY_PROBLEM_TYPE[problem_type][0]
        self.prompt_generator = EvalMetricPromptGenerator(
            data_description=task.metadata["description"], metrics=self.metrics
        )
