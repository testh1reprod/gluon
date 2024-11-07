from typing import Any, Dict, List, Optional, Union
import difflib
import logging
from pathlib import Path

from autogluon.core.utils.utils import infer_problem_type
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage

from autogluon_assistant.prompting import (
    DataFileNamePromptGenerator,
    DescriptionFileNamePromptGenerator,
    EvalMetricPromptGenerator,
    LabelColumnPromptGenerator,
    OutputIDColumnPromptGenerator,
    ProblemTypePromptGenerator,
    TestIDColumnPromptGenerator,
    TrainIDColumnPromptGenerator,
)
from autogluon_assistant.task import TabularPredictionTask

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
    """Base class for parsing data and metadata of a task with the aid of an instruction-tuned LLM."""

    def __init__(self, llm: Any, *args: Any, **kwargs: Any) -> None:
        """Initialize TaskInference.

        Args:
            llm: Language model instance
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        super().__init__(*args, **kwargs)
        self.llm = llm
        self.fallback_value: Optional[str] = None
        self.ignored_value: List[str] = []
        self.prompt_generator: Optional[Any] = None
        self.valid_values: Optional[List[str]] = None

    def initialize_task(self, task: TabularPredictionTask) -> None:
        """Initialize task-specific attributes.

        Args:
            task: TabularPredictionTask instance
        """
        self.prompt_generator = None
        self.valid_values = None

    def log_value(self, key: str, value: Any, max_width: int = 1600) -> None:
        """Log a key-value pair with formatted output.

        Args:
            key: Key to log
            value: Value to log
            max_width: Maximum width of the log message
        """
        if not value:
            logger.info(f"WARNING: Failed to identify the {key} of the task, it is set to None.")
            return

        value_str = str(value).replace("\n", "\\n")
        if len(key) + len(value_str) > max_width:
            value_str = value_str[:max_width - len(key) - 3] + "..."

        logger.info(f"\033[1m{key}\033[0m: {value_str}")

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        """Transform the task using LLM inference.

        Args:
            task: TabularPredictionTask instance

        Returns:
            Modified TabularPredictionTask instance
        """
        self.initialize_task(task)
        parser_output = self._chat_and_parse_prompt_output()
        for key, value in parser_output.items():
            if value in self.ignored_value:
                value = None
            self.log_value(key, value)
            setattr(task, key, self.post_process(task=task, value=value))
        return task

    def post_process(self, task: TabularPredictionTask, value: Any) -> Any:
        """Post-process the parsed value.

        Args:
            task: TabularPredictionTask instance
            value: Value to post-process

        Returns:
            Processed value
        """
        return value

    def parse_output(self, output: BaseMessage) -> Dict[str, str]:
        """Parse LLM output using the prompt generator's parser.

        Args:
            output: LLM output message

        Returns:
            Parsed output dictionary
        """
        if not self.prompt_generator:
            raise ValueError("prompt_generator is not initialized")
        return self.prompt_generator.parser.parse(output.content)

    def _chat_and_parse_prompt_output(self) -> Dict[str, str]:
        """Chat with the LLM and parse the output.

        Returns:
            Dictionary containing parsed output

        Raises:
            OutputParserException: If parsing fails
            ValueError: If parsed value is not in valid_values
        """
        try:
            chat_prompt = self.prompt_generator.generate_chat_prompt()  # type: ignore
            logger.debug(f"LLM chat_prompt:\n{chat_prompt.format_messages()}")
            output = self.llm.invoke(chat_prompt.format_messages())
            logger.debug(f"LLM output:\n{output}")
            parsed_output = self.parse_output(output)
        except OutputParserException as e:
            logger.error(f"Failed to parse output: {e}")
            logger.error(self.llm.describe())
            raise

        if self.valid_values:
            parsed_output = self._validate_and_correct_output(parsed_output)
        return parsed_output

    def _validate_and_correct_output(self, parsed_output: Dict[str, str]) -> Dict[str, str]:
        """Validate and correct parsed output against valid values.

        Args:
            parsed_output: Dictionary of parsed values

        Returns:
            Validated and corrected output dictionary

        Raises:
            ValueError: If no valid match is found and no fallback value is set
        """
        if not self.valid_values:
            return parsed_output

        for key, parsed_value in parsed_output.items():
            if parsed_value in self.valid_values:
                continue

            close_matches = self._get_close_matches(parsed_value)
            
            if not close_matches:
                if self.fallback_value:
                    logger.warning(
                        f"Unrecognized value: {parsed_value} for key {key} parsed by the LLM. "
                        f"Using default value: {self.fallback_value}."
                    )
                    parsed_output[key] = self.fallback_value
                else:
                    raise ValueError(f"Unrecognized value: {parsed_value} for key {key} parsed by the LLM.")
            else:
                parsed_output[key] = close_matches[0]

        return parsed_output

    def _get_close_matches(self, value: Union[str, List[str]]) -> List[str]:
        """Get close matches for a value from valid_values.

        Args:
            value: Value to match against valid_values

        Returns:
            List of close matches
        """
        if isinstance(value, str):
            return difflib.get_close_matches(value, self.valid_values or [])
        elif isinstance(value, list) and len(value) == 1:
            return difflib.get_close_matches(value[0], self.valid_values or [])
        else:
            logger.warning(f"Unrecognized parsed value: {value} with type: {type(value)}.")
            return []


class DescriptionFileNameInference(TaskInference):
    """Infers description filenames using LLM."""

    def initialize_task(self, task: TabularPredictionTask) -> None:
        filenames = [str(path) for path in task.filepaths]
        self.valid_values = filenames + [NO_FILE_IDENTIFIED]
        self.fallback_value = NO_FILE_IDENTIFIED
        self.prompt_generator = DescriptionFileNamePromptGenerator(filenames=filenames)

    def _read_descriptions(self, parser_output: Dict[str, Union[str, List[str]]]) -> str:
        """Read and combine descriptions from identified files.

        Args:
            parser_output: Dictionary containing file paths

        Returns:
            Combined description string
        """
        description_parts = []
        for key, file_paths in parser_output.items():
            if isinstance(file_paths, str):
                file_paths = [file_paths]

            for file_path in file_paths:
                if file_path == NO_FILE_IDENTIFIED:
                    continue
                try:
                    with open(file_path, "r") as file:
                        content = file.read()
                        description_parts.append(f"{key}: {content}")
                except (FileNotFoundError, IOError):
                    continue
        return "\n\n".join(description_parts)

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        self.initialize_task(task)
        parser_output = self._chat_and_parse_prompt_output()
        descriptions = self._read_descriptions(parser_output)
        if descriptions:
            task.metadata["description"] = descriptions
        self.log_value("description", descriptions)
        return task


class DataFileNameInference(TaskInference):
    """Infers data filenames for train, test, and output data."""

    def initialize_task(self, task: TabularPredictionTask) -> None:
        filenames = [str(path) for path in task.filepaths]
        self.valid_values = filenames + [NO_FILE_IDENTIFIED]
        self.fallback_value = NO_FILE_IDENTIFIED
        self.ignored_value = [NO_FILE_IDENTIFIED]
        self.prompt_generator = DataFileNamePromptGenerator(
            data_description=task.metadata["description"],
            filenames=filenames
        )


class LabelColumnInference(TaskInference):
    """Infers label column from data."""

    def initialize_task(self, task: TabularPredictionTask) -> None:
        self.valid_values = list(task.train_data.columns)
        self.prompt_generator = LabelColumnPromptGenerator(
            data_description=task.metadata["description"],
            column_names=self.valid_values
        )


class ProblemTypeInference(TaskInference):
    """Infers problem type from data."""

    def initialize_task(self, task: TabularPredictionTask) -> None:
        self.valid_values = PROBLEM_TYPES
        self.prompt_generator = ProblemTypePromptGenerator(
            data_description=task.metadata["description"]
        )

    def post_process(self, task: TabularPredictionTask, value: str) -> str:
        if value in CLASSIFICATION_PROBLEM_TYPES:
            inferred_type = infer_problem_type(task.train_data[task.label_column], silent=True)
            if inferred_type in CLASSIFICATION_PROBLEM_TYPES:
                return inferred_type
        return value


class BaseIDColumnInference(TaskInference):
    """Base class for ID column inference."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.valid_values = []
        self.fallback_value = NO_ID_COLUMN_IDENTIFIED
        self.prompt_generator = None

    def initialize_task(self, task: TabularPredictionTask, description: Optional[str] = None) -> None:
        data = self.get_data(task)
        if data is None:
            return

        column_names = list(data.columns)[:3]  # Consider only first 3 columns
        self.valid_values = column_names + [NO_ID_COLUMN_IDENTIFIED]
        
        if not description:
            description = task.metadata["description"]
        
        self.prompt_generator = self.get_prompt_generator()(
            data_description=description,
            column_names=column_names,
            label_column=task.metadata["label_column"]
        )

    def get_data(self, task: TabularPredictionTask) -> Any:
        """Get relevant data from task."""
        raise NotImplementedError("Subclasses must implement get_data")

    def get_prompt_generator(self) -> Any:
        """Get appropriate prompt generator."""
        raise NotImplementedError("Subclasses must implement get_prompt_generator")

    def get_id_column_name(self) -> str:
        """Get name of ID column."""
        raise NotImplementedError("Subclasses must implement get_id_column_name")

    def process_id_column(self, task: TabularPredictionTask, id_column: str) -> Optional[str]:
        """Process identified ID column."""
        raise NotImplementedError("Subclasses must implement process_id_column")

    def transform(self, task: TabularPredictionTask) -> TabularPredictionTask:
        if self.get_data(task) is None:
            setattr(task, self.get_id_column_name(), None)
            return task

        self.initialize_task(task)
        parser_output = self._chat_and_parse_prompt_output()
        id_column_name = self.get_id_column_name()

        if parser_output[id_column_name] == NO_ID_COLUMN_IDENTIFIED:
            logger.warning(
                "Failed to infer ID column with data descriptions. Retrying without descriptions."
            )
            self.initialize_task(
                task,
                description="Missing data description. Please infer the ID column based on given column names."
            )
            parser_output = self._chat_and_parse_prompt_output()

        id_column = parser_output[id_column_name]
        id_column = self.process_id_column(task, id_column)
        self.log_value(id_column_name, id_column)
        setattr(task, id_column_name, id_column)
        return task


class TestIDColumnInference(BaseIDColumnInference):
    """Infers test data ID column."""

    def get_data(self, task: TabularPredictionTask) -> Any:
        return task.test_data

    def get_prompt_generator(self) -> Any:
        return TestIDColumnPromptGenerator

    def get_id_column_name(self) -> str:
        return "test_id_column"

    def process_id_column(self, task: TabularPredictionTask, id_column: str) -> str:
        if task.output_id_column != NO_ID_COLUMN_IDENTIFIED:
            if id_column == NO_ID_COLUMN_IDENTIFIED:
                id_column = (
                    task.output_id_column
                    if task.output_id_column not in task.test_data
                    else "id_column"
                )
                new_test_data = task.test_data.copy()
                new_test_data[id_column] = task.sample_submission_data[task.output_id_column]
                task.test_data = new_test_data
        return id_column


class TrainIDColumnInference(BaseIDColumnInference):
    """Infers training data ID column."""

    def get_data(self, task: TabularPredictionTask) -> Any:
        return task.train_data

    def get_prompt_generator(self) -> Any:
        return TrainIDColumnPromptGenerator

    def get_id_column_name(self) -> str:
        return "train_id_column"

    def process_id_column(self, task: TabularPredictionTask, id_column: str) -> str:
        if id_column != NO_ID_COLUMN_IDENTIFIED:
            new_train_data = task.train_data.drop(columns=[id_column])
            task.train_data = new_train_data
            logger.info(f"Dropping ID column {id_column} from training data.")
            task.metadata["dropped_train_id_column"] = True
        return id_column


class OutputIDColumnInference(BaseIDColumnInference):
    """Infers output data ID column."""

    def get_data(self, task: TabularPredictionTask) -> Any:
        """Get sample submission data from task.

        Args:
            task: TabularPredictionTask instance

        Returns:
            Sample submission data
        """
        return task.sample_submission_data

    def get_prompt_generator(self) -> Any:
        """Get OutputIDColumnPromptGenerator.

        Returns:
            OutputIDColumnPromptGenerator class
        """
        return OutputIDColumnPromptGenerator

    def get_id_column_name(self) -> str:
        """Get output ID column name.

        Returns:
            Name of output ID column
        """
        return "output_id_column"

    def process_id_column(self, task: TabularPredictionTask, id_column: str) -> str:
        """Process output ID column (no processing needed).

        Args:
            task: TabularPredictionTask instance
            id_column: Identified ID column name

        Returns:
            Original ID column name
        """
        return id_column


class EvalMetricInference(TaskInference):
    """Infers evaluation metric based on problem type."""

    def initialize_task(self, task: TabularPredictionTask) -> None:
        """Initialize evaluation metric inference.

        Determines available metrics based on problem type and sets up the prompt generator.

        Args:
            task: TabularPredictionTask instance
        """
        problem_type = task.problem_type
        
        # Determine available metrics based on problem type
        self.metrics = (
            list(METRICS_DESCRIPTION.keys())
            if problem_type is None
            else METRICS_BY_PROBLEM_TYPE[problem_type]
        )
        
        # Set valid values for metric validation
        self.valid_values = self.metrics
        
        # Set fallback value if problem type is available
        if problem_type:
            self.fallback_value = METRICS_BY_PROBLEM_TYPE[problem_type][0]
        
        # Initialize prompt generator with available metrics
        self.prompt_generator = EvalMetricPromptGenerator(
            data_description=task.metadata["description"],
            metrics=self.metrics
        )
