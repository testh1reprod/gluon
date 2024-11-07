from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union, Dict

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from pandas import DataFrame

from ..constants import METRICS_DESCRIPTION, NO_FILE_IDENTIFIED, NO_ID_COLUMN_IDENTIFIED, PROBLEM_TYPES
from ..utils import is_text_file, load_pd_quietly
from .utils import get_outer_columns


class PromptGenerator(ABC):
    """
    Abstract base class for generating prompts in data science tasks.
    
    This class provides a framework for creating structured prompts that can be used
    to extract specific information about data science tasks.
    
    Attributes:
        fields (List[str]): List of fields to be extracted from the prompt response
        data_description (str): Description of the data science task
        parser (StructuredOutputParser): Parser for structured output
    """
    
    fields: List[str] = None

    def __init__(self, data_description: str = "") -> None:
        """
        Initialize the prompt generator.

        Args:
            data_description: Description of the data science task
        """
        self.data_description = data_description
        self.parser = self._create_parser()

    @property
    def system_prompt(self) -> str:
        """Return the system prompt for the assistant."""
        return "You are an expert assistant that parses information about data science tasks, such as data science competitions."

    @property
    def basic_intro_prompt(self) -> str:
        """Return the basic introduction prompt."""
        return "The following sections contain descriptive information about a data science task:"

    @property
    def data_description_prompt(self) -> str:
        """Return the data description prompt."""
        return f"# Data Description\n{self.data_description}"

    @abstractmethod
    def generate_prompt(self) -> str:
        """Generate the complete prompt string."""
        pass

    def get_field_parsing_prompt(self) -> str:
        """Generate the prompt for field parsing instructions."""
        return (
            f"Based on the above information, provide the correct values for the following fields strictly "
            f"in valid JSON format: {', '.join(self.fields)}.\n\n"
            "Important:\n"
            "1. Return only valid JSON. No extra explanations, text, or comments.\n"
            "2. Ensure that the output can be parsed by a JSON parser directly.\n"
            "3. Do not include any non-JSON text or formatting outside the JSON object.\n"
            '4. An example is {"<provided_field>": "<correct_value_for_the_field>"}'
        )

    def generate_chat_prompt(self) -> ChatPromptTemplate:
        """Generate a chat prompt template."""
        return ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self.generate_prompt()),
        ])

    def _create_parser(self) -> StructuredOutputParser:
        """Create a structured output parser for the fields."""
        response_schemas = [
            ResponseSchema(name=field, description=f"The {field} for the task")
            for field in self.fields
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)


class DescriptionFileNamePromptGenerator(PromptGenerator):
    """Generator for prompts to identify description and evaluation files."""
    
    fields = ["data_description_file", "evaluation_description_file"]

    def __init__(self, filenames: List[str]) -> None:
        """
        Initialize the description file name prompt generator.

        Args:
            filenames: List of available filenames
        """
        super().__init__()
        self.filenames = filenames

    def read_file_safely(self, filename: Path) -> Optional[str]:
        """
        Safely read a file's contents.

        Args:
            filename: Path to the file

        Returns:
            The file contents or None if the file cannot be read
        """
        try:
            return filename.read_text()
        except UnicodeDecodeError:
            return None

    def generate_prompt(self) -> str:
        """Generate a prompt for identifying description files."""
        file_content_prompts = []
        file_content_prompts.append("# Available Files And Content in The File\n")

        for filename in map(Path, self.filenames):
            if is_text_file(filename):
                content = self.read_file_safely(filename)
                if content is not None:
                    truncated_contents = f"{content[:100].strip()}..."
                    file_content_prompts.append(
                        f"File:\n\n{filename} Truncated Content:\n{truncated_contents}\n"
                    )

        file_content_prompts.append(
            f"Please return the full path of the file to describe the problem settings, "
            f"and response with the value {NO_FILE_IDENTIFIED} if there's no such file."
        )

        return "\n\n".join([
            self.basic_intro_prompt,
            "\n".join(file_content_prompts),
            self.get_field_parsing_prompt(),
        ])


class DataFileNamePromptGenerator(PromptGenerator):
    """Generator for prompts to identify data files."""
    
    fields = ["train_data", "test_data", "sample_submission_data"]

    def __init__(self, data_description: str, filenames: List[str]) -> None:
        """
        Initialize the data file name prompt generator.

        Args:
            data_description: Description of the data
            filenames: List of available filenames
        """
        super().__init__(data_description)
        self.filenames = filenames

    def generate_prompt(self) -> str:
        """Generate a prompt for identifying data files."""
        file_content_prompts = ["# Available Data Files And Columns in The File\n"]

        for filename in self.filenames:
            try:
                content = load_pd_quietly(filename)
                file_content_prompts.append(f"File:\n\n{filename}")
            except Exception as e:
                print(
                    f"Failed to load data as a pandas DataFrame in {filename} "
                    f"with following error (please ignore this if it is not supposed "
                    f"to be a data file): {e}"
                )
                continue

        file_content_prompts.append(
            f"Based on the data description, what are the training, test, and output data? "
            f"The output file may contain keywords such as benchmark, submission, or output. "
            f"Please return the full path of the data files as provided, and response with "
            f"the value {NO_FILE_IDENTIFIED} if there's no such File."
        )

        return "\n\n".join([
            self.basic_intro_prompt,
            "\n".join(file_content_prompts),
            self.get_field_parsing_prompt(),
        ])


class LabelColumnPromptGenerator(PromptGenerator):
    """Generator for prompts to identify label columns."""
    
    fields = ["label_column"]

    def __init__(self, data_description: str, column_names: List[str]) -> None:
        """
        Initialize the label column prompt generator.

        Args:
            data_description: Description of the data
            column_names: List of column names
        """
        super().__init__(data_description)
        self.column_names = get_outer_columns(column_names)

    def generate_prompt(self) -> str:
        """Generate a prompt for identifying label columns."""
        return "\n\n".join([
            self.basic_intro_prompt,
            self.data_description_prompt,
            f"Based on the data description, which one of these columns is likely to be "
            f"the label column:\n{', '.join(self.column_names)}",
            self.get_field_parsing_prompt(),
        ])


class ProblemTypePromptGenerator(PromptGenerator):
    """Generator for prompts to identify problem types."""
    
    fields = ["problem_type"]

    def generate_prompt(self) -> str:
        """Generate a prompt for identifying problem types."""
        return "\n\n".join([
            self.basic_intro_prompt,
            self.data_description_prompt,
            f"Based on the information provided, identify the correct problem_type to be "
            f"used from among these KEYS: {', '.join(PROBLEM_TYPES)}",
            self.get_field_parsing_prompt(),
        ])


class BaseIDColumnPromptGenerator(PromptGenerator):
    """Base class for ID column prompt generators."""
    
    def __init__(self, data_description: str, column_names: List[str], label_column: str) -> None:
        """
        Initialize the base ID column prompt generator.

        Args:
            data_description: Description of the data
            column_names: List of column names
            label_column: Name of the label column
        """
        super().__init__(data_description)
        self.column_names = get_outer_columns(column_names)
        self.label_column = label_column

    def generate_prompt(self) -> str:
        """Generate a prompt for identifying ID columns."""
        return "\n\n".join([
            self.basic_intro_prompt,
            self.data_description_prompt,
            f"Based on the data description, which one of these columns is likely to be "
            f"the Id column:\n{', '.join(self.column_names)}",
            f"If no reasonable Id column is present, for example if all the columns appear "
            f"to be similarly named feature columns, response with the value "
            f"{NO_ID_COLUMN_IDENTIFIED}",
            f"ID columns can't be {self.label_column}",
            self.get_field_parsing_prompt(),
        ])


class IDColumnPromptGenerator(BaseIDColumnPromptGenerator):
    """Generator for prompts to identify general ID columns."""
    fields = ["id_column"]


class TestIDColumnPromptGenerator(BaseIDColumnPromptGenerator):
    """Generator for prompts to identify test data ID columns."""
    fields = ["test_id_column"]


class TrainIDColumnPromptGenerator(BaseIDColumnPromptGenerator):
    """Generator for prompts to identify training data ID columns."""
    fields = ["train_id_column"]


class OutputIDColumnPromptGenerator(BaseIDColumnPromptGenerator):
    """Generator for prompts to identify output data ID columns."""
    fields = ["output_id_column"]


class EvalMetricPromptGenerator(PromptGenerator):
    """Generator for prompts to identify evaluation metrics."""
    
    fields = ["eval_metric"]

    def __init__(self, data_description: str, metrics: str) -> None:
        """
        Initialize the evaluation metric prompt generator.

        Args:
            data_description: Description of the data
            metrics: Available metrics
        """
        super().__init__(data_description)
        self.metrics = metrics

    def generate_prompt(self) -> str:
        """Generate a prompt for identifying evaluation metrics."""
        metric_descriptions = [METRICS_DESCRIPTION[metric] for metric in self.metrics]
        
        return "\n\n".join([
            self.basic_intro_prompt,
            self.data_description_prompt,
            f"""
Based on the information provided, identify the correct evaluation metric to be used 
from among these KEYS: {', '.join(self.metrics)}

The descriptions of these metrics are:
{', '.join(metric_descriptions)}
respectively.

If the exact metric is not in the list provided, then choose the metric that you think 
best approximates the one in the task description.

Only respond with the exact names of the metrics mentioned in KEYS. Do not respond with 
the metric descriptions.
""",
            self.get_field_parsing_prompt(),
        ])
