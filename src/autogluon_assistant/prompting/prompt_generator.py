from abc import ABC, abstractmethod
from typing import Dict, Any, List

from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from ..constants import METRICS_BY_PROBLEM_TYPE, METRICS_DESCRIPTION, NO_ID_COLUMN_IDENTIFIED, PROBLEM_TYPES


class PromptGenerator(ABC):
    fields = None

    def __init__(self, data_description: str):
        self.data_description = data_description
        self.parser = self.create_parser()

    @property
    def system_prompt(self):
        return "You are an expert assistant that parses information about data science tasks, such as data science competitions."

    @property
    def basic_intro_prompt(self):
        return "The following sections contain descriptive information about a data science task:"

    @property
    def data_description_prompt(self):
        return f"# Data Description\n{self.data_description}"

    @abstractmethod
    def generate_prompt(self) -> str:
        pass

    def get_field_parsing_prompt(self) -> str:
        return f"Based on the above information, what are the correct values for the following fields in JSON format: {', '.join(self.fields)}"

    def generate_chat_prompt(self):
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self.generate_prompt()),
            ]
        )
        return chat_prompt

    def create_parser(self):
        response_schemas = [
            ResponseSchema(name=field, description=f"The {field} for the task") for field in self.fields
        ]
        return StructuredOutputParser.from_response_schemas(response_schemas)


class FilenamePromptGenerator(PromptGenerator):
    fields = ["train_data", "test_data", "output_data"]

    def __init__(self, data_description: str, filenames: list):
        super().__init__(data_description)
        self.filenames = filenames

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                f"# Available Files\n{', '.join(self.filenames)}",
                "If there are zip (e.g. .zip or .gz) versions of files and non-zipped versions of the files, choose the non-zip version. For example, return 'train.csv' rather than 'train.csv.zip'.",
                self.get_field_parsing_prompt(),
            ]
        )


class LabelColumnPromptGenerator(PromptGenerator):
    fields = ["label_column"]

    def __init__(self, data_description: str, column_names: list):
        super().__init__(data_description)
        self.column_names = column_names

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                f"Based on the data description, which one of these columns is likely to be the label column:\n{', '.join(self.column_names)}",
                self.get_field_parsing_prompt(),
            ]
        )


class ProblemTypePromptGenerator(PromptGenerator):
    fields = ["problem_type"]

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                f"Based on the information provided, identify the correct problem_type to be used from among these KEYS: {', '.join(PROBLEM_TYPES)}",
                self.get_field_parsing_prompt(),
            ]
        )


class IDColumnPromptGenerator(PromptGenerator):
    fields = ["id_column"]

    def __init__(self, data_description: str, column_names: list):
        super().__init__(data_description)
        self.column_names = column_names

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                f"Based on the data description, which one of these columns is likely to be the Id column:\n{', '.join(self.column_names)}",
                f"If no reasonable Id column is present, for example if all the columns appear to be similarly named feature columns, "
                f"response with the value {NO_ID_COLUMN_IDENTIFIED}",
                self.get_field_parsing_prompt(),
            ]
        )


class TestIDColumnPromptGenerator(IDColumnPromptGenerator):
    fields = ["test_id_column"]


class TrainIDColumnPromptGenerator(IDColumnPromptGenerator):
    fields = ["train_id_column"]


class OutputIDColumnPromptGenerator(IDColumnPromptGenerator):
    fields = ["output_id_column"]


class EvalMetricPromptGenerator(PromptGenerator):
    fields = ["eval_metric"]

    def __init__(self, data_description: str, metrics: str):
        super().__init__(data_description)
        self.metrics = metrics

    def generate_prompt(self) -> str:
        return "\n\n".join(
            [
                self.basic_intro_prompt,
                self.data_description_prompt,
                f"""
Based on the information provided, identify the correct evaluation metric to be used from among these KEYS:
{', '.join(self.metrics)}
The descriptions of these metrics are:
{', '.join([METRICS_DESCRIPTION[metric] for metric in self.metrics])}
respectively.
If the exact metric is not in the list provided, then choose the metric that you think best approximates the one in the task description.
Only respond with the exact names of the metrics mentioned in KEYS. Do not respond with the metric descriptions.
""",
                self.get_field_parsing_prompt(),
            ]
        )
