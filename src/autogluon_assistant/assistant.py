import logging
import os
import signal
from dataclasses import dataclass
from typing import Any, Dict, List, Type, Union

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from autogluon_assistant.llm import AssistantChatBedrock, AssistantChatOpenAI, LLMFactory

from .predictor import AutogluonTabularPredictor
from .task import TabularPredictionTask
from .task_inference import (
    DataFileNameInference,
    DescriptionFileNameInference,
    EvalMetricInference,
    LabelColumnInference,
    OutputIDColumnInference,
    ProblemTypeInference,
    TestIDColumnInference,
    TrainIDColumnInference,
)
from .transformer import TransformTimeoutError

logger = logging.getLogger(__name__)


@dataclass
class TimeoutContext:
    """Context manager for handling operation timeouts."""
    seconds: int
    error_message: str = "Operation timed out"

    def handle_timeout(self, signum: int, frame: Any) -> None:
        """Signal handler for timeout."""
        raise TransformTimeoutError(self.error_message)

    def __enter__(self) -> 'TimeoutContext':
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        signal.alarm(0)


class TabularPredictionAssistant:
    """
    A TabularPredictionAssistant that performs supervised tabular learning tasks.
    
    Attributes:
        config (DictConfig): Configuration for the assistant
        llm (Union[AssistantChatOpenAI, AssistantChatBedrock]): Language model instance
        predictor (AutogluonTabularPredictor): AutoGluon predictor instance
        feature_transformers_config (Any): Configuration for feature transformers
    """

    def __init__(self, config: DictConfig) -> None:
        """
        Initialize the TabularPredictionAssistant.

        Args:
            config (DictConfig): Configuration object containing necessary settings
        """
        self.config = config
        self.llm = LLMFactory.get_chat_model(config.llm)
        self.predictor = AutogluonTabularPredictor(config.autogluon)
        self.feature_transformers_config = config.feature_transformers

    def describe(self) -> Dict[str, Any]:
        """
        Get a description of the assistant's components.

        Returns:
            Dict[str, Any]: Description of predictor, config, and LLM
        """
        return {
            "predictor": self.predictor.describe(),
            "config": OmegaConf.to_container(self.config),
            "llm": self.llm.describe(),
        }

    def handle_exception(self, stage: str, exception: Exception) -> None:
        """
        Handle exceptions by raising them with additional context.

        Args:
            stage (str): The processing stage where the exception occurred
            exception (Exception): The original exception

        Raises:
            Exception: Enhanced exception with stage information
        """
        raise Exception(str(exception), stage)

    def _get_task_inference_preprocessors(self) -> List[Type]:
        """
        Get the list of task inference preprocessors based on configuration.

        Returns:
            List[Type]: List of preprocessor classes
        """
        preprocessors = [
            DescriptionFileNameInference,
            DataFileNameInference,
            LabelColumnInference,
            ProblemTypeInference,
        ]

        if self.config.detect_and_drop_id_column:
            preprocessors.extend([
                OutputIDColumnInference,
                TrainIDColumnInference,
                TestIDColumnInference,
            ])

        if self.config.infer_eval_metric:
            preprocessors.append(EvalMetricInference)

        return preprocessors

    def inference_task(self, task: TabularPredictionTask) -> TabularPredictionTask:
        """
        Perform task inference using configured preprocessors.

        Args:
            task (TabularPredictionTask): The initial task object

        Returns:
            TabularPredictionTask: The processed task object
        """
        logger.info("Task understanding starts...")
        
        for preprocessor_class in self._get_task_inference_preprocessors():
            preprocessor = preprocessor_class(llm=self.llm)
            try:
                with TimeoutContext(
                    seconds=self.config.task_preprocessors_timeout,
                    error_message=f"Task inference preprocessing timed out: {preprocessor_class.__name__}"
                ):
                    task = preprocessor.transform(task)
            except Exception as e:
                self.handle_exception(f"Task inference preprocessing: {preprocessor_class.__name__}", e)

        self._log_token_usage()
        logger.info("Task understanding complete!")
        return task

    def _log_token_usage(self) -> None:
        """Log the token usage statistics."""
        bold_format = lambda text: f"\033[1m{text}\033[0m"
        logger.info(f"{bold_format('Total number of prompt tokens:')} {self.llm.input_}")
        logger.info(f"{bold_format('Total number of completion tokens:')} {self.llm.output_}")

    def preprocess_task(self, task: TabularPredictionTask) -> TabularPredictionTask:
        """
        Preprocess the task using inference and feature transformers.

        Args:
            task (TabularPredictionTask): The task to preprocess

        Returns:
            TabularPredictionTask: The preprocessed task
        """
        task = self.inference_task(task)
        
        if not self.feature_transformers_config:
            logger.info("Automatic feature generation is disabled.")
            return task

        logger.info("Automatic feature generation starts...")
        fe_transformers = self._get_feature_transformers()
        
        for fe_transformer in fe_transformers:
            try:
                with TimeoutContext(
                    seconds=self.config.task_preprocessors_timeout,
                    error_message=f"Task preprocessing timed out: {fe_transformer.name}"
                ):
                    task = fe_transformer.fit_transform(task)
            except Exception as e:
                self.handle_exception(f"Task preprocessing: {fe_transformer.name}", e)

        logger.info("Automatic feature generation complete!")
        return task

    def _get_feature_transformers(self) -> List[Any]:
        """
        Get the list of feature transformers based on configuration and environment.

        Returns:
            List[Any]: List of instantiated feature transformers
        """
        if "OPENAI_API_KEY" not in os.environ:
            logger.info("No OpenAI API keys found, therefore, skip CAAFE")
            return [
                instantiate(ft_config)
                for ft_config in self.feature_transformers_config
                if ft_config["_target_"] != "autogluon_assistant.transformer.CAAFETransformer"
            ]
        
        return [instantiate(ft_config) for ft_config in self.feature_transformers_config]

    def fit_predictor(self, task: TabularPredictionTask) -> None:
        """
        Fit the predictor on the given task.

        Args:
            task (TabularPredictionTask): The task to fit the predictor on
        """
        try:
            self.predictor.fit(task)
        except Exception as e:
            self.handle_exception("Predictor Fit", e)

    def predict(self, task: TabularPredictionTask) -> Any:
        """
        Make predictions using the fitted predictor.

        Args:
            task (TabularPredictionTask): The task to make predictions for

        Returns:
            Any: Predictions from the predictor
        """
        try:
            return self.predictor.predict(task)
        except Exception as e:
            self.handle_exception("Predictor Predict", e)
