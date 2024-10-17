import logging
import signal
from typing import Any, Dict, Union

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


class timeout:
    def __init__(self, seconds=1, error_message="Transform timed out"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TransformTimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class TabularPredictionAssistant:
    """A TabularPredictionAssistant performs a supervised tabular learning task"""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.llm: Union[AssistantChatOpenAI, AssistantChatBedrock] = LLMFactory.get_chat_model(config.llm)
        self.predictor = AutogluonTabularPredictor(config.autogluon)
        self.feature_transformers_config = config.feature_transformers

    def describe(self) -> Dict[str, Any]:
        return {
            "predictor": self.predictor.describe(),
            "config": OmegaConf.to_container(self.config),
            "llm": self.llm.describe(),  # noqa
        }

    def handle_exception(self, stage: str, exception: Exception):
        raise Exception(str(exception), stage)

    def inference_task(self, task: TabularPredictionTask) -> TabularPredictionTask:
        task_inference_preprocessors = [
            DescriptionFileNameInference,
            DataFileNameInference,
            LabelColumnInference,
            ProblemTypeInference,
            OutputIDColumnInference,
            TrainIDColumnInference,
            TestIDColumnInference,
        ]
        if self.config.infer_eval_metric:
            task_inference_preprocessors += [EvalMetricInference]
        for preprocessor_class in task_inference_preprocessors:
            preprocessor = preprocessor_class(llm=self.llm)
            try:
                with timeout(
                    seconds=self.config.task_preprocessors_timeout,
                    error_message=f"Task inference preprocessing timed out: {preprocessor_class}",
                ):
                    task = preprocessor.transform(task)
            except Exception as e:
                self.handle_exception(f"Task inference preprocessing: {preprocessor_class}", e)

        logger.info(f"###LLM Inference Results:###\n{task.metadata}")
        logger.info(f"###Total number of prompt tokens:###\n{self.llm.input_}")
        logger.info(f"###Total number of completion tokens:###\n{self.llm.output_}")

        return task

    def preprocess_task(self, task: TabularPredictionTask) -> TabularPredictionTask:
        # instantiate and run task preprocessors, which infer the problem type, important filenames
        # and columns as well as the feature extractors
        task = self.inference_task(task)
        if self.feature_transformers_config:
            fe_transformers = [instantiate(ft_config) for ft_config in self.feature_transformers_config]
            for fe_transformer in fe_transformers:
                try:
                    with timeout(
                        seconds=self.config.task_preprocessors_timeout,
                        error_message=f"Task preprocessing timed out: {fe_transformer.name}",
                    ):
                        task = fe_transformer.fit_transform(task)
                except Exception as e:
                    self.handle_exception(f"Task preprocessing: {fe_transformer.name}", e)

        return task

    def fit_predictor(self, task: TabularPredictionTask):
        try:
            self.predictor.fit(task)
        except Exception as e:
            self.handle_exception("Predictor Fit", e)

    def predict(self, task: TabularPredictionTask) -> Any:
        try:
            return self.predictor.predict(task)
        except Exception as e:
            self.handle_exception("Predictor Predict", e)
