import logging
import os
import pprint
from typing import Any, Dict, List, Optional, Union

import boto3
import botocore
from langchain.schema import AIMessage, BaseMessage
from langchain_aws import ChatBedrock
from langchain_openai import ChatOpenAI
from omegaconf import DictConfig
from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class ChatModelMixin(BaseModel):
    """Base mixin class for chat models with common functionality."""

    history_: List[Dict[str, Any]] = Field(default_factory=list)
    input_tokens: int = Field(default=0, alias="input_")
    output_tokens: int = Field(default=0, alias="output_")

    def update_token_usage(self, response: AIMessage) -> None:
        """Update token usage based on response metadata."""
        if response.usage_metadata:
            self.input_tokens += response.usage_metadata.get("input_tokens", 0)
            self.output_tokens += response.usage_metadata.get("output_tokens", 0)

    def append_to_history(self, input_messages: List[BaseMessage], response: AIMessage) -> None:
        """Append interaction to history."""
        self.history_.append({
            "input": [{"type": msg.type, "content": msg.content} for msg in input_messages],
            "output": pprint.pformat(dict(response)),
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
        })


class AssistantChatOpenAI(ChatOpenAI, ChatModelMixin):
    """OpenAI chat model with input/output tracing capabilities."""

    def describe(self) -> Dict[str, Any]:
        """Return description of the model configuration and usage."""
        return {
            "model": self.model_name,
            "proxy": self.openai_proxy,
            "history": self.history_,
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def invoke(self, input_messages: List[BaseMessage], **kwargs) -> AIMessage:
        """Invoke the model with retry logic and tracking."""
        response = super().invoke(input_messages, **kwargs)
        
        if isinstance(response, AIMessage):
            self.update_token_usage(response)
            self.append_to_history(input_messages, response)
        
        return response


class AssistantChatBedrock(ChatBedrock, ChatModelMixin):
    """Bedrock chat model with input/output tracing capabilities."""

    def describe(self) -> Dict[str, Any]:
        """Return description of the model configuration and usage."""
        return {
            "model": self.model_id,
            "history": self.history_,
            "prompt_tokens": self.input_tokens,
            "completion_tokens": self.output_tokens,
        }

    @retry(
        stop=stop_after_attempt(50),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def invoke(self, input_messages: List[BaseMessage], **kwargs) -> AIMessage:
        """Invoke the model with retry logic and tracking."""
        try:
            response = super().invoke(input_messages, **kwargs)
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                raise e
            raise

        if isinstance(response, AIMessage):
            self.update_token_usage(response)
            self.append_to_history(input_messages, response)
        
        return response


class LLMFactory:
    """Factory class for creating and managing LLM instances."""

    VALID_PROVIDERS = ["openai", "bedrock"]
    
    @staticmethod
    def get_openai_models() -> List[str]:
        """Fetch available OpenAI models."""
        try:
            client = OpenAI()
            models = client.models.list()
            return [
                model.id 
                for model in models 
                if model.id.startswith(("gpt-3.5", "gpt-4"))
            ]
        except Exception as e:
            logger.error(f"Error fetching OpenAI models: {e}")
            return []

    @staticmethod
    def get_bedrock_models() -> List[str]:
        """Fetch available Bedrock models."""
        try:
            bedrock = boto3.client("bedrock", region_name="us-west-2")
            response = bedrock.list_foundation_models()
            models = [
                model["modelId"]
                for model in response["modelSummaries"]
                if model["modelId"].startswith("anthropic.claude")
            ]
            if not models:
                raise ValueError("No valid Bedrock models found")
            return models
        except Exception as e:
            logger.error(f"Error fetching Bedrock models: {e}")
            return []

    @classmethod
    def get_valid_models(cls, provider: str) -> List[str]:
        """Get valid models for a given provider."""
        if provider not in cls.VALID_PROVIDERS:
            raise ValueError(f"Invalid LLM provider: {provider}")
        
        if provider == "openai":
            return cls.get_openai_models()
        return cls.get_bedrock_models()

    @classmethod
    def get_chat_model(
        cls, 
        config: DictConfig
    ) -> Union[AssistantChatOpenAI, AssistantChatBedrock]:
        """Create a chat model instance based on configuration."""
        provider = config.provider
        model = config.model

        if provider not in cls.VALID_PROVIDERS:
            raise ValueError(
                f"Invalid provider: {provider}. Must be one of {cls.VALID_PROVIDERS}"
            )

        valid_models = cls.get_valid_models(provider)
        if not valid_models:
            raise ValueError(f"No valid models found for provider: {provider}")
        
        if model not in valid_models:
            raise ValueError(
                f"Invalid model: {model}. Must be one of {valid_models}"
            )

        if provider == "openai":
            return cls._create_openai_model(config)
        return cls._create_bedrock_model(config)

    @staticmethod
    def _create_openai_model(config: DictConfig) -> AssistantChatOpenAI:
        """Create an OpenAI chat model instance."""
        if config.api_key_location not in os.environ:
            raise ValueError(f"OpenAI API key not found in environment: {config.api_key_location}")

        logger.info(f"Using OpenAI model: {config.model}")
        
        return AssistantChatOpenAI(
            model_name=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            verbose=config.verbose,
            openai_api_key=os.environ[config.api_key_location],
            openai_api_base=config.proxy_url,
        )

    @staticmethod
    def _create_bedrock_model(config: DictConfig) -> AssistantChatBedrock:
        """Create a Bedrock chat model instance."""
        logger.info(f"Using Bedrock model: {config.model}")
        
        return AssistantChatBedrock(
            model_id=config.model,
            model_kwargs={
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
            },
            region_name="us-west-2",
            verbose=config.verbose,
        )
