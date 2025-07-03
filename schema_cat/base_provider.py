from abc import ABC, abstractmethod
from xml.etree import ElementTree
from typing import List, Dict, Any, Optional, Union


class BaseProvider(ABC):
    """Abstract base class for all providers."""

    @abstractmethod
    async def call(self,
                   model: str,
                   sys_prompt: str,
                   user_prompt: str,
                   xml_schema: Optional[str] = None,
                   max_tokens: int = 8192,
                   temperature: float = 0.0,
                   max_retries: int = 5,
                   initial_delay: float = 1.0,
                   max_delay: float = 60.0) -> Union[ElementTree.XML, str]:
        """
        Make a call to the provider's API.

        Args:
            model: The model name to use
            sys_prompt: System prompt
            user_prompt: User prompt
            xml_schema: XML schema for response validation (optional for freeform responses)
            max_tokens: Maximum tokens in response
            temperature: Temperature for response generation
            max_retries: Maximum number of retries
            initial_delay: Initial delay for retries
            max_delay: Maximum delay for retries

        Returns:
            Parsed XML response as ElementTree.XML if xml_schema is provided,
            or raw string response if xml_schema is None
        """
        pass

    @abstractmethod
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Retrieve all available models from the provider's API.

        Returns:
            List of model dictionaries containing model information.
            Each dictionary should contain at least:
            - 'id': Model identifier/name
            - 'object': Model object type (usually 'model')
            Additional fields may include context_length, pricing, etc.
        """
        pass
