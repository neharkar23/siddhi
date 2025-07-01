# providers/llm_factory.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging
from config.core_config import LLMProvider, config_manager

logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs
        self.client = None
    
    @abstractmethod
    def initialize(self):
        """Initialize the LLM client"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat completion with message history"""
        pass

class OpenAILLM(BaseLLM):
    """OpenAI LLM implementation"""
    
    def initialize(self):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI client initialized with model: {self.model_name}")
        except ImportError:
            raise ImportError("openai package is required for OpenAI provider")
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.client:
            self.initialize()
        
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'temperature']}
            )
            return response.choices[0].text.strip()
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.client:
            self.initialize()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'temperature']}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            raise

class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation"""
    
    def initialize(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini client initialized with model: {self.model_name}")
        except ImportError:
            raise ImportError("google-generativeai package is required for Gemini provider")
    
    def generate(self, prompt: str, **kwargs) -> str:
        if not self.client:
            self.initialize()
        
        try:
            response = self.client.generate_content(
                prompt,
                generation_config={
                    'max_output_tokens': kwargs.get('max_tokens', 1000),
                    'temperature': kwargs.get('temperature', 0.7),
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.client:
            self.initialize()
        
        try:
            # Convert messages to Gemini format
            chat_session = self.client.start_chat(history=[])
            
            # Send all messages except the last one as history
            for msg in messages[:-1]:
                if msg['role'] == 'user':
                    chat_session.send_message(msg['content'])
            
            # Send the last message and get response
            response = chat_session.send_message(
                messages[-1]['content'],
                generation_config={
                    'max_output_tokens': kwargs.get('max_tokens', 1000),
                    'temperature': kwargs.get('temperature', 0.7),
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini chat error: {e}")
            raise

class GroqLLM(BaseLLM):
    """Groq LLM implementation"""
    
    def initialize(self):
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.info(f"Groq client initialized with model: {self.model_name}")
        except ImportError:
            raise ImportError("groq package is required for Groq provider")
    
    def generate(self, prompt: str, **kwargs) -> str:
        # Groq primarily uses chat completions
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, **kwargs)
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        if not self.client:
            self.initialize()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'temperature']}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq chat error: {e}")
            raise

class LLMFactory:
    """Factory class for creating LLM instances"""
    
    _providers = {
        LLMProvider.OPENAI: OpenAILLM,
        LLMProvider.GEMINI: GeminiLLM,
        LLMProvider.GROQ: GroqLLM,
    }
    
    @classmethod
    def create_llm(
        self,
        provider: LLMProvider,
        model_name: str,
        **kwargs
    ) -> BaseLLM:
        """Create LLM instance based on provider"""
        
        if provider not in self._providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        # Get API key for provider
        api_key = config_manager.get_api_key(provider.value)
        if not api_key:
            raise ValueError(f"API key not found for provider: {provider.value}")
        
        # Create and initialize LLM
        llm_class = self._providers[provider]
        llm = llm_class(model_name=model_name, api_key=api_key, **kwargs)
        llm.initialize()
        
        return llm
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available providers"""
        return [provider.value for provider in cls._providers.keys()]
    
    @classmethod
    def list_supported_models(cls) -> list:
        """
        Return a flat list of all supported model names across all providers.
        """
        models_dict = cls.get_default_models()
        # Flatten all model lists into one list
        return [model for models in models_dict.values() for model in models]
    
    @classmethod
    def get_default_models(cls) -> Dict[str, List[str]]:
        """Get default models for each provider"""
        return {
            LLMProvider.OPENAI.value: [
                "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"
            ],
            LLMProvider.GEMINI.value: [
                "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"
            ],
            LLMProvider.GROQ.value: [
                "llama3-8b-8192", "llama3-70b-8192", "llama-3.3-70b-versatile",
                "mixtral-8x7b-32768", "gemma-7b-it"
            ]
        }