"""
Production-grade LiteLLM Framework Implementation
Unified interface for 100+ LLM providers with advanced features
"""

import logging
from typing import Any, Dict, List, Optional
from providers.framework_factory import BaseFramework

logger = logging.getLogger(__name__)

class LiteLLMFramework(BaseFramework):
    """Production LiteLLM Framework with unified LLM interface"""

    def initialize(self):
        """Initialize LiteLLM with comprehensive provider support"""
        try:
            self._import_dependencies()
            self._configure_providers()
            self._setup_callbacks()
            self._configure_caching()
            self._initialized = True
            logger.info("✅ LiteLLM framework initialized successfully")
        except Exception as e:
            logger.error(f"❌ LiteLLM initialization failed: {e}")
            raise

    def _import_dependencies(self):
        """Import LiteLLM dependencies"""
        try:
            import litellm
            from litellm import completion, acompletion, embedding
            from litellm.utils import ModelResponse
            
            self.litellm = litellm
            self.completion = completion
            self.acompletion = acompletion
            self.embedding = embedding
            self.ModelResponse = ModelResponse
            
            # Configure LiteLLM settings
            litellm.drop_params = True  # Drop unsupported params
            litellm.set_verbose = False  # Control logging
            
        except ImportError as e:
            raise ImportError(f"LiteLLM required: pip install litellm")

    def _configure_providers(self):
        """Configure supported LLM providers"""
        self.provider_configs = {
            # OpenAI
            'openai': {
                'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                'api_key_env': 'OPENAI_API_KEY',
                'supports_streaming': True,
                'supports_functions': True
            },
            # Anthropic
            'anthropic': {
                'models': ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307', 'claude-3-opus-20240229'],
                'api_key_env': 'ANTHROPIC_API_KEY',
                'supports_streaming': True,
                'supports_functions': False
            },
            # Google
            'gemini': {
                'models': ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-pro'],
                'api_key_env': 'GEMINI_API_KEY',
                'supports_streaming': True,
                'supports_functions': True
            },
            # Groq
            'groq': {
                'models': ['llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b-32768'],
                'api_key_env': 'GROQ_API_KEY',
                'supports_streaming': True,
                'supports_functions': False
            },
            # Cohere
            'cohere': {
                'models': ['command-r', 'command-r-plus', 'command'],
                'api_key_env': 'COHERE_API_KEY',
                'supports_streaming': True,
                'supports_functions': False
            },
            # Azure OpenAI
            'azure': {
                'models': ['azure/gpt-4o', 'azure/gpt-4-turbo', 'azure/gpt-35-turbo'],
                'api_key_env': 'AZURE_API_KEY',
                'supports_streaming': True,
                'supports_functions': True
            },
            # Hugging Face
            'huggingface': {
                'models': ['huggingface/microsoft/DialoGPT-medium', 'huggingface/google/flan-t5-xl'],
                'api_key_env': 'HUGGINGFACE_API_KEY',
                'supports_streaming': False,
                'supports_functions': False
            },
            # Ollama (local)
            'ollama': {
                'models': ['ollama/llama3.2', 'ollama/mistral', 'ollama/codellama'],
                'api_key_env': None,
                'supports_streaming': True,
                'supports_functions': False
            }
        }

        # Set current provider based on model
        model_name = getattr(self.llm, 'model_name', 'gpt-4o-mini')
        self.current_provider = self._detect_provider(model_name)
        self.current_config = self.provider_configs.get(self.current_provider, {})

    def _detect_provider(self, model_name: str) -> str:
        """Detect provider from model name"""
        if model_name.startswith('gpt-'):
            return 'openai'
        elif model_name.startswith('claude-'):
            return 'anthropic'
        elif model_name.startswith('gemini-'):
            return 'gemini'
        elif model_name.startswith('llama'):
            return 'groq'
        elif model_name.startswith('command'):
            return 'cohere'
        elif model_name.startswith('azure/'):
            return 'azure'
        elif model_name.startswith('huggingface/'):
            return 'huggingface'
        elif model_name.startswith('ollama/'):
            return 'ollama'
        else:
            return 'openai'  # Default fallback

    def _setup_callbacks(self):
        """Setup LiteLLM callbacks for observability"""
        try:
            from litellm.integrations.custom_logger import CustomLogger
            
            class RAGObservabilityLogger(CustomLogger):
                def __init__(self, framework_instance):
                    self.framework = framework_instance
                    super().__init__()

                def log_pre_api_call(self, model, messages, kwargs):
                    logger.info(f"LiteLLM API call: {model}")

                def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
                    duration = end_time - start_time
                    logger.info(f"LiteLLM API response: {duration:.2f}s")

                def log_success_event(self, kwargs, response_obj, start_time, end_time):
                    logger.info("LiteLLM API call successful")

                def log_failure_event(self, kwargs, response_obj, start_time, end_time):
                    logger.error("LiteLLM API call failed")

            # Set up custom logger
            self.custom_logger = RAGObservabilityLogger(self)
            self.litellm.callbacks = [self.custom_logger]
            
        except Exception as e:
            logger.warning(f"⚠️ Could not setup LiteLLM callbacks: {e}")

    def _configure_caching(self):
        """Configure LiteLLM caching"""
        try:
            # Enable semantic caching if Redis is available
            cache_config = {
                'type': 'redis',
                'host': self.kwargs.get('redis_host', 'localhost'),
                'port': self.kwargs.get('redis_port', 6379),
                'password': self.kwargs.get('redis_password'),
                'ttl': self.kwargs.get('cache_ttl', 3600)  # 1 hour
            }
            
            self.litellm.cache = self.litellm.Cache(**cache_config)
            logger.info("✅ LiteLLM caching configured")
            
        except Exception as e:
            logger.warning(f"⚠️ LiteLLM caching not available: {e}")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create LiteLLM RAG chain with advanced features"""
        if not system_prompt:
            system_prompt = """You are an expert Docker assistant powered by LiteLLM's unified interface.

Your capabilities:
1. Access to multiple LLM providers for optimal responses
2. Comprehensive Docker knowledge from retrieved documentation
3. Practical, tested solutions with detailed explanations
4. Security-focused recommendations and best practices
5. Cross-platform compatibility guidance

Instructions:
- Use the provided context to answer Docker questions accurately
- Leverage the best features of the current LLM provider
- Provide step-by-step instructions with command examples
- Include troubleshooting tips and alternative approaches
- Warn about potential issues and suggest preventive measures

Context: {context}
Question: {input}

Provide a comprehensive, provider-optimized response:"""

        try:
            self.system_prompt = system_prompt
            self.rag_chain = {
                'model': getattr(self.llm, 'model_name', 'gpt-4o-mini'),
                'provider': self.current_provider,
                'config': self.current_config,
                'system_prompt': system_prompt,
                'supports_streaming': self.current_config.get('supports_streaming', False),
                'supports_functions': self.current_config.get('supports_functions', False)
            }
            
            logger.info(f"✅ LiteLLM RAG chain created for provider: {self.current_provider}")
            return self.rag_chain

        except Exception as e:
            logger.error(f"❌ Failed to create LiteLLM RAG chain: {e}")
            raise

    def query(self, question: str, **kwargs) -> str:
        """Execute query using LiteLLM unified interface"""
        if not hasattr(self, 'rag_chain'):
            self.create_rag_chain()

        try:
            # Get context from vector store
            context = self._get_context_from_vector_store(question, k=5)
            
            # Format messages
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user", 
                    "content": self.system_prompt.format(context=context, input=question)
                }
            ]

            # Prepare completion parameters
            completion_params = {
                'model': self.rag_chain['model'],
                'messages': messages,
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1000),
                'top_p': kwargs.get('top_p', 0.9),
                'stream': kwargs.get('stream', False)
            }

            # Add provider-specific parameters
            if self.current_provider == 'anthropic':
                completion_params['max_tokens'] = min(completion_params['max_tokens'], 4096)
            elif self.current_provider == 'groq':
                completion_params['max_tokens'] = min(completion_params['max_tokens'], 8192)

            # Execute completion
            if kwargs.get('stream', False) and self.rag_chain['supports_streaming']:
                return self._stream_completion(completion_params)
            else:
                response = self.completion(**completion_params)
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"❌ LiteLLM query failed: {e}")
            return self._fallback_query(question)

    def _stream_completion(self, params: Dict[str, Any]) -> str:
        """Handle streaming completion"""
        try:
            response_chunks = []
            for chunk in self.completion(**params):
                if chunk.choices[0].delta.content:
                    response_chunks.append(chunk.choices[0].delta.content)
            
            return ''.join(response_chunks)
            
        except Exception as e:
            logger.error(f"❌ LiteLLM streaming failed: {e}")
            # Fallback to non-streaming
            params['stream'] = False
            response = self.completion(**params)
            return response.choices[0].message.content

    def _fallback_query(self, question: str) -> str:
        """Fallback query method"""
        try:
            context = self._get_context_from_vector_store(question, k=3)
            prompt = f"""Context: {context}

Question: {question}

Please provide a comprehensive Docker answer based on the context above."""
            
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"❌ LiteLLM fallback failed: {e}")
            return f"Error processing query with LiteLLM: {str(e)}"

    def get_supported_models(self) -> Dict[str, List[str]]:
        """Get all supported models by provider"""
        return {provider: config['models'] for provider, config in self.provider_configs.items()}

    def switch_provider(self, model_name: str) -> bool:
        """Switch to a different provider/model"""
        try:
            new_provider = self._detect_provider(model_name)
            if new_provider in self.provider_configs:
                self.current_provider = new_provider
                self.current_config = self.provider_configs[new_provider]
                self.rag_chain['model'] = model_name
                self.rag_chain['provider'] = new_provider
                logger.info(f"✅ Switched to provider: {new_provider}, model: {model_name}")
                return True
            else:
                logger.error(f"❌ Unsupported provider for model: {model_name}")
                return False
        except Exception as e:
            logger.error(f"❌ Provider switch failed: {e}")
            return False

    def get_provider_status(self) -> Dict[str, Any]:
        """Get current provider status and capabilities"""
        return {
            'current_provider': self.current_provider,
            'current_model': self.rag_chain.get('model', 'unknown'),
            'supports_streaming': self.current_config.get('supports_streaming', False),
            'supports_functions': self.current_config.get('supports_functions', False),
            'available_models': self.current_config.get('models', []),
            'cache_enabled': hasattr(self.litellm, 'cache') and self.litellm.cache is not None
        }

def register():
    """Register LiteLLM framework with factory"""
    return "litellm", LiteLLMFramework
