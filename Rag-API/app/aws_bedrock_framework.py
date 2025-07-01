import json
import logging
from typing import Any, Dict, List, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from providers.framework_factory import BaseFramework

logger = logging.getLogger(__name__)

class AWSBedrockFramework(BaseFramework):
    """Production AWS Bedrock Framework with enterprise features"""

    def initialize(self):
        """Initialize AWS Bedrock with comprehensive configuration"""
        try:
            self._import_dependencies()
            self._setup_aws_client()
            self._configure_models()
            self._setup_streaming()
            self._initialized = True
            logger.info("✅ AWS Bedrock framework initialized successfully")
        except Exception as e:
            logger.error(f"❌ AWS Bedrock initialization failed: {e}")
            raise

    def _import_dependencies(self):
        """Import AWS dependencies with error handling"""
        try:
            import boto3
            from botocore.config import Config
            from botocore.exceptions import ClientError, NoCredentialsError
            
            self.boto3 = boto3
            self.Config = Config
            self.ClientError = ClientError
            self.NoCredentialsError = NoCredentialsError
        except ImportError as e:
            raise ImportError(f"AWS SDK required: pip install boto3 botocore")

    def _setup_aws_client(self):
        """Setup AWS Bedrock client with proper configuration"""
        try:
            # Configure AWS client with retry and timeout settings
            config = self.Config(
                region_name=self.kwargs.get('aws_region', 'us-east-1'),
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                max_pool_connections=50,
                read_timeout=60,
                connect_timeout=10
            )

            # Initialize Bedrock Runtime client
            self.bedrock_client = self.boto3.client(
                'bedrock-runtime',
                config=config,
                aws_access_key_id=self.kwargs.get('aws_access_key_id'),
                aws_secret_access_key=self.kwargs.get('aws_secret_access_key'),
                aws_session_token=self.kwargs.get('aws_session_token')
            )

            # Initialize Bedrock client for model management
            self.bedrock_mgmt_client = self.boto3.client(
                'bedrock',
                config=config,
                aws_access_key_id=self.kwargs.get('aws_access_key_id'),
                aws_secret_access_key=self.kwargs.get('aws_secret_access_key'),
                aws_session_token=self.kwargs.get('aws_session_token')
            )

            # Test connection
            self._test_connection()
            logger.info("✅ AWS Bedrock clients configured successfully")

        except self.NoCredentialsError:
            raise ValueError("AWS credentials not configured. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
        except Exception as e:
            logger.error(f"❌ AWS client setup failed: {e}")
            raise

    def _test_connection(self):
        """Test AWS Bedrock connection"""
        try:
            # List available foundation models to test connection
            response = self.bedrock_mgmt_client.list_foundation_models(
                byOutputModality='TEXT'
            )
            self.available_models = [model['modelId'] for model in response.get('modelSummaries', [])]
            logger.info(f"✅ Connected to AWS Bedrock. Available models: {len(self.available_models)}")
        except Exception as e:
            logger.warning(f"⚠️ Could not list models, but client is configured: {e}")
            self.available_models = []

    def _configure_models(self):
        """Configure supported Bedrock models"""
        self.model_configs = {
            # Anthropic Claude models
            'anthropic.claude-3-5-sonnet-20241022-v2:0': {
                'max_tokens': 4096,
                'temperature_range': (0.0, 1.0),
                'supports_streaming': True,
                'input_cost_per_1k': 0.003,
                'output_cost_per_1k': 0.015
            },
            'anthropic.claude-3-haiku-20240307-v1:0': {
                'max_tokens': 4096,
                'temperature_range': (0.0, 1.0),
                'supports_streaming': True,
                'input_cost_per_1k': 0.00025,
                'output_cost_per_1k': 0.00125
            },
            'anthropic.claude-v2:1': {
                'max_tokens': 4096,
                'temperature_range': (0.0, 1.0),
                'supports_streaming': True,
                'input_cost_per_1k': 0.008,
                'output_cost_per_1k': 0.024
            },
            # Amazon Titan models
            'amazon.titan-text-express-v1': {
                'max_tokens': 8192,
                'temperature_range': (0.0, 1.0),
                'supports_streaming': False,
                'input_cost_per_1k': 0.0008,
                'output_cost_per_1k': 0.0016
            },
            'amazon.titan-text-lite-v1': {
                'max_tokens': 4096,
                'temperature_range': (0.0, 1.0),
                'supports_streaming': False,
                'input_cost_per_1k': 0.0003,
                'output_cost_per_1k': 0.0004
            },
            # Meta Llama models
            'meta.llama3-2-90b-instruct-v1:0': {
                'max_tokens': 2048,
                'temperature_range': (0.0, 1.0),
                'supports_streaming': True,
                'input_cost_per_1k': 0.002,
                'output_cost_per_1k': 0.002
            },
            'meta.llama3-2-11b-instruct-v1:0': {
                'max_tokens': 2048,
                'temperature_range': (0.0, 1.0),
                'supports_streaming': True,
                'input_cost_per_1k': 0.00035,
                'output_cost_per_1k': 0.00035
            },
            # Mistral AI models
            'mistral.mistral-large-2402-v1:0': {
                'max_tokens': 8192,
                'temperature_range': (0.0, 1.0),
                'supports_streaming': True,
                'input_cost_per_1k': 0.008,
                'output_cost_per_1k': 0.024
            }
        }

        # Set current model configuration
        model_id = getattr(self.llm, 'model_name', 'anthropic.claude-3-haiku-20240307-v1:0')
        self.current_model_config = self.model_configs.get(
            model_id, 
            self.model_configs['anthropic.claude-3-haiku-20240307-v1:0']
        )

    def _setup_streaming(self):
        """Setup streaming capabilities"""
        self.supports_streaming = self.current_model_config.get('supports_streaming', False)
        logger.info(f"✅ Streaming support: {self.supports_streaming}")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create AWS Bedrock RAG chain with advanced features"""
        if not system_prompt:
            system_prompt = """You are an expert Docker assistant powered by AWS Bedrock.

Your capabilities:
1. Provide comprehensive Docker guidance using retrieved documentation
2. Explain complex containerization concepts clearly
3. Offer practical, tested solutions with examples
4. Include security best practices and performance optimization
5. Troubleshoot Docker issues systematically

Guidelines:
- Use the provided context to answer Docker questions accurately
- If context is insufficient, clearly state limitations
- Provide step-by-step instructions with command examples
- Include warnings for potentially destructive operations
- Suggest best practices and alternative approaches

Context: {context}
Question: {input}

Provide a detailed, actionable response:"""

        try:
            self.system_prompt = system_prompt
            self.rag_chain = {
                'client': self.bedrock_client,
                'model_id': getattr(self.llm, 'model_name', 'anthropic.claude-3-haiku-20240307-v1:0'),
                'system_prompt': system_prompt,
                'config': self.current_model_config
            }
            
            logger.info("✅ AWS Bedrock RAG chain created successfully")
            return self.rag_chain

        except Exception as e:
            logger.error(f"❌ Failed to create AWS Bedrock RAG chain: {e}")
            raise

    def query(self, question: str, **kwargs) -> str:
        """Execute query using AWS Bedrock with comprehensive error handling"""
        if not hasattr(self, 'rag_chain'):
            self.create_rag_chain()

        try:
            # Get context from vector store
            context = self._get_context_from_vector_store(question, k=5)
            
            # Format prompt with context
            formatted_prompt = self.system_prompt.format(context=context, input=question)
            
            # Determine model type and call appropriate method
            model_id = self.rag_chain['model_id']
            
            if model_id.startswith('anthropic.claude'):
                return self._query_claude(formatted_prompt, **kwargs)
            elif model_id.startswith('amazon.titan'):
                return self._query_titan(formatted_prompt, **kwargs)
            elif model_id.startswith('meta.llama'):
                return self._query_llama(formatted_prompt, **kwargs)
            elif model_id.startswith('mistral.'):
                return self._query_mistral(formatted_prompt, **kwargs)
            else:
                return self._query_generic(formatted_prompt, **kwargs)

        except Exception as e:
            logger.error(f"❌ AWS Bedrock query failed: {e}")
            return self._fallback_query(question)

    def _query_claude(self, prompt: str, **kwargs) -> str:
        """Query Claude models with proper formatting"""
        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": kwargs.get('max_tokens', self.current_model_config['max_tokens']),
                "temperature": kwargs.get('temperature', 0.7),
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.rag_chain['model_id'],
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )

            result = json.loads(response['body'].read())
            return result['content'][0]['text']

        except self.ClientError as e:
            logger.error(f"❌ Claude API error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Claude query error: {e}")
            raise

    def _query_titan(self, prompt: str, **kwargs) -> str:
        """Query Amazon Titan models"""
        try:
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": kwargs.get('max_tokens', self.current_model_config['max_tokens']),
                    "temperature": kwargs.get('temperature', 0.7),
                    "topP": kwargs.get('top_p', 0.9),
                    "stopSequences": kwargs.get('stop_sequences', [])
                }
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.rag_chain['model_id'],
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )

            result = json.loads(response['body'].read())
            return result['results'][0]['outputText']

        except self.ClientError as e:
            logger.error(f"❌ Titan API error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Titan query error: {e}")
            raise

    def _query_llama(self, prompt: str, **kwargs) -> str:
        """Query Meta Llama models"""
        try:
            body = {
                "prompt": prompt,
                "max_gen_len": kwargs.get('max_tokens', self.current_model_config['max_tokens']),
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.9)
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.rag_chain['model_id'],
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )

            result = json.loads(response['body'].read())
            return result['generation']

        except self.ClientError as e:
            logger.error(f"❌ Llama API error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Llama query error: {e}")
            raise

    def _query_mistral(self, prompt: str, **kwargs) -> str:
        """Query Mistral AI models"""
        try:
            body = {
                "prompt": prompt,
                "max_tokens": kwargs.get('max_tokens', self.current_model_config['max_tokens']),
                "temperature": kwargs.get('temperature', 0.7),
                "top_p": kwargs.get('top_p', 0.9),
                "top_k": kwargs.get('top_k', 50)
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.rag_chain['model_id'],
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )

            result = json.loads(response['body'].read())
            return result['outputs'][0]['text']

        except self.ClientError as e:
            logger.error(f"❌ Mistral API error: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Mistral query error: {e}")
            raise

    def _query_generic(self, prompt: str, **kwargs) -> str:
        """Generic query method for unknown models"""
        try:
            # Try Claude format first as it's most common
            return self._query_claude(prompt, **kwargs)
        except:
            # Fallback to Titan format
            return self._query_titan(prompt, **kwargs)

    def _fallback_query(self, question: str) -> str:
        """Fallback query method"""
        try:
            context = self._get_context_from_vector_store(question, k=3)
            prompt = f"""Context: {context}

Question: {question}

Please provide a comprehensive Docker answer based on the context above."""
            
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"❌ AWS Bedrock fallback failed: {e}")
            return f"Error processing query with AWS Bedrock: {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        return {
            'model_id': self.rag_chain['model_id'],
            'config': self.current_model_config,
            'available_models': self.available_models,
            'supports_streaming': self.supports_streaming
        }

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Dict[str, float]:
        """Estimate cost for token usage"""
        config = self.current_model_config
        input_cost = (input_tokens / 1000) * config.get('input_cost_per_1k', 0.001)
        output_cost = (output_tokens / 1000) * config.get('output_cost_per_1k', 0.002)
        
        return {
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': input_cost + output_cost,
            'currency': 'USD'
        }

def register():
    """Register AWS Bedrock framework with factory"""
    return "aws_bedrock", AWSBedrockFramework