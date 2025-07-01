import json
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
import asyncio
import aiohttp
from providers.framework_factory import BaseFramework

logger = logging.getLogger(__name__)

class VercelAIFramework(BaseFramework):
    """Vercel AI Framework with edge optimization"""

    def initialize(self):
        """Initialize Vercel AI with edge computing features"""
        try:
            self._import_dependencies()
            self._configure_edge_runtime()
            self._setup_streaming()
            self._configure_providers()
            self._initialized = True
            logger.info("✅ Vercel AI framework initialized successfully")
        except Exception as e:
            logger.error(f"❌ Vercel AI initialization failed: {e}")
            raise

    def _import_dependencies(self):
        """Import required dependencies"""
        try:
            import aiohttp
            import asyncio
            import json
            
            self.aiohttp = aiohttp
            self.asyncio = asyncio
            self.json = json
            
        except ImportError as e:
            raise ImportError(f"Required packages: pip install aiohttp asyncio")

    def _configure_edge_runtime(self):
        """Configure edge runtime settings"""
        self.edge_config = {
            'timeout': self.kwargs.get('timeout', 30),
            'max_retries': self.kwargs.get('max_retries', 3),
            'edge_regions': self.kwargs.get('edge_regions', ['us-east-1', 'eu-west-1']),
            'cache_ttl': self.kwargs.get('cache_ttl', 300),  # 5 minutes
            'compression': self.kwargs.get('compression', True),
            'streaming_enabled': self.kwargs.get('streaming_enabled', True)
        }

        # Vercel AI endpoint configuration
        self.vercel_endpoint = self.kwargs.get('vercel_endpoint', 'https://api.vercel.com/v1/ai')
        self.vercel_token = self.kwargs.get('vercel_token')
        self.deployment_url = self.kwargs.get('deployment_url')

    def _setup_streaming(self):
        """Setup streaming capabilities for real-time responses"""
        self.streaming_config = {
            'chunk_size': self.kwargs.get('chunk_size', 1024),
            'buffer_size': self.kwargs.get('buffer_size', 8192),
            'stream_timeout': self.kwargs.get('stream_timeout', 60),
            'enable_sse': self.kwargs.get('enable_sse', True)
        }

    def _configure_providers(self):
        """Configure supported AI providers through Vercel"""
        self.provider_mappings = {
            'openai': {
                'endpoint': '/openai/chat/completions',
                'models': ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
                'supports_streaming': True,
                'supports_functions': True
            },
            'anthropic': {
                'endpoint': '/anthropic/messages',
                'models': ['claude-3-5-sonnet-20241022', 'claude-3-haiku-20240307'],
                'supports_streaming': True,
                'supports_functions': False
            },
            'google': {
                'endpoint': '/google/generateContent',
                'models': ['gemini-1.5-pro', 'gemini-1.5-flash'],
                'supports_streaming': True,
                'supports_functions': True
            },
            'cohere': {
                'endpoint': '/cohere/generate',
                'models': ['command-r', 'command-r-plus'],
                'supports_streaming': True,
                'supports_functions': False
            }
        }

        # Detect current provider
        model_name = getattr(self.llm, 'model_name', 'gpt-4o-mini')
        self.current_provider = self._detect_provider(model_name)

    def _detect_provider(self, model_name: str) -> str:
        """Detect provider from model name"""
        if model_name.startswith('gpt-'):
            return 'openai'
        elif model_name.startswith('claude-'):
            return 'anthropic'
        elif model_name.startswith('gemini-'):
            return 'google'
        elif model_name.startswith('command'):
            return 'cohere'
        else:
            return 'openai'  # Default

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create Vercel AI RAG chain with edge optimization"""
        if not system_prompt:
            system_prompt = """You are an expert Docker assistant optimized for edge computing with Vercel AI.

Your capabilities:
1. Ultra-fast responses optimized for edge deployment
2. Comprehensive Docker knowledge from retrieved documentation
3. Real-time streaming responses for better user experience
4. Cross-platform Docker solutions with performance focus
5. Edge-optimized caching and response compression

Instructions:
- Use the provided context to answer Docker questions with minimal latency
- Provide concise yet comprehensive responses optimized for streaming
- Include practical commands and examples that work across environments
- Focus on performance and efficiency in Docker operations
- Suggest edge-compatible deployment strategies

Context: {context}
Question: {input}

Provide an edge-optimized, comprehensive response:"""

        try:
            self.system_prompt = system_prompt
            self.rag_chain = {
                'endpoint': self.vercel_endpoint,
                'provider': self.current_provider,
                'model': getattr(self.llm, 'model_name', 'gpt-4o-mini'),
                'system_prompt': system_prompt,
                'edge_config': self.edge_config,
                'streaming_config': self.streaming_config
            }
            
            logger.info(f"✅ Vercel AI RAG chain created for provider: {self.current_provider}")
            return self.rag_chain

        except Exception as e:
            logger.error(f"❌ Failed to create Vercel AI RAG chain: {e}")
            raise

    def query(self, question: str, **kwargs) -> str:
        """Execute query using Vercel AI with edge optimization"""
        if not hasattr(self, 'rag_chain'):
            self.create_rag_chain()

        try:
            # Get context from vector store
            context = self._get_context_from_vector_store(question, k=5)
            
            # Check if streaming is requested and supported
            if kwargs.get('stream', False) and self.edge_config['streaming_enabled']:
                return self._stream_query(context, question, **kwargs)
            else:
                return self._standard_query(context, question, **kwargs)

        except Exception as e:
            logger.error(f"❌ Vercel AI query failed: {e}")
            return self._fallback_query(question)

    def _standard_query(self, context: str, question: str, **kwargs) -> str:
        """Execute standard (non-streaming) query"""
        try:
            # Prepare request payload
            payload = self._prepare_payload(context, question, **kwargs)
            
            # Execute synchronous request
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response = loop.run_until_complete(self._make_api_request(payload))
                return self._extract_response(response)
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"❌ Standard query failed: {e}")
            raise

    def _stream_query(self, context: str, question: str, **kwargs) -> str:
        """Execute streaming query for real-time responses"""
        try:
            # Prepare streaming payload
            payload = self._prepare_payload(context, question, stream=True, **kwargs)
            
            # Execute streaming request
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                response_chunks = loop.run_until_complete(self._stream_api_request(payload))
                return ''.join(response_chunks)
            finally:
                loop.close()

        except Exception as e:
            logger.error(f"❌ Streaming query failed: {e}")
            # Fallback to standard query
            return self._standard_query(context, question, **kwargs)

    def _prepare_payload(self, context: str, question: str, **kwargs) -> Dict[str, Any]:
        """Prepare API request payload"""
        formatted_prompt = self.system_prompt.format(context=context, input=question)
        
        provider_config = self.provider_mappings[self.current_provider]
        
        if self.current_provider == 'openai':
            payload = {
                'model': self.rag_chain['model'],
                'messages': [
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': formatted_prompt}
                ],
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1000),
                'stream': kwargs.get('stream', False)
            }
        elif self.current_provider == 'anthropic':
            payload = {
                'model': self.rag_chain['model'],
                'max_tokens': kwargs.get('max_tokens', 1000),
                'temperature': kwargs.get('temperature', 0.7),
                'messages': [
                    {'role': 'user', 'content': formatted_prompt}
                ],
                'stream': kwargs.get('stream', False)
            }
        elif self.current_provider == 'google':
            payload = {
                'model': self.rag_chain['model'],
                'contents': [
                    {'parts': [{'text': formatted_prompt}]}
                ],
                'generationConfig': {
                    'temperature': kwargs.get('temperature', 0.7),
                    'maxOutputTokens': kwargs.get('max_tokens', 1000)
                },
                'stream': kwargs.get('stream', False)
            }
        else:  # cohere and others
            payload = {
                'model': self.rag_chain['model'],
                'prompt': formatted_prompt,
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1000),
                'stream': kwargs.get('stream', False)
            }

        return payload

    async def _make_api_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make async API request to Vercel AI"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'RAG-Vercel-Client/1.0'
        }

        if self.vercel_token:
            headers['Authorization'] = f'Bearer {self.vercel_token}'

        # Determine endpoint
        provider_config = self.provider_mappings[self.current_provider]
        endpoint = f"{self.vercel_endpoint}{provider_config['endpoint']}"

        async with self.aiohttp.ClientSession(
            timeout=self.aiohttp.ClientTimeout(total=self.edge_config['timeout'])
        ) as session:
            async with session.post(endpoint, json=payload, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {response.status} - {error_text}")

    async def _stream_api_request(self, payload: Dict[str, Any]) -> List[str]:
        """Make streaming API request to Vercel AI"""
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
            'User-Agent': 'RAG-Vercel-Client/1.0'
        }

        if self.vercel_token:
            headers['Authorization'] = f'Bearer {self.vercel_token}'

        provider_config = self.provider_mappings[self.current_provider]
        endpoint = f"{self.vercel_endpoint}{provider_config['endpoint']}"

        response_chunks = []

        async with self.aiohttp.ClientSession(
            timeout=self.aiohttp.ClientTimeout(total=self.streaming_config['stream_timeout'])
        ) as session:
            async with session.post(endpoint, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            line_str = line.decode('utf-8').strip()
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]  # Remove 'data: ' prefix
                                if data_str != '[DONE]':
                                    try:
                                        chunk_data = json.loads(data_str)
                                        content = self._extract_chunk_content(chunk_data)
                                        if content:
                                            response_chunks.append(content)
                                    except json.JSONDecodeError:
                                        continue
                else:
                    error_text = await response.text()
                    raise Exception(f"Streaming request failed: {response.status} - {error_text}")

        return response_chunks

    def _extract_response(self, response: Dict[str, Any]) -> str:
        """Extract response content based on provider"""
        if self.current_provider == 'openai':
            return response['choices'][0]['message']['content']
        elif self.current_provider == 'anthropic':
            return response['content'][0]['text']
        elif self.current_provider == 'google':
            return response['candidates'][0]['content']['parts'][0]['text']
        elif self.current_provider == 'cohere':
            return response['generations'][0]['text']
        else:
            return str(response)

    def _extract_chunk_content(self, chunk_data: Dict[str, Any]) -> str:
        """Extract content from streaming chunk based on provider"""
        try:
            if self.current_provider == 'openai':
                return chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
            elif self.current_provider == 'anthropic':
                return chunk_data.get('delta', {}).get('text', '')
            elif self.current_provider == 'google':
                candidates = chunk_data.get('candidates', [])
                if candidates:
                    return candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            elif self.current_provider == 'cohere':
                return chunk_data.get('text', '')
            return ''
        except (KeyError, IndexError, TypeError):
            return ''

    def _fallback_query(self, question: str) -> str:
        """Fallback query method"""
        try:
            context = self._get_context_from_vector_store(question, k=3)
            prompt = f"""Context: {context}

Question: {question}

Please provide a comprehensive Docker answer based on the context above."""
            
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"❌ Vercel AI fallback failed: {e}")
            return f"Error processing query with Vercel AI: {str(e)}"

    def get_edge_metrics(self) -> Dict[str, Any]:
        """Get edge performance metrics"""
        return {
            'provider': self.current_provider,
            'model': self.rag_chain.get('model', 'unknown'),
            'edge_regions': self.edge_config['edge_regions'],
            'streaming_enabled': self.edge_config['streaming_enabled'],
            'cache_ttl': self.edge_config['cache_ttl'],
            'timeout': self.edge_config['timeout'],
            'compression': self.edge_config['compression']
        }

    def configure_edge_caching(self, ttl: int = 300, regions: List[str] = None) -> bool:
        """Configure edge caching settings"""
        try:
            self.edge_config['cache_ttl'] = ttl
            if regions:
                self.edge_config['edge_regions'] = regions
            logger.info(f"✅ Edge caching configured: TTL={ttl}s, Regions={regions}")
            return True
        except Exception as e:
            logger.error(f"❌ Edge caching configuration failed: {e}")
            return False

def register():
    """Register Vercel AI framework with factory"""
    return "vercel", VercelAIFramework
