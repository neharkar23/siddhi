# providers/framework_factory.py

"""
AI Framework Factory Module

This module provides a unified interface for various AI frameworks including
LangChain, LlamaIndex, AutoGen, CrewAI, Neo4j, AWS Bedrock, Graphlit, LiteLLM,
Vercel AI SDK, and Cleanlab.

"""
from pathlib import Path
import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
import json
import importlib
import pkgutil
import os

from config.core_config import Framework, config_manager
from providers.llm_factory import BaseLLM
from providers.vector_store_factory import BaseVectorStore

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# BASE FRAMEWORK CLASS
# =============================================================================


class BaseFramework(ABC):
    """
    Abstract base class for AI frameworks
    
    This class defines the common interface that all framework implementations
    must follow, ensuring consistency across different AI frameworks.
    """

    def __init__(self, llm: BaseLLM, vector_store: BaseVectorStore, **kwargs):
        """
        Initialize the framework with required components
        
        Args:
            llm: Language model instance
            vector_store: Vector store instance for document retrieval
            **kwargs: Additional framework-specific configuration
        """
        self.llm = llm
        self.vector_store = vector_store
        self.kwargs = kwargs
        self.client = None
        self._initialized = False

    @abstractmethod
    def initialize(self):
        """Initialize the framework with required dependencies"""
        """Initialize framework with proper error handling"""
        try:
            self._setup_dependencies()
            self._initialized = True
            logger.info(f"{self.__class__.__name__} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            raise

    @abstractmethod
    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create RAG chain for question answering"""
        pass

    @abstractmethod
    def query(self, question: str, **kwargs) -> str:
        """Query the RAG system with a question"""
        pass

    def _get_context_from_vector_store(self, question: str, k: int = 5) -> str:
        """
        Helper method to get context from vector store
        
        Args:
            question: The question to search for
            k: Number of documents to retrieve
            
        Returns:
            Combined context string from retrieved documents
        """
        try:
            docs = self.vector_store.similarity_search(question, k=k)
            return "\n".join([doc[0] if isinstance(doc, tuple) else str(doc) for doc in docs])
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""

    def _create_default_prompt(self, context: str, question: str) -> str:
        """
        Create a default prompt template
        
        Args:
            context: Retrieved context
            question: User question
            
        Returns:
            Formatted prompt string
        """
        return f"""Context: {context}

Question: {question}

Please provide a comprehensive answer based on the context above."""

# =============================================================================
# FRAMEWORK IMPLEMENTATIONS
# =============================================================================

class LangChainFramework(BaseFramework):
    """LangChain framework implementation for building LLM applications"""

    def initialize(self):
        """Initialize LangChain components"""
        try:
            from langchain.chains import RetrievalQA
            from langchain.prompts import PromptTemplate
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chains.combine_documents import create_stuff_documents_chain
            from langchain.chains import create_retrieval_chain

            # Store imported classes
            self.RetrievalQA = RetrievalQA
            self.PromptTemplate = PromptTemplate
            self.ChatPromptTemplate = ChatPromptTemplate
            self.create_stuff_documents_chain = create_stuff_documents_chain
            self.create_retrieval_chain = create_retrieval_chain

            logger.info("LangChain framework initialized successfully")
        except ImportError as e:
            raise ImportError(f"LangChain packages are required: {e}")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create LangChain RAG chain"""
        if not system_prompt:
            system_prompt = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

Context: {context}
Question: {input}
Answer:"""

        try:
            # Create prompt template
            prompt = self.ChatPromptTemplate.from_template(system_prompt)
            
            # Create document chain
            document_chain = self.create_stuff_documents_chain(self.llm, prompt)
            
            # Create retriever
            retriever = (self.vector_store.client.as_retriever() 
                        if hasattr(self.vector_store.client, 'as_retriever') else None)
            
            if retriever:
                self.chain = self.create_retrieval_chain(retriever, document_chain)
            else:
                self.chain = document_chain
                
            return self.chain
        except Exception as e:
            logger.error(f"Failed to create LangChain RAG chain: {e}")
            raise

    def query(self, question: str, **kwargs) -> str:
        """Query using LangChain"""
        if not hasattr(self, 'chain'):
            self.create_rag_chain()

        try:
            if hasattr(self.chain, 'invoke'):
                result = self.chain.invoke({"input": question})
                return result.get('answer', str(result))
            else:
                # Fallback to direct context retrieval
                context = self._get_context_from_vector_store(question)
                prompt = self._create_default_prompt(context, question)
                return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"LangChain query error: {e}")
            raise


class LlamaIndexFramework(BaseFramework):
    """LlamaIndex framework implementation using modern API"""

    def initialize(self):
        """Initialize LlamaIndex components"""
        try:
            from llama_index.core import VectorStoreIndex, Settings
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.schema import Document
            from llama_index.core import get_response_synthesizer

            # Store imported classes
            self.VectorStoreIndex = VectorStoreIndex
            self.Settings = Settings
            self.RetrieverQueryEngine = RetrieverQueryEngine
            self.VectorIndexRetriever = VectorIndexRetriever
            self.Document = Document
            self.get_response_synthesizer = get_response_synthesizer

            # Configure settings
            self._configure_settings()
            logger.info("LlamaIndex framework initialized successfully")
        except ImportError as e:
            raise ImportError(f"LlamaIndex packages are required: {e}")

    def _configure_settings(self):
        """Configure LlamaIndex global settings"""
        try:
            if hasattr(self.llm, 'model_name'):
                self.Settings.llm = LlamaIndexLLMWrapper(self.llm)
            
            self.Settings.context_window = getattr(self.llm, 'context_window', 4096)
            self.Settings.num_output = getattr(self.llm, 'max_tokens', 256)
        except Exception as e:
            logger.warning(f"Could not configure LlamaIndex settings: {e}")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create LlamaIndex RAG chain"""
        try:
            documents = self._convert_vector_store_to_documents()
            
            if documents:
                self.index = self.VectorStoreIndex.from_documents(documents, show_progress=True)
            else:
                self.index = self.VectorStoreIndex([])

            # Create retriever and response synthesizer
            retriever = self.VectorIndexRetriever(index=self.index, similarity_top_k=5)
            response_synthesizer = self.get_response_synthesizer(
                response_mode="compact", use_async=False
            )

            # Create query engine
            self.query_engine = self.RetrieverQueryEngine(
                retriever=retriever, response_synthesizer=response_synthesizer
            )

            logger.info("LlamaIndex RAG chain created successfully")
            return self.query_engine
        except Exception as e:
            logger.warning(f"Could not create LlamaIndex RAG chain: {e}")
            return self._create_fallback_query_engine()

    def _convert_vector_store_to_documents(self) -> List:
        """Convert vector store data to LlamaIndex Document format"""
        try:
            if hasattr(self.vector_store, '_documents'):
                documents = []
                for i, (content, metadata) in enumerate(self.vector_store._documents):
                    doc = self.Document(
                        text=content,
                        metadata=metadata if isinstance(metadata, dict) else {"id": str(i)}
                    )
                    documents.append(doc)
                return documents
            else:
                # Create sample documents for testing
                sample_docs = [
                    "This is a test document about AI frameworks and their capabilities.",
                    "LlamaIndex is a framework for building applications with LLMs and vector stores.",
                    "Vector stores are used for similarity search and retrieval augmented generation.",
                    "RAG combines retrieval and generation for better contextual answers.",
                    "Knowledge graphs and vector databases work together in modern AI systems."
                ]
                return [self.Document(text=doc, metadata={"id": str(i)}) 
                       for i, doc in enumerate(sample_docs)]
        except Exception as e:
            logger.error(f"Error converting vector store to documents: {e}")
            return []

    def _create_fallback_query_engine(self):
        """Create fallback query engine when main creation fails"""
        return FallbackLlamaIndexQueryEngine(self.llm, self.vector_store)

    def query(self, question: str, **kwargs) -> str:
        """Query using LlamaIndex"""
        if not hasattr(self, 'query_engine'):
            self.create_rag_chain()

        try:
            response = self.query_engine.query(question)
            
            # Handle different response types
            if hasattr(response, 'response'):
                return str(response.response)
            elif hasattr(response, 'text'):
                return str(response.text)
            else:
                return str(response)
        except Exception as e:
            logger.error(f"LlamaIndex query error: {e}")
            return self._fallback_query(question)

    def _fallback_query(self, question: str) -> str:
        """Fallback query method"""
        try:
            context = self._get_context_from_vector_store(question)
            prompt = self._create_default_prompt(context, question)
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Fallback query error: {e}")
            return f"Error processing query: {str(e)}"


class AutoGenFramework(BaseFramework):
    """AutoGen framework implementation for multi-agent conversations"""

    def initialize(self):
        """Initialize AutoGen components"""
        try:
            import autogen
            self.autogen = autogen
            logger.info("AutoGen framework initialized successfully")
        except ImportError:
            raise ImportError("pyautogen package is required for AutoGen framework")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create AutoGen agents"""
        if not system_prompt:
            system_prompt = """You are an assistant for question-answering tasks.
Use the retrieved context to answer questions accurately."""

        # Create AutoGen agents
        self.assistant = self.autogen.AssistantAgent(
            name="assistant",
            system_message=system_prompt,
            llm_config={"model": self.llm.model_name}
        )

        self.user_proxy = self.autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            code_execution_config=False
        )

        return self.assistant

    def query(self, question: str, **kwargs) -> str:
        """Query using AutoGen multi-agent system"""
        if not hasattr(self, 'assistant'):
            self.create_rag_chain()

        try:
            context = self._get_context_from_vector_store(question)
            formatted_question = f"""Context: {context}

Question: {question}

Please provide a comprehensive answer based on the context."""

            # Start conversation
            self.user_proxy.initiate_chat(self.assistant, message=formatted_question)

            # Get the last message from assistant
            chat_history = self.user_proxy.chat_messages[self.assistant]
            return chat_history[-1]['content']
        except Exception as e:
            logger.error(f"AutoGen query error: {e}")
            raise


class CrewAIFramework(BaseFramework):
    """CrewAI framework implementation for AI agent crews"""

    def initialize(self):
        """Initialize CrewAI components"""
        try:
            from crewai import Agent, Task, Crew
            self.Agent = Agent
            self.Task = Task
            self.Crew = Crew
            logger.info("CrewAI framework initialized successfully")
        except ImportError:
            raise ImportError("crewai package is required for CrewAI framework")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create CrewAI research agent"""
        if not system_prompt:
            system_prompt = """You are a research assistant specialized in answering questions
based on retrieved documents. Provide accurate and comprehensive answers."""

        self.researcher = self.Agent(
            role='Research Assistant',
            goal='Answer questions based on retrieved context',
            backstory=system_prompt,
            verbose=True,
            allow_delegation=False
        )

        return self.researcher

    def query(self, question: str, **kwargs) -> str:
        """Query using CrewAI agent crew"""
        if not hasattr(self, 'researcher'):
            self.create_rag_chain()

        try:
            context = self._get_context_from_vector_store(question)

            # Create task
            task = self.Task(
                description=f"""Based on the following context, answer the question: {question}

Context: {context}

Provide a comprehensive and accurate answer.""",
                agent=self.researcher
            )

            # Create crew and execute
            crew = self.Crew(
                agents=[self.researcher],
                tasks=[task],
                verbose=True
            )

            result = crew.kickoff()
            return str(result)
        except Exception as e:
            logger.error(f"CrewAI query error: {e}")
            raise


class Neo4jFramework(BaseFramework):
    """Neo4j framework implementation for graph-based RAG"""

    def initialize(self):
        """Initialize Neo4j connection"""
        try:
            from neo4j import GraphDatabase
            
            uri = config_manager.get_api_key('neo4j_uri') or 'bolt://localhost:7687'
            user = config_manager.get_api_key('neo4j_user') or 'neo4j'
            password = config_manager.get_api_key('neo4j_password') or 'password'
            
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Neo4j framework initialized successfully")
        except ImportError:
            raise ImportError("neo4j package is required for Neo4j framework")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Initialize knowledge graph structure"""
        with self.driver.session() as session:
            # Create constraints and indexes
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.content)")
        
        return self.driver

    def query(self, question: str, **kwargs) -> str:
        """Query using Neo4j graph database"""
        try:
            # Get context from vector store
            vector_context = self._get_context_from_vector_store(question)
            
            # Query knowledge graph for relationships
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (d:Document)
                    WHERE d.content CONTAINS $keyword
                    RETURN d.content as content
                    LIMIT 5
                """, keyword=question.split()[0] if question.split() else "")
                
                graph_context = [record["content"] for record in result]

            # Combine vector and graph context
            combined_context = vector_context + "\n" + "\n".join(graph_context)
            prompt = self._create_default_prompt(combined_context, question)
            
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Neo4j query error: {e}")
            raise


class AWSBedrockFramework(BaseFramework):
    """AWS Bedrock framework implementation for cloud-based LLMs"""

    def initialize(self):
        """Initialize AWS Bedrock client"""
        try:
            import boto3
            
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                aws_access_key_id=config_manager.get_api_key('aws_access_key'),
                aws_secret_access_key=config_manager.get_api_key('aws_secret_key'),
                region_name=config_manager.get_api_key('aws_region') or 'us-east-1'
            )
            logger.info("AWS Bedrock framework initialized successfully")
        except ImportError:
            raise ImportError("boto3 package is required for AWS Bedrock framework")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Setup AWS Bedrock RAG chain"""
        return self.bedrock_client

    def query(self, question: str, **kwargs) -> str:
        """Query using AWS Bedrock models"""
        try:
            context = self._get_context_from_vector_store(question)
            prompt = self._create_default_prompt(context, question)

            # Use Bedrock model (example with Claude)
            response = self.bedrock_client.invoke_model(
                modelId='anthropic.claude-v2',
                body=json.dumps({
                    'prompt': f"\n\nHuman: {prompt}\n\nAssistant:",
                    'max_tokens_to_sample': 1000,
                    'temperature': 0.7
                })
            )

            result = json.loads(response['body'].read())
            return result['completion']
        except Exception as e:
            logger.error(f"AWS Bedrock query error: {e}")
            raise


class LiteLLMFramework(BaseFramework):
    """LiteLLM framework implementation for unified LLM interface"""

    def initialize(self):
        """Initialize LiteLLM components"""
        try:
            import litellm
            self.litellm = litellm
            
            # Configure LiteLLM with API keys
            if hasattr(self.llm, 'api_key'):
                litellm.api_key = self.llm.api_key
                
            logger.info("LiteLLM framework initialized successfully")
        except ImportError:
            raise ImportError("litellm package is required for LiteLLM framework")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create LiteLLM RAG chain"""
        if not system_prompt:
            system_prompt = """You are an AI assistant that answers questions based on provided context.
Use the context to provide accurate and helpful responses."""
        
        self.system_prompt = system_prompt
        return self.litellm

    def query(self, question: str, **kwargs) -> str:
        """Query using LiteLLM unified interface"""
        if not hasattr(self, 'system_prompt'):
            self.create_rag_chain()

        try:
            context = self._get_context_from_vector_store(question)
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self._create_default_prompt(context, question)}
            ]

            # Use LiteLLM completion
            response = self.litellm.completion(
                model=self.llm.model_name,
                messages=messages,
                temperature=getattr(self.llm, 'temperature', 0.7),
                max_tokens=getattr(self.llm, 'max_tokens', 1000)
            )

            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LiteLLM query error: {e}")
            # Fallback to direct LLM
            context = self._get_context_from_vector_store(question)
            prompt = self._create_default_prompt(context, question)
            return self.llm.generate(prompt)


class VercelFramework(BaseFramework):
    """Vercel AI SDK framework implementation for edge AI applications"""

    def initialize(self):
        """Initialize Vercel AI SDK components"""
        try:
            # Note: Vercel AI SDK is primarily JavaScript/TypeScript
            # This is a Python wrapper implementation
            import requests
            self.requests = requests
            
            # Configure Vercel AI endpoint
            self.vercel_endpoint = config_manager.get_api_key('vercel_endpoint')
            self.vercel_token = config_manager.get_api_key('vercel_token')
            
            if not self.vercel_endpoint:
                logger.warning("Vercel endpoint not configured, using fallback mode")
                
            logger.info("Vercel AI framework initialized successfully")
        except ImportError:
            raise ImportError("requests package is required for Vercel framework")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create Vercel AI RAG chain"""
        if not system_prompt:
            system_prompt = """You are an AI assistant optimized for edge computing.
Provide fast and efficient responses based on the given context."""
        
        self.system_prompt = system_prompt
        return {"endpoint": self.vercel_endpoint, "system_prompt": system_prompt}

    def query(self, question: str, **kwargs) -> str:
        """Query using Vercel AI SDK"""
        if not hasattr(self, 'system_prompt'):
            self.create_rag_chain()

        try:
            context = self._get_context_from_vector_store(question)
            
            if self.vercel_endpoint and self.vercel_token:
                # Make API call to Vercel AI endpoint
                payload = {
                    "model": self.llm.model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self._create_default_prompt(context, question)}
                    ],
                    "temperature": getattr(self.llm, 'temperature', 0.7),
                    "max_tokens": getattr(self.llm, 'max_tokens', 1000)
                }
                
                headers = {
                    "Authorization": f"Bearer {self.vercel_token}",
                    "Content-Type": "application/json"
                }
                
                response = self.requests.post(
                    self.vercel_endpoint,
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('choices', [{}])[0].get('message', {}).get('content', '')
                else:
                    logger.error(f"Vercel API error: {response.status_code}")
                    raise Exception(f"Vercel API error: {response.status_code}")
            else:
                # Fallback to direct LLM
                prompt = self._create_default_prompt(context, question)
                return self.llm.generate(prompt)
                
        except Exception as e:
            logger.error(f"Vercel query error: {e}")
            # Fallback to direct LLM
            context = self._get_context_from_vector_store(question)
            prompt = self._create_default_prompt(context, question)
            return self.llm.generate(prompt)


class CleanlabFramework(BaseFramework):
    """Cleanlab framework implementation for data-centric AI and quality assessment"""

    def initialize(self):
        """Initialize Cleanlab components"""
        try:
            import cleanlab
            from cleanlab.datalab import Datalab
            
            self.cleanlab = cleanlab
            self.Datalab = Datalab
            
            logger.info("Cleanlab framework initialized successfully")
        except ImportError:
            raise ImportError("cleanlab package is required for Cleanlab framework")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create Cleanlab-enhanced RAG chain with data quality assessment"""
        if not system_prompt:
            system_prompt = """You are an AI assistant that provides high-quality answers 
based on clean, validated data. You assess data quality before providing responses."""
        
        self.system_prompt = system_prompt
        
        # Initialize data quality assessment
        try:
            # Prepare data for quality assessment
            documents = self._prepare_documents_for_cleanlab()
            
            if documents:
                # Create Datalab instance for quality assessment
                self.datalab = self.Datalab(data=documents)
                self.datalab.find_issues()
                
                logger.info("Cleanlab data quality assessment completed")
            else:
                logger.warning("No documents available for Cleanlab assessment")
                
        except Exception as e:
            logger.warning(f"Cleanlab assessment failed, using fallback: {e}")
            self.datalab = None
            
        return self.datalab

    def _prepare_documents_for_cleanlab(self) -> List[Dict]:
        """Prepare documents for Cleanlab quality assessment"""
        try:
            documents = []
            
            if hasattr(self.vector_store, '_documents'):
                for i, (content, metadata) in enumerate(self.vector_store._documents):
                    documents.append({
                        'text': content,
                        'id': metadata.get('id', str(i)) if isinstance(metadata, dict) else str(i),
                        'metadata': metadata if isinstance(metadata, dict) else {}
                    })
            else:
                # Create sample documents
                sample_docs = [
                    "High-quality document about AI frameworks and their applications.",
                    "Comprehensive guide to vector stores and similarity search.",
                    "Detailed explanation of RAG systems and their benefits.",
                    "Technical overview of knowledge graphs in AI systems.",
                    "Best practices for building robust AI applications."
                ]
                
                documents = [
                    {'text': doc, 'id': str(i), 'metadata': {'quality': 'high'}}
                    for i, doc in enumerate(sample_docs)
                ]
                
            return documents
        except Exception as e:
            logger.error(f"Error preparing documents for Cleanlab: {e}")
            return []

    def query(self, question: str, **kwargs) -> str:
        """Query using Cleanlab-enhanced data quality assessment"""
        if not hasattr(self, 'system_prompt'):
            self.create_rag_chain()

        try:
            context = self._get_context_from_vector_store(question)
            
            # Assess query quality if Cleanlab is available
            quality_score = self._assess_query_quality(question)
            
            # Enhance prompt with quality information
            enhanced_prompt = f"""Data Quality Score: {quality_score:.2f}/1.0

{self._create_default_prompt(context, question)}

Note: This response is based on quality-assessed data."""

            response = self.llm.generate(enhanced_prompt)
            
            # Add quality metadata to response
            if quality_score < 0.7:
                response += f"\n\n[Quality Notice: Input data quality score is {quality_score:.2f}. Response reliability may be affected.]"
                
            return response
            
        except Exception as e:
            logger.error(f"Cleanlab query error: {e}")
            # Fallback to standard query
            context = self._get_context_from_vector_store(question)
            prompt = self._create_default_prompt(context, question)
            return self.llm.generate(prompt)

    def _assess_query_quality(self, question: str) -> float:
        """Assess the quality of the input query"""
        try:
            # Simple quality assessment based on question characteristics
            quality_factors = []
            
            # Length check
            if 10 <= len(question) <= 200:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.5)
            
            # Word count check
            word_count = len(question.split())
            if 3 <= word_count <= 50:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.6)
            
            # Question mark check
            if question.strip().endswith('?'):
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.8)
            
            # Calculate average quality score
            return sum(quality_factors) / len(quality_factors)
            
        except Exception as e:
            logger.error(f"Error assessing query quality: {e}")
            return 0.5  # Default moderate quality score


class GraphlitFramework(BaseFramework):
    """Graphlit framework implementation for knowledge graph processing"""

    def initialize(self):
        """Initialize Graphlit components"""
        # Graphlit-specific initialization would go here
        # This is a placeholder implementation
        logger.info("Graphlit framework initialized (placeholder implementation)")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create Graphlit RAG chain"""
        if not system_prompt:
            system_prompt = """You are an AI assistant specialized in processing 
knowledge graphs and providing insights from structured data."""
        
        self.system_prompt = system_prompt
        return None  # Placeholder

    def query(self, question: str, **kwargs) -> str:
        """Query using Graphlit knowledge graph processing"""
        try:
            context = self._get_context_from_vector_store(question)
            prompt = self._create_default_prompt(context, question)
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"Graphlit query error: {e}")
            return f"Error processing query with Graphlit: {str(e)}"

# =============================================================================
# SUPPORTING CLASSES
# =============================================================================

class LlamaIndexLLMWrapper:
    """Wrapper to make LLM compatible with LlamaIndex interface"""

    def __init__(self, llm):
        self.llm = llm
        self.model_name = getattr(llm, 'model_name', 'mock-model')
        self.temperature = getattr(llm, 'temperature', 0.7)
        self.max_tokens = getattr(llm, 'max_tokens', 1000)

    def complete(self, prompt: str, **kwargs) -> str:
        """LlamaIndex complete interface"""
        return self.llm.generate(prompt)

    def chat(self, messages, **kwargs) -> str:
        """LlamaIndex chat interface"""
        if isinstance(messages, list) and len(messages) > 0:
            last_message = messages[-1]
            prompt = (last_message.get('content', str(last_message)) 
                     if isinstance(last_message, dict) else str(last_message))
        else:
            prompt = str(messages)
        return self.llm.generate(prompt)

    def predict(self, prompt: str, **kwargs) -> str:
        """Prediction interface"""
        return self.llm.generate(prompt)

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.llm.generate(prompt)


class FallbackLlamaIndexQueryEngine:
    """Fallback query engine when LlamaIndex creation fails"""

    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store

    def query(self, question: str):
        """Simple query implementation"""
        try:
            docs = self.vector_store.similarity_search(question, k=5)
            context = "\n".join([doc[0] if isinstance(doc, tuple) else str(doc) for doc in docs])
            prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
            response = self.llm.generate(prompt)
            return MockLlamaIndexResponse(response)
        except Exception as e:
            return MockLlamaIndexResponse(f"Error: {str(e)}")


class MockLlamaIndexResponse:
    """Mock response object that mimics LlamaIndex response structure"""

    def __init__(self, response_text: str):
        self.response = response_text
        self.text = response_text
        self.source_nodes = []
        self.metadata = {}

    def __str__(self):
        return self.response

# =============================================================================
# FRAMEWORK FACTORY
# =============================================================================

class FrameworkFactory:
    """
    Factory class for creating framework instances
    
    This factory provides a unified interface for creating different AI framework
    instances with proper initialization and configuration.
    """
    def __init__(self):
        self._frameworks = {}
        self._discover_frameworks()

    
    def _discover_frameworks(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for _, module_name, _ in pkgutil.iter_modules([current_dir]):
            if module_name.endswith("_framework"):
                try:
                    module = importlib.import_module(f"providers.{module_name}")
                    if hasattr(module, "register"):
                        name, cls = module.register()
                        self._frameworks[name] = cls
                        logger.info(f"Registered framework: {name}")
                except Exception as e:
                    logger.warning(f"Failed to import {module_name}: {e}")

    def list_supported_frameworks(self):
        return list(self._frameworks.keys())

    def create_framework(self, provider, llm, vector_store, **kwargs):
        if provider not in self._frameworks:
            raise ValueError(f"Framework '{provider}' not found. Available: {list(self._frameworks.keys())}")
        return self._frameworks[provider](llm, vector_store, **kwargs)

    @classmethod
    def create_framework(
        cls,
        provider: Framework,
        llm: BaseLLM,
        vector_store: BaseVectorStore,
        **kwargs
    ) -> BaseFramework:
        """
        Create framework instance based on provider
        
        Args:
            provider: Framework enum specifying which framework to use
            llm: Language model instance
            vector_store: Vector store instance
            **kwargs: Additional framework-specific configuration
            
        Returns:
            Initialized framework instance
            
        Raises:
            ValueError: If provider is not supported
            ImportError: If required packages are not installed
        """
        if provider not in cls._providers:
            available_providers = [p.value for p in cls._providers.keys()]
            raise ValueError(
                f"Unsupported framework provider: {provider}. "
                f"Available providers: {available_providers}"
            )

        try:
            # Create and initialize framework
            framework_class = cls._providers[provider]
            framework = framework_class(llm=llm, vector_store=vector_store, **kwargs)
            framework.initialize()
            
            logger.info(f"Successfully created {provider.value} framework")
            return framework
            
        except ImportError as e:
            logger.error(f"Failed to create {provider.value} framework: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating {provider.value} framework: {e}")
            raise

    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get list of available framework providers
        
        Returns:
            List of available provider names
        """
        return [provider.value for provider in cls._providers.keys()]

    @classmethod
    def get_provider_info(cls) -> Dict[str, str]:
        """
        Get information about available providers
        
        Returns:
            Dictionary mapping provider names to descriptions
        """
        provider_info = {
            Framework.LANGCHAIN.value: "Framework for building LLM applications with chains",
            Framework.LLAMAINDEX.value: "Data framework for LLM applications with advanced indexing",
            Framework.AUTOGEN.value: "Multi-agent conversation framework",
            Framework.CREWAI.value: "Framework for orchestrating AI agent crews",
            Framework.NEO4J.value: "Graph database framework for knowledge graphs",
            Framework.AWS_BEDROCK.value: "AWS managed service for foundation models",
            Framework.LITELLM.value: "Unified interface for multiple LLM providers",
            Framework.VERCEL.value: "Edge-optimized AI SDK for web applications",
            Framework.CLEANLAB.value: "Data-centric AI framework for quality assessment",
            Framework.GRAPHLIT.value: "Knowledge graph processing framework",
        }
        return provider_info

    @classmethod
    def validate_framework_requirements(cls, provider: Framework) -> Dict[str, bool]:
        """
        Validate if framework requirements are met
        
        Args:
            provider: Framework to validate
            
        Returns:
            Dictionary with validation results
        """
        requirements = {
            Framework.LANGCHAIN: ["langchain", "langchain_core"],
            Framework.LLAMAINDEX: ["llama_index"],
            Framework.AUTOGEN: ["autogen"],
            Framework.CREWAI: ["crewai"],
            Framework.NEO4J: ["neo4j"],
            Framework.AWS_BEDROCK: ["boto3"],
            Framework.LITELLM: ["litellm"],
            Framework.VERCEL: ["requests"],
            Framework.CLEANLAB: ["cleanlab"],
            Framework.GRAPHLIT: [],  # No specific requirements for placeholder
        }

        validation_results = {"provider": provider.value, "requirements_met": True, "missing_packages": []}

        if provider in requirements:
            for package in requirements[provider]:
                try:
                    __import__(package)
                except ImportError:
                    validation_results["requirements_met"] = False
                    validation_results["missing_packages"].append(package)

        return validation_results
