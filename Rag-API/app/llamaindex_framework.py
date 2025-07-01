
import logging
from typing import Any, Dict, List, Optional
from providers.framework_factory import BaseFramework

logger = logging.getLogger(__name__)

class LlamaIndexFramework(BaseFramework):
    """LlamaIndex Framework with advanced indexing capabilities"""

    def initialize(self):
        """Initialize LlamaIndex with comprehensive setup"""
        try:
            self._import_dependencies()
            self._configure_global_settings()
            self._setup_embedding_model()
            self._initialized = True
            logger.info("✅ LlamaIndex framework initialized successfully")
        except ImportError as e:
            raise ImportError(f"LlamaIndex packages required: {e}")
        except Exception as e:
            logger.error(f"❌ LlamaIndex initialization failed: {e}")
            raise

    def _import_dependencies(self):
        """Import LlamaIndex dependencies"""
        from llama_index.core import VectorStoreIndex, Settings, StorageContext
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.core.schema import Document, TextNode
        from llama_index.core import get_response_synthesizer
        from llama_index.core.node_parser import SimpleNodeParser
        from llama_index.core.service_context import ServiceContext
        
        self.VectorStoreIndex = VectorStoreIndex
        self.Settings = Settings
        self.StorageContext = StorageContext
        self.RetrieverQueryEngine = RetrieverQueryEngine
        self.VectorIndexRetriever = VectorIndexRetriever
        self.Document = Document
        self.TextNode = TextNode
        self.get_response_synthesizer = get_response_synthesizer
        self.SimpleNodeParser = SimpleNodeParser

    def _configure_global_settings(self):
        """Configure LlamaIndex global settings"""
        try:
            # Configure LLM
            self.Settings.llm = LlamaIndexLLMWrapper(self.llm)
            self.Settings.context_window = getattr(self.llm, 'context_window', 4096)
            self.Settings.num_output = getattr(self.llm, 'max_tokens', 512)
            
            # Configure chunk settings
            self.Settings.chunk_size = 1024
            self.Settings.chunk_overlap = 200
            
            logger.info("✅ LlamaIndex global settings configured")
        except Exception as e:
            logger.warning(f"⚠️ LlamaIndex settings configuration failed: {e}")

    def _setup_embedding_model(self):
        """Setup embedding model for LlamaIndex"""
        try:
            if hasattr(self.vector_store, 'embedding_model'):
                self.Settings.embed_model = LlamaIndexEmbeddingWrapper(self.vector_store.embedding_model)
            logger.info("✅ LlamaIndex embedding model configured")
        except Exception as e:
            logger.warning(f"⚠️ Embedding model setup failed: {e}")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create advanced LlamaIndex RAG system"""
        if not system_prompt:
            system_prompt = """You are an advanced Docker assistant with deep technical knowledge.

Guidelines:
1. Analyze the provided context thoroughly
2. Provide detailed, technical answers with examples
3. Include Docker best practices and security considerations
4. Explain complex concepts step-by-step
5. Reference specific Docker commands and configurations when relevant

Use the context to provide comprehensive, accurate responses."""

        try:
            # Create documents from vector store
            documents = self._create_documents_from_vector_store()
            
            # Create index
            if documents:
                self.index = self.VectorStoreIndex.from_documents(
                    documents, 
                    show_progress=True,
                    embed_model=self.Settings.embed_model
                )
                logger.info(f"✅ LlamaIndex created from {len(documents)} documents")
            else:
                # Create empty index
                self.index = self.VectorStoreIndex([])
                logger.warning("⚠️ Created empty LlamaIndex (no documents)")

            # Setup advanced retriever
            retriever = self.VectorIndexRetriever(
                index=self.index,
                similarity_top_k=5,
                doc_ids=None,
                alpha=None  # For hybrid search if supported
            )

            # Setup response synthesizer with custom prompt
            response_synthesizer = self.get_response_synthesizer(
                response_mode="compact",
                use_async=False,
                streaming=False,
                text_qa_template=system_prompt
            )

            # Create query engine
            self.query_engine = self.RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer
            )

            logger.info("✅ LlamaIndex RAG chain created successfully")
            return self.query_engine

        except Exception as e:
            logger.warning(f"⚠️ LlamaIndex RAG creation failed: {e}")
            return self._create_fallback_engine(system_prompt)

    def _create_documents_from_vector_store(self) -> List:
        """Convert vector store data to LlamaIndex documents"""
        documents = []
        try:
            # Try to get existing documents from vector store
            if hasattr(self.vector_store, '_documents'):
                for i, (content, metadata) in enumerate(self.vector_store._documents):
                    doc = self.Document(
                        text=content,
                        metadata=metadata if isinstance(metadata, dict) else {"doc_id": str(i)}
                    )
                    documents.append(doc)
                logger.info(f"✅ Converted {len(documents)} documents from vector store")
            else:
                # Create sample Docker documents for testing
                sample_docs = [
                    "Docker is a containerization platform that packages applications and dependencies into containers.",
                    "Docker containers are lightweight, portable, and consistent across different environments.",
                    "Dockerfile is a text file containing instructions to build Docker images automatically.",
                    "Docker Compose allows you to define and run multi-container Docker applications using YAML.",
                    "Docker volumes provide persistent data storage that survives container restarts and removals.",
                    "Docker networks enable communication between containers and external systems.",
                    "Docker registries store and distribute Docker images, with Docker Hub being the default public registry.",
                    "Container orchestration with Docker Swarm or Kubernetes manages containers at scale.",
                ]
                
                documents = [
                    self.Document(
                        text=doc,
                        metadata={"doc_id": str(i), "source": "docker_knowledge", "type": "documentation"}
                    ) for i, doc in enumerate(sample_docs)
                ]
                logger.info(f"✅ Created {len(documents)} sample Docker documents")

        except Exception as e:
            logger.error(f"❌ Document creation failed: {e}")
        
        return documents

    def _create_fallback_engine(self, system_prompt: str):
        """Create fallback query engine"""
        return FallbackLlamaIndexEngine(self.llm, self.vector_store, system_prompt)

    def query(self, question: str, **kwargs) -> str:
        """Execute advanced query using LlamaIndex"""
        if not hasattr(self, 'query_engine'):
            self.create_rag_chain()

        try:
            # Execute query
            response = self.query_engine.query(question)
            
            # Extract response text
            if hasattr(response, 'response'):
                answer = str(response.response)
            elif hasattr(response, 'text'):
                answer = str(response.text)
            else:
                answer = str(response)

            # Add source information if available
            if hasattr(response, 'source_nodes') and response.source_nodes:
                sources = []
                for node in response.source_nodes[:3]:  # Top 3 sources
                    if hasattr(node, 'metadata') and 'source' in node.metadata:
                        sources.append(node.metadata['source'])
                    elif hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                        sources.append(node.node.metadata.get('source', 'Unknown'))
                
                if sources:
                    answer += f"\n\nSources: {', '.join(set(sources))}"

            return answer

        except Exception as e:
            logger.error(f"❌ LlamaIndex query failed: {e}")
            return self._fallback_query(question)

    def _fallback_query(self, question: str) -> str:
        """Fallback query method"""
        try:
            context = self._get_context_from_vector_store(question, k=5)
            prompt = f"""Context: {context}

Question: {question}

Please provide a detailed technical answer based on the context above."""
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"❌ LlamaIndex fallback failed: {e}")
            return f"Error processing query with LlamaIndex: {str(e)}"

    def add_document(self, text: str, metadata: Dict = None) -> bool:
        """Add a new document to the index"""
        try:
            doc = self.Document(text=text, metadata=metadata or {})
            if hasattr(self, 'index'):
                self.index.insert(doc)
                logger.info("✅ Document added to LlamaIndex")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to add document: {e}")
        return False

class LlamaIndexLLMWrapper:
    """Wrapper to make LLM compatible with LlamaIndex"""
    
    def __init__(self, llm):
        self.llm = llm
        self.model_name = getattr(llm, 'model_name', 'wrapped-llm')
        self.temperature = getattr(llm, 'temperature', 0.7)
        self.max_tokens = getattr(llm, 'max_tokens', 512)

    def complete(self, prompt: str, **kwargs) -> str:
        return self.llm.generate(prompt, **kwargs)

    def chat(self, messages, **kwargs) -> str:
        if isinstance(messages, list) and messages:
            prompt = messages[-1].get('content', str(messages[-1])) if isinstance(messages[-1], dict) else str(messages[-1])
        else:
            prompt = str(messages)
        return self.llm.generate(prompt, **kwargs)

class LlamaIndexEmbeddingWrapper:
    """Wrapper for embedding model compatibility"""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def get_text_embedding(self, text: str) -> List[float]:
        if hasattr(self.embedding_model, 'embed_query'):
            return self.embedding_model.embed_query(text)
        return []

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        if hasattr(self.embedding_model, 'embed_documents'):
            return self.embedding_model.embed_documents(texts)
        return [self.get_text_embedding(text) for text in texts]

class FallbackLlamaIndexEngine:
    """Fallback engine when LlamaIndex fails"""
    
    def __init__(self, llm, vector_store, system_prompt):
        self.llm = llm
        self.vector_store = vector_store
        self.system_prompt = system_prompt

    def query(self, question: str):
        try:
            docs = self.vector_store.similarity_search(question, k=5)
            context = "\n".join([doc[0] if isinstance(doc, tuple) else str(doc) for doc in docs])
            prompt = f"{self.system_prompt}\n\nContext: {context}\nQuestion: {question}\nAnswer:"
            return MockLlamaIndexResponse(self.llm.generate(prompt))
        except Exception as e:
            return MockLlamaIndexResponse(f"Error: {str(e)}")

class MockLlamaIndexResponse:
    """Mock response for fallback compatibility"""
    
    def __init__(self, response_text: str):
        self.response = response_text
        self.text = response_text
        self.source_nodes = []

def register():
    """Register LlamaIndex framework with factory"""
    return "llamaindex", LlamaIndexFramework
