from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
from config.core_config import VectorStore, config_manager

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_DIMENSION = 1536
DEFAULT_NLIST = 128
DEFAULT_NPROBE = 10

class VectorStoreError(Exception):
    """Base exception for vector store operations"""
    pass

class VectorStoreDependencyError(VectorStoreError):
    """Raised when required dependencies are missing"""
    pass

class BaseVectorStore(ABC):
    """Abstract base class for vector stores"""
    
    def __init__(self, embedding_model, collection_name: str = "default", **kwargs):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.kwargs = kwargs
        self.client = None
        self.index_path = kwargs.get('index_path')
    
    @abstractmethod
    def initialize(self):
        """Initialize the vector store client"""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        """Add documents to the vector store"""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        pass
    
    @abstractmethod
    def delete_collection(self):
        """Delete the collection"""
        pass

class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation"""
    
    def initialize(self):
        try:
            from langchain_community.vectorstores import FAISS
            self.store_class = FAISS
            
            if self.index_path and os.path.exists(self.index_path):
                self.client = FAISS.load_local(
                    self.index_path, 
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"FAISS index loaded from {self.index_path}")
            else:
                self.client = None
                logger.info("FAISS store initialized (empty)")
        except ImportError as e:
            raise VectorStoreDependencyError("langchain-community package is required for FAISS") from e
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        try:
            from langchain.schema import Document
            
            if not documents:
                logger.warning("No documents provided to add")
                return
                
            docs = [Document(page_content=doc, metadata=meta or {}) 
                    for doc, meta in zip(documents, metadatas or [{}] * len(documents))]
            
            if self.client is None:
                self.client = self.store_class.from_documents(docs, self.embedding_model)
            else:
                self.client.add_documents(docs)
            
            if self.index_path:
                self.client.save_local(self.index_path)
                
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}") from e
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if self.client is None:
            logger.warning("FAISS client not initialized")
            return []
        
        try:
            results = self.client.similarity_search_with_score(query, k=k)
            return [(doc.page_content, score) for doc, score in results]
        except Exception as e:
            logger.error(f"Error during FAISS similarity search: {e}")
            raise VectorStoreError(f"Similarity search failed: {e}") from e
    
    def delete_collection(self):
        try:
            if self.index_path and os.path.exists(self.index_path):
                import shutil
                shutil.rmtree(self.index_path)
                logger.info(f"FAISS index deleted from {self.index_path}")
            self.client = None
        except Exception as e:
            logger.error(f"Error deleting FAISS collection: {e}")
            raise VectorStoreError(f"Failed to delete collection: {e}") from e

class ChromaDBVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation"""
    
    def initialize(self):
        try:
            import chromadb
            from chromadb.config import Settings
            
            persist_directory = self.kwargs.get('persist_directory', './chroma_db')
            
            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(f"ChromaDB collection '{self.collection_name}' loaded")
            except Exception:
                self.collection = self.chroma_client.create_collection(name=self.collection_name)
                logger.info(f"ChromaDB collection '{self.collection_name}' created")
            
        except ImportError as e:
            raise VectorStoreDependencyError("chromadb package is required for ChromaDB") from e
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise VectorStoreError(f"ChromaDB initialization failed: {e}") from e
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return
                
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            if metadatas is None:
                metadatas = [{}] * len(documents)
            
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}") from e
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )
            
            documents = results['documents'][0] if results['documents'] else []
            distances = results['distances'][0] if results['distances'] else []
            
            return [(doc, 1.0 - dist) for doc, dist in zip(documents, distances)]
            
        except Exception as e:
            logger.error(f"Error during ChromaDB similarity search: {e}")
            raise VectorStoreError(f"Similarity search failed: {e}") from e
    
    def delete_collection(self):
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            logger.info(f"ChromaDB collection '{self.collection_name}' deleted")
        except Exception as e:
            logger.error(f"Error deleting ChromaDB collection: {e}")
            raise VectorStoreError(f"Failed to delete collection: {e}") from e

class MilvusVectorStore(BaseVectorStore):
    """Milvus vector store implementation"""
    
    def initialize(self):
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
            
            host = self.kwargs.get('host', 'localhost')
            port = self.kwargs.get('port', 19530)
            dimension = self.kwargs.get('dimension', DEFAULT_DIMENSION)
            
            connections.connect("default", host=host, port=port)
            
            # Define schema if collection doesn't exist
            if not utility.has_collection(self.collection_name):
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
                ]
                schema = CollectionSchema(fields, description="Document collection")
                self.collection = Collection(self.collection_name, schema)
                
                # Create index
                index_params = {
                    "metric_type": "L2",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": DEFAULT_NLIST}
                }
                self.collection.create_index("vector", index_params)
                logger.info(f"Milvus collection '{self.collection_name}' created")
            else:
                self.collection = Collection(self.collection_name)
                logger.info(f"Milvus collection '{self.collection_name}' loaded")
            
            self.collection.load()
            
        except ImportError as e:
            raise VectorStoreDependencyError("pymilvus package is required for Milvus") from e
        except Exception as e:
            logger.error(f"Error initializing Milvus: {e}")
            raise VectorStoreError(f"Milvus initialization failed: {e}") from e
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return
                
            # Generate embeddings
            embeddings = self.embedding_model.embed_documents(documents)
            
            entities = [
                documents,
                embeddings
            ]
            
            self.collection.insert(entities)
            self.collection.flush()
            logger.info(f"Added {len(documents)} documents to Milvus")
            
        except Exception as e:
            logger.error(f"Error adding documents to Milvus: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}") from e
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        try:
            query_embedding = self.embedding_model.embed_query(query)
            
            search_params = {"metric_type": "L2", "params": {"nprobe": DEFAULT_NPROBE}}
            results = self.collection.search(
                [query_embedding], 
                "vector", 
                search_params, 
                limit=k,
                output_fields=["content"]
            )
            
            return [(hit.entity.get("content"), 1.0 / (1.0 + hit.distance)) 
                    for hit in results[0]]
                    
        except Exception as e:
            logger.error(f"Error during Milvus similarity search: {e}")
            raise VectorStoreError(f"Similarity search failed: {e}") from e
    
    def delete_collection(self):
        try:
            from pymilvus import utility
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info(f"Milvus collection '{self.collection_name}' deleted")
        except Exception as e:
            logger.error(f"Error deleting Milvus collection: {e}")
            raise VectorStoreError(f"Failed to delete collection: {e}") from e

class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store implementation"""
    
    def initialize(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
            
            host = self.kwargs.get('host', 'localhost')
            port = self.kwargs.get('port', 6333)
            dimension = self.kwargs.get('dimension', DEFAULT_DIMENSION)
            api_key = config_manager.get_api_key('qdrant')
            
            self.client = QdrantClient(host=host, port=port, api_key=api_key)
            
            # Create collection if it doesn't exist
            try:
                self.client.get_collection(self.collection_name)
                logger.info(f"Qdrant collection '{self.collection_name}' loaded")
            except Exception:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Qdrant collection '{self.collection_name}' created")
            
        except ImportError as e:
            raise VectorStoreDependencyError("qdrant-client package is required for Qdrant") from e
        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            raise VectorStoreError(f"Qdrant initialization failed: {e}") from e
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        try:
            from qdrant_client.http.models import PointStruct
            
            if not documents:
                logger.warning("No documents provided to add")
                return
            
            if ids is None:
                ids = list(range(len(documents)))
            
            if metadatas is None:
                metadatas = [{}] * len(documents)
            
            # Generate embeddings
            embeddings = self.embedding_model.embed_documents(documents)
            
            points = [
                PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={"content": doc, **metadata}
                )
                for idx, doc, embedding, metadata in zip(ids, documents, embeddings, metadatas)
            ]
            
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Added {len(documents)} documents to Qdrant")
            
        except Exception as e:
            logger.error(f"Error adding documents to Qdrant: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}") from e
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        try:
            query_embedding = self.embedding_model.embed_query(query)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k
            )
            
            return [(hit.payload.get("content", ""), hit.score) for hit in results]
            
        except Exception as e:
            logger.error(f"Error during Qdrant similarity search: {e}")
            raise VectorStoreError(f"Similarity search failed: {e}") from e
    
    def delete_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Qdrant collection '{self.collection_name}' deleted")
        except Exception as e:
            logger.error(f"Error deleting Qdrant collection: {e}")
            raise VectorStoreError(f"Failed to delete collection: {e}") from e

class WeaviateVectorStore(BaseVectorStore):
    """Weaviate vector store implementation"""
    
    def initialize(self):
        try:
            import weaviate
            
            url = self.kwargs.get('url', 'http://localhost:8080')
            api_key = config_manager.get_api_key('weaviate')
            
            if api_key:
                auth_config = weaviate.AuthApiKey(api_key=api_key)
                self.client = weaviate.Client(url=url, auth_client_secret=auth_config)
            else:
                self.client = weaviate.Client(url=url)
            
            # Create class/schema if it doesn't exist
            class_name = self.collection_name.capitalize()
            
            if not self.client.schema.exists(class_name):
                class_schema = {
                    "class": class_name,
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"]
                        }
                    ]
                }
                self.client.schema.create_class(class_schema)
                logger.info(f"Weaviate class '{class_name}' created")
            else:
                logger.info(f"Weaviate class '{class_name}' loaded")
            
            self.class_name = class_name
            
        except ImportError as e:
            raise VectorStoreDependencyError("weaviate-client package is required for Weaviate") from e
        except Exception as e:
            logger.error(f"Error initializing Weaviate: {e}")
            raise VectorStoreError(f"Weaviate initialization failed: {e}") from e
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None, ids: List[str] = None):
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return
                
            if metadatas is None:
                metadatas = [{}] * len(documents)
            
            with self.client.batch as batch:
                for doc, metadata in zip(documents, metadatas):
                    data_object = {
                        "content": doc,
                        **metadata
                    }
                    batch.add_data_object(data_object, self.class_name)
            
            logger.info(f"Added {len(documents)} documents to Weaviate")
            
        except Exception as e:
            logger.error(f"Error adding documents to Weaviate: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}") from e
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        try:
            result = (
                self.client.query
                .get(self.class_name, ["content"])
                .with_near_text({"concepts": [query]})
                .with_limit(k)
                .with_additional(["certainty"])
                .do()
            )
            
            objects = result["data"]["Get"][self.class_name]
            return [(obj["content"], obj["_additional"]["certainty"]) for obj in objects]
            
        except Exception as e:
            logger.error(f"Error during Weaviate similarity search: {e}")
            raise VectorStoreError(f"Similarity search failed: {e}") from e
    
    def delete_collection(self):
        try:
            self.client.schema.delete_class(self.class_name)
            logger.info(f"Weaviate class '{self.class_name}' deleted")
        except Exception as e:
            logger.error(f"Error deleting Weaviate class: {e}")
            raise VectorStoreError(f"Failed to delete collection: {e}") from e

class VectorStoreFactory:
    """Factory class for creating vector store instances"""
    
    _providers = {
        VectorStore.FAISS: FAISSVectorStore,
        VectorStore.CHROMADB: ChromaDBVectorStore,
        VectorStore.MILVUS: MilvusVectorStore,
        VectorStore.QDRANT: QdrantVectorStore,
        VectorStore.WEAVIATE: WeaviateVectorStore,
    }
    
    @classmethod
    def create_vector_store(
        cls,
        provider: VectorStore,
        embedding_model,
        collection_name: str = "default",
        **kwargs
    ) -> BaseVectorStore:
        """Create vector store instance based on provider"""
        
        if provider not in cls._providers:
            available_providers = [p.value for p in cls._providers.keys()]
            raise ValueError(f"Unsupported vector store provider: {provider}. "
                           f"Available providers: {available_providers}")
        
        # Validate embedding model
        if not hasattr(embedding_model, 'embed_documents') or not hasattr(embedding_model, 'embed_query'):
            raise ValueError("embedding_model must have 'embed_documents' and 'embed_query' methods")
        
        try:
            # Create and initialize vector store
            store_class = cls._providers[provider]
            store = store_class(
                embedding_model=embedding_model,
                collection_name=collection_name,
                **kwargs
            )
            store.initialize()
            logger.info(f"Successfully created {provider.value} vector store")
            return store
            
        except VectorStoreDependencyError:
            # Re-raise dependency errors as-is
            raise
        except Exception as e:
            logger.error(f"Failed to create {provider.value} vector store: {e}")
            raise VectorStoreError(f"Failed to create vector store: {e}") from e
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available providers"""
        return [provider.value for provider in cls._providers.keys()]
    
    @classmethod
    def list_supported_vector_stores(cls) -> list:
        """
        Return a list of all supported vector store provider names.
        """
        return [provider.value for provider in cls._providers.keys()]
    
    @classmethod
    def validate_dependencies(cls, provider: VectorStore) -> bool:
        """Validate that required dependencies are available for a provider"""
        dependency_map = {
            VectorStore.FAISS: ['langchain_community', 'langchain'],
            VectorStore.CHROMADB: ['chromadb'],
            VectorStore.MILVUS: ['pymilvus'],
            VectorStore.QDRANT: ['qdrant_client'],
            VectorStore.WEAVIATE: ['weaviate'],
        }
        
        required_deps = dependency_map.get(provider, [])
        
        for dep in required_deps:
            try:
                __import__(dep)
            except ImportError:
                logger.warning(f"Missing dependency '{dep}' for {provider.value}")
                return False
        
        return True