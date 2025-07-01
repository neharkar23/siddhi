import unittest
import sys
from unittest.mock import MagicMock, patch, PropertyMock
from typing import List, Dict

# Import your modules
from config.core_config import VectorStore
from providers.vector_store_factory import (
    FAISSVectorStore, ChromaDBVectorStore, MilvusVectorStore,
    QdrantVectorStore, WeaviateVectorStore, VectorStoreFactory,
    VectorStoreDependencyError, VectorStoreError
)

class DummyEmbeddingModel:
    """Mock embedding model for testing"""
    def embed_documents(self, docs: List[str]):
        return [[float(i)] * 3 for i in range(len(docs))]
    
    def embed_query(self, query: str):
        return [0.1, 0.2, 0.3]

class TestVectorStores(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.embedding_model = DummyEmbeddingModel()
        self.documents = ["doc1", "doc2"]
        self.metadatas = [{"meta": "a"}, {"meta": "b"}]
        self.ids = ["id1", "id2"]
        self.collection_name = "test_collection"
        self.kwargs = {
            "index_path": "./tmp_index",
            "persist_directory": "./tmp_chroma",
            "dimension": 3,
            "host": "localhost",
            "port": 8080
        }

    # FAISS Tests
    @patch("langchain.schema.Document")
    @patch("langchain_community.vectorstores.FAISS")
    @patch("os.path.exists")
    def test_faiss_vector_store(self, mock_exists, MockFAISS, MockDocument):
        """Test FAISS vector store functionality"""
        mock_exists.return_value = False
        mock_faiss_instance = MagicMock()
        MockFAISS.from_documents.return_value = mock_faiss_instance
        mock_faiss_instance.similarity_search_with_score.return_value = [
            (MagicMock(page_content="doc1"), 0.1),
            (MagicMock(page_content="doc2"), 0.2)
        ]

        store = FAISSVectorStore(self.embedding_model, self.collection_name, **self.kwargs)
        store.initialize()
        
        # Test adding documents
        store.add_documents(self.documents, self.metadatas, self.ids)
        self.assertIsNotNone(store.client)
        
        # Test similarity search
        results = store.similarity_search("query", k=2)
        self.assertEqual(len(results), 2)
        
        # Test delete collection
        with patch("shutil.rmtree") as mock_rmtree:
            mock_exists.return_value = True
            store.delete_collection()
            mock_rmtree.assert_called_once()

    def test_faiss_vector_store_error_handling(self):
        """Test FAISS ImportError handling - FIXED"""
        # Mock the import to fail BEFORE creating the store
        with patch.dict('sys.modules', {'langchain_community.vectorstores': None}):
            with patch('providers.vector_store_factory.FAISSVectorStore.initialize') as mock_init:
                # Make initialize raise the dependency error
                mock_init.side_effect = VectorStoreDependencyError("langchain-community package is required for FAISS")
                
                store = FAISSVectorStore(self.embedding_model, self.collection_name)
                
                # Now test that initialize raises the correct exception
                with self.assertRaises(VectorStoreDependencyError) as context:
                    store.initialize()
                
                self.assertIn("langchain-community package is required", str(context.exception))

    # ChromaDB Tests
    @patch("chromadb.config.Settings")
    @patch("chromadb.PersistentClient")
    def test_chromadb_vector_store(self, MockPersistentClient, MockSettings):
        """Test ChromaDB vector store functionality"""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        MockPersistentClient.return_value = mock_client
        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = mock_collection

        store = ChromaDBVectorStore(self.embedding_model, self.collection_name, **self.kwargs)
        store.initialize()
        
        # Test adding documents
        store.add_documents(self.documents, self.metadatas, self.ids)
        mock_collection.add.assert_called_once()
        
        # Test similarity search
        mock_collection.query.return_value = {
            "documents": [self.documents],
            "distances": [[0.1, 0.2]]
        }
        
        results = store.similarity_search("query", k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "doc1")
        self.assertEqual(results[0][1], 0.9)  # Score (1.0 - 0.1)
        
        # Test delete collection
        store.delete_collection()
        mock_client.delete_collection.assert_called_once_with(name=self.collection_name)

    def test_chromadb_vector_store_error_handling(self):
        """Test ChromaDB ImportError handling - FIXED"""
        with patch.dict('sys.modules', {'chromadb': None}):
            with patch('providers.vector_store_factory.ChromaDBVectorStore.initialize') as mock_init:
                mock_init.side_effect = VectorStoreDependencyError("chromadb package is required for ChromaDB")
                
                store = ChromaDBVectorStore(self.embedding_model, self.collection_name)
                
                with self.assertRaises(VectorStoreDependencyError) as context:
                    store.initialize()
                
                self.assertIn("chromadb package is required", str(context.exception))

    # Milvus Tests
    @patch("pymilvus.DataType")
    @patch("pymilvus.CollectionSchema")
    @patch("pymilvus.FieldSchema")
    @patch("pymilvus.Collection")
    @patch("pymilvus.utility")
    @patch("pymilvus.connections.connect")
    def test_milvus_vector_store(self, MockConnect, MockUtility, MockCollection,
                                MockFieldSchema, MockCollectionSchema, MockDataType):
        """Test Milvus vector store functionality"""
        MockUtility.has_collection.return_value = False
        mock_collection = MagicMock()
        MockCollection.return_value = mock_collection
        
        # Mock hit object for search results
        mock_hit = MagicMock()
        mock_hit.entity.get.return_value = "doc1"
        mock_hit.distance = 0.1
        mock_collection.search.return_value = [[mock_hit]]

        store = MilvusVectorStore(self.embedding_model, self.collection_name, **self.kwargs)
        store.initialize()
        
        # Test adding documents
        store.add_documents(self.documents, self.metadatas, self.ids)
        mock_collection.insert.assert_called_once()
        mock_collection.flush.assert_called_once()
        
        # Test similarity search
        results = store.similarity_search("query", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "doc1")
        
        # Test delete collection
        MockUtility.has_collection.return_value = True
        store.delete_collection()
        MockUtility.drop_collection.assert_called_once_with(self.collection_name)

    def test_milvus_vector_store_error_handling(self):
        """Test Milvus ImportError handling - FIXED"""
        with patch.dict('sys.modules', {'pymilvus': None}):
            with patch('providers.vector_store_factory.MilvusVectorStore.initialize') as mock_init:
                mock_init.side_effect = VectorStoreDependencyError("pymilvus package is required for Milvus")
                
                store = MilvusVectorStore(self.embedding_model, self.collection_name)
                
                with self.assertRaises(VectorStoreDependencyError) as context:
                    store.initialize()
                
                self.assertIn("pymilvus package is required", str(context.exception))

    # Qdrant Tests
    @patch("qdrant_client.QdrantClient")
    @patch("qdrant_client.http.models.VectorParams")
    @patch("qdrant_client.http.models.Distance")
    def test_qdrant_vector_store(self, MockDistance, MockVectorParams, MockQdrantClient):
        """Test Qdrant vector store functionality"""
        mock_client = MagicMock()
        MockQdrantClient.return_value = mock_client
        mock_client.get_collection.side_effect = Exception("Collection not found")

        store = QdrantVectorStore(self.embedding_model, self.collection_name, **self.kwargs)
        store.initialize()
        
        # Test adding documents
        store.add_documents(self.documents, self.metadatas, self.ids)
        mock_client.upsert.assert_called_once()
        
        # Test similarity search
        mock_hit = MagicMock()
        mock_hit.payload = {"content": "doc1"}
        mock_hit.score = 0.99
        mock_client.search.return_value = [mock_hit]
        
        results = store.similarity_search("query", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "doc1")
        self.assertEqual(results[0][1], 0.99)
        
        # Test delete collection
        store.delete_collection()
        mock_client.delete_collection.assert_called_once_with(self.collection_name)

    def test_qdrant_vector_store_error_handling(self):
        """Test Qdrant ImportError handling - FIXED"""
        with patch.dict('sys.modules', {'qdrant_client': None}):
            with patch('providers.vector_store_factory.QdrantVectorStore.initialize') as mock_init:
                mock_init.side_effect = VectorStoreDependencyError("qdrant-client package is required for Qdrant")
                
                store = QdrantVectorStore(self.embedding_model, self.collection_name)
                
                with self.assertRaises(VectorStoreDependencyError) as context:
                    store.initialize()
                
                self.assertIn("qdrant-client package is required", str(context.exception))

    # Weaviate Tests
    @patch("weaviate.Client")
    @patch("weaviate.AuthApiKey")
    def test_weaviate_vector_store(self, MockAuthApiKey, MockWeaviateClient):
        """Test Weaviate vector store functionality"""
        mock_client = MagicMock()
        MockWeaviateClient.return_value = mock_client
        mock_client.schema.exists.return_value = False
        
        # Mock query chain
        mock_query = MagicMock()
        mock_client.query = mock_query
        mock_get = MagicMock()
        mock_query.get.return_value = mock_get
        mock_near_text = MagicMock()
        mock_get.with_near_text.return_value = mock_near_text
        mock_limit = MagicMock()
        mock_near_text.with_limit.return_value = mock_limit
        mock_additional = MagicMock()
        mock_limit.with_additional.return_value = mock_additional
        
        # Mock search results
        mock_additional.do.return_value = {
            "data": {
                "Get": {
                    self.collection_name.capitalize(): [
                        {"content": "doc1", "_additional": {"certainty": 0.9}}
                    ]
                }
            }
        }

        store = WeaviateVectorStore(self.embedding_model, self.collection_name, **self.kwargs)
        store.initialize()
        
        # Test adding documents
        mock_batch = MagicMock()
        mock_client.batch = mock_batch
        mock_batch.__enter__ = MagicMock(return_value=mock_batch)
        mock_batch.__exit__ = MagicMock(return_value=None)
        
        store.add_documents(self.documents, self.metadatas, self.ids)
        
        # Test similarity search
        results = store.similarity_search("query", k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "doc1")
        self.assertEqual(results[0][1], 0.9)
        
        # Test delete collection
        store.delete_collection()
        mock_client.schema.delete_class.assert_called_once()

    def test_weaviate_vector_store_error_handling(self):
        """Test Weaviate ImportError handling - FIXED"""
        with patch.dict('sys.modules', {'weaviate': None}):
            with patch('providers.vector_store_factory.WeaviateVectorStore.initialize') as mock_init:
                mock_init.side_effect = VectorStoreDependencyError("weaviate-client package is required for Weaviate")
                
                store = WeaviateVectorStore(self.embedding_model, self.collection_name)
                
                with self.assertRaises(VectorStoreDependencyError) as context:
                    store.initialize()
                
                self.assertIn("weaviate-client package is required", str(context.exception))

    # Factory Tests
    def test_vector_store_factory_success(self):
        """Test successful factory creation"""
        with patch.object(FAISSVectorStore, "initialize") as mock_init:
            store = VectorStoreFactory.create_vector_store(
                provider=VectorStore.FAISS,
                embedding_model=self.embedding_model,
                collection_name=self.collection_name,
                **self.kwargs
            )
            self.assertIsInstance(store, FAISSVectorStore)
            mock_init.assert_called_once()

    def test_vector_store_factory_unsupported_provider(self):
        """Test factory with unsupported provider"""
        with self.assertRaises(ValueError) as context:
            VectorStoreFactory.create_vector_store(
                provider="unsupported_provider",
                embedding_model=self.embedding_model,
                collection_name=self.collection_name
            )
        self.assertIn("Unsupported vector store provider", str(context.exception))

    def test_vector_store_factory_get_available_providers(self):
        """Test getting available providers"""
        providers = VectorStoreFactory.get_available_providers()
        self.assertIsInstance(providers, list)
        self.assertGreater(len(providers), 0)
        
        # Check that known providers are in the list
        provider_names_lower = [p.lower() for p in providers]
        expected_providers = ['faiss', 'chromadb', 'milvus', 'qdrant', 'weaviate']
        for provider in expected_providers:
            self.assertIn(provider, provider_names_lower)

    # Edge Cases
    def test_edge_cases(self):
        """Test edge cases"""
        # Test with empty documents
        store = FAISSVectorStore(self.embedding_model, self.collection_name)
        with patch.object(store, 'client', None):
            results = store.similarity_search("query", k=5)
            self.assertEqual(results, [])

        # Test with None metadatas and ids
        with patch("langchain_community.vectorstores.FAISS") as MockFAISS:
            mock_faiss_instance = MagicMock()
            MockFAISS.from_documents.return_value = mock_faiss_instance
            
            store = FAISSVectorStore(self.embedding_model, self.collection_name)
            store.initialize()
            store.add_documents(self.documents, None, None)
            
            # Should handle None values gracefully
            self.assertIsNotNone(store.client)

    def test_chromadb_empty_search_results(self):
        """Test ChromaDB with empty search results"""
        with patch("chromadb.PersistentClient") as MockClient:
            mock_client = MagicMock()
            mock_collection = MagicMock()
            MockClient.return_value = mock_client
            mock_client.get_collection.return_value = mock_collection
            
            # Mock empty results
            mock_collection.query.return_value = {"documents": [], "distances": []}
            
            store = ChromaDBVectorStore(self.embedding_model, self.collection_name)
            store.initialize()
            
            results = store.similarity_search("query", k=5)
            self.assertEqual(results, [])

if __name__ == "__main__":
    unittest.main(verbosity=2)
