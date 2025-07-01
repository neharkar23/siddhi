
import logging
from typing import Any, Dict, List, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.tools import tool
from providers.framework_factory import BaseFramework

logger = logging.getLogger(__name__)

class LangChainFramework(BaseFramework):
    """LangChain Framework with comprehensive RAG capabilities"""

    def initialize(self):
        """Initialize LangChain components with error handling"""
        try:
            # Import all required LangChain components
            self._import_dependencies()
            self._setup_tools()
            self._setup_retriever()
            self._initialized = True
            logger.info("✅ LangChain framework initialized successfully")
        except ImportError as e:
            raise ImportError(f"LangChain packages required: {e}")
        except Exception as e:
            logger.error(f"❌ LangChain initialization failed: {e}")
            raise

    def _import_dependencies(self):
        """Import and store LangChain dependencies"""
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.chains.combine_documents import create_stuff_documents_chain
        from langchain.chains import create_retrieval_chain
        from langchain.memory import ConversationBufferMemory
        
        self.RetrievalQA = RetrievalQA
        self.PromptTemplate = PromptTemplate
        self.ChatPromptTemplate = ChatPromptTemplate
        self.create_stuff_documents_chain = create_stuff_documents_chain
        self.create_retrieval_chain = create_retrieval_chain
        self.ConversationBufferMemory = ConversationBufferMemory

    def _setup_tools(self):
        """Setup LangChain tools for enhanced capabilities"""
        self.tools = []
        
        try:
            # Web search tool
            search_tool = DuckDuckGoSearchResults(num_results=3)
            self.tools.append(search_tool)
            
            # Custom document search tool
            @tool
            def search_documents(query: str) -> str:
                """Search through indexed documents for relevant information"""
                try:
                    docs = self.vector_store.similarity_search(query, k=5)
                    return "\n".join([doc[0] if isinstance(doc, tuple) else str(doc) for doc in docs])
                except Exception as e:
                    return f"Error searching documents: {e}"
            
            self.tools.append(search_documents)
            logger.info(f"✅ LangChain tools initialized: {len(self.tools)} tools")
            
        except Exception as e:
            logger.warning(f"⚠️ Some LangChain tools failed to initialize: {e}")
            self.tools = []

    def _setup_retriever(self):
        """Setup document retriever"""
        try:
            if hasattr(self.vector_store, 'client') and hasattr(self.vector_store.client, 'as_retriever'):
                self.retriever = self.vector_store.client.as_retriever(
                    search_kwargs={"k": 5}
                )
            else:
                self.retriever = None
            logger.info("✅ LangChain retriever configured")
        except Exception as e:
            logger.warning(f"⚠️ Retriever setup failed: {e}")
            self.retriever = None

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create comprehensive LangChain RAG chain"""
        if not system_prompt:
            system_prompt = """You are an expert Docker assistant with comprehensive knowledge.

Instructions:
1. Use the provided context to answer Docker-related questions accurately
2. If context is insufficient, use your general Docker knowledge
3. Provide practical, actionable advice with examples
4. Always prioritize safety in Docker operations
5. Include best practices and warnings when relevant

Context: {context}
Question: {input}

Provide a comprehensive and accurate answer:"""

        try:
            # Create prompt template
            prompt = self.ChatPromptTemplate.from_template(system_prompt)
            
            # Create document processing chain
            document_chain = self.create_stuff_documents_chain(self.llm, prompt)
            
            # Create retrieval chain if retriever is available
            if self.retriever:
                self.chain = self.create_retrieval_chain(self.retriever, document_chain)
                logger.info("✅ LangChain retrieval chain created")
            else:
                # Fallback to document chain with manual context injection
                self.chain = document_chain
                logger.info("✅ LangChain document chain created (no retriever)")
            
            # Add memory for conversation context
            try:
                self.memory = self.ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
            except Exception as e:
                logger.warning(f"⚠️ Memory setup failed: {e}")
                self.memory = None
            
            return self.chain
            
        except Exception as e:
            logger.error(f"❌ Failed to create LangChain RAG chain: {e}")
            raise

    def query(self, question: str, **kwargs) -> str:
        """Execute query using LangChain with comprehensive error handling"""
        if not hasattr(self, 'chain'):
            self.create_rag_chain()

        try:
            # Method 1: Try retrieval chain
            if hasattr(self.chain, 'invoke') and self.retriever:
                result = self.chain.invoke({"input": question})
                answer = result.get('answer', str(result))
                
                # Add sources if available
                if 'source_documents' in result:
                    sources = [doc.metadata.get('source', 'Unknown') for doc in result['source_documents']]
                    if sources:
                        answer += f"\n\nSources: {', '.join(set(sources))}"
                
                return answer
                
            # Method 2: Manual context retrieval
            elif hasattr(self.chain, 'invoke'):
                context = self._get_context_from_vector_store(question, k=5)
                result = self.chain.invoke({"context": context, "input": question})
                return result.get('answer', str(result))
                
            # Method 3: Direct LLM fallback
            else:
                context = self._get_context_from_vector_store(question, k=5)
                prompt = f"""Context: {context}

Question: {question}

Please provide a comprehensive answer based on the context above."""
                return self.llm.generate(prompt)
                
        except Exception as e:
            logger.error(f"❌ LangChain query failed: {e}")
            # Ultimate fallback
            try:
                context = self._get_context_from_vector_store(question, k=3)
                return self.llm.generate(f"Context: {context}\nQuestion: {question}\nAnswer:")
            except Exception as fallback_error:
                logger.error(f"❌ LangChain fallback failed: {fallback_error}")
                return f"Error processing query with LangChain: {str(e)}"

    def add_memory(self, question: str, answer: str):
        """Add conversation to memory"""
        if self.memory:
            try:
                self.memory.save_context({"input": question}, {"output": answer})
            except Exception as e:
                logger.warning(f"⚠️ Failed to save to memory: {e}")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history"""
        if self.memory:
            try:
                return self.memory.chat_memory.messages
            except Exception as e:
                logger.warning(f"⚠️ Failed to get conversation history: {e}")
        return []

def register():
    """Register LangChain framework with factory"""
    return "langchain", LangChainFramework
