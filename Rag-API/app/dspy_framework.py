import logging
from typing import Any, Dict, List, Optional
from providers.framework_factory import BaseFramework
import dspy

logger = logging.getLogger(__name__)

class DSPyFramework(BaseFramework):
    """DSPy Framework with programmatic prompting"""

    def initialize(self):
        """Initialize DSPy framework"""
        try:
            self._import_dependencies()
            self._configure_dspy()
            self._initialized = True
            logger.info("✅ DSPy framework initialized successfully")
        except ImportError as e:
            raise ImportError(f"DSPy packages required: {e}")
        except Exception as e:
            logger.error(f"❌ DSPy initialization failed: {e}")
            raise

    def _import_dependencies(self):
        """Import DSPy dependencies"""

        self.dspy = dspy

    def _configure_dspy(self):
        """Configure DSPy with LLM"""
        try:
            # Create DSPy-compatible LM
            dspy_lm = DSPyLMWrapper(self.llm)
            self.dspy.configure(lm=dspy_lm)
            logger.info("✅ DSPy configured with LLM")
        except Exception as e:
            logger.warning(f"⚠️ DSPy configuration failed: {e}")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create DSPy RAG pipeline"""
        if not system_prompt:
            system_prompt = """Generate comprehensive Docker answers using retrieved context.
Focus on practical solutions, best practices, and clear explanations."""

        try:
            # Create DSPy modules
            self.rag_pipeline = DockerRAGPipeline(
                vector_store=self.vector_store,
                system_prompt=system_prompt
            )
            
            logger.info("✅ DSPy RAG pipeline created")
            return self.rag_pipeline
            
        except Exception as e:
            logger.error(f"❌ DSPy RAG pipeline creation failed: {e}")
            raise

    def query(self, question: str, **kwargs) -> str:
        """Execute query using DSPy pipeline"""
        if not hasattr(self, 'rag_pipeline'):
            self.create_rag_chain()

        try:
            result = self.rag_pipeline(question=question)
            return result.answer if hasattr(result, 'answer') else str(result)
        except Exception as e:
            logger.error(f"❌ DSPy query failed: {e}")
            return self._fallback_query(question)

    def _fallback_query(self, question: str) -> str:
        """Fallback query method"""
        try:
            context = self._get_context_from_vector_store(question, k=5)
            prompt = f"""Context: {context}

Question: {question}

Provide a comprehensive Docker answer based on the context above."""
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"❌ DSPy fallback failed: {e}")
            return f"Error processing query with DSPy: {str(e)}"
        

class DockerRAGPipeline(dspy.Module):
    """DSPy RAG pipeline for Docker questions"""

    def __init__(self, vector_store, system_prompt):
        import dspy
        super().__init__()
        self.vector_store = vector_store
        self.system_prompt = system_prompt
        
        # Define DSPy signatures
        self.retrieve = dspy.Retrieve(k=5)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        """Forward pass through the pipeline"""
        try:
            # Retrieve context
            if hasattr(self, 'retrieve'):
                context = self.retrieve(question).passages
            else:
                # Fallback retrieval
                docs = self.vector_store.similarity_search(question, k=5)
                context = [doc[0] if isinstance(doc, tuple) else str(doc) for doc in docs]
            
            # Generate answer
            context_str = "\n".join(context) if isinstance(context, list) else str(context)
            prediction = self.generate_answer(context=context_str, question=question)
            
            return prediction
            
        except Exception as e:
            logger.error(f"❌ DSPy forward pass failed: {e}")
            # Return fallback
            return type('MockPrediction', (), {'answer': f"Error in DSPy pipeline: {e}"})()

class DSPyLMWrapper:
    """Wrapper to make LLM compatible with DSPy"""
    
    def __init__(self, llm):
        self.llm = llm
        self.model_name = getattr(llm, 'model_name', 'wrapped-llm')

    def generate(self, prompt: str, **kwargs) -> List[str]:
        """DSPy-compatible generate method"""
        try:
            response = self.llm.generate(prompt, **kwargs)
            return [response] if isinstance(response, str) else response
        except Exception as e:
            logger.error(f"❌ DSPy LM generation failed: {e}")
            return [f"Error: {e}"]

    def __call__(self, prompt: str, **kwargs) -> List[str]:
        return self.generate(prompt, **kwargs)

def register():
    """Register DSPy framework with factory"""
    return "dspy", DSPyFramework
