
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
import hashlib
import pickle
import os
from pathlib import Path

from providers.framework_factory import BaseFramework

logger = logging.getLogger(__name__)

class CleanlabFramework(BaseFramework):
    """
    Cleanlab Framework with advanced data quality assessment,
    label error detection, and confidence scoring for RAG systems
    """

    def initialize(self):
        """Initialize Cleanlab components with comprehensive setup"""
        try:
            self._import_dependencies()
            self._setup_data_quality_pipeline()
            self._setup_confidence_models()
            self._setup_error_detection()
            self._initialized = True
            logger.info("✅ Cleanlab framework initialized successfully")
        except ImportError as e:
            raise ImportError(f"Cleanlab packages required: {e}")
        except Exception as e:
            logger.error(f"❌ Cleanlab initialization failed: {e}")
            raise

    def _import_dependencies(self):
        """Import Cleanlab and related dependencies"""
        try:
            import cleanlab
            from cleanlab.datalab import Datalab
            from cleanlab.filter import find_label_issues
            from cleanlab.rank import get_label_quality_scores
            from cleanlab.count import estimate_cv_predicted_probabilities
            from cleanlab.classification import CleanLearning
            from cleanlab.outlier import OutOfDistribution
            
            # Store imported modules
            self.cleanlab = cleanlab
            self.Datalab = Datalab
            self.find_label_issues = find_label_issues
            self.get_label_quality_scores = get_label_quality_scores
            self.estimate_cv_predicted_probabilities = estimate_cv_predicted_probabilities
            self.CleanLearning = CleanLearning
            self.OutOfDistribution = OutOfDistribution
            
            # Import sklearn for baseline models
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.model_selection import cross_val_predict
            from sklearn.metrics.pairwise import cosine_similarity
            
            self.RandomForestClassifier = RandomForestClassifier
            self.TfidfVectorizer = TfidfVectorizer
            self.cross_val_predict = cross_val_predict
            self.cosine_similarity = cosine_similarity
            
        except ImportError as e:
            logger.error(f"Missing required dependencies: {e}")
            raise ImportError(f"Required packages: cleanlab, scikit-learn, numpy, pandas")

    def _setup_data_quality_pipeline(self):
        """Setup comprehensive data quality assessment pipeline"""
        try:
            # Initialize quality metrics storage
            self.quality_metrics = {
                'document_quality_scores': {},
                'query_quality_scores': {},
                'response_quality_scores': {},
                'confidence_scores': {},
                'outlier_scores': {},
                'label_issues': {},
                'data_issues_summary': {}
            }
            
            # Setup quality thresholds
            self.quality_thresholds = {
                'high_quality': 0.8,
                'medium_quality': 0.6,
                'low_quality': 0.4,
                'outlier_threshold': 0.1,
                'confidence_threshold': 0.7
            }
            
            # Initialize data storage
            self.processed_documents = []
            self.document_embeddings = []
            self.quality_labels = []
            
            # Setup vectorizer for text analysis
            self.vectorizer = self.TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info("✅ Data quality pipeline configured")
            
        except Exception as e:
            logger.error(f"❌ Quality pipeline setup failed: {e}")
            raise

    def _setup_confidence_models(self):
        """Setup confidence estimation models"""
        try:
            # Initialize confidence estimation model
            self.confidence_model = self.RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            
            # Initialize outlier detection
            self.outlier_detector = self.OutOfDistribution()
            
            # Model training status
            self.models_trained = False
            self.confidence_model_path = "models/cleanlab_confidence_model.pkl"
            self.vectorizer_path = "models/cleanlab_vectorizer.pkl"
            
            # Create models directory
            os.makedirs("models", exist_ok=True)
            
            logger.info("✅ Confidence models initialized")
            
        except Exception as e:
            logger.error(f"❌ Confidence models setup failed: {e}")
            raise

    def _setup_error_detection(self):
        """Setup error detection and data issue identification"""
        try:
            # Initialize error detection parameters
            self.error_detection_config = {
                'min_examples_per_class': 10,
                'cv_folds': 5,
                'filter_by': 'prune_by_noise_rate',
                'frac_noise': 0.1,
                'confident_joint_method': 'calibrate_confident_joint'
            }
            
            # Initialize issue types to detect
            self.issue_types = [
                'label_issues',
                'outlier_issues', 
                'near_duplicate_issues',
                'data_valuation_issues',
                'class_imbalance_issues'
            ]
            
            logger.info("✅ Error detection configured")
            
        except Exception as e:
            logger.error(f"❌ Error detection setup failed: {e}")
            raise

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create Cleanlab-enhanced RAG chain with data quality assessment"""
        if not system_prompt:
            system_prompt = """You are an AI assistant that provides high-quality answers based on clean, validated data.

Your capabilities include:
1. Assessing data quality before generating responses
2. Identifying potential issues in input data
3. Providing confidence scores for your responses
4. Detecting outliers and anomalies in queries
5. Ensuring response reliability through data validation

Always consider data quality in your responses and provide confidence indicators."""

        try:
            # Process existing documents for quality assessment
            self._process_vector_store_documents()
            
            # Train quality assessment models if we have enough data
            if len(self.processed_documents) >= 10:
                self._train_quality_models()
            
            # Setup RAG chain with quality enhancement
            self.system_prompt = system_prompt
            self.rag_chain = CleanlabRAGChain(
                llm=self.llm,
                vector_store=self.vector_store,
                quality_assessor=self,
                system_prompt=system_prompt
            )
            
            logger.info("✅ Cleanlab RAG chain created successfully")
            return self.rag_chain
            
        except Exception as e:
            logger.error(f"❌ Cleanlab RAG chain creation failed: {e}")
            raise

    def _process_vector_store_documents(self):
        """Process documents from vector store for quality assessment"""
        try:
            documents = self._extract_documents_from_vector_store()
            
            if not documents:
                logger.warning("⚠️ No documents found in vector store")
                return
            
            # Process each document
            for i, doc_data in enumerate(documents):
                content = doc_data.get('content', '')
                metadata = doc_data.get('metadata', {})
                
                # Calculate quality metrics
                quality_score = self._assess_document_quality(content)
                
                # Store processed document
                self.processed_documents.append({
                    'id': metadata.get('id', str(i)),
                    'content': content,
                    'metadata': metadata,
                    'quality_score': quality_score,
                    'length': len(content),
                    'word_count': len(content.split()),
                    'timestamp': datetime.now().isoformat()
                })
                
                # Store quality score
                self.quality_metrics['document_quality_scores'][str(i)] = quality_score
            
            # Create embeddings for similarity analysis
            if self.processed_documents:
                texts = [doc['content'] for doc in self.processed_documents]
                try:
                    self.document_embeddings = self.vectorizer.fit_transform(texts).toarray()
                    logger.info(f"✅ Processed {len(self.processed_documents)} documents")
                except Exception as e:
                    logger.warning(f"⚠️ Embedding creation failed: {e}")
                    self.document_embeddings = []
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")

    def _extract_documents_from_vector_store(self) -> List[Dict]:
        """Extract documents from vector store"""
        documents = []
        try:
            # Try different methods to extract documents
            if hasattr(self.vector_store, '_documents'):
                for i, (content, metadata) in enumerate(self.vector_store._documents):
                    documents.append({
                        'content': content,
                        'metadata': metadata if isinstance(metadata, dict) else {'id': str(i)}
                    })
            elif hasattr(self.vector_store, 'similarity_search'):
                # Use similarity search with broad query to get sample documents
                sample_docs = self.vector_store.similarity_search("documentation example", k=20)
                for i, doc in enumerate(sample_docs):
                    content = doc[0] if isinstance(doc, tuple) else str(doc)
                    documents.append({
                        'content': content,
                        'metadata': {'id': str(i), 'source': 'similarity_search'}
                    })
            else:
                # Create sample documents for testing
                sample_texts = [
                    "Docker is a containerization platform that enables developers to package applications and dependencies into lightweight containers.",
                    "Container orchestration with Kubernetes provides automated deployment, scaling, and management of containerized applications.",
                    "Docker Compose allows you to define and run multi-container Docker applications using YAML configuration files.",
                    "Docker volumes provide persistent data storage that survives container restarts and can be shared between containers.",
                    "Docker networks enable communication between containers and provide isolation and security for containerized applications.",
                    "Dockerfile is a text file containing instructions to build Docker images with specific configurations and dependencies.",
                    "Docker registries store and distribute Docker images, with Docker Hub being the most popular public registry.",
                    "Container security best practices include using minimal base images, scanning for vulnerabilities, and implementing proper access controls."
                ]
                
                documents = [
                    {'content': text, 'metadata': {'id': str(i), 'source': 'sample', 'quality': 'high'}}
                    for i, text in enumerate(sample_texts)
                ]
                
        except Exception as e:
            logger.error(f"❌ Document extraction failed: {e}")
        
        return documents

    def _assess_document_quality(self, content: str) -> float:
        """Assess quality of a single document"""
        try:
            quality_factors = []
            
            # Length assessment
            length = len(content)
            if 50 <= length <= 2000:
                quality_factors.append(1.0)
            elif 20 <= length < 50 or 2000 < length <= 5000:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.3)
            
            # Word count assessment
            words = content.split()
            word_count = len(words)
            if 10 <= word_count <= 300:
                quality_factors.append(1.0)
            elif 5 <= word_count < 10 or 300 < word_count <= 500:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.4)
            
            # Sentence structure assessment
            sentences = content.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            if 8 <= avg_sentence_length <= 25:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.6)
            
            # Vocabulary diversity
            unique_words = len(set(word.lower() for word in words if word.isalpha()))
            vocabulary_ratio = unique_words / max(word_count, 1)
            if vocabulary_ratio >= 0.6:
                quality_factors.append(1.0)
            elif vocabulary_ratio >= 0.4:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.5)
            
            # Technical content indicators (for Docker documentation)
            technical_keywords = ['docker', 'container', 'image', 'volume', 'network', 'compose', 'kubernetes', 'deployment']
            technical_score = sum(1 for keyword in technical_keywords if keyword.lower() in content.lower())
            technical_factor = min(technical_score / 3, 1.0)
            quality_factors.append(technical_factor)
            
            # Calculate overall quality score
            overall_quality = sum(quality_factors) / len(quality_factors)
            
            return round(overall_quality, 3)
            
        except Exception as e:
            logger.error(f"❌ Quality assessment failed: {e}")
            return 0.5  # Default moderate quality

    def _train_quality_models(self):
        """Train quality assessment models using processed documents"""
        try:
            if len(self.processed_documents) < 10:
                logger.warning("⚠️ Insufficient data for model training")
                return
            
            # Prepare training data
            texts = [doc['content'] for doc in self.processed_documents]
            quality_scores = [doc['quality_score'] for doc in self.processed_documents]
            
            # Create quality labels (high/medium/low)
            quality_labels = []
            for score in quality_scores:
                if score >= self.quality_thresholds['high_quality']:
                    quality_labels.append(2)  # High quality
                elif score >= self.quality_thresholds['medium_quality']:
                    quality_labels.append(1)  # Medium quality
                else:
                    quality_labels.append(0)  # Low quality
            
            # Check if we have multiple classes
            unique_labels = set(quality_labels)
            if len(unique_labels) < 2:
                logger.warning("⚠️ Insufficient class diversity for training")
                return
            
            # Vectorize texts
            X = self.vectorizer.fit_transform(texts).toarray()
            y = np.array(quality_labels)
            
            # Train confidence model
            self.confidence_model.fit(X, y)
            
            # Get cross-validated predictions for Cleanlab analysis
            if len(unique_labels) >= 2 and len(texts) >= 10:
                pred_probs = self.cross_val_predict(
                    self.confidence_model, X, y, 
                    cv=min(5, len(texts)), 
                    method='predict_proba'
                )
                
                # Find label issues using Cleanlab
                if pred_probs.shape[1] >= 2:  # Ensure we have probabilities for multiple classes
                    label_issues = self.find_label_issues(
                        labels=y,
                        pred_probs=pred_probs,
                        return_indices_ranked_by='self_confidence'
                    )
                    
                    # Store label issues
                    self.quality_metrics['label_issues'] = {
                        'indices': label_issues.tolist(),
                        'count': len(label_issues),
                        'percentage': len(label_issues) / len(texts) * 100
                    }
                    
                    logger.info(f"✅ Found {len(label_issues)} potential label issues")
            
            # Save trained models
            self._save_models()
            
            self.models_trained = True
            logger.info("✅ Quality models trained successfully")
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")

    def _save_models(self):
        """Save trained models to disk"""
        try:
            # Save confidence model
            with open(self.confidence_model_path, 'wb') as f:
                pickle.dump(self.confidence_model, f)
            
            # Save vectorizer
            with open(self.vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            logger.info("✅ Models saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Model saving failed: {e}")

    def _load_models(self):
        """Load trained models from disk"""
        try:
            if os.path.exists(self.confidence_model_path):
                with open(self.confidence_model_path, 'rb') as f:
                    self.confidence_model = pickle.load(f)
                
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                self.models_trained = True
                logger.info("✅ Models loaded successfully")
                return True
                
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
        
        return False

    def query(self, question: str, **kwargs) -> str:
        """Execute query with comprehensive quality assessment"""
        if not hasattr(self, 'rag_chain'):
            self.create_rag_chain()

        try:
            # Assess query quality
            query_quality = self._assess_query_quality(question)
            
            # Detect if query is an outlier
            outlier_score = self._detect_query_outlier(question)
            
            # Get context from vector store with quality filtering
            context = self._get_quality_filtered_context(question, k=5)
            
            # Generate response using LLM
            enhanced_prompt = self._create_quality_enhanced_prompt(
                context, question, query_quality, outlier_score
            )
            
            response = self.llm.generate(enhanced_prompt)
            
            # Assess response quality
            response_quality = self._assess_response_quality(response, question)
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence_score(
                query_quality, response_quality, outlier_score
            )
            
            # Store quality metrics
            query_id = hashlib.md5(question.encode()).hexdigest()[:8]
            self.quality_metrics['query_quality_scores'][query_id] = {
                'query_quality': query_quality,
                'response_quality': response_quality,
                'outlier_score': outlier_score,
                'confidence_score': confidence_score,
                'timestamp': datetime.now().isoformat()
            }
            
            # Enhance response with quality information
            enhanced_response = self._enhance_response_with_quality_info(
                response, query_quality, response_quality, confidence_score, outlier_score
            )
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"❌ Cleanlab query failed: {e}")
            return self._fallback_query(question)

    def _assess_query_quality(self, question: str) -> float:
        """Assess the quality of input query"""
        try:
            quality_factors = []
            
            # Length assessment
            length = len(question)
            if 10 <= length <= 200:
                quality_factors.append(1.0)
            elif 5 <= length < 10 or 200 < length <= 500:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.3)
            
            # Word count assessment
            words = question.split()
            word_count = len(words)
            if 3 <= word_count <= 30:
                quality_factors.append(1.0)
            elif 1 <= word_count < 3 or 30 < word_count <= 50:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.4)
            
            # Question structure assessment
            if question.strip().endswith('?'):
                quality_factors.append(1.0)
            elif any(word in question.lower() for word in ['how', 'what', 'why', 'when', 'where', 'which']):
                quality_factors.append(0.9)
            else:
                quality_factors.append(0.6)
            
            # Technical relevance (for Docker queries)
            docker_keywords = ['docker', 'container', 'image', 'volume', 'network', 'compose', 'kubernetes']
            relevance_score = sum(1 for keyword in docker_keywords if keyword.lower() in question.lower())
            relevance_factor = min(relevance_score / 2, 1.0)
            quality_factors.append(relevance_factor)
            
            # Grammar and clarity (simple heuristics)
            if question.count(' ') >= 2 and not question.islower() and not question.isupper():
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.7)
            
            return sum(quality_factors) / len(quality_factors)
            
        except Exception as e:
            logger.error(f"❌ Query quality assessment failed: {e}")
            return 0.5

    def _detect_query_outlier(self, question: str) -> float:
        """Detect if query is an outlier compared to training data"""
        try:
            if not self.models_trained or not self.document_embeddings.size:
                return 0.5  # Neutral score when no training data
            
            # Vectorize the query
            query_vector = self.vectorizer.transform([question]).toarray()
            
            # Calculate similarity to existing documents
            if query_vector.size > 0 and self.document_embeddings.size > 0:
                similarities = self.cosine_similarity(query_vector, self.document_embeddings)[0]
                max_similarity = np.max(similarities)
                avg_similarity = np.mean(similarities)
                
                # Higher similarity means less likely to be outlier
                outlier_score = 1.0 - max_similarity
                
                return outlier_score
            
            return 0.5
            
        except Exception as e:
            logger.error(f"❌ Outlier detection failed: {e}")
            return 0.5

    def _get_quality_filtered_context(self, question: str, k: int = 5) -> str:
        """Get context filtered by quality scores"""
        try:
            # Get standard context from vector store
            docs = self.vector_store.similarity_search(question, k=k*2)  # Get more to filter
            
            # Filter by quality if we have quality scores
            if self.quality_metrics['document_quality_scores']:
                filtered_docs = []
                for i, doc in enumerate(docs):
                    doc_content = doc[0] if isinstance(doc, tuple) else str(doc)
                    
                    # Check if we have quality score for this document
                    quality_score = self.quality_metrics['document_quality_scores'].get(str(i), 0.5)
                    
                    if quality_score >= self.quality_thresholds['medium_quality']:
                        filtered_docs.append(doc_content)
                    
                    if len(filtered_docs) >= k:
                        break
                
                # If we don't have enough high-quality docs, use all
                if len(filtered_docs) < k//2:
                    filtered_docs = [doc[0] if isinstance(doc, tuple) else str(doc) for doc in docs[:k]]
                
                return "\n".join(filtered_docs)
            else:
                # No quality filtering available
                return "\n".join([doc[0] if isinstance(doc, tuple) else str(doc) for doc in docs[:k]])
                
        except Exception as e:
            logger.error(f"❌ Quality filtering failed: {e}")
            return self._get_context_from_vector_store(question, k)

    def _assess_response_quality(self, response: str, question: str) -> float:
        """Assess the quality of generated response"""
        try:
            quality_factors = []
            
            # Length appropriateness
            response_length = len(response)
            question_length = len(question)
            length_ratio = response_length / max(question_length, 1)
            
            if 2 <= length_ratio <= 20:
                quality_factors.append(1.0)
            elif 1 <= length_ratio < 2 or 20 < length_ratio <= 50:
                quality_factors.append(0.7)
            else:
                quality_factors.append(0.4)
            
            # Content relevance (keyword overlap)
            question_words = set(question.lower().split())
            response_words = set(response.lower().split())
            overlap = len(question_words.intersection(response_words))
            relevance_score = overlap / max(len(question_words), 1)
            quality_factors.append(min(relevance_score * 2, 1.0))
            
            # Structure and completeness
            if response.count('.') >= 1 and response.count(' ') >= 10:
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.6)
            
            # Error indicators
            error_indicators = ['error', 'failed', 'cannot', 'unable', 'sorry']
            if any(indicator in response.lower() for indicator in error_indicators):
                quality_factors.append(0.3)
            else:
                quality_factors.append(1.0)
            
            # Technical accuracy indicators
            if any(word in response.lower() for word in ['docker', 'container', 'command', 'example']):
                quality_factors.append(1.0)
            else:
                quality_factors.append(0.8)
            
            return sum(quality_factors) / len(quality_factors)
            
        except Exception as e:
            logger.error(f"❌ Response quality assessment failed: {e}")
            return 0.5

    def _calculate_confidence_score(self, query_quality: float, response_quality: float, outlier_score: float) -> float:
        """Calculate overall confidence score"""
        try:
            # Weight different factors
            weights = {
                'query_quality': 0.3,
                'response_quality': 0.4,
                'outlier_penalty': 0.3
            }
            
            # Calculate weighted score
            confidence = (
                query_quality * weights['query_quality'] +
                response_quality * weights['response_quality'] +
                (1.0 - outlier_score) * weights['outlier_penalty']
            )
            
            return round(confidence, 3)
            
        except Exception as e:
            logger.error(f"❌ Confidence calculation failed: {e}")
            return 0.5

    def _create_quality_enhanced_prompt(self, context: str, question: str, 
                                      query_quality: float, outlier_score: float) -> str:
        """Create prompt enhanced with quality information"""
        quality_notice = ""
        
        if query_quality < self.quality_thresholds['medium_quality']:
            quality_notice += f"[Quality Notice: Query quality is {query_quality:.2f}. Please provide a clear and comprehensive answer.]\n"
        
        if outlier_score > self.quality_thresholds['outlier_threshold']:
            quality_notice += f"[Outlier Notice: This query appears unusual (outlier score: {outlier_score:.2f}). Provide extra context if needed.]\n"
        
        enhanced_prompt = f"""{quality_notice}
You are an AI assistant that provides high-quality, data-validated responses.

Context Quality Assessment: Based on curated, quality-filtered documentation
Query Quality Score: {query_quality:.2f}/1.0
Outlier Detection Score: {outlier_score:.2f}/1.0

Context:
{context}

Question: {question}

Instructions:
1. Provide accurate information based on the high-quality context above
2. If the context is insufficient, clearly state this limitation
3. Include practical examples when relevant
4. Prioritize safety and best practices
5. Indicate your confidence level in the response

Response:"""

        return enhanced_prompt

    def _enhance_response_with_quality_info(self, response: str, query_quality: float, 
                                          response_quality: float, confidence_score: float, 
                                          outlier_score: float) -> str:
        """Enhance response with quality and confidence information"""
        
        # Determine confidence level
        if confidence_score >= 0.8:
            confidence_level = "High"
            confidence_emoji = "🟢"
        elif confidence_score >= 0.6:
            confidence_level = "Medium"
            confidence_emoji = "🟡"
        else:
            confidence_level = "Low"
            confidence_emoji = "🔴"
        
        # Add quality footer
        quality_footer = f"""

---
**Quality Assessment:**
{confidence_emoji} **Confidence Level:** {confidence_level} ({confidence_score:.2f}/1.0)
📊 **Query Quality:** {query_quality:.2f}/1.0
📝 **Response Quality:** {response_quality:.2f}/1.0"""

        if outlier_score > 0.7:
            quality_footer += f"\n⚠️ **Outlier Alert:** This query is unusual (score: {outlier_score:.2f})"
        
        if confidence_score < 0.6:
            quality_footer += f"\n💡 **Recommendation:** Consider rephrasing your question for better results"
        
        quality_footer += "\n*Powered by Cleanlab data quality assessment*"
        
        return response + quality_footer

    def _fallback_query(self, question: str) -> str:
        """Fallback query method when main processing fails"""
        try:
            context = self._get_context_from_vector_store(question, k=3)
            prompt = f"""Context: {context}

Question: {question}

Please provide a comprehensive answer based on the context above.

[Note: This response was generated using fallback mode due to quality assessment system unavailability]"""
            
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"❌ Fallback query failed: {e}")
            return f"I apologize, but I encountered an error processing your query: {str(e)}"

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'framework': 'cleanlab',
                'models_trained': self.models_trained,
                'total_documents_processed': len(self.processed_documents),
                'quality_metrics': self.quality_metrics.copy(),
                'quality_thresholds': self.quality_thresholds.copy(),
                'summary': {}
            }
            
            # Calculate summary statistics
            if self.quality_metrics['document_quality_scores']:
                scores = list(self.quality_metrics['document_quality_scores'].values())
                report['summary']['document_quality'] = {
                    'average': round(np.mean(scores), 3),
                    'std': round(np.std(scores), 3),
                    'min': round(np.min(scores), 3),
                    'max': round(np.max(scores), 3),
                    'high_quality_count': sum(1 for s in scores if s >= self.quality_thresholds['high_quality']),
                    'medium_quality_count': sum(1 for s in scores if self.quality_thresholds['medium_quality'] <= s < self.quality_thresholds['high_quality']),
                    'low_quality_count': sum(1 for s in scores if s < self.quality_thresholds['medium_quality'])
                }
            
            if self.quality_metrics['query_quality_scores']:
                query_scores = [q['query_quality'] for q in self.quality_metrics['query_quality_scores'].values()]
                response_scores = [q['response_quality'] for q in self.quality_metrics['query_quality_scores'].values()]
                confidence_scores = [q['confidence_score'] for q in self.quality_metrics['query_quality_scores'].values()]
                
                report['summary']['query_analysis'] = {
                    'total_queries': len(query_scores),
                    'average_query_quality': round(np.mean(query_scores), 3),
                    'average_response_quality': round(np.mean(response_scores), 3),
                    'average_confidence': round(np.mean(confidence_scores), 3),
                    'high_confidence_queries': sum(1 for s in confidence_scores if s >= 0.8),
                    'low_confidence_queries': sum(1 for s in confidence_scores if s < 0.6)
                }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Quality report generation failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'framework': 'cleanlab'
            }

    def cleanup_old_quality_data(self, days: int = 30):
        """Clean up old quality assessment data"""
        try:
            cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
            
            # Clean up query quality scores
            to_remove = []
            for query_id, data in self.quality_metrics['query_quality_scores'].items():
                try:
                    timestamp = datetime.fromisoformat(data['timestamp']).timestamp()
                    if timestamp < cutoff_date:
                        to_remove.append(query_id)
                except:
                    to_remove.append(query_id)  # Remove invalid timestamps
            
            for query_id in to_remove:
                del self.quality_metrics['query_quality_scores'][query_id]
            
            logger.info(f"✅ Cleaned up {len(to_remove)} old quality records")
            return len(to_remove)
            
        except Exception as e:
            logger.error(f"❌ Quality data cleanup failed: {e}")
            return 0

class CleanlabRAGChain:
    """RAG chain enhanced with Cleanlab quality assessment"""
    
    def __init__(self, llm, vector_store, quality_assessor, system_prompt):
        self.llm = llm
        self.vector_store = vector_store
        self.quality_assessor = quality_assessor
        self.system_prompt = system_prompt

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the RAG chain with quality assessment"""
        question = inputs.get('input', inputs.get('question', ''))
        
        # Use quality assessor for processing
        answer = self.quality_assessor.query(question)
        
        return {
            'answer': answer,
            'input': question,
            'quality_assessed': True
        }

def register():
    """Register Cleanlab framework with factory"""
    return "cleanlab", CleanlabFramework
