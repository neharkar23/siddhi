
import logging
from typing import Any, Dict, List, Optional, Tuple
from providers.framework_factory import BaseFramework
import json
import time

logger = logging.getLogger(__name__)

class Neo4jFramework(BaseFramework):
    """Neo4j Framework with advanced graph-based RAG capabilities"""

    def initialize(self):
        """Initialize Neo4j connection and graph schema"""
        try:
            self._import_dependencies()
            self._setup_connection()
            self._setup_graph_schema()
            self._setup_vector_index()
            self._initialized = True
            logger.info("✅ Neo4j framework initialized successfully")
        except ImportError as e:
            raise ImportError(f"Neo4j packages required: {e}")
        except Exception as e:
            logger.error(f"❌ Neo4j initialization failed: {e}")
            raise

    def _import_dependencies(self):
        """Import Neo4j dependencies"""
        from neo4j import GraphDatabase
        from neo4j.exceptions import ServiceUnavailable, AuthError
        
        self.GraphDatabase = GraphDatabase
        self.ServiceUnavailable = ServiceUnavailable
        self.AuthError = AuthError

    def _setup_connection(self):
        """Setup Neo4j database connection"""
        try:
            # Get connection details from config
            uri = self.kwargs.get('uri', 'bolt://localhost:7687')
            username = self.kwargs.get('username', 'neo4j')
            password = self.kwargs.get('password', 'password')
            database = self.kwargs.get('database', 'neo4j')
            
            self.driver = self.GraphDatabase.driver(
                uri, 
                auth=(username, password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            
            # Test connection
            with self.driver.session(database=database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    logger.info("✅ Neo4j connection established successfully")
                    
            self.database = database
            
        except self.AuthError as e:
            logger.error(f"❌ Neo4j authentication failed: {e}")
            raise
        except self.ServiceUnavailable as e:
            logger.error(f"❌ Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Neo4j connection failed: {e}")
            raise

    def _setup_graph_schema(self):
        """Setup graph schema for Docker knowledge"""
        try:
            with self.driver.session(database=self.database) as session:
                # Create constraints
                constraints = [
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
                    "CREATE CONSTRAINT IF NOT EXISTS FOR (cmd:Command) REQUIRE cmd.name IS UNIQUE"
                ]
                
                for constraint in constraints:
                    session.run(constraint)
                
                # Create indexes for performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.content)",
                    "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.title)",
                    "CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.description)",
                    "CREATE INDEX IF NOT EXISTS FOR (t:Topic) ON (t.category)"
                ]
                
                for index in indexes:
                    session.run(index)
                    
                logger.info("✅ Neo4j graph schema created successfully")
                
        except Exception as e:
            logger.error(f"❌ Failed to setup graph schema: {e}")
            raise

    def _setup_vector_index(self):
        """Setup vector similarity index in Neo4j"""
        try:
            with self.driver.session(database=self.database) as session:
                # Create vector index for semantic search
                vector_index_query = """
                CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
                FOR (d:Document) ON (d.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 1536,
                        `vector.similarity_function`: 'cosine'
                    }
                }
                """
                session.run(vector_index_query)
                logger.info("✅ Neo4j vector index created successfully")
                
        except Exception as e:
            logger.warning(f"⚠️ Vector index creation failed (may not be supported): {e}")

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create Neo4j graph-enhanced RAG chain"""
        if not system_prompt:
            system_prompt = """You are a Docker expert with access to a comprehensive knowledge graph.

Your capabilities:
1. Access to structured Docker knowledge in a graph database
2. Understanding of relationships between Docker concepts
3. Ability to trace dependencies and connections
4. Knowledge of Docker commands, concepts, and best practices

Use both the graph relationships and document content to provide comprehensive answers."""

        try:
            # Initialize knowledge graph with Docker concepts
            self._populate_initial_knowledge()
            self.system_prompt = system_prompt
            
            logger.info("✅ Neo4j RAG chain created successfully")
            return self.driver
            
        except Exception as e:
            logger.error(f"❌ Failed to create Neo4j RAG chain: {e}")
            raise

    def _populate_initial_knowledge(self):
        """Populate graph with initial Docker knowledge"""
        try:
            with self.driver.session(database=self.database) as session:
                # Create Docker concepts and relationships
                docker_knowledge = [
                    {
                        "concept": "Container",
                        "description": "Lightweight, standalone executable package",
                        "related_commands": ["docker run", "docker start", "docker stop"],
                        "topics": ["Runtime", "Isolation"]
                    },
                    {
                        "concept": "Image",
                        "description": "Read-only template for creating containers",
                        "related_commands": ["docker build", "docker pull", "docker push"],
                        "topics": ["Build", "Registry"]
                    },
                    {
                        "concept": "Dockerfile",
                        "description": "Text file with instructions to build images",
                        "related_commands": ["docker build"],
                        "topics": ["Build", "Configuration"]
                    },
                    {
                        "concept": "Volume",
                        "description": "Persistent data storage for containers",
                        "related_commands": ["docker volume create", "docker volume ls"],
                        "topics": ["Storage", "Persistence"]
                    },
                    {
                        "concept": "Network",
                        "description": "Communication layer between containers",
                        "related_commands": ["docker network create", "docker network ls"],
                        "topics": ["Networking", "Communication"]
                    }
                ]
                
                for item in docker_knowledge:
                    # Create concept node
                    session.run("""
                        MERGE (c:Concept {name: $name})
                        SET c.description = $description
                        """, name=item["concept"], description=item["description"])
                    
                    # Create command relationships
                    for cmd in item["related_commands"]:
                        session.run("""
                            MERGE (c:Concept {name: $concept})
                            MERGE (cmd:Command {name: $command})
                            MERGE (c)-[:USES_COMMAND]->(cmd)
                            """, concept=item["concept"], command=cmd)
                    
                    # Create topic relationships
                    for topic in item["topics"]:
                        session.run("""
                            MERGE (c:Concept {name: $concept})
                            MERGE (t:Topic {name: $topic})
                            MERGE (c)-[:BELONGS_TO]->(t)
                            """, concept=item["concept"], topic=topic)
                
                logger.info("✅ Initial Docker knowledge populated in graph")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to populate initial knowledge: {e}")

    def query(self, question: str, **kwargs) -> str:
        """Execute query using Neo4j graph-enhanced RAG"""
        if not hasattr(self, 'system_prompt'):
            self.create_rag_chain()

        try:
            # Step 1: Get vector similarity context
            vector_context = self._get_context_from_vector_store(question, k=5)
            
            # Step 2: Get graph-based context
            graph_context = self._get_graph_context(question)
            
            # Step 3: Get related concepts and commands
            related_info = self._get_related_concepts(question)
            
            # Step 4: Combine all contexts
            combined_context = self._combine_contexts(vector_context, graph_context, related_info)
            
            # Step 5: Generate response
            enhanced_prompt = f"""{self.system_prompt}

Vector Context:
{vector_context}

Graph Relationships:
{graph_context}

Related Concepts:
{related_info}

Question: {question}

Provide a comprehensive answer using both the document context and graph relationships:"""

            response = self.llm.generate(enhanced_prompt)
            
            # Step 6: Store query and response in graph for learning
            self._store_query_response(question, response)
            
            return response

        except Exception as e:
            logger.error(f"❌ Neo4j query failed: {e}")
            return self._fallback_query(question)

    def _get_graph_context(self, question: str) -> str:
        """Get context from graph relationships"""
        try:
            with self.driver.session(database=self.database) as session:
                # Extract key terms from question
                key_terms = self._extract_key_terms(question)
                
                graph_results = []
                for term in key_terms:
                    # Find related concepts
                    result = session.run("""
                        MATCH (c:Concept)
                        WHERE toLower(c.name) CONTAINS toLower($term) 
                           OR toLower(c.description) CONTAINS toLower($term)
                        OPTIONAL MATCH (c)-[:USES_COMMAND]->(cmd:Command)
                        OPTIONAL MATCH (c)-[:BELONGS_TO]->(t:Topic)
                        RETURN c.name as concept, c.description as description,
                               collect(DISTINCT cmd.name) as commands,
                               collect(DISTINCT t.name) as topics
                        LIMIT 5
                        """, term=term)
                    
                    for record in result:
                        concept_info = {
                            "concept": record["concept"],
                            "description": record["description"],
                            "commands": [cmd for cmd in record["commands"] if cmd],
                            "topics": [topic for topic in record["topics"] if topic]
                        }
                        graph_results.append(concept_info)
                
                # Format graph context
                if graph_results:
                    context_parts = []
                    for info in graph_results:
                        part = f"Concept: {info['concept']} - {info['description']}"
                        if info['commands']:
                            part += f"\nRelated Commands: {', '.join(info['commands'])}"
                        if info['topics']:
                            part += f"\nTopics: {', '.join(info['topics'])}"
                        context_parts.append(part)
                    
                    return "\n\n".join(context_parts)
                else:
                    return "No specific graph relationships found for this query."
                    
        except Exception as e:
            logger.error(f"❌ Graph context retrieval failed: {e}")
            return ""

    def _get_related_concepts(self, question: str) -> str:
        """Get related concepts and their relationships"""
        try:
            with self.driver.session(database=self.database) as session:
                # Find concept relationships
                result = session.run("""
                    MATCH (c1:Concept)-[r]->(c2:Concept)
                    WHERE toLower(c1.name) CONTAINS any(term IN $terms WHERE toLower(c1.name) CONTAINS term)
                       OR toLower(c2.name) CONTAINS any(term IN $terms WHERE toLower(c2.name) CONTAINS term)
                    RETURN c1.name as from_concept, type(r) as relationship, c2.name as to_concept
                    LIMIT 10
                    """, terms=self._extract_key_terms(question))
                
                relationships = []
                for record in result:
                    relationships.append(f"{record['from_concept']} {record['relationship']} {record['to_concept']}")
                
                return "Concept Relationships:\n" + "\n".join(relationships) if relationships else ""
                
        except Exception as e:
            logger.error(f"❌ Related concepts retrieval failed: {e}")
            return ""

    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key Docker-related terms from question"""
        docker_terms = [
            "container", "image", "dockerfile", "volume", "network", "compose",
            "build", "run", "start", "stop", "pull", "push", "exec", "logs",
            "registry", "hub", "swarm", "service", "stack", "node"
        ]
        
        question_lower = question.lower()
        found_terms = [term for term in docker_terms if term in question_lower]
        
        # Also extract potential concept words (capitalized or technical terms)
        words = question.split()
        technical_terms = [word.strip('.,!?') for word in words 
                          if len(word) > 3 and (word.isupper() or word.istitle())]
        
        return list(set(found_terms + technical_terms))

    def _combine_contexts(self, vector_context: str, graph_context: str, related_info: str) -> str:
        """Combine different context sources"""
        contexts = []
        if vector_context:
            contexts.append(f"Document Context:\n{vector_context}")
        if graph_context:
            contexts.append(f"Graph Context:\n{graph_context}")
        if related_info:
            contexts.append(f"Related Information:\n{related_info}")
        
        return "\n\n".join(contexts)

    def _store_query_response(self, question: str, response: str):
        """Store query and response for future learning"""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("""
                    CREATE (q:Query {
                        text: $question,
                        response: $response,
                        timestamp: datetime(),
                        id: randomUUID()
                    })
                    """, question=question, response=response)
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to store query/response: {e}")

    def _fallback_query(self, question: str) -> str:
        """Fallback query method"""
        try:
            context = self._get_context_from_vector_store(question, k=3)
            prompt = f"""Context: {context}

Question: {question}

Please provide a comprehensive Docker answer based on the context above."""
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"❌ Neo4j fallback failed: {e}")
            return f"Error processing query with Neo4j: {str(e)}"

    def add_document_to_graph(self, content: str, metadata: Dict = None) -> bool:
        """Add document to both vector store and graph"""
        try:
            with self.driver.session(database=self.database) as session:
                doc_id = metadata.get('id', str(time.time())) if metadata else str(time.time())
                title = metadata.get('title', 'Untitled') if metadata else 'Untitled'
                
                # Add to graph
                session.run("""
                    CREATE (d:Document {
                        id: $id,
                        title: $title,
                        content: $content,
                        timestamp: datetime(),
                        metadata: $metadata
                    })
                    """, id=doc_id, title=title, content=content, 
                    metadata=json.dumps(metadata or {}))
                
                # Extract and link concepts
                self._extract_and_link_concepts(session, doc_id, content)
                
                logger.info(f"✅ Document added to Neo4j graph: {doc_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to add document to graph: {e}")
            return False

    def _extract_and_link_concepts(self, session, doc_id: str, content: str):
        """Extract concepts from document and create relationships"""
        try:
            # Simple concept extraction (can be enhanced with NLP)
            docker_concepts = ["container", "image", "dockerfile", "volume", "network"]
            found_concepts = [concept for concept in docker_concepts 
                            if concept.lower() in content.lower()]
            
            for concept in found_concepts:
                session.run("""
                    MATCH (d:Document {id: $doc_id})
                    MERGE (c:Concept {name: $concept})
                    MERGE (d)-[:MENTIONS]->(c)
                    """, doc_id=doc_id, concept=concept.title())
                    
        except Exception as e:
            logger.warning(f"⚠️ Concept extraction failed: {e}")

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph database statistics"""
        try:
            with self.driver.session(database=self.database) as session:
                stats = {}
                
                # Count nodes by type
                result = session.run("""
                    MATCH (n)
                    RETURN labels(n) as labels, count(n) as count
                    """)
                
                node_counts = {}
                for record in result:
                    label = record["labels"][0] if record["labels"] else "Unknown"
                    node_counts[label] = record["count"]
                
                stats["node_counts"] = node_counts
                
                # Count relationships
                result = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as relationship_type, count(r) as count
                    """)
                
                relationship_counts = {}
                for record in result:
                    relationship_counts[record["relationship_type"]] = record["count"]
                
                stats["relationship_counts"] = relationship_counts
                
                return stats
                
        except Exception as e:
            logger.error(f"❌ Failed to get graph statistics: {e}")
            return {}

    def close(self):
        """Close Neo4j connection"""
        if hasattr(self, 'driver') and self.driver:
            self.driver.close()
            logger.info("✅ Neo4j connection closed")

def register():
    """Register Neo4j framework with factory"""
    return "neo4j", Neo4jFramework
