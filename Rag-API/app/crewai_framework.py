
import logging
from typing import Any, Dict, List, Optional
from providers.framework_factory import BaseFramework

logger = logging.getLogger(__name__)

class CrewAIFramework(BaseFramework):
    """CrewAI Framework with advanced agent crews"""

    def initialize(self):
        """Initialize CrewAI agent crews"""
        try:
            self._import_dependencies()
            self._setup_tools()
            self._initialized = True
            logger.info("✅ CrewAI framework initialized successfully")
        except ImportError as e:
            raise ImportError(f"CrewAI packages required: {e}")
        except Exception as e:
            logger.error(f"❌ CrewAI initialization failed: {e}")
            raise

    def _import_dependencies(self):
        """Import CrewAI dependencies"""
        from crewai import Agent, Task, Crew, Process
        from crewai.tools import BaseTool
        
        self.Agent = Agent
        self.Task = Task
        self.Crew = Crew
        self.Process = Process
        self.BaseTool = BaseTool

    def _setup_tools(self):
        """Setup custom tools for CrewAI agents"""
        try:
            # Document search tool
            class DocumentSearchTool(self.BaseTool):
                name: str = "document_search"
                description: str = "Search through Docker documentation and knowledge base"
                
                def _run(self, query: str) -> str:
                    try:
                        docs = self.vector_store.similarity_search(query, k=5)
                        return "\n".join([doc[0] if isinstance(doc, tuple) else str(doc) for doc in docs])
                    except Exception as e:
                        return f"Error searching documents: {e}"

            # Web search tool (if available)
            class WebSearchTool(self.BaseTool):
                name: str = "web_search"
                description: str = "Search the web for latest Docker information"
                
                def _run(self, query: str) -> str:
                    try:
                        # Use DuckDuckGo or other web search
                        from duckduckgo_search import ddg
                        results = ddg(f"Docker {query}", max_results=3)
                        return "\n".join([f"{r['title']}: {r['body']}" for r in results])
                    except:
                        return "Web search not available"

            self.document_search_tool = DocumentSearchTool()
            self.web_search_tool = WebSearchTool()
            self.tools = [self.document_search_tool]
            
            logger.info("✅ CrewAI tools initialized")
            
        except Exception as e:
            logger.warning(f"⚠️ CrewAI tools setup failed: {e}")
            self.tools = []

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create CrewAI specialized crew"""
        if not system_prompt:
            system_prompt = """You are part of a specialized Docker assistance crew.
Work collaboratively to provide comprehensive, accurate Docker solutions."""

        try:
            # Create specialized agents
            self._create_docker_architect()
            self._create_security_specialist()
            self._create_devops_engineer()
            self._create_crew()
            
            logger.info("✅ CrewAI crew created successfully")
            return self.crew
            
        except Exception as e:
            logger.error(f"❌ CrewAI crew creation failed: {e}")
            raise

    def _create_docker_architect(self):
        """Create Docker architect agent"""
        self.docker_architect = self.Agent(
            role='Docker Architect',
            goal='Design optimal Docker solutions and architectures',
            backstory="""You are a senior Docker architect with 10+ years of experience.
You specialize in:
- Container architecture design
- Performance optimization
- Scalability planning
- Best practices implementation
- Complex multi-container applications""",
            verbose=True,
            allow_delegation=True,
            tools=self.tools,
            llm=CrewAILLMWrapper(self.llm)
        )

    def _create_security_specialist(self):
        """Create security specialist agent"""
        self.security_specialist = self.Agent(
            role='Docker Security Specialist',
            goal='Ensure Docker deployments are secure and follow security best practices',
            backstory="""You are a cybersecurity expert focused on container security.
Your expertise includes:
- Container vulnerability assessment
- Docker security hardening
- Access control and permissions
- Network security for containers
- Compliance and audit requirements""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=CrewAILLMWrapper(self.llm)
        )

    def _create_devops_engineer(self):
        """Create DevOps engineer agent"""
        self.devops_engineer = self.Agent(
            role='DevOps Engineer',
            goal='Provide practical Docker implementation and operational guidance',
            backstory="""You are a DevOps engineer with extensive Docker experience.
You focus on:
- CI/CD pipeline integration
- Container orchestration
- Monitoring and logging
- Troubleshooting and debugging
- Production deployment strategies""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=CrewAILLMWrapper(self.llm)
        )

    def _create_crew(self):
        """Create the collaborative crew"""
        self.crew = self.Crew(
            agents=[
                self.docker_architect,
                self.security_specialist,
                self.devops_engineer
            ],
            process=self.Process.sequential,
            verbose=True,
            memory=True,
            embedder={
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small"
                }
            } if hasattr(self.llm, 'api_key') else None
        )

    def query(self, question: str, **kwargs) -> str:
        """Execute query using CrewAI crew"""
        if not hasattr(self, 'crew'):
            self.create_rag_chain()

        try:
            # Get context from vector store
            context = self._get_context_from_vector_store(question, k=5)
            
            # Create tasks for the crew
            tasks = self._create_tasks_for_question(question, context)
            
            # Execute crew with tasks
            result = self.crew.kickoff(tasks=tasks)
            
            # Format and return result
            if hasattr(result, 'raw'):
                return str(result.raw)
            else:
                return str(result)

        except Exception as e:
            logger.error(f"❌ CrewAI query failed: {e}")
            return self._fallback_query(question)

    def _create_tasks_for_question(self, question: str, context: str) -> List:
        """Create appropriate tasks based on the question"""
        tasks = []
        
        try:
            # Research task
            research_task = self.Task(
                description=f"""Research the following Docker question using available documentation:
                
Question: {question}
Context: {context}

Provide comprehensive research findings including:
1. Relevant Docker concepts and components
2. Best practices and recommendations
3. Common pitfalls and how to avoid them
4. Related documentation references""",
                agent=self.devops_engineer,
                expected_output="Detailed research findings with practical insights"
            )
            
            # Architecture task
            architecture_task = self.Task(
                description=f"""Based on the research findings, design the optimal Docker solution for:

Question: {question}

Provide:
1. Architectural recommendations
2. Container design patterns
3. Performance considerations
4. Scalability factors
5. Implementation roadmap""",
                agent=self.docker_architect,
                expected_output="Comprehensive architectural solution with implementation details"
            )
            
            # Security review task
            security_task = self.Task(
                description=f"""Review the Docker solution for security considerations:

Question: {question}

Analyze and provide:
1. Security best practices
2. Vulnerability assessments
3. Access control recommendations
4. Network security considerations
5. Compliance requirements""",
                agent=self.security_specialist,
                expected_output="Security analysis with actionable recommendations"
            )
            
            tasks = [research_task, architecture_task, security_task]
            
        except Exception as e:
            logger.error(f"❌ Task creation failed: {e}")
            # Create simple fallback task
            fallback_task = self.Task(
                description=f"Answer this Docker question: {question}",
                agent=self.docker_architect,
                expected_output="Comprehensive Docker answer"
            )
            tasks = [fallback_task]
        
        return tasks

    def _fallback_query(self, question: str) -> str:
        """Fallback query method"""
        try:
            context = self._get_context_from_vector_store(question, k=3)
            prompt = f"""As a Docker expert crew, provide a comprehensive answer:

Context: {context}
Question: {question}

Include architectural, security, and operational perspectives."""
            
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"❌ CrewAI fallback failed: {e}")
            return f"Error processing query with CrewAI: {str(e)}"

    def get_crew_memory(self) -> Dict[str, Any]:
        """Get crew memory and insights"""
        try:
            if hasattr(self.crew, 'memory') and self.crew.memory:
                return {
                    "total_tasks": len(self.crew.memory.get('tasks', [])),
                    "agents_used": len(self.crew.agents),
                    "last_execution": "Available"
                }
        except Exception as e:
            logger.warning(f"⚠️ Failed to get crew memory: {e}")
        return {"status": "Memory not available"}

class CrewAILLMWrapper:
    """Wrapper to make LLM compatible with CrewAI"""
    
    def __init__(self, llm):
        self.llm = llm
        self.model_name = getattr(llm, 'model_name', 'wrapped-llm')

    def generate(self, prompt: str, **kwargs) -> str:
        return self.llm.generate(prompt, **kwargs)

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.llm.generate(prompt, **kwargs)

def register():
    """Register CrewAI framework with factory"""
    return "crewai", CrewAIFramework
