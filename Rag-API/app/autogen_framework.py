"""
Production-grade AutoGen Framework Implementation
Multi-agent conversation and collaboration system
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from providers.framework_factory import BaseFramework

logger = logging.getLogger(__name__)

class AutoGenFramework(BaseFramework):
    """Production AutoGen Framework with multi-agent capabilities"""

    def initialize(self):
        """Initialize AutoGen multi-agent system"""
        try:
            self._import_dependencies()
            self._setup_agent_config()
            self._initialized = True
            logger.info("✅ AutoGen framework initialized successfully")
        except ImportError as e:
            raise ImportError(f"AutoGen packages required: {e}")
        except Exception as e:
            logger.error(f"❌ AutoGen initialization failed: {e}")
            raise

    def _import_dependencies(self):
        """Import AutoGen dependencies"""
        import autogen
        self.autogen = autogen

    def _setup_agent_config(self):
        """Setup agent configuration"""
        self.agent_config = {
            "temperature": getattr(self.llm, 'temperature', 0.7),
            "timeout": 300,
            "cache_seed": None,
            "model": getattr(self.llm, 'model_name', 'gpt-4'),
            "api_key": getattr(self.llm, 'api_key', None)
        }

    def create_rag_chain(self, system_prompt: str = None) -> Any:
        """Create AutoGen agent system"""
        if not system_prompt:
            system_prompt = """You are a Docker expert assistant working in a multi-agent team.

Your role:
1. Provide accurate Docker technical assistance
2. Collaborate with other agents when needed
3. Search through documentation before answering
4. Validate solutions with team members
5. Provide practical, tested Docker solutions

Always be thorough and consider security and best practices."""

        try:
            # Create specialist agents
            self._create_docker_expert()
            self._create_research_assistant()
            self._create_user_proxy()
            self._create_group_chat()
            
            logger.info("✅ AutoGen agent system created")
            return self.group_chat_manager
            
        except Exception as e:
            logger.error(f"❌ AutoGen agent creation failed: {e}")
            raise

    def _create_docker_expert(self):
        """Create Docker expert agent"""
        self.docker_expert = self.autogen.AssistantAgent(
            name="docker_expert",
            system_message="""You are a Docker expert with deep knowledge of:
- Container architecture and runtime
- Dockerfile best practices and optimization
- Docker Compose for multi-container applications
- Docker networking and security
- Container orchestration basics
- Troubleshooting common Docker issues

Always provide practical, tested solutions with examples.""",
            llm_config=self.agent_config,
            human_input_mode="NEVER"
        )

    def _create_research_assistant(self):
        """Create research assistant agent"""
        self.research_assistant = self.autogen.AssistantAgent(
            name="research_assistant",
            system_message="""You are a research assistant specializing in finding Docker information.

Your tasks:
1. Search through available documentation
2. Find relevant examples and tutorials
3. Verify information accuracy
4. Provide source references
5. Suggest additional resources when helpful

Always cite sources and provide comprehensive research results.""",
            llm_config=self.agent_config,
            human_input_mode="NEVER"
        )

    def _create_user_proxy(self):
        """Create user proxy agent"""
        self.user_proxy = self.autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={
                "work_dir": "docker_workspace",
                "use_docker": False  # Don't run Docker in Docker
            },
            system_message="""You coordinate between the user and the Docker experts.
Your role is to:
1. Present questions clearly to the team
2. Ensure comprehensive answers are provided
3. Ask for clarification when needed
4. Summarize team discussions
5. Terminate conversations when questions are fully answered"""
        )

    def _create_group_chat(self):
        """Create group chat for agent collaboration"""
        try:
            self.group_chat = self.autogen.GroupChat(
                agents=[self.docker_expert, self.research_assistant, self.user_proxy],
                messages=[],
                max_round=10,
                speaker_selection_method="round_robin"
            )
            
            self.group_chat_manager = self.autogen.GroupChatManager(
                groupchat=self.group_chat,
                llm_config=self.agent_config
            )
            
        except Exception as e:
            logger.warning(f"⚠️ Group chat setup failed, using simple agents: {e}")
            self.group_chat_manager = self.docker_expert

    def query(self, question: str, **kwargs) -> str:
        """Execute query using AutoGen multi-agent system"""
        if not hasattr(self, 'group_chat_manager'):
            self.create_rag_chain()

        try:
            # Get context from vector store
            context = self._get_context_from_vector_store(question, k=5)
            
            # Format comprehensive prompt
            enhanced_question = f"""Context from documentation:
{context}

User question: {question}

Please provide a comprehensive Docker solution using the above context and your expertise. 
Include practical examples, best practices, and any relevant warnings or considerations.

Collaborate as a team to ensure the answer is thorough and accurate."""

            # Start group conversation
            if hasattr(self.group_chat_manager, 'groupchat'):
                # Use group chat
                self.user_proxy.initiate_chat(
                    self.group_chat_manager,
                    message=enhanced_question
                )
                
                # Extract final response
                messages = self.group_chat.messages
                if messages:
                    # Get the most comprehensive response from Docker expert
                    for msg in reversed(messages):
                        if (msg.get("name") == "docker_expert" and 
                            len(msg.get("content", "")) > 100):
                            return self._clean_response(msg["content"])
                    
                    # Fallback to last message
                    return self._clean_response(messages[-1].get("content", "No response generated"))
                else:
                    return "No response generated from agents"
            else:
                # Use single agent fallback
                self.user_proxy.initiate_chat(
                    self.docker_expert,
                    message=enhanced_question
                )
                
                # Get response from chat history
                chat_history = self.user_proxy.chat_messages.get(self.docker_expert, [])
                if chat_history:
                    return self._clean_response(chat_history[-1].get("content", ""))
                else:
                    return "No response generated"

        except Exception as e:
            logger.error(f"❌ AutoGen query failed: {e}")
            return self._fallback_query(question)

    def _clean_response(self, response: str) -> str:
        """Clean and format agent response"""
        if not response:
            return "No response generated"
        
        # Remove agent-specific markers
        response = response.replace("TERMINATE", "").strip()
        
        # Remove excessive whitespace
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        cleaned = '\n'.join(lines)
        
        return cleaned if cleaned else "Response generated but content was empty"

    def _fallback_query(self, question: str) -> str:
        """Fallback query method"""
        try:
            context = self._get_context_from_vector_store(question, k=3)
            prompt = f"""As a Docker expert, please answer this question:

Context: {context}
Question: {question}

Provide a comprehensive answer with examples and best practices."""
            
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"❌ AutoGen fallback failed: {e}")
            return f"Error processing query with AutoGen: {str(e)}"

    def get_agent_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history from agents"""
        try:
            if hasattr(self, 'group_chat') and self.group_chat.messages:
                return self.group_chat.messages
            elif hasattr(self, 'user_proxy') and hasattr(self.user_proxy, 'chat_messages'):
                all_messages = []
                for agent_messages in self.user_proxy.chat_messages.values():
                    all_messages.extend(agent_messages)
                return all_messages
        except Exception as e:
            logger.warning(f"⚠️ Failed to get conversation history: {e}")
        return []

    def reset_agents(self):
        """Reset agent conversation history"""
        try:
            if hasattr(self, 'group_chat'):
                self.group_chat.messages = []
            if hasattr(self, 'user_proxy'):
                self.user_proxy.chat_messages = {}
            logger.info("✅ AutoGen agents reset")
        except Exception as e:
            logger.warning(f"⚠️ Failed to reset agents: {e}")

def register():
    """Register AutoGen framework with factory"""
    return "autogen", AutoGenFramework
