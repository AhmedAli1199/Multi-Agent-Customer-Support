"""
Base agent class for all specialized agents.
Provides common functionality for Gemini LLM integration.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import google.generativeai as genai
from config import GEMINI_API_KEY, ModelConfig

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, name: str, model_name: str, system_prompt: str):
        """
        Initialize base agent.

        Args:
            name: Agent name
            model_name: Gemini model to use (pro or flash)
            system_prompt: System prompt defining agent behavior
        """
        self.name = name
        self.model_name = model_name
        self.system_prompt = system_prompt

        # Initialize Gemini model
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=genai.GenerationConfig(
                temperature=ModelConfig.TEMPERATURE,
                max_output_tokens=ModelConfig.MAX_OUTPUT_TOKENS,
                top_p=ModelConfig.TOP_P,
                top_k=ModelConfig.TOP_K,
            )
        )

        print(f"[OK] {self.name} initialized with model: {self.model_name}")

    def generate(self, prompt: str, context: Optional[Dict] = None) -> str:
        """
        Generate response using Gemini.

        Args:
            prompt: User prompt
            context: Optional context dictionary

        Returns:
            Generated response
        """
        # Build full prompt
        full_prompt = f"{self.system_prompt}\n\n{prompt}"

        if context:
            full_prompt = f"{full_prompt}\n\nContext: {context}"

        # Generate response
        response = self.model.generate_content(full_prompt)

        return response.text

    @abstractmethod
    def process(self, customer_query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Process customer query and return agent response.
        Must be implemented by subclasses.

        Args:
            customer_query: Customer's question/request
            conversation_history: Previous messages in conversation

        Returns:
            Dict with agent response and metadata
        """
        pass

    def format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history for prompt"""
        if not history:
            return "No previous conversation."

        formatted = "Previous conversation:\n"
        for msg in history[-5:]:  # Last 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"{role.capitalize()}: {content}\n"

        return formatted

    def log_interaction(self, query: str, response: str):
        """Log agent interaction (for debugging and prompt collection)"""
        print(f"\n[{self.name}]")
        print(f"Query: {query[:100]}...")
        print(f"Response: {response[:100]}...")
