"""
Knowledge Agent: RAG-based FAQ and policy query handler.
"""
from typing import Dict, List
from agents.base_agent import BaseAgent
from config import ModelConfig
from tools.knowledge_retrieval import knowledge_retriever

KNOWLEDGE_SYSTEM_PROMPT = """You are a Knowledge Agent for customer support. Your role is to:
1. Answer customer questions using the provided knowledge base
2. Provide accurate, helpful, and empathetic responses
3. Cite sources when possible
4. Admit when you don't know something rather than making up information

Guidelines:
- Be concise but thorough
- Use friendly, professional tone
- If the knowledge base doesn't have the answer, say so clearly
- Offer to escalate complex questions
- Always prioritize customer satisfaction"""

class KnowledgeAgent(BaseAgent):
    """Knowledge agent for FAQ and information queries"""

    def __init__(self):
        super().__init__(
            name="Knowledge Agent",
            model_name=ModelConfig.SECONDARY_MODEL,  # Uses SECONDARY_MODEL for cost efficiency
            system_prompt=KNOWLEDGE_SYSTEM_PROMPT
        )
        self.retriever = knowledge_retriever

    def process(self, customer_query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Process query using RAG.

        Returns:
            Dict with agent response
        """
        # Retrieve relevant documents
        context = self.retriever.get_formatted_context(customer_query)

        # Build prompt
        history_context = self.format_conversation_history(conversation_history) if conversation_history else ""

        prompt = f"""Using the knowledge base below, answer the customer's question.

Knowledge Base:
{context}

{history_context}

Customer Question: "{customer_query}"

Provide a helpful, accurate response. If the knowledge base doesn't contain the answer, politely say so and offer to help in another way or escalate to a human agent."""

        # Generate response
        response_text = self.generate(prompt)

        self.log_interaction(customer_query, response_text)

        # Check if knowledge was sufficient
        insufficient_indicators = [
            "don't have information",
            "not in the knowledge base",
            "unable to find",
            "don't know",
            "recommend speaking with"
        ]

        needs_escalation = any(indicator in response_text.lower() for indicator in insufficient_indicators)

        return {
            "agent": self.name,
            "response": response_text,
            "context_used": context[:200] + "..." if len(context) > 200 else context,
            "confidence": "low" if needs_escalation else "high",
            "needs_escalation": needs_escalation
        }
