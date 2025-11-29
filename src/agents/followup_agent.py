"""
Follow-Up Agent: Proactive engagement and satisfaction checks.
"""
from typing import Dict, List
from agents.base_agent import BaseAgent
from config import ModelConfig

FOLLOWUP_SYSTEM_PROMPT = """You are a Follow-Up Agent for customer support. Your role is to:
1. Check if customer's issue was resolved satisfactorily
2. Generate appropriate follow-up messages
3. Collect feedback and satisfaction ratings
4. Identify unresolved issues that need escalation

Guidelines:
- Be empathetic and customer-focused
- Keep messages concise and friendly
- Offer additional help proactively
- Thank customers for their patience

Message Types:
- SATISFACTION_CHECK: Ask if issue was resolved
- FOLLOW_UP_OFFER: Offer additional assistance
- FEEDBACK_REQUEST: Request feedback/rating
- CLOSURE: Close conversation positively"""

class FollowUpAgent(BaseAgent):
    """Follow-up agent for customer satisfaction"""

    def __init__(self):
        super().__init__(
            name="Follow-Up Agent",
            model_name=ModelConfig.GEMINI_FLASH,
            system_prompt=FOLLOWUP_SYSTEM_PROMPT
        )

    def process(self, customer_query: str, conversation_history: List[Dict] = None, resolution_summary: str = None) -> Dict:
        """
        Generate follow-up message.

        Args:
            customer_query: Original query
            conversation_history: Full conversation
            resolution_summary: Summary of how issue was resolved

        Returns:
            Dict with follow-up message
        """
        history_context = self.format_conversation_history(conversation_history) if conversation_history else ""

        prompt = f"""Generate a follow-up message for this customer interaction:

{history_context}

Original Issue: "{customer_query}"

Resolution Summary: {resolution_summary or "Issue addressed"}

Create a brief, friendly follow-up message to:
1. Confirm the issue was resolved
2. Offer additional help if needed
3. Thank the customer

Keep it under 50 words."""

        response_text = self.generate(prompt)

        self.log_interaction(customer_query, response_text)

        return {
            "agent": self.name,
            "follow_up_message": response_text,
            "message_type": "SATISFACTION_CHECK"
        }
