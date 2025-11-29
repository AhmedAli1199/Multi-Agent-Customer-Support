"""
Escalation Agent: Handles human handoff for complex issues.
"""
from typing import Dict, List
from agents.base_agent import BaseAgent
from config import ModelConfig

ESCALATION_SYSTEM_PROMPT = """You are an Escalation Agent for customer support. Your role is to:
1. Prepare comprehensive summaries for human agents
2. Identify why automation couldn't resolve the issue
3. Suggest recommended next steps
4. Prioritize based on urgency and customer impact

When escalating, provide:
- Clear summary of the issue
- What was attempted
- Why it needs human intervention
- Recommended priority level
- Customer sentiment assessment

Be thorough but concise."""

class EscalationAgent(BaseAgent):
    """Escalation agent for human handoff"""

    def __init__(self):
        super().__init__(
            name="Escalation Agent",
            model_name=ModelConfig.GEMINI_PRO,
            system_prompt=ESCALATION_SYSTEM_PROMPT
        )

    def process(self, customer_query: str, conversation_history: List[Dict] = None, escalation_reason: str = None) -> Dict:
        """
        Create escalation summary for human agent.

        Args:
            customer_query: Original query
            conversation_history: Full conversation
            escalation_reason: Why escalation is needed

        Returns:
            Dict with escalation summary
        """
        history_context = self.format_conversation_history(conversation_history) if conversation_history else ""

        prompt = f"""Create an escalation summary for a human agent:

{history_context}

Customer Issue: "{customer_query}"

Escalation Reason: {escalation_reason or "Complex issue requiring human judgment"}

Provide:
1. Brief summary (2-3 sentences)
2. Key points the human agent should know
3. Recommended priority (LOW/MEDIUM/HIGH/CRITICAL)
4. Suggested next steps

Format as clear, actionable information for the human agent."""

        response_text = self.generate(prompt)

        # Customer-facing message
        customer_message = f"""I understand this situation requires special attention. I'm connecting you with one of our specialist team members who can provide the personalized assistance you need.

They'll review your case shortly and will have all the context from our conversation. Thank you for your patience."""

        self.log_interaction(customer_query, response_text)

        return {
            "agent": self.name,
            "escalation_summary": response_text,
            "customer_message": customer_message,
            "priority": "MEDIUM",  # Would be extracted from response in production
            "requires_human": True
        }
