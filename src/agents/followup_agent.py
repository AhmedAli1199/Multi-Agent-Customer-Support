"""
Follow-Up Agent: Proactive engagement and satisfaction checks.
"""
from typing import Dict, List
from agents.base_agent import BaseAgent
from config import ModelConfig

FOLLOWUP_SYSTEM_PROMPT = """Decide if a follow-up question is needed. Reply with JSON only.

Rules:
- needs_followup=true: Only for complex resolved actions (refunds, cancellations) or frustrated customers
- needs_followup=false: For simple info queries, product questions, policy answers

Reply format: {"needs_followup": true/false, "message": "..." or ""}

Examples:
- Simple policy question → {"needs_followup": false, "message": ""}
- Order cancelled → {"needs_followup": true, "message": "Is there anything else I can help with?"}
- Product info → {"needs_followup": false, "message": ""}"""

class FollowUpAgent(BaseAgent):
    """Follow-up agent for customer satisfaction"""

    def __init__(self):
        super().__init__(
            name="Follow-Up Agent",
            model_name=ModelConfig.SECONDARY_MODEL,  # Uses SECONDARY_MODEL for cost efficiency
            system_prompt=FOLLOWUP_SYSTEM_PROMPT
        )

    def process(self, customer_query: str, conversation_history: List[Dict] = None, resolution_summary: str = None, agent_sequence: List[str] = None) -> Dict:
        """
        Intelligently decide if follow-up is needed and generate message if appropriate.

        Args:
            customer_query: Original query
            conversation_history: Full conversation
            resolution_summary: Summary of how issue was resolved
            agent_sequence: Which agents handled the query

        Returns:
            Dict with follow-up decision and optional message
        """
        import json
        import re

        history_context = self.format_conversation_history(conversation_history) if conversation_history else ""
        agents_used = ', '.join(agent_sequence) if agent_sequence else "unknown"

        # Simple, concise prompt for small LLMs
        prompt = f"""Query: "{customer_query}"
Agents used: {agents_used}
Resolution: {resolution_summary or "Answered"}

Should we follow up? Reply JSON only."""

        response_text = self.generate(prompt)

        # Parse JSON response
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text

            decision = json.loads(json_str)
        except:
            # Fallback: only follow-up if action agent was used
            decision = {
                "needs_followup": "action" in agents_used.lower(),
                "reason": "Fallback decision based on agent sequence",
                "message": "Is there anything else I can help you with today?" if "action" in agents_used.lower() else ""
            }

        self.log_interaction(customer_query, json.dumps(decision, indent=2))

        return {
            "agent": self.name,
            "needs_followup": decision.get("needs_followup", False),
            "follow_up_message": decision.get("message", ""),
            "reason": decision.get("reason", ""),
            "message_type": "CONDITIONAL_FOLLOWUP"
        }
