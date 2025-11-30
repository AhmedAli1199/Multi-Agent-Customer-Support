"""
Follow-Up Agent: Proactive engagement and satisfaction checks.
"""
from typing import Dict, List
from agents.base_agent import BaseAgent
from config import ModelConfig

FOLLOWUP_SYSTEM_PROMPT = """You are a Follow-Up Agent for customer support. Your role is to INTELLIGENTLY decide if a follow-up message is needed.

CRITICAL RULES - Follow-up is ONLY needed when:
1. Complex issue was just resolved (order cancellation, refund, account change)
2. Multi-step resolution that needs confirmation
3. Customer showed frustration/negative sentiment that needs smoothing
4. Action was taken that customer should acknowledge

DO NOT send follow-up for:
- Simple information queries (policies, product info, contact details)
- Questions that were just answered with information
- Ongoing conversations (customer will respond if they need more)
- Cases where the main response already ended conversationally

Response Format:
{
  "needs_followup": true/false,
  "reason": "brief explanation",
  "message": "follow-up message (only if needs_followup=true, otherwise empty)"
}

If needs_followup=true, keep message under 30 words and conversational."""

class FollowUpAgent(BaseAgent):
    """Follow-up agent for customer satisfaction"""

    def __init__(self):
        super().__init__(
            name="Follow-Up Agent",
            model_name=ModelConfig.GEMINI_FLASH,
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

        prompt = f"""Analyze this customer interaction and decide if a follow-up message is needed:

{history_context}

Customer Query: "{customer_query}"
Resolution: {resolution_summary or "Issue addressed"}
Agents Used: {agents_used}

Analyze:
1. Was this a simple info query or complex action?
2. Did the response already end conversationally?
3. Does customer likely need reassurance or confirmation?
4. Was there negative sentiment that needs smoothing?

Respond in JSON format as specified in your instructions."""

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
