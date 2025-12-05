"""
Triage Agent: Intent classification, entity extraction, urgency scoring, and routing.
"""
import json
import re
from typing import Dict, List
from agents.base_agent import BaseAgent
from config import ModelConfig
from utils.helpers import extract_order_id, calculate_sentiment_score

TRIAGE_SYSTEM_PROMPT = """You are a Triage Agent for customer support. Your role is to:
1. Classify the customer's intent from their query
2. Extract important entities (order IDs, product names, dates, amounts)
3. Assess urgency level (LOW, MEDIUM, HIGH, CRITICAL)
4. Determine which specialist agent should handle this query

Intent Categories:
- INFO_QUERY: General questions about products, policies, shipping
- ACTION_REQUEST: Requests requiring backend actions (cancel, refund, modify order)
- COMPLAINT: Customer complaints or issues
- ESCALATION_NEEDED: Complex issues requiring human intervention

Routing Rules:
- INFO_QUERY → Knowledge Agent
- ACTION_REQUEST → Action Agent
- COMPLAINT (if solvable) → Action Agent
- COMPLAINT (if complex) → Escalation Agent
- ESCALATION_NEEDED → Escalation Agent

Respond in JSON format:
{
  "intent": "primary intent",
  "sub_intents": ["any additional intents"],
  "entities": {
    "order_id": "extracted order ID or null",
    "product": "product name or null",
    "amount": "monetary amount or null"
  },
  "urgency": "LOW|MEDIUM|HIGH|CRITICAL",
  "sentiment": "POSITIVE|NEUTRAL|NEGATIVE",
  "route_to": "knowledge|action|escalation",
  "reasoning": "brief explanation of routing decision"
}"""

class TriageAgent(BaseAgent):
    """Triage agent for intent classification and routing"""

    def __init__(self):
        super().__init__(
            name="Triage Agent",
            model_name=ModelConfig.PRIMARY_MODEL,  # Uses PRIMARY_MODEL from config
            system_prompt=TRIAGE_SYSTEM_PROMPT
        )

    def process(self, customer_query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Process query and determine routing.

        Returns:
            Dict with intent, entities, urgency, routing decision
        """
        # Build prompt
        history_context = self.format_conversation_history(conversation_history) if conversation_history else ""

        prompt = f"""Analyze this customer query and provide routing decision:

{history_context}

Customer Query: "{customer_query}"

Provide your analysis in JSON format."""

        # Generate response
        response_text = self.generate(prompt)

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text

            analysis = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback parsing
            analysis = self._fallback_analysis(customer_query)

        # Enhance with additional processing
        if not analysis.get("entities", {}).get("order_id"):
            extracted_id = extract_order_id(customer_query)
            if extracted_id:
                analysis["entities"]["order_id"] = extracted_id

        # Calculate sentiment if not provided
        if "sentiment" not in analysis:
            sentiment_score = calculate_sentiment_score(customer_query)
            if sentiment_score < -0.3:
                analysis["sentiment"] = "NEGATIVE"
            elif sentiment_score > 0.3:
                analysis["sentiment"] = "POSITIVE"
            else:
                analysis["sentiment"] = "NEUTRAL"

        self.log_interaction(customer_query, json.dumps(analysis, indent=2))

        return {
            "agent": self.name,
            "analysis": analysis,
            "next_agent": analysis.get("route_to", "knowledge")
        }

    def _fallback_analysis(self, query: str) -> Dict:
        """Fallback analysis if JSON parsing fails"""
        query_lower = query.lower()

        # Simple keyword-based classification
        if any(word in query_lower for word in ["cancel", "refund", "return", "change"]):
            intent = "ACTION_REQUEST"
            route_to = "action"
        elif any(word in query_lower for word in ["angry", "terrible", "worst", "complaint"]):
            intent = "COMPLAINT"
            route_to = "escalation"
        else:
            intent = "INFO_QUERY"
            route_to = "knowledge"

        return {
            "intent": intent,
            "sub_intents": [],
            "entities": {"order_id": extract_order_id(query)},
            "urgency": "MEDIUM",
            "sentiment": "NEUTRAL" if calculate_sentiment_score(query) > -0.3 else "NEGATIVE",
            "route_to": route_to,
            "reasoning": "Fallback classification based on keywords"
        }
