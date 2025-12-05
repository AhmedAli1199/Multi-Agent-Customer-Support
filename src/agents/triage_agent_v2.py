"""
Triage Agent V2: Improved routing logic to reduce over-escalation.
More specific criteria for when to escalate vs. when to use Action/Knowledge agents.
"""
import json
import re
from typing import Dict, List
from agents.base_agent import BaseAgent
from config import ModelConfig
from utils.helpers import extract_order_id, calculate_sentiment_score

# Optimized for smaller LLMs - concise with few-shot examples
TRIAGE_SYSTEM_PROMPT_V2 = """Route customer queries to the correct agent. Reply ONLY with JSON.

AGENTS:
- knowledge: Product info, policies, recommendations, FAQs
- action: Cancel/modify orders, refunds, check status, account changes, ANY order-related task
- escalation: ONLY if customer EXPLICITLY demands human OR security issue OR legal threat

IMPORTANT: Consider conversation history! If user provides an order ID or number after being asked for one, route to "action".

EXAMPLES:
User: "Cancel order 12345" → {"route_to":"action","intent":"CANCEL_ORDER","order_id":"12345"}
User: "What's your return policy?" → {"route_to":"knowledge","intent":"POLICY_QUESTION","order_id":null}
User: "I want to speak to a manager" → {"route_to":"escalation","intent":"HUMAN_REQUEST","order_id":null}
User: "Check my order status" → {"route_to":"action","intent":"ORDER_STATUS","order_id":null}
User: "Show me laptops" → {"route_to":"knowledge","intent":"PRODUCT_SEARCH","order_id":null}
User: "I want a refund" → {"route_to":"action","intent":"REFUND_REQUEST","order_id":null}
User: "12345" (after being asked for order ID) → {"route_to":"action","intent":"PROVIDE_ORDER_ID","order_id":"12345"}
User: "its 99999" → {"route_to":"action","intent":"PROVIDE_ORDER_ID","order_id":"99999"}

DEFAULT: If user provides a number/ID, assume it's an order ID and route to "action".
Only use "escalation" for EXPLICIT human requests or security issues. NEVER escalate just because something wasn't found.

Reply with JSON only: {"route_to":"...", "intent":"...", "order_id":"..." or null}"""


class TriageAgentV2(BaseAgent):
    """Improved Triage Agent with better routing logic"""

    def __init__(self):
        super().__init__(
            name="Triage Agent V2",
            model_name=ModelConfig.PRIMARY_MODEL,  # Uses PRIMARY_MODEL from config
            system_prompt=TRIAGE_SYSTEM_PROMPT_V2
        )

    def process(self, customer_query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Process query and determine routing with reduced over-escalation.

        Returns:
            Dict with intent, entities, urgency, routing decision
        """
        # Check if this looks like an order ID being provided (continuation of previous request)
        query_stripped = customer_query.strip().lower()
        
        # Detect if user is providing an order ID (number, or "its <number>", "order <number>", etc.)
        order_id_patterns = [
            r'^(?:its?|it\'s|order|order\s*(?:id|number)?[:\s]*)?[\s]*([a-z0-9\-]{3,})$',
            r'^#?(\d{4,})$',
            r'^([a-z0-9\-]{5,})$'
        ]
        
        is_order_id_response = False
        extracted_id = None
        for pattern in order_id_patterns:
            match = re.match(pattern, query_stripped, re.IGNORECASE)
            if match:
                extracted_id = match.group(1) if match.groups() else query_stripped
                # Check if previous conversation was asking for order ID
                if conversation_history and len(conversation_history) > 0:
                    last_msg = conversation_history[-1].get("content", "").lower()
                    if any(phrase in last_msg for phrase in ["order id", "order number", "provide", "what is your order"]):
                        is_order_id_response = True
                        break
                # Also route to action if it looks like just an order ID even without history
                if re.match(r'^[\d\-]{5,}$', extracted_id):
                    is_order_id_response = True
                    break
        
        # Fast path: if user is providing an order ID, route directly to action
        if is_order_id_response and extracted_id:
            analysis = {
                "route_to": "action",
                "intent": "PROVIDE_ORDER_ID",
                "order_id": extracted_id,
                "sentiment": "NEUTRAL",
                "reasoning": "User provided order ID in response to previous request"
            }
            self.log_interaction(customer_query, json.dumps(analysis, indent=2))
            return {
                "agent": self.name,
                "analysis": analysis,
                "next_agent": "action"
            }

        # Build context from conversation history for better routing
        history_context = ""
        if conversation_history and len(conversation_history) > 0:
            recent = conversation_history[-2:]  # Last 2 messages
            history_context = "\nRecent conversation:\n" + "\n".join([
                f"- {msg.get('role', 'user')}: {msg.get('content', '')[:100]}" 
                for msg in recent
            ])

        # Simple prompt for small LLMs
        prompt = f"""Customer: "{customer_query}"{history_context}

Route this query. Reply with JSON only."""

        # Generate response
        response_text = self.generate(prompt)

        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis = json.loads(json_str)
            else:
                analysis = self._improved_fallback_analysis(customer_query)
        except json.JSONDecodeError:
            # Fallback parsing with improved logic
            analysis = self._improved_fallback_analysis(customer_query)

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

        # Override escalation if routing logic is too aggressive
        route = analysis.get("route_to", "knowledge")
        if route == "escalation":
            # Check if escalation is truly necessary
            query_lower = customer_query.lower()
            escalation_keywords = [
                "speak to manager",
                "talk to human",
                "transfer me",
                "real person",
                "lawyer",
                "legal action",
                "sue",
                "fraud",
                "hacked",
                "unauthorized",
                "security breach"
            ]

            needs_escalation = any(keyword in query_lower for keyword in escalation_keywords)

            if not needs_escalation:
                # Reroute based on query content
                action_keywords = ["cancel", "refund", "return", "modify", "change", "update", "reset"]
                has_order_id = bool(extract_order_id(customer_query))

                if any(keyword in query_lower for keyword in action_keywords) or has_order_id:
                    analysis["route_to"] = "action"
                    analysis["reasoning"] = "Overridden: Action request, does not require escalation"
                else:
                    analysis["route_to"] = "knowledge"
                    analysis["reasoning"] = "Overridden: Informational query, does not require escalation"

        self.log_interaction(customer_query, json.dumps(analysis, indent=2))

        return {
            "agent": self.name,
            "analysis": analysis,
            "next_agent": analysis.get("route_to", "knowledge")
        }

    def _improved_fallback_analysis(self, query: str) -> Dict:
        """Improved fallback analysis with better routing logic"""
        query_lower = query.lower()

        # First check if it looks like an order ID
        extracted_id = extract_order_id(query)
        if extracted_id or re.match(r'^[\s]*(its?|it\'s)?[\s]*[\d\-]{4,}[\s]*$', query_lower):
            return {
                "intent": "PROVIDE_ORDER_ID",
                "sub_intents": [],
                "entities": {"order_id": extracted_id or re.sub(r'[^\d\-]', '', query)},
                "urgency": "MEDIUM",
                "sentiment": "NEUTRAL",
                "route_to": "action",
                "reasoning": "User provided order ID",
                "confidence": "high"
            }

        # Check for explicit escalation requests
        escalation_keywords = [
            "speak to manager", "talk to human", "transfer me", "real person",
            "lawyer", "legal", "sue", "fraud", "hacked", "security"
        ]
        if any(keyword in query_lower for keyword in escalation_keywords):
            return {
                "intent": "ESCALATION_NEEDED",
                "sub_intents": [],
                "entities": {"order_id": extract_order_id(query)},
                "urgency": "HIGH",
                "sentiment": "NEGATIVE",
                "route_to": "escalation",
                "reasoning": "Explicit escalation request or severe issue",
                "confidence": "high"
            }

        # Check for action requests
        action_keywords = ["cancel", "refund", "return", "modify", "change", "update", "reset"]
        if any(keyword in query_lower for keyword in action_keywords):
            return {
                "intent": "ACTION_REQUEST",
                "sub_intents": [],
                "entities": {"order_id": extract_order_id(query)},
                "urgency": "MEDIUM",
                "sentiment": "NEUTRAL",
                "route_to": "action",
                "reasoning": "Action request requiring system operation",
                "confidence": "high"
            }

        # Default to knowledge for informational queries
        return {
            "intent": "INFO_QUERY",
            "sub_intents": [],
            "entities": {"order_id": extract_order_id(query)},
            "urgency": "LOW",
            "sentiment": "NEUTRAL" if calculate_sentiment_score(query) > -0.3 else "NEGATIVE",
            "route_to": "knowledge",
            "reasoning": "Informational query for Knowledge Agent",
            "confidence": "medium"
        }
