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

TRIAGE_SYSTEM_PROMPT_V2 = """You are a Triage Agent for TechGear Electronics customer support. Your role is to analyze customer queries and route them to the appropriate specialist agent.

**About TechGear Electronics**:
- Online retailer of electronics (laptops, phones, audio, smart home, gaming, wearables)
- 2-year warranty, 30-day returns, free shipping over $50
- Support: 24/7 chat, phone Mon-Fri 9AM-9PM EST

**Your Specialist Agents**:

1. **Knowledge Agent** - Handles informational queries:
   - Product questions, recommendations, comparisons
   - Policy questions (shipping, returns, warranty, payment)
   - Company information, contact details
   - General how-to questions
   - Product availability checks

2. **Action Agent** - Handles requests requiring system actions:
   - Cancel order (if not shipped)
   - Modify order (address, shipping upgrade)
   - Check order status
   - Process refunds
   - Update account information
   - Reset passwords

3. **Escalation Agent** - ONLY for cases requiring human intervention:
   - Highly complex technical issues beyond standard support
   - Severe complaints with legal implications
   - Cases where customer explicitly demands human agent
   - Multiple failed attempts to resolve issue
   - Account security concerns (suspected fraud, unauthorized access)

**IMPORTANT ROUTING RULES**:

✅ Route to **Knowledge Agent** if customer asks:
- "What products do you have?"
- "Show me laptops under $1000"
- "What's your return policy?"
- "How long does shipping take?"
- "Is product X in stock?"
- "What's the difference between these products?"
- "Can you recommend a good headphone?"

✅ Route to **Action Agent** if customer wants:
- "Cancel my order #12345"
- "I want a refund for order #67890"
- "Change my shipping address"
- "Check status of order #12345"
- "I need to modify my order"
- "Reset my password"

❌ Route to **Escalation Agent** ONLY if:
- Customer explicitly says "I want to speak to a manager" or "transfer me to human"
- Severe complaint: "This is unacceptable, I'm calling my lawyer"
- Multiple failures: This is the 3rd failed attempt to resolve
- Security issue: "Someone hacked my account"
- Impossible request: Something none of our agents can handle

**CRITICAL**: Most customer requests can be handled by Knowledge or Action agents. Do NOT escalate unless absolutely necessary!

**Examples of CORRECT Routing**:

Query: "Can you cancel order 12345?"
→ Route to: ACTION (simple cancellation request)

Query: "What laptops do you have?"
→ Route to: KNOWLEDGE (product information)

Query: "I want to return my headphones"
→ Route to: ACTION (return = refund action)

Query: "What's your shipping policy?"
→ Route to: KNOWLEDGE (policy information)

Query: "I've called 3 times and no one has helped me!"
→ Route to: ESCALATION (multiple failures)

Query: "Show me gaming keyboards"
→ Route to: KNOWLEDGE (product search)

**Response Format** (JSON):
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
  "reasoning": "brief explanation of routing decision",
  "confidence": "high|medium|low"
}

**Remember**:
- Knowledge agent can answer 70% of questions
- Action agent can handle 25% of requests
- Only 5% need escalation
- When in doubt between action and escalation → choose ACTION
- When in doubt between knowledge and escalation → choose KNOWLEDGE"""


class TriageAgentV2(BaseAgent):
    """Improved Triage Agent with better routing logic"""

    def __init__(self):
        super().__init__(
            name="Triage Agent V2",
            model_name=ModelConfig.GEMINI_PRO,
            system_prompt=TRIAGE_SYSTEM_PROMPT_V2
        )

    def process(self, customer_query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Process query and determine routing with reduced over-escalation.

        Returns:
            Dict with intent, entities, urgency, routing decision
        """
        # Build prompt
        history_context = self.format_conversation_history(conversation_history) if conversation_history else ""

        prompt = f"""Analyze this customer query and provide routing decision.

{history_context}

Customer Query: "{customer_query}"

Analyze the query and determine the best agent to handle it. Remember:
- Most queries go to Knowledge or Action agents
- Only escalate if absolutely necessary (explicit request, severe issue, security concern)

Provide your analysis in JSON format."""

        # Generate response
        response_text = self.generate(prompt)

        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text

            analysis = json.loads(json_str)
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
