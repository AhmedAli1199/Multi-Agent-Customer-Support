"""
Baseline Single-Agent System for comparison with multi-agent approach.
This agent handles all tasks (triage, knowledge, actions, follow-up) in a single LLM call.
"""
from typing import Dict, List
import google.generativeai as genai
from config import GEMINI_API_KEY, ModelConfig
from tools.knowledge_retrieval import knowledge_retriever
from tools.mock_apis import order_api, refund_api, account_api
import json
import re

genai.configure(api_key=GEMINI_API_KEY)

SINGLE_AGENT_PROMPT = """You are a customer support agent. Handle all customer queries comprehensively:

1. Understand the customer's intent and needs
2. Search the knowledge base if needed
3. Take appropriate actions (order management, refunds, etc.)
4. Provide helpful, empathetic responses
5. Follow up to ensure satisfaction

Available Actions:
- check_order_status(order_id)
- cancel_order(order_id, reason)
- modify_order(order_id, changes)
- initiate_refund(order_id, amount, reason)
- update_address(customer_id, new_address)
- reset_password(customer_id)

Knowledge Base: You have access to a knowledge base via context provided.

Respond in JSON format:
{
  "intent": "primary intent detected",
  "actions_taken": ["list of actions executed"],
  "response": "your response to the customer",
  "needs_escalation": true/false,
  "escalation_reason": "reason if escalation needed",
  "confidence": 0.0-1.0
}

Be professional, empathetic, and thorough. Always confirm before executing irreversible actions.
"""

class SingleAgent:
    """Baseline single-agent for comparison with multi-agent system"""

    def __init__(self):
        self.model = genai.GenerativeModel(
            model_name=ModelConfig.GEMINI_PRO,
            generation_config={
                "temperature": ModelConfig.TEMPERATURE,
                "max_output_tokens": ModelConfig.MAX_OUTPUT_TOKENS,
                "top_p": ModelConfig.TOP_P,
                "top_k": ModelConfig.TOP_K,
            }
        )
        print(f"[OK] Single Agent initialized with model: {ModelConfig.GEMINI_PRO}")

    def process(self, customer_query: str, conversation_history: List[Dict] = None, auto_execute: bool = False) -> Dict:
        """
        Process customer query in a single LLM call.

        Args:
            customer_query: Customer's query
            conversation_history: Previous conversation messages
            auto_execute: Whether to automatically execute actions

        Returns:
            Dict with processing results
        """
        # Retrieve knowledge base context
        kb_context = knowledge_retriever.get_formatted_context(customer_query)

        # Format conversation history
        history_text = ""
        if conversation_history:
            history_text = "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                for msg in conversation_history[-5:]  # Last 5 messages
            ])

        # Build comprehensive prompt
        prompt = f"""{SINGLE_AGENT_PROMPT}

{f"Conversation History:\\n{history_text}\\n" if history_text else ""}
Knowledge Base Context:
{kb_context}

Customer Query: "{customer_query}"

Analyze the query, use the knowledge base context, determine actions, and provide a complete response in JSON format.
"""

        # Generate response
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text

            # Parse JSON response
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text

            result = json.loads(json_str)

            # Execute actions if auto_execute is enabled
            if auto_execute and result.get("actions_taken"):
                execution_results = self._execute_actions(result.get("actions_taken", []))
                result["execution_results"] = execution_results

            return {
                "agent": "Single Agent",
                "intent": result.get("intent"),
                "response": result.get("response"),
                "actions_taken": result.get("actions_taken", []),
                "needs_escalation": result.get("needs_escalation", False),
                "escalation_reason": result.get("escalation_reason"),
                "confidence": result.get("confidence", 0.5),
                "execution_results": result.get("execution_results", {})
            }

        except Exception as e:
            print(f"[ERROR] Single agent processing failed: {e}")
            return {
                "agent": "Single Agent",
                "intent": "UNKNOWN",
                "response": "I apologize, but I'm having trouble processing your request. Let me connect you with a human agent.",
                "needs_escalation": True,
                "escalation_reason": f"Processing error: {str(e)}",
                "confidence": 0.0
            }

    def _execute_actions(self, actions: List[str]) -> Dict:
        """Execute requested actions"""
        results = {}

        for action_str in actions:
            try:
                # Parse action string (e.g., "cancel_order(12345, Customer request)")
                action_match = re.match(r'(\w+)\((.*?)\)', action_str)
                if not action_match:
                    results[action_str] = {"success": False, "error": "Invalid action format"}
                    continue

                action_name = action_match.group(1)
                params_str = action_match.group(2)

                # Execute based on action name
                if action_name == "check_order_status":
                    order_id = params_str.strip().strip("'\"")
                    results[action_str] = order_api.check_order_status(order_id)

                elif action_name == "cancel_order":
                    params = [p.strip().strip("'\"") for p in params_str.split(",")]
                    order_id = params[0] if len(params) > 0 else ""
                    reason = params[1] if len(params) > 1 else "Customer request"
                    results[action_str] = order_api.cancel_order(order_id, reason)

                elif action_name == "modify_order":
                    params = [p.strip().strip("'\"") for p in params_str.split(",")]
                    order_id = params[0] if len(params) > 0 else ""
                    changes = {"note": params[1]} if len(params) > 1 else {}
                    results[action_str] = order_api.modify_order(order_id, changes)

                elif action_name == "initiate_refund":
                    params = [p.strip().strip("'\"") for p in params_str.split(",")]
                    order_id = params[0] if len(params) > 0 else ""
                    amount = float(params[1]) if len(params) > 1 else 0.0
                    reason = params[2] if len(params) > 2 else "Customer request"
                    results[action_str] = refund_api.initiate_refund(order_id, amount, reason)

                elif action_name == "update_address":
                    params = [p.strip().strip("'\"") for p in params_str.split(",")]
                    customer_id = params[0] if len(params) > 0 else ""
                    new_address = params[1] if len(params) > 1 else ""
                    results[action_str] = account_api.update_address(customer_id, new_address)

                elif action_name == "reset_password":
                    customer_id = params_str.strip().strip("'\"")
                    results[action_str] = account_api.reset_password(customer_id)

                else:
                    results[action_str] = {"success": False, "error": f"Unknown action: {action_name}"}

            except Exception as e:
                results[action_str] = {"success": False, "error": str(e)}

        return results

# Global instance
single_agent = SingleAgent()
