"""
Action Agent: Executes backend operations (refunds, order modifications, etc.)
"""
from typing import Dict, List
from agents.base_agent import BaseAgent
from config import ModelConfig
from tools.mock_apis import order_api, refund_api, account_api
import json

ACTION_SYSTEM_PROMPT = """You are an Action Agent for customer support. Your role is to:
1. Execute backend operations safely and accurately
2. Validate all actions before execution
3. Provide clear confirmation of actions taken
4. Handle errors gracefully

Available Actions:
- check_order_status(order_id)
- cancel_order(order_id, reason)
- modify_order(order_id, changes)
- initiate_refund(order_id, amount, reason)
- update_address(customer_id, new_address)
- reset_password(customer_id)

Safety Rules:
- Always confirm action details with customer first
- Validate order_id and customer_id
- Check eligibility before processing refunds
- Never execute irreversible actions without confirmation

Respond in JSON format:
{
  "action_needed": "name of action",
  "parameters": {required parameters},
  "confirmation_needed": true/false,
  "response_to_customer": "message to send"
}"""

class ActionAgent(BaseAgent):
    """Action agent for backend operations"""

    def __init__(self):
        super().__init__(
            name="Action Agent",
            model_name=ModelConfig.GEMINI_PRO,  # Use Pro for reliability
            system_prompt=ACTION_SYSTEM_PROMPT
        )

    def process(self, customer_query: str, conversation_history: List[Dict] = None, auto_execute: bool = False) -> Dict:
        """
        Process action request and execute if appropriate.

        Args:
            customer_query: Customer request
            conversation_history: Previous messages
            auto_execute: Whether to automatically execute actions (for demo)

        Returns:
            Dict with action results
        """
        history_context = self.format_conversation_history(conversation_history) if conversation_history else ""

        prompt = f"""Analyze this customer request and determine what action to take:

{history_context}

Customer Request: "{customer_query}"

Determine the appropriate action and provide response in JSON format."""

        # Generate action plan
        response_text = self.generate(prompt)

        # Parse action plan
        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text

            action_plan = json.loads(json_str)
        except:
            action_plan = {
                "action_needed": "none",
                "response_to_customer": response_text
            }

        # Execute action if auto_execute is True (for demo)
        if auto_execute and action_plan.get("action_needed") != "none":
            execution_result = self._execute_action(action_plan)
            action_plan["execution_result"] = execution_result

        self.log_interaction(customer_query, json.dumps(action_plan, indent=2))

        return {
            "agent": self.name,
            "action_plan": action_plan,
            "response": action_plan.get("response_to_customer", "Action processed")
        }

    def _execute_action(self, action_plan: Dict) -> Dict:
        """Execute the planned action"""
        action = action_plan.get("action_needed")
        params = action_plan.get("parameters", {})

        try:
            if action == "check_order_status":
                result = order_api.check_order_status(params.get("order_id"))
            elif action == "cancel_order":
                result = order_api.cancel_order(params.get("order_id"), params.get("reason", "Customer request"))
            elif action == "modify_order":
                result = order_api.modify_order(params.get("order_id"), params.get("changes", {}))
            elif action == "initiate_refund":
                result = refund_api.initiate_refund(
                    params.get("order_id"),
                    params.get("amount"),
                    params.get("reason", "Customer request")
                )
            elif action == "update_address":
                result = account_api.update_address(params.get("customer_id"), params.get("new_address"))
            elif action == "reset_password":
                result = account_api.reset_password(params.get("customer_id"))
            else:
                result = {"success": False, "error": f"Unknown action: {action}"}

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
