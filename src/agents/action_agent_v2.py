"""
Action Agent V2: Uses proper LangChain tool calling with function calling.
This version actually calls tools instead of just planning actions.
"""
from typing import Dict, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from config import ModelConfig
from utils.llm_client import get_chat_model
from tools.action_tools import ACTION_TOOLS

ACTION_SYSTEM_PROMPT = """You are a customer support agent that helps with order actions.

TOOLS:
- check_order_status: Check order status (needs order_id)
- cancel_order: Cancel an order (needs order_id)
- initiate_refund: Process refund (needs order_id, amount, reason)
- modify_order: Modify order (needs order_id, new_address or shipping_upgrade)

CRITICAL RULES:
1. If no order_id provided, ask for it: "I'll need your order ID to help. Could you provide it?"
2. Call ONE tool, get result, then respond to customer
3. DO NOT call the same tool twice
4. If "ORDER_NOT_FOUND", tell customer to check their confirmation email
5. Be brief - 1-2 sentences

Example:
- Customer: "Cancel my order 12345"
- You: Call cancel_order with order_id="12345"
- Tool returns success
- You: "Your order #12345 has been cancelled. Refund in 5-7 days."
"""

class ActionAgentV2:
    """Action agent with proper tool calling capabilities"""

    def __init__(self):
        """Initialize Action Agent with tools"""
        self.name = "Action Agent V2"

        # Initialize LLM via unified client (uses PRIMARY_MODEL for complex tasks)
        self.llm = get_chat_model(
            model_name=ModelConfig.PRIMARY_MODEL,
            temperature=ModelConfig.TEMPERATURE
        )
        
        # Bind tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(ACTION_TOOLS)

        print(f"[OK] {self.name} initialized with {len(ACTION_TOOLS)} tools (provider: {ModelConfig.LLM_PROVIDER})")
        print(f"  Model: {ModelConfig.PRIMARY_MODEL}")
        print(f"  Tools: {', '.join(tool.name for tool in ACTION_TOOLS)}")

    def process(self, customer_query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Process customer request and execute actions using tools.

        Args:
            customer_query: Customer request
            conversation_history: Previous conversation messages

        Returns:
            Dict with agent response and metadata
        """
        # Build messages - start fresh each time to avoid tool_call issues
        messages = [SystemMessage(content=ACTION_SYSTEM_PROMPT)]
        
        # Add simplified conversation history (just the content, no tool calls)
        if conversation_history:
            for msg in conversation_history[-3:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if content:  # Only add if there's content
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    else:
                        messages.append(AIMessage(content=content))
        
        # Add current query
        messages.append(HumanMessage(content=customer_query))

        try:
            # Get response with potential tool calls
            response = self.llm_with_tools.invoke(messages)
            
            tool_calls = []
            response_text = response.content if response.content else ""
            
            # Check if LLM wants to call tools
            if response.tool_calls:
                # Execute all tool calls and collect results
                tool_messages = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_call_id = tool_call["id"]
                    
                    # Find and execute the tool
                    tool_output = "Tool not found"
                    for tool in ACTION_TOOLS:
                        if tool.name == tool_name:
                            try:
                                tool_output = tool.invoke(tool_args)
                            except Exception as te:
                                tool_output = f"Error executing tool: {str(te)}"
                            break
                    
                    tool_calls.append({
                        "tool": tool_name,
                        "input": tool_args,
                        "output": tool_output
                    })
                    
                    # Create proper ToolMessage for OpenAI
                    tool_messages.append(ToolMessage(
                        content=str(tool_output),
                        tool_call_id=tool_call_id
                    ))
                
                # If we got tool outputs, generate a final response
                if tool_calls:
                    # Add the assistant response with tool calls
                    messages.append(response)
                    # Add all tool results as ToolMessages
                    messages.extend(tool_messages)
                    # Get final response from LLM
                    final_response = self.llm.invoke(messages)
                    response_text = final_response.content
            
            # FALLBACK: If no response, provide helpful message
            if not response_text:
                if tool_calls and tool_calls[-1]["output"]:
                    response_text = str(tool_calls[-1]["output"])
                else:
                    response_text = "I'll need your order ID to help with that. Could you provide it?"

            return {
                "agent": self.name,
                "response": response_text,
                "tool_calls": tool_calls,
                "success": True
            }

        except Exception as e:
            print(f"[ACTION AGENT ERROR] {str(e)}")
            error_msg = "I had trouble processing that. Could you please try again with your order ID?"

            return {
                "agent": self.name,
                "response": error_msg,
                "tool_calls": [],
                "success": True,  # Don't trigger escalation on errors
                "error": str(e)
            }
