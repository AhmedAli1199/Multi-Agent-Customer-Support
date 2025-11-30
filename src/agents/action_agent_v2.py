"""
Action Agent V2: Uses proper LangChain tool calling with Gemini function calling.
This version actually calls tools instead of just planning actions.
"""
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import GEMINI_API_KEY, ModelConfig
from tools.action_tools import ACTION_TOOLS

ACTION_SYSTEM_PROMPT = """You are an Action Agent for TechGear Electronics customer support. Your role is to:

1. Execute backend operations safely and accurately
2. Help customers with order management, refunds, and account updates
3. Provide clear confirmation of actions taken
4. Handle errors gracefully and explain issues to customers

**Available Actions** (via tools):
- Check order status
- Cancel orders (if not yet delivered)
- Modify orders (change address, upgrade shipping)
- Process refunds
- Check refund status
- Update customer addresses
- Reset passwords
- Get account information

**TechGear Policies to Remember**:
- Orders can be cancelled if not yet shipped
- Refunds take 5-7 business days to process
- Cannot modify orders already shipped
- 30-day return policy on all items
- Free shipping on orders $50+

**Safety Rules**:
- Always confirm order ID and customer details before taking action
- Explain what you're doing before and after using tools
- If action fails, explain why and suggest alternatives
- Be empathetic and customer-focused

**Communication Style**:
- Be friendly and professional
- Use clear, simple language
- Confirm successful actions
- Apologize for any issues and offer solutions

When a customer asks you to perform an action, use the appropriate tool and then confirm the results in a clear, customer-friendly way."""


class ActionAgentV2:
    """Action agent with proper tool calling capabilities"""

    def __init__(self):
        """Initialize Action Agent with tools"""
        self.name = "Action Agent V2"

        # Initialize Gemini with function calling
        self.llm = ChatGoogleGenerativeAI(
            model=ModelConfig.GEMINI_PRO,
            google_api_key=GEMINI_API_KEY,
            temperature=ModelConfig.TEMPERATURE,
            convert_system_message_to_human=True  # Required for Gemini
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", ACTION_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create agent with tools
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=ACTION_TOOLS,
            prompt=self.prompt
        )

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=ACTION_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

        print(f"[OK] {self.name} initialized with {len(ACTION_TOOLS)} tools")
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
        # Format conversation history
        chat_history = []
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    chat_history.append(("human", content))
                else:
                    chat_history.append(("ai", content))

        try:
            # Execute agent
            result = self.agent_executor.invoke({
                "input": customer_query,
                "chat_history": chat_history
            })

            response_text = result.get("output", "I apologize, but I was unable to process your request.")

            # Extract tool calls from intermediate steps
            tool_calls = []
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if len(step) >= 2:
                        action, observation = step[0], step[1]
                        tool_calls.append({
                            "tool": action.tool,
                            "input": action.tool_input,
                            "output": observation
                        })

            return {
                "agent": self.name,
                "response": response_text,
                "tool_calls": tool_calls,
                "success": True
            }

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error while processing your request: {str(e)}\n\nPlease try again or contact our support team at 1-800-TECHGEAR for immediate assistance."

            return {
                "agent": self.name,
                "response": error_msg,
                "tool_calls": [],
                "success": False,
                "error": str(e)
            }
