"""
Knowledge Agent V2: Enhanced with product search tools and company context.
Uses RAG + product tools for comprehensive information retrieval.
"""
from typing import Dict, List
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import ModelConfig
from utils.llm_client import get_chat_model
from tools.knowledge_retrieval import knowledge_retriever
from tools.product_tools import PRODUCT_TOOLS

KNOWLEDGE_SYSTEM_PROMPT = """You are a helpful and knowledgeable customer support agent for TechGear Electronics.

YOUR MISSION: ALWAYS help the customer. You have powerful tools - USE THEM!

AVAILABLE TOOLS:
- **search_faqs**: Search comprehensive FAQ database (payment, complaints, sign-up, contact, refunds, etc.)
- **get_company_info**: Get policies/info (types: "returns", "shipping", "warranty", "contact", "payment", "general")  
- **search_products**: Find products by keyword
- **get_product_details**: Get specifications for a specific product
- **check_product_availability**: Check if product is in stock

WHICH TOOL TO USE:

1. **For Customer Service Questions** (toll-free number, payment issues, complaints, sign-up errors, how to purchase, feedback):
   → Use **search_faqs** - it has comprehensive answers!
   
2. **For Policy Questions** (general shipping/returns/warranty):
   → Use **get_company_info** with type parameter

3. **For Product Questions** (find laptops, check availability):
   → Use **search_products** or **get_product_details**

CRITICAL RULES:
- ALWAYS call a tool before answering - don't guess!
- For questions about contact info, complaints, payments, sign-ups → USE search_faqs FIRST
- If search_faqs doesn't help, try get_company_info(type="contact")
- NEVER say "I don't have information" - provide what you DO find
- Keep responses helpful, brief (2-4 sentences), and actionable

Examples:
- "toll-free number?" → search_faqs("toll-free number") → "You can reach us at 1-800-TECHGEAR..."
- "payment issue?" → search_faqs("payment issue") → "For billing issues, contact billing@techgear.com..."
- "how to complain?" → search_faqs("complaint") → "File a complaint via phone 1-800-TECHGEAR or email complaints@techgear.com..."
- "sign-up error?" → search_faqs("sign up error") → "For sign-up help, contact support@techgear.com or call 1-800-TECHGEAR..."
"""

class KnowledgeAgentV2:
    """Enhanced Knowledge Agent with product search and company info tools"""

    def __init__(self):
        """Initialize Knowledge Agent with RAG and product tools"""
        self.name = "Knowledge Agent V2"
        self.retriever = knowledge_retriever

        # Initialize LLM via unified client (uses SECONDARY_MODEL for cost efficiency)
        self.llm = get_chat_model(
            model_name=ModelConfig.SECONDARY_MODEL,
            temperature=0.3  # Lower temperature for factual responses
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", KNOWLEDGE_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create agent with product tools
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=PRODUCT_TOOLS,
            prompt=self.prompt
        )

        # Create agent executor - allow 3 iterations for: tool call -> observation -> final answer
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=PRODUCT_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,  # Need at least 3: think -> tool -> answer
            early_stopping_method="generate",
            return_intermediate_steps=True
        )

        print(f"[OK] {self.name} initialized with {len(PRODUCT_TOOLS)} tools (provider: {ModelConfig.LLM_PROVIDER})")
        print(f"  Model: {ModelConfig.SECONDARY_MODEL}")
        print(f"  Tools: {', '.join(tool.name for tool in PRODUCT_TOOLS)}")
        print(f"  Key Tool: search_faqs (comprehensive FAQ database access)")

    def process(self, customer_query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Process customer query using RAG and product tools.

        Args:
            customer_query: Customer question
            conversation_history: Previous conversation

        Returns:
            Dict with agent response and metadata
        """
        # Format conversation history - only last 3 messages to reduce latency
        chat_history = []
        if conversation_history:
            for msg in conversation_history[-3:]:  # Reduced from 5 to 3
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    chat_history.append(("human", content))
                else:
                    chat_history.append(("ai", content))

        try:
            # Execute agent with just the customer query (no confusing context injection)
            result = self.agent_executor.invoke({
                "input": customer_query,
                "chat_history": chat_history
            })

            response_text = result.get("output", "")

            # Extract tool calls
            tool_calls = []
            last_tool_output = ""
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if len(step) >= 2:
                        action, observation = step[0], step[1]
                        tool_calls.append({
                            "tool": action.tool,
                            "input": action.tool_input,
                            "output": observation[:200] + "..." if len(observation) > 200 else observation
                        })
                        last_tool_output = observation

            # FALLBACK: If no response but we got tool output, use that
            if not response_text or response_text == "Agent stopped due to iteration limit or time limit.":
                if last_tool_output:
                    response_text = last_tool_output
                else:
                    response_text = "I apologize, but I was unable to find that information. Please try asking in a different way."

            # Check if escalation is needed - ONLY for explicit requests
            escalation_indicators = [
                "speak with a human",
                "talk to a manager", 
                "transfer to agent",
                "i cannot help with this"
            ]
            needs_escalation = any(indicator in response_text.lower() for indicator in escalation_indicators)

            return {
                "agent": self.name,
                "response": response_text,
                "tool_calls": tool_calls,
                "needs_escalation": needs_escalation,
                "confidence": "low" if needs_escalation else "high"
            }

        except Exception as e:
            # Log error but DON'T escalate - try to help the customer
            print(f"[KNOWLEDGE AGENT ERROR] {str(e)}")
            error_msg = "I apologize, I had trouble processing that request. Could you please rephrase your question?"

            return {
                "agent": self.name,
                "response": error_msg,
                "tool_calls": [],
                "needs_escalation": False,  # Don't escalate on errors - let customer try again
                "confidence": "medium",
                "error": str(e)
            }
