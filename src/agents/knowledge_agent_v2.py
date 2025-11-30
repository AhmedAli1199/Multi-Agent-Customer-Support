"""
Knowledge Agent V2: Enhanced with product search tools and company context.
Uses RAG + product tools for comprehensive information retrieval.
"""
from typing import Dict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import GEMINI_API_KEY, ModelConfig
from tools.knowledge_retrieval import knowledge_retriever
from tools.product_tools import PRODUCT_TOOLS

KNOWLEDGE_SYSTEM_PROMPT = """You are a Knowledge Agent for TechGear Electronics customer support. Your role is to:

1. Answer customer questions about products, policies, and services
2. Help customers find the right products for their needs
3. Provide accurate information from our knowledge base and product catalog
4. Offer excellent customer service with a friendly, professional tone

**About TechGear Electronics**:
- Leading online retailer of premium electronics since 2015
- Headquarters: San Francisco, CA
- Tagline: "Your Trusted Technology Partner"
- We sell laptops, smartphones, audio equipment, smart home devices, gaming gear, wearables, and accessories
- All products include 2-year warranty and 30-day return policy
- Free shipping on orders $50+

**Your Tools**:
- **search_products**: Find products by keyword or category
- **get_product_details**: Get full specifications for a specific product
- **check_product_availability**: Check if product is in stock
- **get_product_categories**: Show all product categories
- **compare_products**: Compare two products side-by-side
- **get_company_info**: Get company policies, contact info, shipping, returns, etc.

**Key Policies**:
- **Returns**: 30-day return window, no restocking fee
- **Shipping**: Free over $50, Standard ($5.99, 5-7 days), Express ($14.99, 2-3 days), Overnight ($29.99, next day)
- **Warranty**: 2-year standard on all electronics
- **Support**: 24/7 chat, phone Mon-Fri 9AM-9PM EST
- **Price Match**: Yes, within 14 days with proof

**How to Help Customers**:
1. **Product Questions**: Use search_products or get_product_details
2. **Policy Questions**: Use get_company_info with appropriate type
3. **Availability**: Use check_product_availability
4. **General Questions**: Answer using your knowledge of TechGear

**Communication Style**:
- Be warm, friendly, and helpful
- Use clear, jargon-free language
- Provide specific product recommendations when asked
- Always include relevant details (price, availability, shipping)
- Offer to help with next steps

**Important**:
- If you don't have information, admit it honestly
- Offer to escalate complex questions to human support
- Always prioritize customer satisfaction
- Use tools to get accurate, up-to-date information

Remember: You represent TechGear Electronics, so be professional and knowledgeable!"""


class KnowledgeAgentV2:
    """Enhanced Knowledge Agent with product search and company info tools"""

    def __init__(self):
        """Initialize Knowledge Agent with RAG and product tools"""
        self.name = "Knowledge Agent V2"
        self.retriever = knowledge_retriever

        # Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(
            model=ModelConfig.GEMINI_FLASH,  # Flash for cost efficiency
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,  # Lower temperature for factual responses
            convert_system_message_to_human=True
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

        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=PRODUCT_TOOLS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=4
        )

        print(f"[OK] {self.name} initialized with {len(PRODUCT_TOOLS)} tools")
        print(f"  Tools: {', '.join(tool.name for tool in PRODUCT_TOOLS)}")

    def process(self, customer_query: str, conversation_history: List[Dict] = None) -> Dict:
        """
        Process customer query using RAG and product tools.

        Args:
            customer_query: Customer question
            conversation_history: Previous conversation

        Returns:
            Dict with agent response and metadata
        """
        # Retrieve relevant context from knowledge base
        kb_context = self.retriever.get_formatted_context(customer_query, top_k=3)

        # Build enhanced query with context
        enhanced_query = f"""Customer Question: {customer_query}

Relevant Knowledge Base Information:
{kb_context}

Please answer the customer's question using the available tools and knowledge base information. Be helpful and specific."""

        # Format conversation history
        chat_history = []
        if conversation_history:
            for msg in conversation_history[-5:]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    chat_history.append(("human", content))
                else:
                    chat_history.append(("ai", content))

        try:
            # Execute agent
            result = self.agent_executor.invoke({
                "input": enhanced_query,
                "chat_history": chat_history
            })

            response_text = result.get("output", "I apologize, but I was unable to find information about that.")

            # Extract tool calls
            tool_calls = []
            if "intermediate_steps" in result:
                for step in result["intermediate_steps"]:
                    if len(step) >= 2:
                        action, observation = step[0], step[1]
                        tool_calls.append({
                            "tool": action.tool,
                            "input": action.tool_input,
                            "output": observation[:200] + "..." if len(observation) > 200 else observation
                        })

            # Check if escalation is needed
            insufficient_indicators = [
                "don't have information",
                "don't know",
                "unable to find",
                "contact support",
                "speak with"
            ]
            needs_escalation = any(indicator in response_text.lower() for indicator in insufficient_indicators)

            return {
                "agent": self.name,
                "response": response_text,
                "tool_calls": tool_calls,
                "kb_context_used": kb_context[:200] + "..." if len(kb_context) > 200 else kb_context,
                "needs_escalation": needs_escalation,
                "confidence": "low" if needs_escalation else "high"
            }

        except Exception as e:
            error_msg = f"I apologize, but I encountered an issue retrieving that information. Please try rephrasing your question or contact our support team at 1-800-TECHGEAR.\n\nError: {str(e)}"

            return {
                "agent": self.name,
                "response": error_msg,
                "tool_calls": [],
                "needs_escalation": True,
                "confidence": "low",
                "error": str(e)
            }
