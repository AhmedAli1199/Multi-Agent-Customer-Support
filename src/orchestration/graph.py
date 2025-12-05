"""
LangGraph workflow for multi-agent orchestration.
Defines the agent workflow and routing logic.
"""
from typing import Dict
from langgraph.graph import StateGraph, END
from orchestration.state import AgentState

# Import agents
from agents.triage_agent import TriageAgent
from agents.knowledge_agent_simple import SimpleKnowledgeAgent  # Use simple, reliable agent
from agents.action_agent import ActionAgent
from agents.followup_agent import FollowUpAgent
from agents.escalation_agent import EscalationAgent

# Initialize agents
triage_agent = TriageAgent()
knowledge_agent = SimpleKnowledgeAgent()  # Switched to simple agent
action_agent = ActionAgent()
followup_agent = FollowUpAgent()
escalation_agent = EscalationAgent()

def triage_node(state: AgentState) -> AgentState:
    """Triage node: Classify intent and route"""
    result = triage_agent.process(
        state["customer_query"],
        state.get("conversation_history")
    )

    state["triage_result"] = result
    state["intent"] = result["analysis"].get("intent")
    state["entities"] = result["analysis"].get("entities")
    state["urgency"] = result["analysis"].get("urgency")
    state["sentiment"] = result["analysis"].get("sentiment")
    state["next_agent"] = result.get("next_agent")
    state["current_agent"] = "triage"
    state["agent_sequence"] = ["triage"]

    return state

def knowledge_node(state: AgentState) -> AgentState:
    """Knowledge node: Answer using RAG"""
    result = knowledge_agent.process(
        state["customer_query"],
        state.get("conversation_history")
    )

    state["knowledge_result"] = result
    state["final_response"] = result["response"]
    state["current_agent"] = "knowledge"
    state["agent_sequence"].append("knowledge")

    # Check if escalation needed
    if result.get("needs_escalation"):
        state["needs_escalation"] = True
        state["next_agent"] = "escalation"
    else:
        state["needs_escalation"] = False
        state["next_agent"] = "followup"
        state["resolution_status"] = "resolved"

    return state

def action_node(state: AgentState) -> AgentState:
    """Action node: Execute backend operations"""
    result = action_agent.process(
        state["customer_query"],
        state.get("conversation_history"),
        auto_execute=True  # Auto-execute for demo
    )

    state["action_result"] = result
    state["final_response"] = result["response"]
    state["current_agent"] = "action"
    state["agent_sequence"].append("action")

    # Determine next step
    if result["action_plan"].get("execution_result", {}).get("success"):
        state["resolution_status"] = "resolved"
        state["next_agent"] = "followup"
    else:
        state["resolution_status"] = "partial"
        state["needs_escalation"] = True
        state["next_agent"] = "escalation"

    return state

def followup_node(state: AgentState) -> AgentState:
    """Follow-up node: Customer satisfaction check"""
    # Create resolution summary
    resolution_summary = state.get("final_response", "Issue addressed")

    result = followup_agent.process(
        state["customer_query"],
        state.get("conversation_history"),
        resolution_summary=resolution_summary
    )

    state["followup_result"] = result
    # Append follow-up to response
    state["final_response"] = f"{state['final_response']}\n\n{result['follow_up_message']}"
    state["current_agent"] = "followup"
    state["agent_sequence"].append("followup")
    state["next_agent"] = None  # End workflow

    return state

def escalation_node(state: AgentState) -> AgentState:
    """Escalation node: Prepare for human handoff"""
    escalation_reason = "Complex issue requiring human judgment"

    knowledge_result = state.get("knowledge_result")
    action_result = state.get("action_result")

    if knowledge_result and knowledge_result.get("needs_escalation"):
        escalation_reason = "Knowledge base insufficient"
    elif action_result and action_result.get("action_plan", {}).get("execution_result", {}).get("success") == False:
        escalation_reason = "Action execution failed"

    result = escalation_agent.process(
        state["customer_query"],
        state.get("conversation_history"),
        escalation_reason=escalation_reason
    )

    state["escalation_result"] = result
    state["final_response"] = result["customer_message"]
    state["resolution_status"] = "escalated"
    state["current_agent"] = "escalation"
    state["agent_sequence"].append("escalation")
    state["next_agent"] = None  # End workflow

    return state

# Routing functions
def route_after_triage(state: AgentState) -> str:
    """Route after triage based on intent"""
    next_agent = state.get("next_agent", "knowledge")

    if next_agent == "knowledge":
        return "knowledge"
    elif next_agent == "action":
        return "action"
    elif next_agent == "escalation":
        return "escalation"
    else:
        return "knowledge"  # Default

def route_after_knowledge(state: AgentState) -> str:
    """Route after knowledge agent"""
    if state.get("needs_escalation"):
        return "escalation"
    else:
        return "followup"

def route_after_action(state: AgentState) -> str:
    """Route after action agent"""
    if state.get("needs_escalation"):
        return "escalation"
    else:
        return "followup"

# Build workflow graph
def create_workflow():
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("knowledge", knowledge_node)
    workflow.add_node("action", action_node)
    workflow.add_node("followup", followup_node)
    workflow.add_node("escalation", escalation_node)

    # Set entry point
    workflow.set_entry_point("triage")

    # Add conditional edges
    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "knowledge": "knowledge",
            "action": "action",
            "escalation": "escalation"
        }
    )

    workflow.add_conditional_edges(
        "knowledge",
        route_after_knowledge,
        {
            "followup": "followup",
            "escalation": "escalation"
        }
    )

    workflow.add_conditional_edges(
        "action",
        route_after_action,
        {
            "followup": "followup",
            "escalation": "escalation"
        }
    )

    # Follow-up and escalation end the workflow
    workflow.add_edge("followup", END)
    workflow.add_edge("escalation", END)

    return workflow.compile()

# Create compiled workflow
multi_agent_workflow = create_workflow()

print("[OK] Multi-agent workflow created successfully")
print("  Agents: Triage -> Knowledge/Action -> Follow-Up/Escalation")
