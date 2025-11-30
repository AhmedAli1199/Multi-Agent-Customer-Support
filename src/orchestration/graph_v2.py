"""
LangGraph workflow V2 for enhanced multi-agent orchestration.
Uses V2 agents with proper tool calling and improved routing.
"""
from typing import Dict
from langgraph.graph import StateGraph, END
from orchestration.state import AgentState

# Import V2 agents
from agents.triage_agent_v2 import TriageAgentV2
from agents.knowledge_agent_v2 import KnowledgeAgentV2
from agents.action_agent_v2 import ActionAgentV2
from agents.followup_agent import FollowUpAgent
from agents.escalation_agent import EscalationAgent

# Initialize agents
print("\n" + "="*60)
print("Initializing Enhanced Multi-Agent System V2")
print("="*60)

triage_agent = TriageAgentV2()
knowledge_agent = KnowledgeAgentV2()
action_agent = ActionAgentV2()
followup_agent = FollowUpAgent()
escalation_agent = EscalationAgent()

print("\n[OK] All agents initialized successfully!")
print("="*60 + "\n")


def triage_node(state: AgentState) -> AgentState:
    """Triage node: Classify intent and route with improved logic"""
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

    print(f"\n[TRIAGE] Routing to: {state['next_agent']}")
    print(f"[TRIAGE] Intent: {state['intent']}, Urgency: {state['urgency']}")

    return state


def knowledge_node(state: AgentState) -> AgentState:
    """Knowledge node: Answer using RAG + product tools"""
    print("\n[KNOWLEDGE AGENT] Processing query...")

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
        print("[KNOWLEDGE AGENT] Escalation needed")
        state["needs_escalation"] = True
        state["next_agent"] = "escalation"
    else:
        print("[KNOWLEDGE AGENT] Query resolved successfully")
        state["needs_escalation"] = False
        state["next_agent"] = "followup"
        state["resolution_status"] = "resolved"

    return state


def action_node(state: AgentState) -> AgentState:
    """Action node: Execute backend operations using tools"""
    print("\n[ACTION AGENT] Executing action...")

    result = action_agent.process(
        state["customer_query"],
        state.get("conversation_history")
    )

    state["action_result"] = result
    state["final_response"] = result["response"]
    state["current_agent"] = "action"
    state["agent_sequence"].append("action")

    # Determine next step based on success
    if result.get("success"):
        print("[ACTION AGENT] Action executed successfully")
        state["resolution_status"] = "resolved"
        state["next_agent"] = "followup"
    else:
        print("[ACTION AGENT] Action failed, escalating")
        state["resolution_status"] = "partial"
        state["needs_escalation"] = True
        state["next_agent"] = "escalation"

    return state


def followup_node(state: AgentState) -> AgentState:
    """Follow-up node: Intelligent satisfaction check (only when needed)"""
    print("\n[FOLLOW-UP AGENT] Analyzing if follow-up is needed...")

    resolution_summary = state.get("final_response", "Issue addressed")

    result = followup_agent.process(
        state["customer_query"],
        state.get("conversation_history"),
        resolution_summary=resolution_summary,
        agent_sequence=state.get("agent_sequence", [])
    )

    state["followup_result"] = result

    # Only append follow-up if agent decided it's needed
    if result.get("needs_followup"):
        print(f"[FOLLOW-UP AGENT] Follow-up needed: {result.get('reason')}")
        state["final_response"] = f"{state['final_response']}\n\n{result['follow_up_message']}"
        state["agent_sequence"].append("followup")
    else:
        print(f"[FOLLOW-UP AGENT] No follow-up needed: {result.get('reason')}")
        # Don't add to agent_sequence if no follow-up sent

    state["current_agent"] = "followup"
    state["next_agent"] = None  # End workflow

    print("[FOLLOW-UP AGENT] Conversation complete")

    return state


def escalation_node(state: AgentState) -> AgentState:
    """Escalation node: Prepare for human handoff"""
    print("\n[ESCALATION AGENT] Preparing human handoff...")

    escalation_reason = "Complex issue requiring human judgment"

    knowledge_result = state.get("knowledge_result")
    action_result = state.get("action_result")

    if knowledge_result and knowledge_result.get("needs_escalation"):
        escalation_reason = "Knowledge base insufficient"
    elif action_result and not action_result.get("success"):
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

    print("[ESCALATION AGENT] Escalation prepared")

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
def create_workflow_v2():
    """Create the enhanced LangGraph workflow with V2 agents"""
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
multi_agent_workflow_v2 = create_workflow_v2()

print("[OK] Enhanced Multi-Agent Workflow V2 created successfully")
print("  Workflow: Triage -> Knowledge/Action (with tools) -> Follow-Up/Escalation")
print("  Features: Proper tool calling, product search, reduced escalation")
