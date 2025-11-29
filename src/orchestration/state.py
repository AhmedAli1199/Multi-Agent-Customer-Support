"""
State schema for LangGraph orchestration.
Defines the shared state passed between agents.
"""
from typing import TypedDict, List, Dict, Optional, Literal

class AgentState(TypedDict):
    """State passed between agents in the workflow"""

    # Core conversation
    customer_query: str
    conversation_history: List[Dict[str, str]]

    # Triage results
    intent: Optional[str]
    entities: Optional[Dict]
    urgency: Optional[str]
    sentiment: Optional[str]

    # Agent routing
    current_agent: str
    next_agent: Optional[str]
    agent_sequence: List[str]

    # Agent outputs
    triage_result: Optional[Dict]
    knowledge_result: Optional[Dict]
    action_result: Optional[Dict]
    followup_result: Optional[Dict]
    escalation_result: Optional[Dict]

    # Final response
    final_response: Optional[str]
    resolution_status: Literal["resolved", "partial", "escalated", "unresolved"]

    # Metadata
    needs_escalation: bool
    confidence_score: Optional[float]
    metadata: Optional[Dict]
