"""
Quick test of the multi-agent system
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from orchestration.graph import multi_agent_workflow
from orchestration.state import AgentState

def test_multi_agent():
    """Test the multi-agent workflow"""
    print("="*60)
    print("Testing Multi-Agent Customer Support System")
    print("="*60)

    # Test query
    test_query = "I want to cancel my order #12345"

    print(f"\nCustomer Query: {test_query}\n")

    # Create initial state
    initial_state = {
        "customer_query": test_query,
        "conversation_history": [],
        "current_agent": None,
        "next_agent": None,
        "agent_sequence": [],
        "needs_escalation": False,
        "resolution_status": "unresolved",
        "triage_result": None,
        "knowledge_result": None,
        "action_result": None,
        "followup_result": None,
        "escalation_result": None,
        "final_response": None,
        "intent": None,
        "entities": None,
        "urgency": None,
        "sentiment": None,
        "confidence_score": None,
        "metadata": None
    }

    # Run workflow
    print("Running multi-agent workflow...\n")
    final_state = multi_agent_workflow.invoke(initial_state)

    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nAgent Sequence: {' -> '.join(final_state['agent_sequence'])}")
    print(f"Intent: {final_state.get('intent')}")
    print(f"Resolution Status: {final_state.get('resolution_status')}")
    print(f"\nFinal Response:\n{final_state.get('final_response')}")
    print("\n" + "="*60)

if __name__ == "__main__":
    test_multi_agent()
