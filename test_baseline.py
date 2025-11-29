"""
Test and compare baseline single-agent with multi-agent system
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from baseline.single_agent import single_agent
from orchestration.graph import multi_agent_workflow
from orchestration.state import AgentState
import time

def test_single_agent():
    """Test baseline single-agent system"""
    print("="*60)
    print("Testing SINGLE-AGENT System")
    print("="*60)

    test_query = "I want to cancel my order #12345"
    print(f"\nCustomer Query: {test_query}\n")

    start_time = time.time()
    result = single_agent.process(test_query, auto_execute=True)
    end_time = time.time()

    print("\n" + "="*60)
    print("SINGLE-AGENT RESULTS")
    print("="*60)
    print(f"Response Time: {end_time - start_time:.2f}s")
    print(f"Intent: {result.get('intent')}")
    print(f"Actions Taken: {result.get('actions_taken')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"\nResponse:\n{result.get('response')}")
    print("="*60)

    return result, end_time - start_time

def test_multi_agent():
    """Test multi-agent system"""
    print("\n" + "="*60)
    print("Testing MULTI-AGENT System")
    print("="*60)

    test_query = "I want to cancel my order #12345"
    print(f"\nCustomer Query: {test_query}\n")

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

    start_time = time.time()
    final_state = multi_agent_workflow.invoke(initial_state)
    end_time = time.time()

    print("\n" + "="*60)
    print("MULTI-AGENT RESULTS")
    print("="*60)
    print(f"Response Time: {end_time - start_time:.2f}s")
    print(f"Agent Sequence: {' -> '.join(final_state['agent_sequence'])}")
    print(f"Intent: {final_state.get('intent')}")
    print(f"Resolution Status: {final_state.get('resolution_status')}")
    print(f"\nFinal Response:\n{final_state.get('final_response')}")
    print("="*60)

    return final_state, end_time - start_time

def compare_systems():
    """Compare both systems side by side"""
    print("\n" + "="*80)
    print("COMPARISON: Single-Agent vs Multi-Agent")
    print("="*80)

    # Test single-agent
    single_result, single_time = test_single_agent()

    # Test multi-agent
    multi_result, multi_time = test_multi_agent()

    # Comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<30} {'Single-Agent':<25} {'Multi-Agent':<25}")
    print("-"*80)
    print(f"{'Response Time':<30} {f'{single_time:.2f}s':<25} {f'{multi_time:.2f}s':<25}")
    print(f"{'Intent Detection':<30} {str(single_result.get('intent')):<25} {str(multi_result.get('intent')):<25}")
    print(f"{'Actions Executed':<30} {str(len(single_result.get('actions_taken', []))):<25} {'1 (cancel_order)':<25}")
    print(f"{'Agent Coordination':<30} {'N/A (single)':<25} {' -> '.join(multi_result.get('agent_sequence', [])):<25}")
    print("="*80)

if __name__ == "__main__":
    compare_systems()
