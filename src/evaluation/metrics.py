"""
Evaluation metrics for comparing multi-agent and single-agent systems
"""
from typing import Dict, List, Tuple
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.graph import multi_agent_workflow
from baseline.single_agent import single_agent
from config import TEST_DATASET_FILE, EvalConfig
import statistics

class MetricsEvaluator:
    """Evaluate and compare agent systems on key metrics"""

    def __init__(self):
        # Load test dataset
        with open(TEST_DATASET_FILE, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)
        print(f"[OK] Loaded {len(self.test_data)} test conversations")

    def evaluate_first_contact_resolution(self, results: List[Dict]) -> float:
        """
        Calculate First-Contact Resolution (FCR) rate

        FCR = (Queries resolved without escalation) / Total queries
        """
        resolved_count = sum(
            1 for r in results
            if r.get('resolution_status') in ['resolved', 'partial'] and not r.get('needs_escalation')
        )
        return (resolved_count / len(results)) * 100 if results else 0.0

    def evaluate_average_response_time(self, results: List[Dict]) -> float:
        """
        Calculate Average Response Time (ART) in seconds
        """
        times = [r.get('processing_time', 0) for r in results if 'processing_time' in r]
        return statistics.mean(times) if times else 0.0

    def evaluate_escalation_rate(self, results: List[Dict]) -> float:
        """
        Calculate Escalation Rate (ER)

        ER = (Queries escalated to human) / Total queries
        """
        escalated_count = sum(1 for r in results if r.get('needs_escalation'))
        return (escalated_count / len(results)) * 100 if results else 0.0

    def evaluate_csat(self, results: List[Dict]) -> float:
        """
        Calculate simulated Customer Satisfaction (CSAT) score

        Based on:
        - Resolution status
        - Confidence score
        - Response appropriateness
        """
        scores = []
        for r in results:
            score = 3.0  # Baseline neutral

            # Resolution bonus
            if r.get('resolution_status') == 'resolved':
                score += 1.5
            elif r.get('resolution_status') == 'partial':
                score += 0.5

            # Confidence bonus
            confidence = r.get('confidence_score', 0.5)
            score += confidence * 0.5

            # Escalation penalty
            if r.get('needs_escalation'):
                score -= 1.0

            # Clamp to 1-5 scale
            score = max(1.0, min(5.0, score))
            scores.append(score)

        return statistics.mean(scores) if scores else 0.0

    def evaluate_intent_accuracy(self, results: List[Dict]) -> float:
        """
        Calculate intent classification accuracy

        Compare detected intent with ground truth from dataset
        """
        correct = 0
        total = 0

        for i, result in enumerate(results):
            if i >= len(self.test_data):
                break

            expected_intent = self.test_data[i].get('intent', '').upper()
            detected_intent = result.get('intent', '').upper()

            # Map intent categories
            intent_map = {
                'ACCOUNT_ACCESS': ['ACCOUNT', 'LOGIN', 'PASSWORD'],
                'ORDER_ISSUES': ['ORDER', 'CANCEL', 'MODIFY', 'TRACK'],
                'PAYMENT_REFUND': ['PAYMENT', 'REFUND', 'BILLING'],
                'GENERAL_INQUIRY': ['GENERAL', 'INFO', 'QUESTION'],
                'ACTION_REQUEST': ['ACTION', 'REQUEST']
            }

            # Check if intents match (with mapping)
            match = False
            if expected_intent == detected_intent:
                match = True
            else:
                for category, keywords in intent_map.items():
                    if any(k in expected_intent for k in keywords) and any(k in detected_intent for k in keywords):
                        match = True
                        break

            if match:
                correct += 1
            total += 1

        return (correct / total) * 100 if total > 0 else 0.0

    def run_evaluation(self, system_type: str, sample_size: int = None) -> Tuple[List[Dict], Dict]:
        """
        Run evaluation on specified system

        Args:
            system_type: "multi-agent" or "single-agent"
            sample_size: Number of test queries (None = use all)

        Returns:
            Tuple of (results list, metrics dict)
        """
        if sample_size is None:
            sample_size = min(len(self.test_data), EvalConfig.TEST_SAMPLE_SIZE)

        test_queries = self.test_data[:sample_size]
        results = []

        print(f"\n[INFO] Evaluating {system_type} on {sample_size} queries...")

        for i, test_case in enumerate(test_queries):
            query = test_case['customer_query']

            try:
                start_time = time.time()

                if system_type == "multi-agent":
                    # Run multi-agent system
                    initial_state = {
                        "customer_query": query,
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

                    final_state = multi_agent_workflow.invoke(initial_state)

                    result = {
                        "query": query,
                        "response": final_state.get("final_response"),
                        "intent": final_state.get("intent"),
                        "resolution_status": final_state.get("resolution_status"),
                        "needs_escalation": final_state.get("needs_escalation", False),
                        "confidence_score": final_state.get("confidence_score", 0.7),
                        "processing_time": time.time() - start_time,
                        "agent_sequence": final_state.get("agent_sequence", [])
                    }

                else:  # single-agent
                    result_data = single_agent.process(query, auto_execute=False)

                    result = {
                        "query": query,
                        "response": result_data.get("response"),
                        "intent": result_data.get("intent"),
                        "resolution_status": "resolved" if not result_data.get("needs_escalation") else "escalated",
                        "needs_escalation": result_data.get("needs_escalation", False),
                        "confidence_score": result_data.get("confidence", 0.5),
                        "processing_time": time.time() - start_time,
                        "agent_sequence": ["single-agent"]
                    }

                results.append(result)

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{sample_size} queries...")

            except Exception as e:
                print(f"[ERROR] Failed on query {i}: {e}")
                continue

        # Calculate metrics
        metrics = {
            "first_contact_resolution": self.evaluate_first_contact_resolution(results),
            "average_response_time": self.evaluate_average_response_time(results),
            "escalation_rate": self.evaluate_escalation_rate(results),
            "csat_score": self.evaluate_csat(results),
            "intent_accuracy": self.evaluate_intent_accuracy(results),
            "total_queries": len(results)
        }

        print(f"\n[OK] Evaluation complete for {system_type}")
        return results, metrics

    def compare_systems(self, sample_size: int = 50) -> Dict:
        """
        Compare multi-agent and single-agent systems

        Returns:
            Comparison results with metrics for both systems
        """
        print("="*80)
        print("SYSTEM COMPARISON EVALUATION")
        print("="*80)

        # Evaluate both systems
        multi_results, multi_metrics = self.run_evaluation("multi-agent", sample_size)
        single_results, single_metrics = self.run_evaluation("single-agent", sample_size)

        # Create comparison
        comparison = {
            "sample_size": sample_size,
            "multi_agent": {
                "results": multi_results,
                "metrics": multi_metrics
            },
            "single_agent": {
                "results": single_results,
                "metrics": single_metrics
            },
            "improvements": {
                "fcr_improvement": multi_metrics["first_contact_resolution"] - single_metrics["first_contact_resolution"],
                "art_improvement": single_metrics["average_response_time"] - multi_metrics["average_response_time"],
                "er_improvement": single_metrics["escalation_rate"] - multi_metrics["escalation_rate"],
                "csat_improvement": multi_metrics["csat_score"] - single_metrics["csat_score"],
                "accuracy_improvement": multi_metrics["intent_accuracy"] - single_metrics["intent_accuracy"]
            }
        }

        return comparison

    def print_comparison(self, comparison: Dict):
        """Print formatted comparison results"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS")
        print("="*80)

        multi_m = comparison["multi_agent"]["metrics"]
        single_m = comparison["single_agent"]["metrics"]
        improvements = comparison["improvements"]

        print(f"\n{'Metric':<35} {'Multi-Agent':<20} {'Single-Agent':<20} {'Improvement':<15}")
        print("-"*90)

        print(f"{'First-Contact Resolution (FCR)':<35} {multi_m['first_contact_resolution']:<20.2f}% {single_m['first_contact_resolution']:<20.2f}% {improvements['fcr_improvement']:>+14.2f}%")
        print(f"{'Average Response Time (ART)':<35} {multi_m['average_response_time']:<20.2f}s {single_m['average_response_time']:<20.2f}s {improvements['art_improvement']:>+14.2f}s")
        print(f"{'Escalation Rate (ER)':<35} {multi_m['escalation_rate']:<20.2f}% {single_m['escalation_rate']:<20.2f}% {improvements['er_improvement']:>+14.2f}%")
        print(f"{'CSAT Score':<35} {multi_m['csat_score']:<20.2f}/5 {single_m['csat_score']:<20.2f}/5 {improvements['csat_improvement']:>+14.2f}")
        print(f"{'Intent Classification Accuracy':<35} {multi_m['intent_accuracy']:<20.2f}% {single_m['intent_accuracy']:<20.2f}% {improvements['accuracy_improvement']:>+14.2f}%")

        print("="*90)

# Global instance
evaluator = MetricsEvaluator()
