"""
Ablation Study: Analyze impact of individual agent components
With rate limit protection for LLM API calls.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List
import json
import time
from config import TEST_DATASET_FILE, DATA_DIR, EvalConfig, ModelConfig
from orchestration.state import AgentState
from agents.triage_agent import TriageAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.action_agent import ActionAgent
from agents.followup_agent import FollowUpAgent
from agents.escalation_agent import EscalationAgent
from baseline.single_agent import SingleAgent
from utils.rate_limit_handler import is_rate_limit_error

class AblationStudy:
    """
    Conduct ablation studies to measure individual agent contributions

    Configurations:
    1. Full Multi-Agent (All 5 agents)
    2. No Follow-Up Agent
    3. No Knowledge Agent (Action only)
    4. Triage + Action only (Minimal)
    5. Single-Agent Baseline
    """

    def __init__(self):
        with open(TEST_DATASET_FILE, 'r', encoding='utf-8') as f:
            self.test_data = json.load(f)

        # Initialize agents
        self.triage_agent = TriageAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.action_agent = ActionAgent()
        self.followup_agent = FollowUpAgent()
        self.escalation_agent = EscalationAgent()
        self.single_agent = SingleAgent()

        print(f"[OK] Loaded {len(self.test_data)} test cases for ablation study")

    def run_full_system(self, query: str) -> Dict:
        """Configuration 1: Full multi-agent system"""
        start_time = time.time()

        # Triage
        triage_result = self.triage_agent.process(query)
        route = triage_result.get('routing', {}).get('route_to', 'knowledge')

        # Route to appropriate agent
        if route == 'escalation':
            escalation_result = self.escalation_agent.process(query)
            # Escalation agent returns 'customer_message', not 'response'
            final_response = escalation_result.get('customer_message', escalation_result.get('escalation_summary', 'Escalated to human agent'))
            agents_used = ['triage', 'escalation']
        elif route == 'action':
            action_result = self.action_agent.process(query, auto_execute=True)
            followup_result = self.followup_agent.process(query)
            # Action agent has 'response', followup has 'follow_up_message'
            action_response = action_result.get('response', 'Action completed')
            followup_message = followup_result.get('follow_up_message', '')
            final_response = action_response + ("\n\n" + followup_message if followup_message else "")
            agents_used = ['triage', 'action', 'followup']
        else:  # knowledge
            knowledge_result = self.knowledge_agent.process(query)
            followup_result = self.followup_agent.process(query)
            # Knowledge agent has 'response', followup has 'follow_up_message'
            knowledge_response = knowledge_result.get('response', 'Response provided')
            followup_message = followup_result.get('follow_up_message', '')
            final_response = knowledge_response + ("\n\n" + followup_message if followup_message else "")
            agents_used = ['triage', 'knowledge', 'followup']

        return {
            'response': final_response,
            'agents_used': agents_used,
            'processing_time': time.time() - start_time,
            'configuration': 'full_system'
        }

    def run_no_followup(self, query: str) -> Dict:
        """Configuration 2: No Follow-Up Agent"""
        start_time = time.time()

        triage_result = self.triage_agent.process(query)
        route = triage_result.get('routing', {}).get('route_to', 'knowledge')

        if route == 'escalation':
            escalation_result = self.escalation_agent.process(query)
            # Escalation agent returns 'customer_message', not 'response'
            final_response = escalation_result.get('customer_message', escalation_result.get('escalation_summary', 'Escalated to human agent'))
            agents_used = ['triage', 'escalation']
        elif route == 'action':
            action_result = self.action_agent.process(query, auto_execute=True)
            final_response = action_result.get('response', 'Action completed')
            agents_used = ['triage', 'action']
        else:
            knowledge_result = self.knowledge_agent.process(query)
            final_response = knowledge_result.get('response', 'Response provided')
            agents_used = ['triage', 'knowledge']

        return {
            'response': final_response,
            'agents_used': agents_used,
            'processing_time': time.time() - start_time,
            'configuration': 'no_followup'
        }

    def run_action_only(self, query: str) -> Dict:
        """Configuration 3: Triage + Action (no knowledge agent)"""
        start_time = time.time()

        triage_result = self.triage_agent.process(query)
        action_result = self.action_agent.process(query, auto_execute=True)

        return {
            'response': action_result.get('response', 'Action completed'),
            'agents_used': ['triage', 'action'],
            'processing_time': time.time() - start_time,
            'configuration': 'action_only'
        }

    def run_minimal(self, query: str) -> Dict:
        """Configuration 4: Triage + Direct routing (minimal agents)"""
        start_time = time.time()

        triage_result = self.triage_agent.process(query)
        route = triage_result.get('routing', {}).get('route_to', 'knowledge')

        if route == 'action':
            action_result = self.action_agent.process(query, auto_execute=True)
            final_response = action_result.get('response', 'Action completed')
            agents_used = ['triage', 'action']
        else:
            knowledge_result = self.knowledge_agent.process(query)
            final_response = knowledge_result.get('response', 'Response provided')
            agents_used = ['triage', 'knowledge']

        return {
            'response': final_response,
            'agents_used': agents_used,
            'processing_time': time.time() - start_time,
            'configuration': 'minimal'
        }

    def run_baseline(self, query: str) -> Dict:
        """Configuration 5: Single-agent baseline"""
        start_time = time.time()

        result = self.single_agent.process(query, auto_execute=False)

        return {
            'response': result.get('response', 'Response provided'),
            'agents_used': ['single-agent'],
            'processing_time': time.time() - start_time,
            'configuration': 'baseline'
        }

    def evaluate_configuration(self, config_name: str, config_func, sample_size: int = 30) -> Dict:
        """Evaluate a specific configuration with rate limit protection"""
        print(f"\n[INFO] Evaluating {config_name}...")
        print(f"[INFO] Using: {ModelConfig.LLM_PROVIDER} - {ModelConfig.PRIMARY_MODEL}")

        results = []
        test_queries = self.test_data[:sample_size]
        
        max_retries = EvalConfig.RATE_LIMIT_MAX_RETRIES
        retry_delay = EvalConfig.RATE_LIMIT_RETRY_DELAY

        for i, test_case in enumerate(test_queries):
            query = test_case['customer_query']

            for attempt in range(max_retries + 1):
                try:
                    result = config_func(query)
                    results.append(result)

                    # Rate limit delay between calls
                    time.sleep(EvalConfig.RATE_LIMIT_DELAY)

                    if (i + 1) % 5 == 0:
                        print(f"  Processed {i + 1}/{sample_size} queries...")
                    
                    # Success - break retry loop
                    break

                except Exception as e:
                    if is_rate_limit_error(e):
                        if attempt < max_retries:
                            wait_time = retry_delay * (1.5 ** attempt)
                            print(f"\n[RATE LIMIT] Query {i+1}: Hit rate limit (attempt {attempt + 1}/{max_retries + 1})")
                            print(f"[RATE LIMIT] Waiting {wait_time:.1f}s before retry...")
                            time.sleep(wait_time)
                        else:
                            print(f"[ERROR] Query {i+1}: Rate limit exceeded after {max_retries} retries")
                            # Add error result and continue
                            results.append({
                                'response': f"RATE_LIMIT_ERROR: {str(e)}",
                                'agents_used': [],
                                'processing_time': 0,
                                'configuration': config_name,
                                'error': str(e)
                            })
                            time.sleep(retry_delay * 2)
                            break
                    else:
                        print(f"[ERROR] Query {i+1} failed: {e}")
                        results.append({
                            'response': f"ERROR: {str(e)}",
                            'agents_used': [],
                            'processing_time': 0,
                            'configuration': config_name,
                            'error': str(e)
                        })
                        break

        # Filter successful results for metrics
        valid_results = [r for r in results if 'error' not in r]
        
        # Calculate metrics
        avg_time = sum(r['processing_time'] for r in valid_results) / len(valid_results) if valid_results else 0
        avg_agents = sum(len(r['agents_used']) for r in valid_results) / len(valid_results) if valid_results else 0

        metrics = {
            'configuration': config_name,
            'avg_processing_time': avg_time,
            'avg_agents_used': avg_agents,
            'total_queries': len(results),
            'successful_queries': len(valid_results),
            'failed_queries': len(results) - len(valid_results),
            'results': results
        }

        print(f"[OK] {config_name}: Avg Time={avg_time:.2f}s, Avg Agents={avg_agents:.1f}, Success={len(valid_results)}/{len(results)}")
        return metrics

    def run_ablation_study(self, sample_size: int = 30) -> Dict:
        """Run complete ablation study with rate limit protection"""
        print("="*80)
        print("ABLATION STUDY")
        print("="*80)
        print(f"Sample size: {sample_size} queries")
        print(f"\nLLM Configuration:")
        print(f"  - Provider: {ModelConfig.LLM_PROVIDER}")
        print(f"  - Primary Model: {ModelConfig.PRIMARY_MODEL}")
        print(f"  - Secondary Model: {ModelConfig.SECONDARY_MODEL}")
        print(f"\nRate Limit Settings:")
        print(f"  - Delay between calls: {EvalConfig.RATE_LIMIT_DELAY}s")
        print(f"  - Retry delay: {EvalConfig.RATE_LIMIT_RETRY_DELAY}s")
        print(f"  - Max retries: {EvalConfig.RATE_LIMIT_MAX_RETRIES}")

        configurations = {
            'Full System (5 agents)': self.run_full_system,
            'No Follow-Up (4 agents)': self.run_no_followup,
            'Action Only (2 agents)': self.run_action_only,
            'Minimal (2 agents)': self.run_minimal,
            'Baseline (single-agent)': self.run_baseline
        }

        all_results = {}

        for config_name, config_func in configurations.items():
            all_results[config_name] = self.evaluate_configuration(
                config_name, config_func, sample_size
            )

        return all_results

    def print_ablation_results(self, results: Dict):
        """Print formatted ablation study results"""
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS")
        print("="*80)

        print(f"\n{'Configuration':<35} {'Avg Time (s)':<20} {'Avg Agents':<20}")
        print("-"*75)

        for config_name, metrics in results.items():
            print(f"{config_name:<35} {metrics['avg_processing_time']:<20.2f} {metrics['avg_agents_used']:<20.1f}")

        print("="*75)

        # Analyze impact
        print("\n[ANALYSIS]")
        if 'Full System (5 agents)' in results and 'No Follow-Up (4 agents)' in results:
            full = results['Full System (5 agents)']['avg_processing_time']
            no_followup = results['No Follow-Up (4 agents)']['avg_processing_time']
            time_diff = full - no_followup
            print(f"  - Follow-Up Agent adds {time_diff:.2f}s processing time")

        if 'Full System (5 agents)' in results and 'Baseline (single-agent)' in results:
            full = results['Full System (5 agents)']['avg_processing_time']
            baseline = results['Baseline (single-agent)']['avg_processing_time']
            speedup = (baseline - full) / baseline * 100 if baseline > 0 else 0
            print(f"  - Multi-agent vs Single-agent: {speedup:+.1f}% time difference")

    def save_results(self, results: Dict, filename: str = "ablation_results.json"):
        """Save ablation study results to file"""
        output_path = DATA_DIR / filename

        # Remove detailed results for cleaner output
        summary = {
            "llm_config": {
                "provider": ModelConfig.LLM_PROVIDER,
                "primary_model": ModelConfig.PRIMARY_MODEL,
                "secondary_model": ModelConfig.SECONDARY_MODEL,
                "temperature": ModelConfig.TEMPERATURE
            },
            "rate_limit_config": {
                "delay_between_calls": EvalConfig.RATE_LIMIT_DELAY,
                "retry_delay": EvalConfig.RATE_LIMIT_RETRY_DELAY,
                "max_retries": EvalConfig.RATE_LIMIT_MAX_RETRIES
            },
            "configurations": {}
        }
        
        for config_name, metrics in results.items():
            summary["configurations"][config_name] = {
                'avg_processing_time': metrics['avg_processing_time'],
                'avg_agents_used': metrics['avg_agents_used'],
                'total_queries': metrics['total_queries'],
                'successful_queries': metrics.get('successful_queries', metrics['total_queries']),
                'failed_queries': metrics.get('failed_queries', 0)
            }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[OK] Results saved to {output_path}")

# Main execution
if __name__ == "__main__":
    ablation = AblationStudy()
    results = ablation.run_ablation_study(sample_size=30)
    ablation.print_ablation_results(results)
    ablation.save_results(results)
