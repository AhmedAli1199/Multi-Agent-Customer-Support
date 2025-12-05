"""
Complete evaluation pipeline for multi-agent customer support system
Runs both metrics comparison and ablation study with rate limit protection.

Uses unified LLM client from config and .env for consistent model access.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation.metrics import evaluator
from evaluation.ablation_study import AblationStudy
from config import ModelConfig, EvalConfig
from utils.rate_limit_handler import RateLimitedEvaluator, RateLimitError
import json
import time


def main():
    """Run complete evaluation pipeline with rate limit protection"""
    print("="*80)
    print("MULTI-AGENT CUSTOMER SUPPORT SYSTEM - EVALUATION PIPELINE")
    print("="*80)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Configuration
    SAMPLE_SIZE = 20  # Increased to 20 for better statistical significance

    print(f"Configuration:")
    print(f"  - Sample size: {SAMPLE_SIZE} queries")
    print(f"  - Metrics: FCR, ART, ER, CSAT, Accuracy")
    print(f"  - Ablation configs: 5 (Full, No Follow-Up, Action Only, Minimal, Baseline)")
    print(f"\nLLM Configuration (from .env):")
    print(f"  - Provider: {ModelConfig.LLM_PROVIDER}")
    print(f"  - Primary Model: {ModelConfig.PRIMARY_MODEL}")
    print(f"  - Secondary Model: {ModelConfig.SECONDARY_MODEL}")
    print(f"  - Temperature: {ModelConfig.TEMPERATURE}")
    print(f"\nRate Limit Settings:")
    print(f"  - Delay between calls: {EvalConfig.RATE_LIMIT_DELAY}s")
    print(f"  - Retry delay on rate limit: {EvalConfig.RATE_LIMIT_RETRY_DELAY}s")
    print(f"  - Max retries: {EvalConfig.RATE_LIMIT_MAX_RETRIES}")
    print("\n" + "="*80)

    # Initialize rate limit tracker
    rate_limiter = RateLimitedEvaluator(
        delay_between_calls=EvalConfig.RATE_LIMIT_DELAY,
        retry_delay=EvalConfig.RATE_LIMIT_RETRY_DELAY,
        max_retries=EvalConfig.RATE_LIMIT_MAX_RETRIES
    )
    rate_limiter.start_evaluation()

    # Part 1: System Comparison
    print("\n[PART 1] MULTI-AGENT VS SINGLE-AGENT COMPARISON")
    print("="*80)

    try:
        comparison = evaluator.compare_systems(sample_size=SAMPLE_SIZE)
        evaluator.print_comparison(comparison)

        # Save comparison results
        output_file = Path(__file__).parent / "data" / "comparison_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            # Remove detailed results for cleaner output
            summary = {
                "sample_size": comparison["sample_size"],
                "llm_config": {
                    "provider": ModelConfig.LLM_PROVIDER,
                    "primary_model": ModelConfig.PRIMARY_MODEL,
                    "secondary_model": ModelConfig.SECONDARY_MODEL,
                    "temperature": ModelConfig.TEMPERATURE
                },
                "multi_agent_metrics": comparison["multi_agent"]["metrics"],
                "single_agent_metrics": comparison["single_agent"]["metrics"],
                "improvements": comparison["improvements"]
            }
            json.dump(summary, f, indent=2)

        print(f"\n[OK] Comparison results saved to {output_file}")

    except RateLimitError as e:
        print(f"\n[RATE LIMIT ERROR] Comparison evaluation stopped: {e}")
        print("[INFO] Consider reducing SAMPLE_SIZE or increasing RATE_LIMIT_DELAY in .env")
    except Exception as e:
        print(f"\n[ERROR] Comparison evaluation failed: {e}")
        import traceback
        traceback.print_exc()  # Print full error trace

    # Part 2: Ablation Study
    print("\n\n[PART 2] ABLATION STUDY")
    print("="*80)

    try:
        ablation = AblationStudy()
        ablation_results = ablation.run_ablation_study(sample_size=SAMPLE_SIZE)
        ablation.print_ablation_results(ablation_results)
        ablation.save_results(ablation_results)

    except RateLimitError as e:
        print(f"\n[RATE LIMIT ERROR] Ablation study stopped: {e}")
        print("[INFO] Consider reducing SAMPLE_SIZE or increasing RATE_LIMIT_DELAY in .env")
    except Exception as e:
        print(f"\n[ERROR] Ablation study failed: {e}")
        import traceback
        traceback.print_exc()  # Print full error trace

    # Print rate limit stats
    rate_limiter.print_stats()

    # Summary
    print("\n\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nLLM Used: {ModelConfig.LLM_PROVIDER} - {ModelConfig.PRIMARY_MODEL}")
    print("\nGenerated files:")
    print("  - data/comparison_results.json")
    print("  - data/ablation_results.json")
    print("\nNext steps:")
    print("  1. Review evaluation results")
    print("  2. Include metrics in research paper")
    print("  3. Create visualizations (optional)")
    print("  4. Run full evaluation with larger sample size")
    print("\nTo adjust rate limits, modify these in .env:")
    print("  - RATE_LIMIT_DELAY=2.0  (delay between calls)")
    print("  - RATE_LIMIT_RETRY_DELAY=20.0  (wait on rate limit)")
    print("  - RATE_LIMIT_MAX_RETRIES=3  (retry attempts)")
    print("="*80)


if __name__ == "__main__":
    main()
