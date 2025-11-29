"""
Complete evaluation pipeline for multi-agent customer support system
Runs both metrics comparison and ablation study
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation.metrics import evaluator
from evaluation.ablation_study import AblationStudy
import json
import time

def main():
    """Run complete evaluation pipeline"""
    print("="*80)
    print("MULTI-AGENT CUSTOMER SUPPORT SYSTEM - EVALUATION PIPELINE")
    print("="*80)
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Configuration
    SAMPLE_SIZE = 20  # Reduced for faster testing, increase for full evaluation

    print(f"Configuration:")
    print(f"  - Sample size: {SAMPLE_SIZE} queries")
    print(f"  - Metrics: FCR, ART, ER, CSAT, Accuracy")
    print(f"  - Ablation configs: 5 (Full, No Follow-Up, Action Only, Minimal, Baseline)")
    print("\n" + "="*80)

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
                "multi_agent_metrics": comparison["multi_agent"]["metrics"],
                "single_agent_metrics": comparison["single_agent"]["metrics"],
                "improvements": comparison["improvements"]
            }
            json.dump(summary, f, indent=2)

        print(f"\n[OK] Comparison results saved to {output_file}")

    except Exception as e:
        print(f"\n[ERROR] Comparison evaluation failed: {e}")

    # Part 2: Ablation Study
    print("\n\n[PART 2] ABLATION STUDY")
    print("="*80)

    try:
        ablation = AblationStudy()
        ablation_results = ablation.run_ablation_study(sample_size=SAMPLE_SIZE)
        ablation.print_ablation_results(ablation_results)
        ablation.save_results(ablation_results)

    except Exception as e:
        print(f"\n[ERROR] Ablation study failed: {e}")

    # Summary
    print("\n\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files:")
    print("  - data/comparison_results.json")
    print("  - data/ablation_results.json")
    print("\nNext steps:")
    print("  1. Review evaluation results")
    print("  2. Include metrics in research paper")
    print("  3. Create visualizations (optional)")
    print("  4. Run full evaluation with larger sample size")
    print("="*80)

if __name__ == "__main__":
    main()
