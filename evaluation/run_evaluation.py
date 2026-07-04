# evaluation/run_evaluation.py
import json
import os
from evaluation.evaluator import LLMAsAJudgePipeline

def main():
    suite_path = "evaluation/test_suite.json"
    output_dir = "data/eval_results"
    output_file = os.path.join(output_dir, "suite_report.json")
    
    if not os.path.exists(suite_path):
        print(f"[-] Error: Could not find test suite dataset at {suite_path}")
        return

    # Load test dataset
    with open(suite_path, "r") as f:
        test_suite = json.load(f)
        
    print(f"[+] Loaded {len(test_suite)} evaluation test cases from test_suite.json.")
    print("[+] Initiating LLM-as-a-Judge Pipeline with dual-order position bias mitigation...")
    
    # Initialize the judge engine
    evaluator = LLMAsAJudgePipeline()
    
    # Run evaluations
    report = evaluator.run_suite(test_suite)
    
    # Ensure save directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Export full metrics
    with open(output_file, "w") as out:
        json.dump(report, out, indent=2)
        
    print("\n================ EVALUATION SUITE REPORT CARD ================")
    summary = report["suite_summary"]
    print(f" Total Cases Evaluated : {summary['total_evaluation_cases']}")
    print(f" Config V1 Total Wins  : {summary['config_v1_total_wins']}")
    print(f" Config V2 Total Wins  : {summary['config_v2_total_wins']}")
    print(f" Neutral Ties Declared : {summary['total_ties']}")
    print(f" Calculated Flip Rate  : {summary['calculated_position_flip_rate']}  <-- [Position Bias Metric]")
    print(f" Declared Clear Winner : {summary['overall_declared_winner']}")
    print("==============================================================")
    print(f"[+] Full analytical JSON report saved cleanly to: {output_file}\n")

if __name__ == "__main__":
    main()