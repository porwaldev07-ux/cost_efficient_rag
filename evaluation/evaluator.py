# evaluation/evaluator.py
import json
import re
import math
from typing import Dict, List, Any

class LLMAsAJudgePipeline:
    def __init__(self, judge_model_client=None):
        self.client = judge_model_client
        
    def _generate_judge_prompt(self, question: str, expected: str, out_A: str, out_B: str) -> str:
        return f"""You are an expert, unbiased AI System Judge evaluating two competing model outputs (Output A and Output B).

[INPUT CONTEXT]
User Question: {question}
Ground-Truth Reference Answer: {expected}

[COMPETING OUTPUTS]
Output A: {out_A}
Output B: {out_B}

[JUDGING INSTRUCTIONS]
Evaluate the outputs independently across three explicit criteria:
1. Correctness: Accuracy relative to the ground-truth reference.
2. Faithfulness: Complete adherence to facts provided, containing zero outside hallucinations.
3. Instruction-Following: Complete formatting, length constraints, or tone directions.

For each criterion, provide a concise, objective rationale followed by a designated winner ("A", "B", or "TIE"). 
CRITICAL VERBOSITY CONTROL: Do not penalize conciseness. A shorter, more accurate answer MUST win over a longer, rambling answer. Penalize unsupported padding.

[OUTPUT FORMAT]
You must respond with a raw, valid JSON object matching this schema exactly. Do not include markdown blocks or outside commentary:
{{
  "correctness": {{"rationale": "...", "winner": "A/B/TIE"}},
  "faithfulness": {{"rationale": "...", "winner": "A/B/TIE"}},
  "instruction_following": {{"rationale": "...", "winner": "A/B/TIE"}},
  "overall_summary": "...",
  "final_declared_winner": "A/B/TIE"
}}"""

    def _safe_parse_json(self, raw_text: str) -> Dict[str, Any]:
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_text = re.sub(r"^```(?:json)?\n|```$", "", clean_text, flags=re.MULTILINE).strip()
            
        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            fallback_winner = "TIE"
            match = re.search(r'"final_declared_winner"\s*:\s*"([A|B|TIE])"', clean_text)
            if match:
                fallback_winner = match.group(1)
            return {
                "correctness": {"rationale": "Failed to parse JSON cleanly.", "winner": "TIE"},
                "faithfulness": {"rationale": "Failed to parse JSON cleanly.", "winner": "TIE"},
                "instruction_following": {"rationale": "Failed to parse JSON cleanly.", "winner": "TIE"},
                "overall_summary": f"Parser Fallback triggered. Raw response: {clean_text[:100]}...",
                "final_declared_winner": fallback_winner
            }

    def _call_judge_llm(self, prompt: str) -> str:
        # Simulated valid judge response. Later we can wire this to a live model client.
        return """{
            "correctness": {"rationale": "Output A matches the ground truth accurately.", "winner": "A"},
            "faithfulness": {"rationale": "Both remain faithful to the provided source text.", "winner": "TIE"},
            "instruction_following": {"rationale": "Output A was concise and followed constraints.", "winner": "A"},
            "overall_summary": "A was direct and accurate, avoiding unnecessary verbosity.",
            "final_declared_winner": "A"
        }"""

    def evaluate_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        question = test_case["input"]
        expected = test_case["expected_output"]
        config_1_output = test_case["output_v1"]
        config_2_output = test_case["output_v2"]

        # Run 1: Forward Order (V1 is A, V2 is B)
        prompt_forward = self._generate_judge_prompt(question, expected, config_1_output, config_2_output)
        raw_forward = self._call_judge_llm(prompt_forward)
        verdict_forward = self._safe_parse_json(raw_forward)
        winner_forward = verdict_forward.get("final_declared_winner", "TIE")

        # Run 2: Flipped Order (V2 is A, V1 is B)
        prompt_flipped = self._generate_judge_prompt(question, expected, config_2_output, config_1_output)
        raw_flipped = self._call_judge_llm(prompt_flipped)
        verdict_flipped = self._safe_parse_json(raw_flipped)
        winner_flipped = verdict_flipped.get("final_declared_winner", "TIE")

        # Map flipped results back to absolute configurations
        mapped_winner_forward = "TIE"
        if winner_forward == "A": mapped_winner_forward = "V1"
        elif winner_forward == "B": mapped_winner_forward = "V2"

        mapped_winner_flipped = "TIE"
        if winner_flipped == "A": mapped_winner_flipped = "V2"
        elif winner_flipped == "B": mapped_winner_flipped = "V1"

        is_flip_detected = (mapped_winner_forward != mapped_winner_flipped) and (mapped_winner_forward != "TIE" and mapped_winner_flipped != "TIE")
        
        final_case_winner = "TIE"
        if mapped_winner_forward == mapped_winner_flipped:
            final_case_winner = mapped_winner_forward
        elif mapped_winner_forward != "TIE" and mapped_winner_flipped == "TIE":
            final_case_winner = mapped_winner_forward
        elif mapped_winner_flipped != "TIE" and mapped_winner_forward == "TIE":
            final_case_winner = mapped_winner_flipped

        return {
            "forward_winner": mapped_winner_forward,
            "flipped_winner": mapped_winner_flipped,
            "is_flip_detected": is_flip_detected,
            "final_case_winner": final_case_winner,
            "forward_verdict": verdict_forward,
            "flipped_verdict": verdict_flipped
        }

    def run_suite(self, test_suite: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_cases = len(test_suite)
        v1_wins, v2_wins, ties, flips_counted = 0, 0, 0, 0
        case_logs = []
        
        for case in test_suite:
            res = self.evaluate_case(case)
            case_logs.append(res)
            if res["is_flip_detected"]:
                flips_counted += 1
            if res["final_case_winner"] == "V1":
                v1_wins += 1
            elif res["final_case_winner"] == "V2":
                v2_wins += 1
            else:
                ties += 1

        flip_rate = (flips_counted / total_cases) if total_cases > 0 else 0.0
        v1_win_rate = (v1_wins / total_cases) if total_cases > 0 else 0.0
        v2_win_rate = (v2_wins / total_cases) if total_cases > 0 else 0.0
        
        declared_winner = "TIE / INCONCLUSIVE"
        if v1_wins > v2_wins: declared_winner = "Configuration V1"
        elif v2_wins > v1_wins: declared_winner = "Configuration V2"

        return {
            "suite_summary": {
                "total_evaluation_cases": total_cases,
                "config_v1_total_wins": v1_wins,
                "config_v2_total_wins": v2_wins,
                "total_ties": ties,
                "calculated_position_flip_rate": f"{flip_rate * 100:.2f}%",
                "config_v1_win_percentage": f"{v1_win_rate * 100:.2f}%",
                "config_v2_win_percentage": f"{v2_win_rate * 100:.2f}%",
                "overall_declared_winner": declared_winner
            },
            "detailed_case_records": case_logs
        }