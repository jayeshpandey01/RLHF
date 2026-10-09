"""LLM-as-a-Judge Evaluation Engine with Rubric Calibration & Bias Mitigation (Milestone 4).

Implements:
- 4-Dimensional Rubric Calibration (Accuracy, Faithfulness, Clarity, Conciseness)
- Dual-Engine Architecture: Generative LLM backend with Chain-of-Thought parsing
  and deterministic analytical heuristic fallback engine.
- Systematic Bias Mitigation:
  * Symmetric Two-Pass Position-Swap Debiasing with Inconsistency Tracking (kappa_pos)
  * Asymmetric Soft-Threshold Exponential Length Penalty
- Seamless integration with GoldenTestCase (Phase 4.1) and CitationVerifier (Phase 4.2).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from loguru import logger

from evaluation.benchmark_suite import GoldenTestCase
from evaluation.citation_verifier import CitationVerifier


@dataclass
class JudgeScore:
    """Quantitative multi-dimensional evaluation score emitted by the LLM Judge."""

    accuracy: float            # [1.0, 5.0] Factual truthfulness and golden alignment
    faithfulness: float        # [1.0, 5.0] Grounding in reference context / source docs
    clarity: float             # [1.0, 5.0] Structural flow, coherence, grammar
    conciseness: float         # [1.0, 5.0] Information density, lack of fluff
    raw_composite: float       # Convex weighted sum before length penalty [1.0, 5.0]
    calibrated_score: float    # Post-length-penalty calibrated scalar score [1.0, 5.0]
    length_penalty: float      # Multiplier in (0.0, 1.0]
    rationale: str = ""        # Chain-of-thought justification
    mode: str = "model"        # "model" or "heuristic"

    def to_dict(self) -> dict[str, Any]:
        """Serialize score to dictionary."""
        return {
            "accuracy": round(self.accuracy, 4),
            "faithfulness": round(self.faithfulness, 4),
            "clarity": round(self.clarity, 4),
            "conciseness": round(self.conciseness, 4),
            "raw_composite": round(self.raw_composite, 4),
            "calibrated_score": round(self.calibrated_score, 4),
            "length_penalty": round(self.length_penalty, 4),
            "rationale": self.rationale,
            "mode": self.mode,
        }


@dataclass
class PairwiseJudgeResult:
    """Position-debiased pairwise evaluation outcome between Candidate A and Candidate B."""

    winner: str                      # 'A', 'B', or 'TIE'
    score_a: float                   # Debiased calibrated score for Candidate A [1.0, 5.0]
    score_b: float                   # Debiased calibrated score for Candidate B [1.0, 5.0]
    score_delta: float               # score_a - score_b
    position_inconsistency: float    # kappa_pos in [0.0, 1.0]
    pass1_winner: str                # Winner when presented in sequence (A, B)
    pass2_winner: str                # Winner when presented in sequence (B, A)
    score_a_pass1: float
    score_a_pass2: float
    score_b_pass1: float
    score_b_pass2: float
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize pairwise result to dictionary."""
        return {
            "winner": self.winner,
            "score_a": round(self.score_a, 4),
            "score_b": round(self.score_b, 4),
            "score_delta": round(self.score_delta, 4),
            "position_inconsistency": round(self.position_inconsistency, 4),
            "pass1_winner": self.pass1_winner,
            "pass2_winner": self.pass2_winner,
            "score_a_pass1": round(self.score_a_pass1, 4),
            "score_a_pass2": round(self.score_a_pass2, 4),
            "score_b_pass1": round(self.score_b_pass1, 4),
            "score_b_pass2": round(self.score_b_pass2, 4),
            "rationale": self.rationale,
        }


class LLMJudge:
    """Production LLM-as-a-Judge engine with rubric calibration and bias mitigation."""

    DEFAULT_WEIGHTS = {
        "accuracy": 0.35,
        "faithfulness": 0.35,
        "clarity": 0.15,
        "conciseness": 0.15,
    }

    def __init__(
        self,
        model_fn: Optional[Callable[[str], str]] = None,
        weights: Optional[dict[str, float]] = None,
        len_tau: float = 1.8,
        len_lambda: float = 0.15,
        tie_margin: float = 0.15,
        citation_verifier: Optional[CitationVerifier] = None,
    ):
        """Initialize the LLM Judge.

        Args:
            model_fn: Optional callable taking a prompt string and returning model generation.
                      If None, the judge operates in deterministic heuristic mode.
            weights: Rubric dimension weights summing to 1.0. Defaults to 35/35/15/15.
            len_tau: Length ratio threshold before verbosity penalty applies. Default 1.8.
            len_lambda: Exponential decay factor for verbosity penalty. Default 0.15.
            tie_margin: Minimum absolute score difference to declare a winner. Default 0.15.
            citation_verifier: Optional CitationVerifier instance for factual entailment.
        """
        self.model_fn = model_fn
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        total_w = sum(self.weights.values())
        if abs(total_w - 1.0) > 1e-4:
            # Normalize weights to sum strictly to 1.0
            self.weights = {k: v / total_w for k, v in self.weights.items()}

        self.len_tau = float(len_tau)
        self.len_lambda = float(len_lambda)
        self.tie_margin = float(tie_margin)
        self.citation_verifier = citation_verifier or CitationVerifier()

    # -------------------------------------------------------------------------
    # Length Penalty & Bias Calculations
    # -------------------------------------------------------------------------

    def compute_length_penalty(self, response: str, golden_reference: str) -> float:
        """Calculate soft-threshold exponential length penalty multiplier phi(r_L).

        phi(r_L) = 1.0 if r_L <= tau else exp(-lambda * (r_L - tau))

        Args:
            response: Candidate response text.
            golden_reference: Gold standard reference text.

        Returns:
            Penalty multiplier in (0.0, 1.0].
        """
        if not golden_reference or not golden_reference.strip():
            return 1.0

        resp_len = max(len(response.split()), 1)
        gold_len = max(len(golden_reference.split()), 1)
        r_l = resp_len / gold_len

        if r_l <= self.len_tau:
            return 1.0
        return float(math.exp(-self.len_lambda * (r_l - self.len_tau)))

    def compute_composite_score(
        self,
        accuracy: float,
        faithfulness: float,
        clarity: float,
        conciseness: float,
        length_penalty: float = 1.0,
    ) -> tuple[float, float]:
        """Compute convex composite raw score and length-calibrated score in [1.0, 5.0]."""
        raw = (
            self.weights["accuracy"] * accuracy
            + self.weights["faithfulness"] * faithfulness
            + self.weights["clarity"] * clarity
            + self.weights["conciseness"] * conciseness
        )
        raw_clamped = max(1.0, min(5.0, raw))
        calibrated = max(1.0, min(5.0, raw_clamped * length_penalty))
        return raw_clamped, calibrated

    # -------------------------------------------------------------------------
    # Prompt Formatting & Parsing
    # -------------------------------------------------------------------------

    def format_single_prompt(
        self,
        prompt: str,
        response: str,
        golden_reference: str = "",
        context: str = "",
        required_keywords: Optional[list[str]] = None,
    ) -> str:
        """Format structured Chain-of-Thought ChatML prompt for single evaluation."""
        keywords_str = ", ".join(required_keywords) if required_keywords else "N/A"
        return (
            "<|im_start|>system\n"
            "You are an expert impartial AI evaluator. Rate the assistant's response on a 1.0 to 5.0 scale "
            "across four criteria: Accuracy, Faithfulness, Clarity, and Conciseness.\n"
            "Output your evaluation strictly as a valid JSON object with the following schema:\n"
            "{\n"
            '  "rationale": "<step-by-step reasoning>",\n'
            '  "accuracy": <float 1.0-5.0>,\n'
            '  "faithfulness": <float 1.0-5.0>,\n'
            '  "clarity": <float 1.0-5.0>,\n'
            '  "conciseness": <float 1.0-5.0>\n'
            "}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"[User Prompt]\n{prompt}\n\n"
            f"[Reference Context]\n{context or 'No explicit context provided.'}\n\n"
            f"[Golden Target Answer]\n{golden_reference or 'No golden target provided.'}\n\n"
            f"[Mandatory Key Terms]\n{keywords_str}\n\n"
            f"[Candidate Response to Evaluate]\n{response}\n\n"
            "Provide your evaluation JSON now:\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def format_pairwise_prompt(
        self,
        prompt: str,
        cand1_text: str,
        cand2_text: str,
        golden_reference: str = "",
        context: str = "",
    ) -> str:
        """Format structured Chain-of-Thought ChatML prompt for pairwise comparison."""
        return (
            "<|im_start|>system\n"
            "You are an expert impartial AI evaluator. Compare Candidate 1 and Candidate 2 on a 1.0 to 5.0 scale "
            "across four criteria: Accuracy, Faithfulness, Clarity, and Conciseness.\n"
            "Output your evaluation strictly as a valid JSON object with the following schema:\n"
            "{\n"
            '  "rationale": "<comparative reasoning>",\n'
            '  "candidate_1": {\n'
            '    "accuracy": <float 1.0-5.0>,\n'
            '    "faithfulness": <float 1.0-5.0>,\n'
            '    "clarity": <float 1.0-5.0>,\n'
            '    "conciseness": <float 1.0-5.0>\n'
            "  },\n"
            '  "candidate_2": {\n'
            '    "accuracy": <float 1.0-5.0>,\n'
            '    "faithfulness": <float 1.0-5.0>,\n'
            '    "clarity": <float 1.0-5.0>,\n'
            '    "conciseness": <float 1.0-5.0>\n'
            "  }\n"
            "}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"[User Prompt]\n{prompt}\n\n"
            f"[Reference Context]\n{context or 'No explicit context provided.'}\n\n"
            f"[Golden Target Answer]\n{golden_reference or 'No golden target provided.'}\n\n"
            f"[Candidate 1]\n{cand1_text}\n\n"
            f"[Candidate 2]\n{cand2_text}\n\n"
            "Provide your evaluation JSON now:\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def parse_single_output(self, output: str) -> tuple[float, float, float, float, str]:
        """Robustly extract scores and rationale from model output with regex fallback.

        Returns:
            (accuracy, faithfulness, clarity, conciseness, rationale)
        """
        # 1. Try standard JSON parsing
        try:
            # Extract JSON substring if wrapped in markdown code fence
            json_text = output.strip()
            fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, re.DOTALL)
            if fence_match:
                json_text = fence_match.group(1)
            else:
                brace_match = re.search(r"\{.*\}", output, re.DOTALL)
                if brace_match:
                    json_text = brace_match.group(0)

            data = json.loads(json_text)
            acc = float(data.get("accuracy", 3.0))
            fai = float(data.get("faithfulness", 3.0))
            cla = float(data.get("clarity", 3.0))
            con = float(data.get("conciseness", 3.0))
            rat = str(data.get("rationale", ""))
            return (
                max(1.0, min(5.0, acc)),
                max(1.0, min(5.0, fai)),
                max(1.0, min(5.0, cla)),
                max(1.0, min(5.0, con)),
                rat,
            )
        except Exception:
            pass

        # 2. Defensive regex fallback
        def _find_val(dim: str) -> float:
            pat = rf'"{dim}"\s*:\s*([0-9]+(?:\.[0-9]+)?)'
            m = re.search(pat, output, re.IGNORECASE)
            if not m:
                pat2 = rf'{dim}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)'
                m = re.search(pat2, output, re.IGNORECASE)
            if m:
                try:
                    return max(1.0, min(5.0, float(m.group(1))))
                except ValueError:
                    return 3.0
            return 3.0

        acc = _find_val("accuracy")
        fai = _find_val("faithfulness")
        cla = _find_val("clarity")
        con = _find_val("conciseness")
        rat = output.strip()
        return acc, fai, cla, con, rat

    def parse_pairwise_output(
        self, output: str
    ) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float], str]:
        """Robustly parse pairwise model output for candidate 1 and candidate 2."""
        # 1. Try JSON parsing
        try:
            fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, re.DOTALL)
            json_text = fence_match.group(1) if fence_match else output
            if not fence_match:
                brace_match = re.search(r"\{.*\}", output, re.DOTALL)
                if brace_match:
                    json_text = brace_match.group(0)

            data = json.loads(json_text)
            c1 = data.get("candidate_1", {})
            c2 = data.get("candidate_2", {})
            s1 = (
                max(1.0, min(5.0, float(c1.get("accuracy", 3.0)))),
                max(1.0, min(5.0, float(c1.get("faithfulness", 3.0)))),
                max(1.0, min(5.0, float(c1.get("clarity", 3.0)))),
                max(1.0, min(5.0, float(c1.get("conciseness", 3.0)))),
            )
            s2 = (
                max(1.0, min(5.0, float(c2.get("accuracy", 3.0)))),
                max(1.0, min(5.0, float(c2.get("faithfulness", 3.0)))),
                max(1.0, min(5.0, float(c2.get("clarity", 3.0)))),
                max(1.0, min(5.0, float(c2.get("conciseness", 3.0)))),
            )
            return s1, s2, str(data.get("rationale", ""))
        except Exception:
            pass

        # 2. Fallback to single output parser
        s_single = self.parse_single_output(output)
        s1 = (s_single[0], s_single[1], s_single[2], s_single[3])
        return s1, s1, output.strip()

    # -------------------------------------------------------------------------
    # Deterministic Heuristic Fallback Engine
    # -------------------------------------------------------------------------

    def evaluate_heuristic_single(
        self,
        prompt: str,
        response: str,
        golden_reference: str = "",
        context: str = "",
        required_keywords: Optional[list[str]] = None,
    ) -> JudgeScore:
        """Deterministically score a response across the 4-dimensional rubric.

        Used when model_fn is None or for offline deterministic regression testing.
        """
        if not response or not response.strip():
            return JudgeScore(
                accuracy=1.0,
                faithfulness=1.0,
                clarity=1.0,
                conciseness=1.0,
                raw_composite=1.0,
                calibrated_score=1.0,
                length_penalty=1.0,
                rationale="Empty response evaluated with minimum score.",
                mode="heuristic",
            )

        resp_clean = response.strip()

        # 1. Accuracy: Keyword coverage & Jaccard token overlap against golden reference
        kw_coverage = 1.0
        if required_keywords:
            lowered = resp_clean.lower()
            matched = 0
            for kw in required_keywords:
                kw_low = kw.lower()
                kw_words = re.findall(r"\w+", kw_low)
                if kw_low in lowered or (kw_words and all(w in lowered for w in kw_words)):
                    matched += 1
            kw_coverage = matched / len(required_keywords)

        jaccard = 1.0
        if golden_reference:
            toks_resp = set(re.findall(r"\w+", resp_clean.lower()))
            toks_gold = set(re.findall(r"\w+", golden_reference.lower()))
            if toks_resp and toks_gold:
                raw_jaccard = len(toks_resp & toks_gold) / len(toks_resp | toks_gold)
                jaccard = min(1.0, raw_jaccard / 0.35)

        accuracy_norm = 0.70 * kw_coverage + 0.30 * jaccard
        accuracy = max(1.0, min(5.0, 1.0 + 4.0 * accuracy_norm))

        # 2. Faithfulness: CitationVerifier sentence-level entailment
        if context and context.strip():
            metrics = self.citation_verifier.verify_response(
                response=resp_clean,
                ground_truth_context=context,
            )
            faithfulness_norm = metrics.faithfulness_score
        else:
            # When context is not specified, faithfulness aligns with golden alignment
            faithfulness_norm = accuracy_norm
        faithfulness = max(1.0, min(5.0, 1.0 + 4.0 * faithfulness_norm))

        # 3. Clarity: Repetition penalty & formatting structure
        words = resp_clean.split()
        if len(words) >= 6:
            trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
            rep_ratio = 1.0 - (len(set(trigrams)) / max(len(trigrams), 1))
            clarity = max(1.0, min(5.0, 5.0 - 4.0 * rep_ratio))
        else:
            clarity = 4.0

        # 4. Conciseness: Monotonic length ratio adherence
        gold_len = max(len(golden_reference.split()), 1) if golden_reference else max(len(words), 1)
        r_l = len(words) / gold_len
        if r_l <= 1.2:
            conciseness = 5.0
        elif r_l <= self.len_tau:
            conciseness = max(1.0, 5.0 - 2.0 * (r_l - 1.2))
        else:
            conciseness = max(1.0, 3.8 - 2.0 * (r_l - self.len_tau))

        # Length Penalty & Calibration
        len_pen = self.compute_length_penalty(resp_clean, golden_reference)
        raw_comp, calib = self.compute_composite_score(
            accuracy, faithfulness, clarity, conciseness, len_pen
        )

        rationale = (
            f"Deterministic heuristic evaluation: Acc={accuracy:.2f} (kw={kw_coverage:.2f}, "
            f"jaccard={jaccard:.2f}), Faith={faithfulness:.2f}, Clar={clarity:.2f}, "
            f"Conc={conciseness:.2f}, LenPen={len_pen:.3f}."
        )

        return JudgeScore(
            accuracy=accuracy,
            faithfulness=faithfulness,
            clarity=clarity,
            conciseness=conciseness,
            raw_composite=raw_comp,
            calibrated_score=calib,
            length_penalty=len_pen,
            rationale=rationale,
            mode="heuristic",
        )

    # -------------------------------------------------------------------------
    # Public Evaluation API
    # -------------------------------------------------------------------------

    def evaluate_single(
        self,
        prompt: str,
        response: str,
        golden_reference: str = "",
        context: str = "",
        required_keywords: Optional[list[str]] = None,
    ) -> JudgeScore:
        """Evaluate a single response against golden domain reference and context."""
        if not response or not response.strip():
            return JudgeScore(
                accuracy=1.0,
                faithfulness=1.0,
                clarity=1.0,
                conciseness=1.0,
                raw_composite=1.0,
                calibrated_score=1.0,
                length_penalty=1.0,
                rationale="Empty response.",
                mode="empty",
            )

        if self.model_fn is None:
            return self.evaluate_heuristic_single(
                prompt=prompt,
                response=response,
                golden_reference=golden_reference,
                context=context,
                required_keywords=required_keywords,
            )

        # Generative model evaluation
        prompt_text = self.format_single_prompt(
            prompt=prompt,
            response=response,
            golden_reference=golden_reference,
            context=context,
            required_keywords=required_keywords,
        )
        try:
            model_out = self.model_fn(prompt_text)
            acc, fai, cla, con, rationale = self.parse_single_output(model_out)
        except Exception as e:
            logger.warning(f"LLMJudge model invocation failed ({e}). Falling back to heuristic.")
            return self.evaluate_heuristic_single(
                prompt=prompt,
                response=response,
                golden_reference=golden_reference,
                context=context,
                required_keywords=required_keywords,
            )

        len_pen = self.compute_length_penalty(response, golden_reference)
        raw_comp, calib = self.compute_composite_score(acc, fai, cla, con, len_pen)

        return JudgeScore(
            accuracy=acc,
            faithfulness=fai,
            clarity=cla,
            conciseness=con,
            raw_composite=raw_comp,
            calibrated_score=calib,
            length_penalty=len_pen,
            rationale=rationale,
            mode="model",
        )

    def evaluate_pair(
        self,
        prompt: str,
        response_a: str,
        response_b: str,
        golden_reference: str = "",
        context: str = "",
        required_keywords: Optional[list[str]] = None,
    ) -> PairwiseJudgeResult:
        """Perform symmetric two-pass position-swap pairwise evaluation between A and B."""
        # Calculate individual length penalties
        len_pen_a = self.compute_length_penalty(response_a, golden_reference)
        len_pen_b = self.compute_length_penalty(response_b, golden_reference)

        if self.model_fn is None:
            # Deterministic heuristic engine evaluates both candidates symmetrically
            score_a_obj = self.evaluate_heuristic_single(
                prompt=prompt,
                response=response_a,
                golden_reference=golden_reference,
                context=context,
                required_keywords=required_keywords,
            )
            score_b_obj = self.evaluate_heuristic_single(
                prompt=prompt,
                response=response_b,
                golden_reference=golden_reference,
                context=context,
                required_keywords=required_keywords,
            )

            s_a = score_a_obj.calibrated_score
            s_b = score_b_obj.calibrated_score
            delta = s_a - s_b

            if delta > self.tie_margin:
                winner = "A"
            elif delta < -self.tie_margin:
                winner = "B"
            else:
                winner = "TIE"

            return PairwiseJudgeResult(
                winner=winner,
                score_a=s_a,
                score_b=s_b,
                score_delta=delta,
                position_inconsistency=0.0,
                pass1_winner=winner,
                pass2_winner=winner,
                score_a_pass1=s_a,
                score_a_pass2=s_a,
                score_b_pass1=s_b,
                score_b_pass2=s_b,
                rationale=f"Heuristic comparison: A={s_a:.3f}, B={s_b:.3f}, Delta={delta:.3f}",
            )

        # Pass 1: Order (A, B) -> Candidate 1 = A, Candidate 2 = B
        prompt_pass1 = self.format_pairwise_prompt(
            prompt=prompt,
            cand1_text=response_a,
            cand2_text=response_b,
            golden_reference=golden_reference,
            context=context,
        )
        try:
            out_p1 = self.model_fn(prompt_pass1)
            (acc_a1, fai_a1, cla_a1, con_a1), (acc_b1, fai_b1, cla_b1, con_b1), rat1 = (
                self.parse_pairwise_output(out_p1)
            )
        except Exception as e:
            logger.warning(f"Pass 1 model invocation failed ({e}). Falling back to heuristic.")
            return self.evaluate_pair(
                prompt=prompt,
                response_a=response_a,
                response_b=response_b,
                golden_reference=golden_reference,
                context=context,
                required_keywords=required_keywords,
            )

        _, cal_a1 = self.compute_composite_score(acc_a1, fai_a1, cla_a1, con_a1, len_pen_a)
        _, cal_b1 = self.compute_composite_score(acc_b1, fai_b1, cla_b1, con_b1, len_pen_b)
        pass1_win = "A" if (cal_a1 - cal_b1) > self.tie_margin else ("B" if (cal_b1 - cal_a1) > self.tie_margin else "TIE")

        # Pass 2: Order (B, A) [Position-Swap] -> Candidate 1 = B, Candidate 2 = A
        prompt_pass2 = self.format_pairwise_prompt(
            prompt=prompt,
            cand1_text=response_b,
            cand2_text=response_a,
            golden_reference=golden_reference,
            context=context,
        )
        try:
            out_p2 = self.model_fn(prompt_pass2)
            (acc_b2, fai_b2, cla_b2, con_b2), (acc_a2, fai_a2, cla_a2, con_a2), rat2 = (
                self.parse_pairwise_output(out_p2)
            )
        except Exception as e:
            logger.warning(f"Pass 2 model invocation failed ({e}). Falling back to heuristic.")
            return self.evaluate_pair(
                prompt=prompt,
                response_a=response_a,
                response_b=response_b,
                golden_reference=golden_reference,
                context=context,
                required_keywords=required_keywords,
            )

        _, cal_a2 = self.compute_composite_score(acc_a2, fai_a2, cla_a2, con_a2, len_pen_a)
        _, cal_b2 = self.compute_composite_score(acc_b2, fai_b2, cla_b2, con_b2, len_pen_b)
        pass2_win = "A" if (cal_a2 - cal_b2) > self.tie_margin else ("B" if (cal_b2 - cal_a2) > self.tie_margin else "TIE")

        # Position-Swap Debiased Averages
        score_a_deb = (cal_a1 + cal_a2) / 2.0
        score_b_deb = (cal_b1 + cal_b2) / 2.0
        delta_deb = score_a_deb - score_b_deb

        # Position Inconsistency Coefficient kappa_pos in [0.0, 1.0]
        # Max theoretical difference is 4.0 (5.0 - 1.0)
        kappa_pos = 0.5 * (abs(cal_a1 - cal_a2) + abs(cal_b1 - cal_b2)) / 4.0

        if delta_deb > self.tie_margin:
            winner = "A"
        elif delta_deb < -self.tie_margin:
            winner = "B"
        else:
            winner = "TIE"

        if kappa_pos > 0.25:
            logger.warning(
                f"LLMJudge high position inconsistency detected: kappa_pos={kappa_pos:.3f} "
                f"(Pass1: {pass1_win}, Pass2: {pass2_win})."
            )

        combined_rationale = (
            f"Pass 1 Order (A,B): ScoreA={cal_a1:.2f}, ScoreB={cal_b1:.2f} [{rat1}]. "
            f"Pass 2 Order (B,A): ScoreA={cal_a2:.2f}, ScoreB={cal_b2:.2f} [{rat2}]."
        )

        return PairwiseJudgeResult(
            winner=winner,
            score_a=score_a_deb,
            score_b=score_b_deb,
            score_delta=delta_deb,
            position_inconsistency=kappa_pos,
            pass1_winner=pass1_win,
            pass2_winner=pass2_win,
            score_a_pass1=cal_a1,
            score_a_pass2=cal_a2,
            score_b_pass1=cal_b1,
            score_b_pass2=cal_b2,
            rationale=combined_rationale,
        )

    def evaluate_benchmark_case(
        self,
        case: GoldenTestCase,
        candidate_response: str,
        baseline_response: Optional[str] = None,
    ) -> Union[JudgeScore, PairwiseJudgeResult]:
        """Directly evaluate a GoldenTestCase instance from Phase 4.1.

        Args:
            case: GoldenTestCase with prompt, golden reference, context, and keywords.
            candidate_response: The new candidate model generation to evaluate.
            baseline_response: Optional baseline model generation for pairwise comparison.

        Returns:
            JudgeScore if evaluating single response, or PairwiseJudgeResult if comparing.
        """
        prompt = getattr(case, "question", getattr(case, "prompt", ""))
        golden_ref = getattr(case, "reference_answer", getattr(case, "golden_reference", ""))
        context = getattr(case, "ground_truth_context", getattr(case, "context_document", ""))
        keywords = getattr(case, "golden_keywords", getattr(case, "required_keywords", None))

        if baseline_response is not None:
            return self.evaluate_pair(
                prompt=prompt,
                response_a=candidate_response,
                response_b=baseline_response,
                golden_reference=golden_ref,
                context=context,
                required_keywords=keywords,
            )
        return self.evaluate_single(
            prompt=prompt,
            response=candidate_response,
            golden_reference=golden_ref,
            context=context,
            required_keywords=keywords,
        )
