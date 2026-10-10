# GAP-08: Statistical Release Gate Decision Loopholes & Flaws

## 1. Landmark Academic Literature & Foundations

1. **Efron, B. (1979)**. *Bootstrap Methods: Another Look at the Jackknife*. The Annals of Statistics.
2. **McNemar, Q. (1947)**. *Note on the sampling error of the difference between correlated proportions or percentages*. Psychometrika.
3. **Edwards, A. L. (1948)**. *Note on the "correction for continuity" in testing the significance of the difference between correlated proportions*. Psychometrika.
4. **Wilcoxon, F. (1945)**. *Individual comparisons by ranking methods*. Biometrics Bulletin.
5. **Dror, R., Baumer, G., Shlomov, S., & Reichart, R. (2018)**. *The Hitchhiker’s Guide to Testing Statistical Significance in Natural Language Processing*. ACL 2018.

---

## 2. Exhaustive Audit of Gate Decision Rules

In [`evaluation/statistical_gating.py`](file:///Users/jayesh/Documents/RLHF/evaluation/statistical_gating.py#L384-L485), the `evaluate_gate` method assesses candidate performance across a golden benchmark suite. Our audit identified **six critical flaws and vulnerabilities** in this logic.

---

### Flaw 1: Edge-Case Delta Bounds Check is Completely Ignored in Decisions

In lines 416–420:
```python
        # Criterion 4: Edge Case Delta Bounds
        if edge_delta >= -0.15:
            passed_checks.append(f"Edge Case Delta Bounds: PASSED (delta={edge_delta:+.3f} >= -0.15).")
        else:
            violated_checks.append(f"Edge Case Delta Bounds: FAILED (delta={edge_delta:+.3f} < -0.15).")
```
When `edge_delta < -0.15`, the failure message is appended to `violated_checks`.

Now observe the actual decision logic in lines 450–475:
```python
        is_stat_sig_regress = (
            overall_boot.ci_upper < 0.0
            or (overall_hypo.wilcoxon_p < self.alpha and overall_boot.mean_delta < 0.0)
        )
        is_core_or_regr_violation = core_reg > 0 or regr_reg > 0 or core_delta < -self.parity_delta

        if is_core_or_regr_violation or is_stat_sig_regress:
            decision = GateDecision.TRIGGER_ROLLBACK
            ...
        elif ci_width > 0.80 or judge_inconsistency > 0.25:
            decision = GateDecision.ALERT_DRIFT
            ...
        elif (
            core_reg == 0
            and regr_reg == 0
            and adv_mean >= self.pass_threshold
            and (
                overall_boot.ci_lower > 0.0  # Statistically superior
                or (overall_boot.ci_lower >= -self.parity_delta and overall_boot.mean_delta >= 0.0)
            )
        ):
            decision = GateDecision.PROCEED_DEPLOY
```

**Where is `edge_delta` checked? Nowhere.**
Even if edge cases suffer catastrophic regressions (e.g. `edge_delta = -0.40`), if the core domain shows enough improvement to keep `overall_boot.ci_lower > 0`, the gate approves **`PROCEED_DEPLOY`**!
We proved this empirically in [`test_gate_decision_loopholes_gap.py`](file:///Users/jayesh/Documents/RLHF/actual_code_gaps/reproductions/test_gate_decision_loopholes_gap.py#L38-L56):
```
[Gate Gap Verification 1] Edge-case failure ignored:
  Decision: GateDecision.PROCEED_DEPLOY
  Violated checks: ['Edge Case Delta Bounds: FAILED (delta=-0.400 < -0.15).']
  Edge case mean delta: -0.40
```

---

### Flaw 2: McNemar Discordance Test is Never Used in Decisions

In lines 343–356:
```python
m_stat, m_p, b, c, t_p, t_f, r_ratio = compute_mcnemar_discordance(
    cand_list, base_list, tau_pass=self.pass_threshold
)
overall_hypo = HypothesisTestResult(
    wilcoxon_stat=w_stat,
    wilcoxon_p=w_p,
    mcnemar_stat=m_stat,
    mcnemar_p=m_p,
    ...
)
```
The Edwards continuity-corrected McNemar test evaluates whether the discordant proportions ($b$: candidate pass / base fail vs $c$: candidate fail / base pass) are statistically significant.
However, in lines 446–485, **neither `m_p` nor `m_stat` is ever checked**.
Even if the candidate has statistically significant regressions under the McNemar test ($p < 0.001$), the decision matrix ignores it.

---

### Flaw 3: Promotion of No-Op or Pure Noise Candidates

In lines 470–472:
```python
        (
            overall_boot.ci_lower > 0.0  # Statistically superior
            or (overall_boot.ci_lower >= -self.parity_delta and overall_boot.mean_delta >= 0.0)  # Non-inferior parity
        )
```
- If a candidate produces identical scores to the baseline (a no-op candidate):
  - `mean_delta = 0.0`
  - `ci_lower = 0.0 >= -0.05`
  - `core_reg = 0`, `regr_reg = 0`
  - Decision: **`PROCEED_DEPLOY`**!
- If a candidate's delta is $+0.001$ from random token jitter:
  - Decision: **`PROCEED_DEPLOY`**!
The gate actively promotes models that provide zero measurable benefit.

---

### Flaw 4: Insufficient Statistical Power with $N=30$ Benchmark Cases

The default golden benchmark suite contains exactly **30 cases** (12 Core Domain, 6 Edge Cases, 6 Adversarial Bait, 6 Regression Replay):
- In non-parametric bootstrap resampling on $N=30$, a sample variance of $\sigma \approx 0.5$ yields a standard error $\text{SE} \approx \frac{0.5}{\sqrt{30}} \approx 0.091$.
- The 95% confidence interval width is approximately $\pm 1.96 \times 0.091 \approx \pm 0.18$.
- When the confidence interval width is $\approx 0.36$, a parity threshold of `parity_delta = 0.05` cannot statistically distinguish genuine degradation from sample noise without a statistical power analysis ($\beta$-risk / Type II error).

---

### Flaw 5: Empty Bucket Vulnerability

In lines 385–386:
```python
core_reg = bucket_breakdown.get("core_domain", {}).get("regressions", 0)
regr_reg = bucket_breakdown.get("regression_replay", {}).get("regressions", 0)
```
If a custom or filtered benchmark suite has **zero test cases** in `core_domain` or `regression_replay`:
- `bucket_breakdown.get("core_domain")` is empty `{}`.
- `core_reg` defaults to `0`.
- The gate treats 0 test cases as "0 regressions" and passes the invariant!

---

### Flaw 6: Erroneous `TRIGGER_ROLLBACK` Semantics for Offline Candidates

In [`orchestration/loop_orchestrator.py`](file:///Users/jayesh/Documents/RLHF/orchestration/loop_orchestrator.py#L441-L447):
```python
        elif decision == GateDecision.TRIGGER_ROLLBACK.value:
            with self._model_lock:
                self.active_model = self.anchor_model
                self.model_version = 0  # Reverted to anchor
            logger.warning("ContinualLearningOrchestrator: TRIGGER_ROLLBACK -> Reverted to golden anchor baseline!")
            status = "ROLLED_BACK"
```
When an offline training worker evaluates an unreleased candidate model:
- The candidate was **never deployed**; it exists only in a scratch folder.
- If the candidate performs poorly (e.g. failing the gate), `evaluate_gate` returns `TRIGGER_ROLLBACK`.
- The orchestrator reacts by **rolling back the currently healthy active serving model to version 0**!
- An offline training experiment failure causes a production outage / rollback of the active serving model.
The appropriate response for an unreleased candidate failure is **`BLOCK_RELEASE`** (reject candidate, keep serving active model). `TRIGGER_ROLLBACK` should only be emitted during post-deployment production drift monitoring.

---

## 3. Note on Benchmark Case Count Discrepancy

A pattern match for `case_id=` in `evaluation/benchmark_suite.py` returns 32 lines. This is because:
- Line 24: `case_id: str` (type annotation)
- Line 37: `case_id=self.case_id` (serialization assignment)
- 30 instantiated `GoldenTestCase` definitions.
The actual loaded count is verified to be **exactly 30 cases**.

