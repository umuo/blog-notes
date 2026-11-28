"""Fixed-model Agent Harness evaluation lab inspired by FrontierHarness Eval.

Runs a deterministic, zero-dependency local simulation comparing 4 harness designs
across 2 base models on 4 refund-CSV engineering tasks (2 attempts each, 64 scored
trials total), plus reward-hacking and infrastructure-error handling checks.
"""

from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

# Unified pricing table (USD per 1,000,000 tokens, exact Decimal arithmetic)
MILLION = Decimal("1000000")
PRICE_UNCACHED_IN = Decimal("1.00") / MILLION
PRICE_CACHED_IN = Decimal("0.10") / MILLION  # 90% prompt-cache discount
PRICE_OUTPUT = Decimal("4.00") / MILLION

TASKS = (
    {"id": "refund-null-mean", "base_turns": 4, "hard": False},
    {"id": "refund-zero-guard", "base_turns": 5, "hard": False},
    {"id": "refund-currency-round", "base_turns": 6, "hard": True},
    {"id": "refund-approval-lock", "base_turns": 7, "hard": True},
)

# Two base models with distinct behavioral traits
MODELS = {
    "model-alpha": {
        "base_pass_easy": 0.90,
        "base_pass_hard": 0.65,
        "overconfident_rate": 0.75,
        "loop_prone": False,
    },
    "model-beta": {
        "base_pass_easy": 0.70,
        "base_pass_hard": 0.30,
        "overconfident_rate": 0.80,
        "loop_prone": True,
    },
}

# Four harness configurations isolating prefix stability, bloat, and verify gates
HARNESSES = {
    "H1_minimal_stable": {
        "static_prefix_tokens": 2000,
        "dynamic_prefix_breaks_cache": False,
        "turn_delta_tokens": 600,
        "output_per_turn": 350,
        "extra_turns": 0,
        "seconds_per_turn": 6.0,
        "verify_gate": False,
    },
    "H2_cache_breaker": {
        "static_prefix_tokens": 2000,
        "dynamic_prefix_breaks_cache": True,  # injects timestamp at prompt start
        "turn_delta_tokens": 600,
        "output_per_turn": 350,
        "extra_turns": 0,
        "seconds_per_turn": 7.5,  # slower prefill without prefix cache
        "verify_gate": False,
    },
    "H3_chatty_cached": {
        "static_prefix_tokens": 18000,  # huge static manual & 40 tool schemas
        "dynamic_prefix_breaks_cache": False,
        "turn_delta_tokens": 900,
        "output_per_turn": 600,
        "extra_turns": 4,
        "seconds_per_turn": 8.0,
        "verify_gate": False,
    },
    "H4_verify_gate": {
        "static_prefix_tokens": 3200,
        "dynamic_prefix_breaks_cache": False,
        "turn_delta_tokens": 650,
        "output_per_turn": 380,
        "extra_turns": 1,  # runs local deterministic verifier before exit
        "seconds_per_turn": 6.2,
        "verify_gate": True,
    },
}


def deterministic_outcome(model_name: str, harness_name: str, task: dict, attempt: int) -> tuple[bool, str]:
    """Deterministic pass/fail outcome modeling Model x Harness interaction."""
    hard = task["hard"]
    if model_name == "model-alpha":
        if harness_name in ("H1_minimal_stable", "H2_cache_breaker"):
            # Easy: 4/4 pass, Hard: 2/4 pass -> 6/8 (75.0%)
            passed = (not hard) or (attempt == 0)
        elif harness_name == "H3_chatty_cached":
            # Long noisy context hurts hard tasks -> 4/8 (50.0%)
            passed = not hard
        else:  # H4_verify_gate
            # Catches 1 extra hard attempt -> 7/8 (87.5%), but costs extra prefix+turn
            passed = (not hard) or (attempt == 0) or (task["id"] == "refund-currency-round")
    else:  # model-beta (prone to confident premature exit without verify gate)
        if harness_name in ("H1_minimal_stable", "H2_cache_breaker"):
            # Easy: 2/4 pass, Hard: 1/4 pass -> 3/8 (37.5%)
            passed = (attempt == 0 and not hard) or (task["id"] == "refund-currency-round" and attempt == 0)
        elif harness_name == "H3_chatty_cached":
            # Bloated context collapses hard tasks -> 2/8 (25.0%)
            passed = (attempt == 0 and not hard)
        else:  # H4_verify_gate rescues loop-prone model-beta -> 7/8 (87.5%)
            passed = (not hard) or (attempt == 0) or (task["id"] == "refund-currency-round")

    if passed:
        return True, "verified_pass"
    # Most failures end as confident wrong answers; a minority hit timeout
    if task["id"] == "refund-approval-lock" and attempt == 1:
        return False, "timeout"
    return False, "confident_wrong_exit"


def simulate_trial(model_name: str, harness_name: str, task: dict, attempt: int) -> dict:
    cfg = HARNESSES[harness_name]
    passed, exit_reason = deterministic_outcome(model_name, harness_name, task, attempt)
    turns = task["base_turns"] + cfg["extra_turns"] + (0 if passed else 2)
    if exit_reason == "timeout":
        turns += 3

    total_cached = 0
    total_uncached = 0
    total_output = 0
    history_tokens = 0

    for turn in range(turns):
        prompt_tokens = cfg["static_prefix_tokens"] + history_tokens
        if turn == 0 or cfg["dynamic_prefix_breaks_cache"]:
            cached = 0
            uncached = prompt_tokens
        else:
            # Everything already sent in prior turns hits provider prefix cache;
            # only the new observation/output appended this turn is uncached.
            prev_prompt = cfg["static_prefix_tokens"] + (history_tokens - cfg["turn_delta_tokens"])
            cached = prev_prompt
            uncached = prompt_tokens - cached

        total_cached += cached
        total_uncached += uncached
        total_output += cfg["output_per_turn"]
        history_tokens += cfg["turn_delta_tokens"]

    cost_usd = float(
        Decimal(total_uncached) * PRICE_UNCACHED_IN
        + Decimal(total_cached) * PRICE_CACHED_IN
        + Decimal(total_output) * PRICE_OUTPUT
    )
    duration_s = round(turns * cfg["seconds_per_turn"], 2)
    return {
        "model": model_name,
        "harness": harness_name,
        "task_id": task["id"],
        "attempt": attempt,
        "turns": turns,
        "passed": passed,
        "exit_reason": exit_reason,
        "cached_input_tokens": total_cached,
        "uncached_input_tokens": total_uncached,
        "output_tokens": total_output,
        "cost_usd": round(cost_usd, 6),
        "duration_s": duration_s,
    }


def score_cell(trials: list[dict]) -> dict:
    n = len(trials)
    passes = sum(1 for t in trials if t["passed"])
    total_cost = sum(Decimal(str(t["cost_usd"])) for t in trials)
    failed_cost = sum(Decimal(str(t["cost_usd"])) for t in trials if not t["passed"])
    total_time = sum(Decimal(str(t["duration_s"])) for t in trials)
    total_cached = sum(t["cached_input_tokens"] for t in trials)
    total_input = sum(t["cached_input_tokens"] + t["uncached_input_tokens"] for t in trials)
    failed_trials = [t for t in trials if not t["passed"]]
    confident_failures = sum(1 for t in failed_trials if t["exit_reason"] == "confident_wrong_exit")

    return {
        "trials": n,
        "passes": passes,
        "pass_rate": round(passes / n, 4),
        "cost_per_task_usd": round(float(total_cost / n), 5),
        "cost_per_pass_usd": round(float(total_cost / passes), 5) if passes else None,
        "cache_hit_rate": round(total_cached / total_input, 4) if total_input else 0.0,
        "time_per_pass_s": round(float(total_time / passes), 2) if passes else None,
        "failed_spend_share": round(float(failed_cost / total_cost), 4) if total_cost else 0.0,
        "confident_failure_share": (
            round(confident_failures / len(failed_trials), 4) if failed_trials else 0.0
        ),
    }


def audit_special_cases() -> dict:
    """Verify anti-reward-hacking zeroing and infra-error exclusion."""
    raw_records = [
        {"id": "probe-ok", "verifier_pass": True, "edited_tests": False, "infra_error": None},
        {"id": "probe-hack", "verifier_pass": True, "edited_tests": True, "infra_error": None},
        {"id": "probe-infra", "verifier_pass": False, "edited_tests": False, "infra_error": "HTTP_503"},
    ]
    scored = []
    excluded_infra = []
    flagged_hacks = []
    for rec in raw_records:
        if rec["infra_error"]:
            excluded_infra.append(rec["id"])
            continue
        if rec["edited_tests"]:
            flagged_hacks.append(rec["id"])
            scored.append(0)
            continue
        scored.append(1 if rec["verifier_pass"] else 0)
    return {
        "scored_count": len(scored),
        "verified_passes": sum(scored),
        "pass_rate": round(sum(scored) / len(scored), 4),
        "flagged_reward_hacks": flagged_hacks,
        "excluded_infra_errors": excluded_infra,
    }


def main() -> None:
    all_trials = [
        simulate_trial(model, harness, task, attempt)
        for model in MODELS
        for harness in HARNESSES
        for task in TASKS
        for attempt in range(2)
    ]

    by_model: dict[str, dict[str, dict]] = {}
    for model in MODELS:
        cells = {}
        for harness in HARNESSES:
            subset = [t for t in all_trials if t["model"] == model and t["harness"] == harness]
            cells[harness] = score_cell(subset)
        ranked = sorted(cells, key=lambda h: cells[h]["cost_per_pass_usd"])
        for rank, harness in enumerate(ranked, 1):
            cells[harness]["cost_rank"] = rank
        by_model[model] = cells

    overall_by_harness = {
        harness: score_cell([t for t in all_trials if t["harness"] == harness])
        for harness in HARNESSES
    }
    audit = audit_special_cases()

    # Core FrontierHarness-style invariants verified by assertions
    alpha = by_model["model-alpha"]
    beta = by_model["model-beta"]

    # 1. Model x Harness interaction flips the #1 cheapest-per-pass harness
    assert alpha["H1_minimal_stable"]["cost_rank"] == 1
    assert beta["H4_verify_gate"]["cost_rank"] == 1
    assert beta["H1_minimal_stable"]["cost_rank"] > 1

    # 2. Breaking prefix cache inflates cost_per_pass even at identical pass rate
    assert alpha["H1_minimal_stable"]["pass_rate"] == alpha["H2_cache_breaker"]["pass_rate"]
    assert alpha["H2_cache_breaker"]["cache_hit_rate"] == 0.0
    assert alpha["H2_cache_breaker"]["cost_per_pass_usd"] > alpha["H1_minimal_stable"]["cost_per_pass_usd"] * 1.8

    # 3. Highest cache hit rate does NOT imply lowest cost per pass
    best_cache_h = max(alpha, key=lambda h: alpha[h]["cache_hit_rate"])
    assert best_cache_h == "H3_chatty_cached"
    assert alpha["H3_chatty_cached"]["cost_per_pass_usd"] > alpha["H1_minimal_stable"]["cost_per_pass_usd"] * 2.5

    # 4. Reward hacking zeroed out & infra 503 excluded from denominator
    assert audit == {
        "scored_count": 2,
        "verified_passes": 1,
        "pass_rate": 0.5,
        "flagged_reward_hacks": ["probe-hack"],
        "excluded_infra_errors": ["probe-infra"],
    }

    output = {
        "total_scored_trials": len(all_trials),
        "overall_by_harness": overall_by_harness,
        "by_model": by_model,
        "audit_probes": audit,
    }
    out_path = Path(__file__).with_name("harness-eval-results.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
