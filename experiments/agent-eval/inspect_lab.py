"""Run the CSV workflow as a real Inspect AI task without model generation."""

import argparse
import json
from importlib.metadata import version

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer
from inspect_ai.solver import solver

from eval_lab import REQUEST, evaluate


@solver
def run_refund_workflow():
    async def solve(state, generate):
        evidence = evaluate(plan=state.metadata["plan"])
        state.store.set("evidence", evidence)
        return state

    return solve


@scorer(metrics=[accuracy()])
def verify_business_state():
    async def score(state, target):
        evidence = state.store.get("evidence")
        checks = evidence["checks"]
        return Score(
            value=CORRECT if all(checks.values()) else INCORRECT,
            answer=evidence["claim"],
            explanation=json.dumps(checks, sort_keys=True),
            metadata={"reported": evidence["report"], "archive_after": evidence["archive_after"]},
        )

    return score


@task
def refund_archive():
    return Task(
        dataset=[
            Sample(id="verified", input=REQUEST, target="C", metadata={"plan": "normal"}),
            Sample(id="claim-only", input=REQUEST, target="C", metadata={"plan": "claim_only"}),
        ],
        solver=run_refund_workflow(),
        scorer=verify_business_state(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()
    log, = eval(refund_archive(), model="mockllm/model", log_dir=args.log_dir, display="none")
    assert log.status == "success"
    samples = {sample.id: next(iter(sample.scores.values())).value for sample in log.samples}
    metric, = log.results.scores
    accuracy_value = metric.metrics["accuracy"].value
    assert samples == {"verified": "C", "claim-only": "I"}
    assert accuracy_value == 0.5 and not log.stats.model_usage
    print(json.dumps({
        "inspect_ai": version("inspect-ai"),
        "sample_scores": samples,
        "accuracy": accuracy_value,
        "model_usage": log.stats.model_usage,
    }, indent=4))


if __name__ == "__main__":
    main()
