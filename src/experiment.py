"""
Experiment runner for VirtueBench.

Evaluates models on the four cardinal virtues, optionally with text injected
into the system prompt (for A/B experiments).

Usage:
    python -m src                             # full benchmark, all virtues
    python -m src --quick                     # smoke test (10 samples per virtue)
    python -m src --subset prudence           # single virtue
    python -m src --inject path/to/text.txt   # A/B with injected text
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog

from .tasks import make_virtue_task, VIRTUES
from .analysis import print_comparison_table, print_delta_table


MODELS = [
    "anthropic/claude-sonnet-4-20250514",
    "openai/gpt-4o",
]

RESULTS_DIR = Path(__file__).parent.parent / "results"


def extract_score(log: EvalLog) -> dict:
    """Pull accuracy and metadata from an EvalLog."""
    results = log.results
    metrics = {}
    if results and results.scores:
        for score in results.scores:
            metrics.update({k: v.value for k, v in score.metrics.items()})
    return {
        "model": str(log.eval.model),
        "accuracy": metrics.get("accuracy", None),
        "stderr": metrics.get("stderr", None),
        "samples": log.eval.dataset.samples if log.eval.dataset else None,
        "status": str(log.status),
    }


def run_condition(
    virtue: str,
    model: str,
    system_prompt: str | None,
    limit: int | None,
    seed: int,
    log_dir: str,
    condition_label: str,
) -> dict:
    """Run a single experimental condition and return the result dict."""
    task = make_virtue_task(
        virtue,
        system_prompt=system_prompt,
        limit=limit,
        seed=seed,
    )

    logs = inspect_eval(
        task,
        model=model,
        log_dir=log_dir,
    )

    log = logs[0]
    result = extract_score(log)
    result["virtue"] = virtue
    result["condition"] = condition_label
    return result


def run_experiment(
    virtues: list[str],
    models: list[str],
    injection_text: str | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Run full experiment across virtues and models.

    If injection_text is provided, runs A/B: vanilla vs injected.
    Otherwise, runs vanilla only.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = str(RESULTS_DIR / "logs" / timestamp)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "logs").mkdir(exist_ok=True)

    all_results = []

    for model in models:
        for virtue in virtues:
            print(f"\n{'='*60}")
            print(f"Model: {model} | Virtue: {virtue}")
            print(f"{'='*60}")

            # Vanilla
            print(f"\n--- Vanilla ---")
            result_a = run_condition(
                virtue, model, None, limit, seed, log_dir, "vanilla",
            )
            all_results.append(result_a)
            print(f"  Accuracy: {result_a['accuracy']}")

            # Injected (if text provided)
            if injection_text:
                print(f"\n--- Injected ---")
                result_b = run_condition(
                    virtue, model, injection_text, limit, seed, log_dir, "injected",
                )
                all_results.append(result_b)
                print(f"  Accuracy: {result_b['accuracy']}")

                if result_a["accuracy"] is not None and result_b["accuracy"] is not None:
                    delta = result_b["accuracy"] - result_a["accuracy"]
                    sign = "+" if delta >= 0 else ""
                    print(f"\n  Delta: {sign}{delta:.4f}")

    # Save results
    results_file = RESULTS_DIR / f"results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run VirtueBench: cardinal virtues evaluation benchmark"
    )
    parser.add_argument(
        "--subset",
        choices=VIRTUES + ["all"],
        default="all",
        help="Virtue subset to evaluate (default: all)",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=MODELS,
        help="Model(s) to evaluate",
    )
    parser.add_argument(
        "--inject",
        type=str,
        default=None,
        help="Path to text file to inject in system prompt (enables A/B experiment)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for scenario shuffling",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of eval samples per virtue",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test: 10 samples per virtue",
    )

    args = parser.parse_args()

    if args.quick:
        args.limit = 10

    virtues = VIRTUES if args.subset == "all" else [args.subset]

    injection_text = None
    if args.inject:
        injection_text = Path(args.inject).read_text(encoding="utf-8")

    print(f"Models: {args.model}")
    print(f"Virtues: {virtues}")
    print(f"Limit: {args.limit or 'all'}")
    print(f"Injection: {'yes' if injection_text else 'no'}")

    results = run_experiment(
        virtues=virtues,
        models=args.model,
        injection_text=injection_text,
        limit=args.limit,
        seed=args.seed,
    )

    print_comparison_table(results)
    if injection_text:
        print_delta_table(results)
