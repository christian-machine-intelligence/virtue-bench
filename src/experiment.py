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


def extract_score(log: EvalLog, detailed: bool = False) -> dict:
    """Pull accuracy and metadata from an EvalLog.

    If detailed=True, also extract per-sample model responses and rationales.
    """
    results = log.results
    metrics = {}
    if results and results.scores:
        for score in results.scores:
            metrics.update({k: v.value for k, v in score.metrics.items()})
    result = {
        "model": str(log.eval.model),
        "accuracy": metrics.get("accuracy", None),
        "stderr": metrics.get("stderr", None),
        "samples": log.eval.dataset.samples if log.eval.dataset else None,
        "status": str(log.status),
    }

    if detailed and log.samples:
        sample_details = []
        for sample in log.samples:
            # Extract the prompt text
            prompt = ""
            if sample.input:
                if isinstance(sample.input, str):
                    prompt = sample.input
                elif isinstance(sample.input, list):
                    prompt = " ".join(
                        m.text for m in sample.input
                        if hasattr(m, "text") and m.text
                    )

            # Extract model response
            model_response = ""
            if sample.output and sample.output.completion:
                model_response = sample.output.completion.strip()

            # Extract score details
            answer = None
            correct = None
            explanation = None
            if sample.scores:
                for scorer_name, score_val in sample.scores.items():
                    answer = score_val.answer
                    correct = score_val.value == "C"
                    explanation = score_val.explanation
                    break

            detail = {
                "id": sample.id,
                "prompt": prompt,
                "target": sample.target,
                "model_response": model_response,
                "model_answer": answer,
                "correct": correct,
                "explanation": explanation,
                "metadata": sample.metadata if sample.metadata else {},
            }
            sample_details.append(detail)
        result["sample_details"] = sample_details

    return result


def run_condition(
    virtue: str,
    model: str,
    system_prompt: str | None,
    limit: int | None,
    seed: int,
    log_dir: str,
    condition_label: str,
    detailed: bool = False,
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
    result = extract_score(log, detailed=detailed)
    result["virtue"] = virtue
    result["condition"] = condition_label
    return result


def run_experiment(
    virtues: list[str],
    models: list[str],
    injection_text: str | None = None,
    limit: int | None = None,
    seed: int = 42,
    detailed: bool = False,
    output_name: str | None = None,
) -> list[dict]:
    """Run full experiment across virtues and models.

    If injection_text is provided, runs A/B: vanilla vs injected.
    Otherwise, runs vanilla only.
    If detailed is True, includes per-sample model responses and rationales.
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
                detailed=detailed,
            )
            all_results.append(result_a)
            print(f"  Accuracy: {result_a['accuracy']}")

            # Injected (if text provided)
            if injection_text:
                print(f"\n--- Injected ---")
                result_b = run_condition(
                    virtue, model, injection_text, limit, seed, log_dir, "injected",
                    detailed=detailed,
                )
                all_results.append(result_b)
                print(f"  Accuracy: {result_b['accuracy']}")

                if result_a["accuracy"] is not None and result_b["accuracy"] is not None:
                    delta = result_b["accuracy"] - result_a["accuracy"]
                    sign = "+" if delta >= 0 else ""
                    print(f"\n  Delta: {sign}{delta:.4f}")

    # Save results
    filename = output_name or f"results_{timestamp}"
    if not filename.endswith(".json"):
        filename += ".json"
    results_file = RESULTS_DIR / filename
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
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for results JSON (saved in results/)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include per-sample model responses and rationales in output",
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
        detailed=args.detailed,
        output_name=args.output,
    )

    print_comparison_table(results)
    if injection_text:
        print_delta_table(results)
