"""
Standalone VirtueBench runner using Claude Code's pipe mode (claude -p).

This lets you run the benchmark with a Claude Max subscription instead of
needing an API key. Each sample is sent to `claude -p` via subprocess.

Usage:
    python -m src.run_cli                          # full benchmark
    python -m src.run_cli --quick                   # smoke test (10 per virtue)
    python -m src.run_cli --subset courage          # single virtue
    python -m src.run_cli --model opus              # specific model
    python -m src.run_cli --inject path/to/text.txt # A/B with injected text
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .tasks import load_virtue_csv, BASE_INSTRUCTION, VIRTUES
from .analysis import print_comparison_table, print_delta_table


RESULTS_DIR = Path(__file__).parent.parent / "results"


def query_claude(prompt: str, system_prompt: str, model: str) -> str:
    """Send a prompt to Claude via `claude -p` and return the response."""
    full_prompt = f"<system>\n{system_prompt}\n</system>\n\n{prompt}"
    result = subprocess.run(
        ["claude", "-p", "--model", model],
        input=full_prompt,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"  [error] claude -p failed: {result.stderr.strip()}", file=sys.stderr)
        return ""
    return result.stdout.strip()


def score_response(response: str, target: str) -> bool:
    """Extract the first 0 or 1 from the response and compare to target."""
    for ch in response:
        if ch in ("0", "1"):
            return ch == target
    return False


def run_virtue(
    virtue: str,
    model: str,
    system_prompt: str,
    limit: int | None,
    seed: int,
    condition_label: str,
) -> dict:
    """Run all samples for a single virtue and return a result dict."""
    samples = load_virtue_csv(virtue, limit=limit, seed=seed)
    correct = 0
    total = len(samples)

    for i, sample in enumerate(samples, 1):
        print(f"  [{i}/{total}] ", end="", flush=True)
        response = query_claude(sample.input, system_prompt, model)

        if score_response(response, sample.target):
            correct += 1
            print("correct")
        else:
            print("incorrect")

    accuracy = correct / total if total > 0 else 0.0
    return {
        "model": f"claude-cli/{model}",
        "accuracy": accuracy,
        "stderr": None,
        "samples": total,
        "status": "success",
        "virtue": virtue,
        "condition": condition_label,
    }


def run_experiment(
    virtues: list[str],
    model: str,
    injection_text: str | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> list[dict]:
    """Run the full experiment across virtues."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for virtue in virtues:
        print(f"\n{'='*60}")
        print(f"Model: {model} (claude -p) | Virtue: {virtue}")
        print(f"{'='*60}")

        # Vanilla
        print(f"\n--- Vanilla ---")
        result_a = run_virtue(virtue, model, BASE_INSTRUCTION, limit, seed, "vanilla")
        all_results.append(result_a)
        print(f"  Accuracy: {result_a['accuracy']:.4f}")

        # Injected (if text provided)
        if injection_text:
            injected_prompt = injection_text + "\n\n---\n\n" + BASE_INSTRUCTION
            print(f"\n--- Injected ---")
            result_b = run_virtue(virtue, model, injected_prompt, limit, seed, "injected")
            all_results.append(result_b)
            print(f"  Accuracy: {result_b['accuracy']:.4f}")

            if result_a["accuracy"] is not None and result_b["accuracy"] is not None:
                delta = result_b["accuracy"] - result_a["accuracy"]
                sign = "+" if delta >= 0 else ""
                print(f"\n  Delta: {sign}{delta:.4f}")

    # Save results
    results_file = RESULTS_DIR / f"results_cli_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run VirtueBench using Claude Code pipe mode (claude -p)"
    )
    parser.add_argument(
        "--subset",
        choices=VIRTUES + ["all"],
        default="all",
        help="Virtue subset to evaluate (default: all)",
    )
    parser.add_argument(
        "--model",
        default="sonnet",
        help="Claude model name for claude -p (default: sonnet)",
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

    print(f"Model: {args.model} (via claude -p)")
    print(f"Virtues: {virtues}")
    print(f"Limit: {args.limit or 'all'}")
    print(f"Injection: {'yes' if injection_text else 'no'}")

    # Check that claude CLI is available
    try:
        subprocess.run(["claude", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: 'claude' CLI not found. Install Claude Code first:", file=sys.stderr)
        print("  https://docs.anthropic.com/en/docs/claude-code", file=sys.stderr)
        sys.exit(1)

    results = run_experiment(
        virtues=virtues,
        model=args.model,
        injection_text=injection_text,
        limit=args.limit,
        seed=args.seed,
    )

    print_comparison_table(results)
    if injection_text:
        print_delta_table(results)


if __name__ == "__main__":
    main()
