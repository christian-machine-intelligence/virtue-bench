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
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from .data import load_virtue_csv, parse_answer, BASE_INSTRUCTION, VIRTUES


RESULTS_DIR = Path(__file__).parent.parent / "results"


async def query_claude(prompt: str, system_prompt: str, model: str, retries: int = 1) -> str:
    """Send a prompt to Claude via `claude -p` and return the response."""
    for attempt in range(1 + retries):
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "-p", "--model", model, "--system-prompt", system_prompt, "--no-session-persistence", "--setting-sources", "user",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(prompt.encode()), timeout=120)
        except asyncio.TimeoutError:
            if attempt < retries:
                continue
            return ""
        if proc.returncode != 0:
            if attempt < retries:
                continue
            return ""
        response = stdout.decode().strip()
        if response or attempt == retries:
            return response
    return ""


async def run_virtue(
    virtue: str,
    model: str,
    system_prompt: str,
    limit: int | None,
    seed: int,
    condition_label: str,
    trace: bool = False,
    concurrency: int = 5,
) -> dict:
    """Run all samples for a single virtue concurrently."""
    samples = load_virtue_csv(virtue, limit=limit, seed=seed)
    total = len(samples)
    sem = asyncio.Semaphore(concurrency)
    results = [None] * total

    async def process(i: int, sample):
        async with sem:
            response = await query_claude(sample.input, system_prompt, model)
        answer = parse_answer(response)
        is_correct = answer == sample.target
        results[i] = {
            "response": response,
            "answer": answer,
            "correct": is_correct,
            "sample": sample,
        }
        status = "correct" if is_correct else "incorrect"
        print(f"  [{i+1}/{total}] {status}", flush=True)

    await asyncio.gather(*(process(i, s) for i, s in enumerate(samples)))

    correct = sum(1 for r in results if r["correct"])
    trace_data = []
    if trace:
        for i, r in enumerate(results):
            trace_data.append({
                "id": i + 1,
                "prompt": r["sample"].input,
                "target": r["sample"].target,
                "model_response": r["response"],
                "model_answer": r["answer"],
                "correct": r["correct"],
                "explanation": r["response"],
                "metadata": r["sample"].metadata,
            })

    accuracy = correct / total if total > 0 else 0.0
    result = {
        "model": f"claude-cli/{model}",
        "accuracy": accuracy,
        "stderr": None,
        "samples": total,
        "status": "success",
        "virtue": virtue,
        "condition": condition_label,
    }
    if trace:
        result["sample_details"] = trace_data
    return result


async def run_experiment(
    virtues: list[str],
    model: str,
    injection_text: str | None = None,
    limit: int | None = None,
    seed: int = 42,
    trace: bool = False,
    concurrency: int = 5,
) -> list[dict]:
    """Run the full experiment across virtues."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for virtue in virtues:
        print(f"\n{'='*60}")
        print(f"Model: {model} (claude -p, concurrency={concurrency}) | Virtue: {virtue}")
        print(f"{'='*60}")

        print(f"\n--- Vanilla ---")
        result_a = await run_virtue(virtue, model, BASE_INSTRUCTION, limit, seed, "vanilla", trace, concurrency)
        all_results.append(result_a)
        print(f"  Accuracy: {result_a['accuracy']:.4f}")

        if injection_text:
            injected_prompt = injection_text + "\n\n---\n\n" + BASE_INSTRUCTION
            print(f"\n--- Injected ---")
            result_b = await run_virtue(virtue, model, injected_prompt, limit, seed, "injected", trace, concurrency)
            all_results.append(result_b)
            print(f"  Accuracy: {result_b['accuracy']:.4f}")

            if result_a["accuracy"] is not None and result_b["accuracy"] is not None:
                delta = result_b["accuracy"] - result_a["accuracy"]
                sign = "+" if delta >= 0 else ""
                print(f"\n  Delta: {sign}{delta:.4f}")

    virtues_label = "-".join(virtues)
    results_file = RESULTS_DIR / f"results_cli_{model}_{virtues_label}_{timestamp}.json"
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
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include per-sample answers, rationales, and correctness in results",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent claude -p calls (default: 5)",
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
    print(f"Concurrency: {args.concurrency}")
    print(f"Injection: {'yes' if injection_text else 'no'}")

    results = asyncio.run(run_experiment(
        virtues=virtues,
        model=args.model,
        injection_text=injection_text,
        limit=args.limit,
        seed=args.seed,
        trace=args.detailed,
        concurrency=args.concurrency,
    ))

    print(f"\n{'Model':<25} {'Virtue':<12} {'Condition':<10} {'Accuracy':>8} {'Samples':>8}")
    print("-" * 67)
    for r in results:
        acc = f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "N/A"
        print(f"{r.get('model',''):<25} {r.get('virtue',''):<12} {r.get('condition',''):<10} {acc:>8} {r.get('samples',''):>8}")


if __name__ == "__main__":
    main()
