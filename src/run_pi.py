"""
VirtueBench runner for OpenAI models using pi's print mode (pi -p).

Uses the openai-codex provider (ChatGPT Pro subscription) with all
tools, extensions, skills, and sessions disabled for clean eval.

Usage:
    python -m src.run_pi                          # full benchmark
    python -m src.run_pi --quick                   # smoke test (10 per virtue)
    python -m src.run_pi --subset courage          # single virtue
    python -m src.run_pi --model gpt-5.4           # specific model
    python -m src.run_pi --detailed                # write per-sample debug logs
"""

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from .data import load_virtue_csv, parse_answer, FRAMES, VIRTUES
from .result_artifacts import write_result_artifacts


RESULTS_DIR = Path(__file__).parent.parent / "results"
NEUTRAL_CWD = "/tmp"


async def query_pi(
    prompt: str,
    system_prompt: str,
    model: str,
    provider: str = "openai-codex",
    thinking: str = "off",
    retries: int = 2,
    timeout: int = 120,
) -> dict:
    """Send a prompt to an OpenAI model via `pi -p` and return outcome metadata."""
    last_error = "unknown"

    for attempt in range(1, retries + 2):
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "pi", "-p",
                "--provider", provider,
                "--model", model,
                "--system-prompt", system_prompt,
                "--no-tools",
                "--no-extensions",
                "--no-skills",
                "--no-session",
                "--no-prompt-templates",
                "--thinking", thinking,
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=NEUTRAL_CWD,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            if proc is not None:
                proc.kill()
                try:
                    await proc.communicate()
                except Exception:
                    pass
            last_error = "timeout"
            if attempt <= retries:
                continue
            break
        except FileNotFoundError:
            last_error = "pi_not_found"
            break
        except Exception as exc:
            last_error = f"spawn_error:{exc.__class__.__name__}"
            if attempt <= retries:
                continue
            break

        response = stdout.decode(errors="replace").strip()
        if proc.returncode != 0:
            last_error = "nonzero_exit"
            if attempt <= retries:
                continue
            break

        if not response:
            last_error = "blank_response"
            if attempt <= retries:
                continue
            break

        return {
            "response": response,
            "infra_error": None,
        }

    return {
        "response": "",
        "infra_error": last_error,
    }


async def run_virtue(
    virtue: str,
    model: str,
    provider: str,
    thinking: str,
    system_prompt: str,
    limit: int | None,
    seed: int,
    condition_label: str,
    trace: bool = False,
    concurrency: int = 5,
    retries: int = 2,
    timeout: int = 120,
) -> dict:
    """Run all samples for a single virtue concurrently."""
    samples = load_virtue_csv(virtue, limit=limit, seed=seed)
    total = len(samples)
    sem = asyncio.Semaphore(concurrency)
    results = [None] * total

    async def process(i: int, sample):
        async with sem:
            outcome = await query_pi(
                sample.input,
                system_prompt,
                model,
                provider=provider,
                thinking=thinking,
                retries=retries,
                timeout=timeout,
            )
        response = outcome["response"]
        answer = parse_answer(response) if outcome["infra_error"] is None else None
        is_correct = answer == sample.target if outcome["infra_error"] is None else None
        results[i] = {
            "response": response,
            "answer": answer,
            "correct": is_correct,
            "sample": sample,
            "infra_error": outcome["infra_error"],
        }
        if outcome["infra_error"] is not None:
            status = f"infra:{outcome['infra_error']}"
        else:
            status = "correct" if is_correct else "incorrect"
        print(f"  [{i+1}/{total}] {status}", flush=True)

    await asyncio.gather(*(process(i, s) for i, s in enumerate(samples)))

    scored_results = [r for r in results if r["infra_error"] is None]
    infra_results = [r for r in results if r["infra_error"] is not None]
    correct = sum(1 for r in scored_results if r["correct"])
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

    accuracy = correct / total if not infra_results and total > 0 else None
    result = {
        "model": f"pi/{model}" + (f":{thinking}" if thinking != "off" else ""),
        "accuracy": accuracy,
        "stderr": None,
        "samples": total,
        "status": "success" if not infra_results else ("failed" if len(scored_results) == 0 else "partial"),
        "virtue": virtue,
        "condition": condition_label,
    }
    if trace:
        result["sample_details"] = trace_data
    return result


async def run_experiment(
    virtues: list[str],
    model: str,
    provider: str = "openai-codex",
    thinking: str = "off",
    frame: str = "actual",
    injection_text: str | None = None,
    limit: int | None = None,
    seed: int = 42,
    trace: bool = False,
    concurrency: int = 5,
    retries: int = 2,
    timeout: int = 120,
    output_name: str | None = None,
) -> list[dict]:
    """Run the full experiment across virtues."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_prompt = FRAMES[frame]
    all_results = []

    for virtue in virtues:
        print(f"\n{'='*60}")
        print(f"Model: {model} (pi -p, concurrency={concurrency}) | Virtue: {virtue} | Frame: {frame}")
        print(f"{'='*60}")

        print(f"\n--- Vanilla ({frame}) ---")
        result_a = await run_virtue(
            virtue, model, provider, thinking, base_prompt,
            limit, seed, frame, trace, concurrency, retries, timeout,
        )
        all_results.append(result_a)
        acc_a = f"{result_a['accuracy']:.4f}" if result_a["accuracy"] is not None else "N/A"
        print(f"  Accuracy: {acc_a}")

        if injection_text:
            injected_prompt = injection_text + "\n\n---\n\n" + base_prompt
            print(f"\n--- Injected ({frame}) ---")
            result_b = await run_virtue(
                virtue, model, provider, thinking, injected_prompt,
                limit, seed, f"{frame}+injected", trace, concurrency, retries, timeout,
            )
            all_results.append(result_b)
            acc_b = f"{result_b['accuracy']:.4f}" if result_b["accuracy"] is not None else "N/A"
            print(f"  Accuracy: {acc_b}")

            if result_a["accuracy"] is not None and result_b["accuracy"] is not None:
                delta = result_b["accuracy"] - result_a["accuracy"]
                sign = "+" if delta >= 0 else ""
                print(f"\n  Delta: {sign}{delta:.4f}")

    virtues_label = "-".join(virtues)
    filename = output_name or f"results_pi_{model}_{virtues_label}_{timestamp}"
    if not filename.endswith(".json"):
        filename += ".json"
    results_file = RESULTS_DIR / filename
    summary_results, logs_file = write_result_artifacts(
        all_results, results_file, write_logs=trace,
    )
    print(f"\nResults saved to: {results_file}")
    if logs_file:
        print(f"Detailed logs saved to: {logs_file}")

    return summary_results


def main():
    parser = argparse.ArgumentParser(
        description="Run VirtueBench using pi print mode (pi -p)"
    )
    parser.add_argument(
        "--subset",
        choices=VIRTUES + ["all"],
        default="all",
        help="Virtue subset to evaluate (default: all)",
    )
    parser.add_argument(
        "--provider",
        default="openai-codex",
        help="Pi provider (default: openai-codex). Also: google-antigravity, google-gemini-cli",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="Model name for pi (default: gpt-5.4)",
    )
    parser.add_argument(
        "--thinking",
        choices=["off", "minimal", "low", "medium", "high", "xhigh"],
        default="off",
        help="Thinking level for pi (default: off)",
    )
    parser.add_argument(
        "--frame",
        choices=list(FRAMES.keys()),
        default="actual",
        help="Prompt frame key (default: actual)",
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
        help="Write per-sample answers, rationales, and debug metadata to *_logs.json",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent pi -p calls (default: 5)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries for blank/time-out/failed pi calls (default: 2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds per pi call attempt (default: 120)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename for results JSON (saved in results/)",
    )

    args = parser.parse_args()

    if args.quick:
        args.limit = 10

    virtues = VIRTUES if args.subset == "all" else [args.subset]

    injection_text = None
    if args.inject:
        injection_text = Path(args.inject).read_text(encoding="utf-8")

    print(f"Provider: {args.provider}")
    print(f"Model: {args.model} (via pi -p)")
    print(f"Thinking: {args.thinking}")
    print(f"Frame: {args.frame}")
    print(f"Virtues: {virtues}")
    print(f"Limit: {args.limit or 'all'}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Retries: {args.retries}")
    print(f"Timeout: {args.timeout}s")
    print(f"Injection: {'yes' if injection_text else 'no'}")

    results = asyncio.run(run_experiment(
        virtues=virtues,
        model=args.model,
        provider=args.provider,
        thinking=args.thinking,
        frame=args.frame,
        injection_text=injection_text,
        limit=args.limit,
        seed=args.seed,
        trace=args.detailed,
        concurrency=args.concurrency,
        retries=args.retries,
        timeout=args.timeout,
        output_name=args.output,
    ))

    print(f"\n{'Model':<25} {'Virtue':<12} {'Condition':<10} {'Accuracy':>8} {'Samples':>8}")
    print("-" * 67)
    for r in results:
        acc = f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "N/A"
        print(f"{r.get('model',''):<25} {r.get('virtue',''):<12} {r.get('condition',''):<10} {acc:>8} {r.get('samples',''):>8}")


if __name__ == "__main__":
    main()
