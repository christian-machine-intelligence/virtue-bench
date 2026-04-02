"""
VirtueBench runner using Codex app-server (JSON-RPC over stdio).

Spawns a single codex app-server process and starts a fresh thread per sample.
This keeps samples isolated while still amortizing process startup cost.

Usage:
    python -m src.run_codex                          # full benchmark
    python -m src.run_codex --quick                   # smoke test (10 per virtue)
    python -m src.run_codex --subset courage          # single virtue
    python -m src.run_codex --model gpt-5.4           # specific model
    python -m src.run_codex --detailed                # write per-sample debug logs
"""

import argparse
from collections import deque
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .data import load_virtue_csv, parse_answer, BASE_INSTRUCTION, VIRTUES
from .result_artifacts import write_result_artifacts


RESULTS_DIR = Path(__file__).parent.parent / "results"
NEUTRAL_CWD = "/tmp"


class CodexAppServer:
    """Manages a codex app-server process over stdio JSON-RPC."""

    def __init__(self, model: str):
        self.model = model
        self.proc = None
        self.thread_id = None
        self._next_id = 0
        self._reader_thread = None
        self._responses = {}
        self._notifications = []
        self._lock = threading.Lock()
        self._events = {}
        self._turn_done = threading.Event()
        self._notification_event = threading.Event()
        self._stderr_thread = None
        self._stderr_lines = deque(maxlen=200)

    def _next_request_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _send(self, method: str, params: dict, request_id: int | None = None) -> int | None:
        msg = {"method": method, "params": params}
        if request_id is not None:
            msg["id"] = request_id
        line = json.dumps(msg) + "\n"
        self.proc.stdin.write(line)
        self.proc.stdin.flush()
        return request_id

    def _request(self, method: str, params: dict, timeout: int = 120) -> dict:
        rid = self._next_request_id()
        event = threading.Event()
        self._events[rid] = event
        self._send(method, params, rid)
        if not event.wait(timeout=timeout):
            with self._lock:
                self._events.pop(rid, None)
            raise TimeoutError(f"{method} timed out after {timeout}s")
        with self._lock:
            self._events.pop(rid, None)
            return self._responses.pop(rid, {})

    def _notify(self, method: str, params: dict):
        self._send(method, params)

    def _reader_loop(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                rid = msg["id"]
                with self._lock:
                    self._responses[rid] = msg
                if rid in self._events:
                    self._events[rid].set()
            else:
                with self._lock:
                    self._notifications.append(msg)
                self._notification_event.set()
                if msg.get("method") in ("turn/completed", "item/completed"):
                    self._turn_done.set()

    def _stderr_loop(self):
        for line in self.proc.stderr:
            line = line.rstrip()
            if not line:
                continue
            with self._lock:
                self._stderr_lines.append(line)

    def _stderr_snapshot(self) -> str | None:
        with self._lock:
            if not self._stderr_lines:
                return None
            return "\n".join(self._stderr_lines)

    def start(self):
        self.proc = subprocess.Popen(
            ["codex", "app-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=NEUTRAL_CWD,
        )
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_thread.start()

        # Initialize
        resp = self._request("initialize", {
            "clientInfo": {
                "name": "virtue_bench",
                "title": "VirtueBench",
                "version": "0.1.0",
            },
        })
        if "error" in resp:
            raise RuntimeError(f"initialize failed: {resp['error']}")

        self._notify("initialized", {})

        # Warm connection with a first thread.
        resp = self._request("thread/start", {
            "model": self.model,
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "ephemeral": True,
            "cwd": NEUTRAL_CWD,
        })
        if "error" in resp:
            raise RuntimeError(f"thread/start failed: {resp['error']}")
        self.thread_id = resp["result"]["thread"]["id"]

    def new_thread(self, system_prompt: str):
        """Start a fresh thread (resets conversation history)."""
        resp = self._request("thread/start", {
            "model": self.model,
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "ephemeral": True,
            "cwd": NEUTRAL_CWD,
            "developerInstructions": system_prompt,
        })
        if "error" in resp:
            raise RuntimeError(f"thread/start failed: {resp['error']}")
        self.thread_id = resp["result"]["thread"]["id"]

    def _extract_matching_message(self, thread_id: str, turn_id: str) -> str | None:
        with self._lock:
            for idx, notif in enumerate(self._notifications):
                method = notif.get("method")
                params = notif.get("params", {})

                if params.get("threadId") != thread_id:
                    continue

                if method == "item/completed" and params.get("turnId") == turn_id:
                    item = params.get("item", {})
                    if item.get("type") == "agentMessage":
                        self._notifications.pop(idx)
                        return item.get("text", "").strip()

                if method == "turn/completed":
                    turn = params.get("turn", {})
                    if turn.get("id") != turn_id:
                        continue
                    self._notifications.pop(idx)
                    for item in turn.get("items", []):
                        if item.get("type") == "agentMessage":
                            return item.get("text", "").strip()
                    return ""
        return None

    def query(
        self,
        prompt: str,
        system_prompt: str,
        retries: int = 2,
        timeout: int = 120,
    ) -> dict:
        """Run a single isolated Codex query and return outcome metadata."""
        started_at = time.perf_counter()
        last_error = "unknown"

        for attempt in range(1, retries + 2):
            try:
                response = self._query_once(prompt, system_prompt, timeout=timeout)
            except TimeoutError:
                last_error = "timeout"
                if attempt <= retries:
                    continue
                break
            except Exception as exc:
                last_error = f"appserver_error:{exc.__class__.__name__}"
                if attempt <= retries:
                    continue
                break

            if response:
                return {
                    "response": response,
                    "infra_error": None,
                    "attempts": attempt,
                    "latency_ms": round((time.perf_counter() - started_at) * 1000),
                    "stderr": self._stderr_snapshot(),
                }

            last_error = "blank_response"
            if attempt <= retries:
                continue
            break

        return {
            "response": "",
            "infra_error": last_error,
            "attempts": retries + 1,
            "latency_ms": round((time.perf_counter() - started_at) * 1000),
            "stderr": self._stderr_snapshot(),
        }

    def _query_once(self, prompt: str, system_prompt: str, timeout: int = 120) -> str:
        """Send a single turn on a fresh thread and wait for the matching response."""
        self.new_thread(system_prompt)
        self._turn_done.clear()
        self._notification_event.clear()
        with self._lock:
            self._notifications.clear()

        resp = self._request("turn/start", {
            "threadId": self.thread_id,
            "input": [{"type": "text", "text": prompt}],
        }, timeout=timeout)
        if "error" in resp:
            raise RuntimeError(f"turn/start failed: {resp['error'].get('message', '')}")

        turn_id = resp["result"]["turn"]["id"]
        deadline = time.monotonic() + timeout

        while True:
            message = self._extract_matching_message(self.thread_id, turn_id)
            if message is not None:
                return message

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"turn/start timed out after {timeout}s")

            if self.proc.poll() is not None:
                raise RuntimeError(f"codex app-server exited with code {self.proc.returncode}")

            self._notification_event.wait(timeout=min(remaining, 1.0))
            self._notification_event.clear()

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)


def run_virtue(
    server: CodexAppServer,
    virtue: str,
    system_prompt: str,
    limit: int | None,
    seed: int,
    condition_label: str,
    trace: bool = False,
    retries: int = 2,
    timeout: int = 120,
) -> dict:
    """Run all samples for a single virtue and return a result dict."""
    samples = load_virtue_csv(virtue, limit=limit, seed=seed)
    correct = 0
    total = len(samples)
    trace_data = []
    sample_results = []

    for i, sample in enumerate(samples, 1):
        print(f"  [{i}/{total}] ", end="", flush=True)
        outcome = server.query(
            sample.input,
            system_prompt,
            retries=retries,
            timeout=timeout,
        )
        response = outcome["response"]
        answer = parse_answer(response) if outcome["infra_error"] is None else None
        is_correct = answer == sample.target if outcome["infra_error"] is None else None

        sample_result = {
            "response": response,
            "answer": answer,
            "correct": is_correct,
            "sample": sample,
            "infra_error": outcome["infra_error"],
            "attempts": outcome["attempts"],
            "latency_ms": outcome["latency_ms"],
            "stderr": outcome["stderr"],
        }
        sample_results.append(sample_result)

        if outcome["infra_error"] is not None:
            print(f"infra:{outcome['infra_error']}")
        elif is_correct:
            correct += 1
            print("correct")
        else:
            print("incorrect")

        if trace:
            trace_data.append({
                "id": i,
                "prompt": sample.input,
                "target": sample.target,
                "model_response": response,
                "model_answer": answer,
                "correct": is_correct,
                "explanation": response,
                "metadata": sample.metadata,
            })

    scored_results = [r for r in sample_results if r["infra_error"] is None]
    infra_results = [r for r in sample_results if r["infra_error"] is not None]
    scored_total = len(scored_results)
    accuracy = correct / scored_total if scored_total > 0 else None
    result = {
        "model": f"codex-appserver/{server.model}",
        "accuracy": accuracy,
        "stderr": None,
        "samples": total,
        "status": "success" if not infra_results else ("failed" if scored_total == 0 else "partial"),
        "virtue": virtue,
        "condition": condition_label,
    }
    if trace:
        result["sample_details"] = trace_data
    return result


def run_experiment(
    virtues: list[str],
    model: str,
    injection_text: str | None = None,
    limit: int | None = None,
    seed: int = 42,
    trace: bool = False,
    retries: int = 2,
    timeout: int = 120,
    output_name: str | None = None,
) -> list[dict]:
    """Run the full experiment across virtues."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    server = CodexAppServer(model)
    print("Starting codex app-server...", flush=True)
    server.start()
    print(f"Connected. Thread: {server.thread_id}")

    all_results = []

    try:
        for virtue in virtues:
            print(f"\n{'='*60}")
            print(f"Model: {model} (codex app-server) | Virtue: {virtue}")
            print(f"{'='*60}")

            print(f"\n--- Vanilla ---")
            result_a = run_virtue(
                server,
                virtue,
                BASE_INSTRUCTION,
                limit,
                seed,
                "vanilla",
                trace,
                retries,
                timeout,
            )
            all_results.append(result_a)
            acc_a = f"{result_a['accuracy']:.4f}" if result_a["accuracy"] is not None else "N/A"
            print(f"  Accuracy: {acc_a}")

            if injection_text:
                injected_prompt = injection_text + "\n\n---\n\n" + BASE_INSTRUCTION
                print(f"\n--- Injected ---")
                result_b = run_virtue(
                    server,
                    virtue,
                    injected_prompt,
                    limit,
                    seed,
                    "injected",
                    trace,
                    retries,
                    timeout,
                )
                all_results.append(result_b)
                acc_b = f"{result_b['accuracy']:.4f}" if result_b["accuracy"] is not None else "N/A"
                print(f"  Accuracy: {acc_b}")

                if result_a["accuracy"] is not None and result_b["accuracy"] is not None:
                    delta = result_b["accuracy"] - result_a["accuracy"]
                    sign = "+" if delta >= 0 else ""
                    print(f"\n  Delta: {sign}{delta:.4f}")
    finally:
        server.stop()

    virtues_label = "-".join(virtues)
    filename = output_name or f"results_codex_{model}_{virtues_label}_{timestamp}"
    if not filename.endswith(".json"):
        filename += ".json"
    results_file = RESULTS_DIR / filename
    summary_results, logs_file = write_result_artifacts(
        all_results,
        results_file,
        write_logs=trace,
    )
    print(f"\nResults saved to: {results_file}")
    if logs_file:
        print(f"Detailed logs saved to: {logs_file}")

    return summary_results


def main():
    parser = argparse.ArgumentParser(
        description="Run VirtueBench using Codex app-server (JSON-RPC)"
    )
    parser.add_argument(
        "--subset",
        choices=VIRTUES + ["all"],
        default="all",
        help="Virtue subset to evaluate (default: all)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="Model name for codex (default: gpt-5.4)",
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
        "--retries",
        type=int,
        default=2,
        help="Retries for blank/time-out/failed app-server calls (default: 2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds per app-server call attempt (default: 120)",
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

    print(f"Model: {args.model} (via codex app-server)")
    print(f"Virtues: {virtues}")
    print(f"Limit: {args.limit or 'all'}")
    print(f"Retries: {args.retries}")
    print(f"Timeout: {args.timeout}s")
    print(f"Injection: {'yes' if injection_text else 'no'}")

    results = run_experiment(
        virtues=virtues,
        model=args.model,
        injection_text=injection_text,
        limit=args.limit,
        seed=args.seed,
        trace=args.detailed,
        retries=args.retries,
        timeout=args.timeout,
        output_name=args.output,
    )

    print(f"\n{'Model':<25} {'Virtue':<12} {'Condition':<10} {'Accuracy':>8} {'Samples':>8}")
    print("-" * 67)
    for r in results:
        acc = f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "N/A"
        print(f"{r.get('model',''):<25} {r.get('virtue',''):<12} {r.get('condition',''):<10} {acc:>8} {r.get('samples',''):>8}")


if __name__ == "__main__":
    main()
