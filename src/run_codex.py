"""
VirtueBench runner using Codex app-server (JSON-RPC over stdio).

Spawns a single codex app-server process and sends all samples as turns
on one thread. Much faster than codex exec (one startup vs N startups).

Usage:
    python -m src.run_codex                          # full benchmark
    python -m src.run_codex --quick                   # smoke test (10 per virtue)
    python -m src.run_codex --subset courage          # single virtue
    python -m src.run_codex --model gpt-5.4           # specific model
    python -m src.run_codex --trace                   # per-sample recording
"""

import argparse
import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from .data import load_virtue_csv, BASE_INSTRUCTION, VIRTUES


RESULTS_DIR = Path(__file__).parent.parent / "results"


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

    def _request(self, method: str, params: dict) -> dict:
        rid = self._next_request_id()
        self._events[rid] = threading.Event()
        self._send(method, params, rid)
        self._events[rid].wait(timeout=120)
        with self._lock:
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

    def start(self):
        self.proc = subprocess.Popen(
            ["codex", "app-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

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

        # Start thread
        resp = self._request("thread/start", {
            "model": self.model,
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "ephemeral": True,
        })
        if "error" in resp:
            raise RuntimeError(f"thread/start failed: {resp['error']}")
        self.thread_id = resp["result"]["thread"]["id"]

    def query(self, prompt: str, retries: int = 1) -> str:
        """Send a turn and wait for the agent message response."""
        for attempt in range(1 + retries):
            response = self._query_once(prompt)
            if response or attempt == retries:
                return response
            print("[blank, retrying] ", end="", flush=True)
        return ""

    def _query_once(self, prompt: str) -> str:
        """Send a single turn and wait for the agent message response."""
        with self._lock:
            self._notifications.clear()

        resp = self._request("turn/start", {
            "threadId": self.thread_id,
            "input": [{"type": "text", "text": prompt}],
        })
        if "error" in resp:
            return f"[error] {resp['error'].get('message', '')}"

        # Wait for turn/completed notification
        import time
        for _ in range(600):  # up to 60s
            time.sleep(0.1)
            with self._lock:
                for notif in self._notifications:
                    if notif.get("method") == "turn/completed":
                        turn = notif.get("params", {}).get("turn", {})
                        for item in turn.get("items", []):
                            if item.get("type") == "agentMessage":
                                return item.get("text", "").strip()
                for notif in self._notifications:
                    if notif.get("method") == "item/completed":
                        item = notif.get("params", {}).get("item", {})
                        if item.get("type") == "agentMessage":
                            return item.get("text", "").strip()
        return ""

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)


def parse_response(response: str) -> tuple[str | None, str]:
    """Extract the leading A/B answer and the rationale from a response."""
    answer = None
    for ch in response:
        if ch in ("A", "B"):
            answer = ch
            break
    rationale = response
    for sep in (" — ", " - ", "—"):
        if sep in response:
            rationale = response.split(sep, 1)[1].strip()
            break
    return answer, rationale


def run_virtue(
    server: CodexAppServer,
    virtue: str,
    system_prompt: str,
    limit: int | None,
    seed: int,
    condition_label: str,
    trace: bool = False,
) -> dict:
    """Run all samples for a single virtue and return a result dict."""
    samples = load_virtue_csv(virtue, limit=limit, seed=seed)
    correct = 0
    total = len(samples)
    trace_data = []

    for i, sample in enumerate(samples, 1):
        print(f"  [{i}/{total}] ", end="", flush=True)
        full_prompt = f"{system_prompt}\n\n{sample.input}"
        response = server.query(full_prompt)
        answer, rationale = parse_response(response)
        is_correct = answer == sample.target

        if is_correct:
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

    accuracy = correct / total if total > 0 else 0.0
    result = {
        "model": f"codex-appserver/{server.model}",
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


def run_experiment(
    virtues: list[str],
    model: str,
    injection_text: str | None = None,
    limit: int | None = None,
    seed: int = 42,
    trace: bool = False,
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
            result_a = run_virtue(server, virtue, BASE_INSTRUCTION, limit, seed, "vanilla", trace)
            all_results.append(result_a)
            print(f"  Accuracy: {result_a['accuracy']:.4f}")

            if injection_text:
                injected_prompt = injection_text + "\n\n---\n\n" + BASE_INSTRUCTION
                print(f"\n--- Injected ---")
                result_b = run_virtue(server, virtue, injected_prompt, limit, seed, "injected", trace)
                all_results.append(result_b)
                print(f"  Accuracy: {result_b['accuracy']:.4f}")

                if result_a["accuracy"] is not None and result_b["accuracy"] is not None:
                    delta = result_b["accuracy"] - result_a["accuracy"]
                    sign = "+" if delta >= 0 else ""
                    print(f"\n  Delta: {sign}{delta:.4f}")
    finally:
        server.stop()

    virtues_label = "-".join(virtues)
    results_file = RESULTS_DIR / f"results_codex_{model}_{virtues_label}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    return all_results


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
        help="Include per-sample answers, rationales, and correctness in results",
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
    print(f"Injection: {'yes' if injection_text else 'no'}")

    try:
        subprocess.run(["codex", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("Error: 'codex' CLI not found. Install Codex first:", file=sys.stderr)
        print("  https://github.com/openai/codex", file=sys.stderr)
        sys.exit(1)

    results = run_experiment(
        virtues=virtues,
        model=args.model,
        injection_text=injection_text,
        limit=args.limit,
        seed=args.seed,
        trace=args.detailed,
    )

    print(f"\n{'Model':<25} {'Virtue':<12} {'Condition':<10} {'Accuracy':>8} {'Samples':>8}")
    print("-" * 67)
    for r in results:
        acc = f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "N/A"
        print(f"{r.get('model',''):<25} {r.get('virtue',''):<12} {r.get('condition',''):<10} {acc:>8} {r.get('samples',''):>8}")


if __name__ == "__main__":
    main()
