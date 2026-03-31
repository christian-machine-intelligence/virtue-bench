"""
Analysis utilities: comparison tables, summary statistics, and result formatting.
"""

from tabulate import tabulate


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table of experiment results."""
    headers = ["Model", "Virtue", "Condition", "Accuracy", "Stderr", "Samples"]
    rows = []
    for r in results:
        rows.append([
            r.get("model", ""),
            r.get("virtue", ""),
            r.get("condition", ""),
            f"{r['accuracy']:.4f}" if r.get("accuracy") is not None else "N/A",
            f"{r['stderr']:.4f}" if r.get("stderr") is not None else "N/A",
            r.get("samples", ""),
        ])
    print("\n" + tabulate(rows, headers=headers, tablefmt="github"))


def print_delta_table(results: list[dict]) -> None:
    """Print deltas between vanilla and injected conditions.

    Expects results to alternate: vanilla, injected, vanilla, injected, ...
    """
    headers = ["Model", "Virtue", "Vanilla", "Injected", "Delta"]
    rows = []
    for i in range(0, len(results) - 1, 2):
        vanilla = results[i]
        injected = results[i + 1]
        acc_v = vanilla.get("accuracy")
        acc_i = injected.get("accuracy")
        if acc_v is not None and acc_i is not None:
            delta = acc_i - acc_v
            sign = "+" if delta >= 0 else ""
            rows.append([
                vanilla.get("model", ""),
                vanilla.get("virtue", ""),
                f"{acc_v:.4f}",
                f"{acc_i:.4f}",
                f"{sign}{delta:.4f}",
            ])
    print("\n" + tabulate(rows, headers=headers, tablefmt="github"))


def summarize_by_virtue(results: list[dict]) -> dict[str, dict]:
    """Group results by virtue and compute mean accuracy per condition."""
    summary: dict[str, dict] = {}
    for r in results:
        virtue = r.get("virtue", "unknown")
        condition = r.get("condition", "unknown")
        acc = r.get("accuracy")
        if acc is None:
            continue
        key = f"{virtue}_{condition}"
        if key not in summary:
            summary[key] = {"virtue": virtue, "condition": condition, "accuracies": []}
        summary[key]["accuracies"].append(acc)

    for v in summary.values():
        accs = v["accuracies"]
        v["mean_accuracy"] = sum(accs) / len(accs) if accs else 0
        v["n"] = len(accs)

    return summary
