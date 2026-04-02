"""
Analysis utilities: comparison tables, summary statistics, and frame analysis.
"""

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any

try:
    from tabulate import tabulate
except ModuleNotFoundError:
    def tabulate(rows, headers, tablefmt="github"):
        """Minimal github-table fallback when tabulate is unavailable."""
        str_rows = [[str(cell) for cell in row] for row in rows]
        widths = [
            max(len(str(header)), *(len(row[i]) for row in str_rows))
            for i, header in enumerate(headers)
        ]

        def fmt_row(row):
            return "| " + " | ".join(
                cell.ljust(widths[i]) for i, cell in enumerate(row)
            ) + " |"

        header_row = fmt_row([str(header) for header in headers])
        divider = "| " + " | ".join("-" * width for width in widths) + " |"
        body = "\n".join(fmt_row(row) for row in str_rows)
        return "\n".join(part for part in [header_row, divider, body] if part)


RESULTS_DIR = Path(__file__).parent.parent / "results"
DEFAULT_FRAME_COMPARISONS = [
    ("preserve", "actual"),
    ("actual", "bare"),
    ("actual", "character"),
    ("actual", "duty"),
    ("actual", "resist"),
]
DEFAULT_SHARED_FLIP_COMPARISONS = [
    ("preserve", "actual"),
    ("actual", "resist"),
]
DEFAULT_STABLE_FAILURE_CONDITIONS = ["resist"]


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


def load_results_file(path: Path) -> list[dict]:
    """Load a JSON results file and return the row list."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} does not contain a result row list")
    return data


def find_result_row(results: list[dict], virtue: str, condition: str) -> dict | None:
    """Find the first result row matching virtue and condition."""
    for row in results:
        if row.get("virtue") == virtue and row.get("condition") == condition:
            return row
    return None


def require_sample_details(row: dict, path: Path, virtue: str, condition: str) -> list[dict]:
    """Return sample_details or raise a helpful error."""
    sample_details = row.get("sample_details")
    if not isinstance(sample_details, list):
        raise ValueError(
            f"{path} is missing sample_details for virtue={virtue}, condition={condition}. "
            "Use a *_logs.json result file."
        )
    return sample_details


def sample_map(sample_details: list[dict]) -> dict[Any, dict]:
    """Index sample details by sample id."""
    return {detail["id"]: detail for detail in sample_details}


def exact_two_sided_binomial_pvalue(improve: int, regress: int) -> float:
    """Two-sided exact p-value for discordant paired counts.

    Equivalent to an exact sign / McNemar-style test on discordant pairs.
    """
    n = improve + regress
    if n == 0:
        return 1.0
    k = min(improve, regress)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_frame_result(
    results: list[dict],
    path: Path,
    virtue: str,
    from_condition: str,
    to_condition: str,
) -> dict | None:
    """Compute paired movement between two frame conditions for one result file."""
    from_row = find_result_row(results, virtue, from_condition)
    to_row = find_result_row(results, virtue, to_condition)
    if from_row is None or to_row is None:
        return None

    from_samples = sample_map(require_sample_details(from_row, path, virtue, from_condition))
    to_samples = sample_map(require_sample_details(to_row, path, virtue, to_condition))

    improve = regress = same_right = same_wrong = 0
    for sample_id, from_detail in from_samples.items():
        if sample_id not in to_samples:
            raise ValueError(
                f"{path} has mismatched sample ids for {from_condition} vs {to_condition}"
            )
        from_correct = bool(from_detail.get("correct"))
        to_correct = bool(to_samples[sample_id].get("correct"))
        if (not from_correct) and to_correct:
            improve += 1
        elif from_correct and (not to_correct):
            regress += 1
        elif from_correct and to_correct:
            same_right += 1
        else:
            same_wrong += 1

    return {
        "file": path.name,
        "model": from_row.get("model", ""),
        "virtue": virtue,
        "from_condition": from_condition,
        "to_condition": to_condition,
        "improve": improve,
        "regress": regress,
        "same_right": same_right,
        "same_wrong": same_wrong,
        "p_value": exact_two_sided_binomial_pvalue(improve, regress),
    }


def compute_paired_frame_results(
    paths: list[Path],
    virtue: str,
    comparisons: list[tuple[str, str]],
) -> list[dict]:
    """Compute paired frame movement results across files and comparisons."""
    rows: list[dict] = []
    for path in paths:
        results = load_results_file(path)
        for from_condition, to_condition in comparisons:
            row = paired_frame_result(results, path, virtue, from_condition, to_condition)
            if row is not None:
                rows.append(row)
    return rows


def aggregate_paired_results(rows: list[dict]) -> list[dict]:
    """Aggregate paired movement rows across models for each comparison."""
    grouped: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["from_condition"], row["to_condition"])
        bucket = grouped.setdefault(key, {
            "from_condition": row["from_condition"],
            "to_condition": row["to_condition"],
            "models": [],
            "improve": 0,
            "regress": 0,
            "same_right": 0,
            "same_wrong": 0,
        })
        bucket["models"].append(row["model"])
        bucket["improve"] += row["improve"]
        bucket["regress"] += row["regress"]
        bucket["same_right"] += row["same_right"]
        bucket["same_wrong"] += row["same_wrong"]

    summary_rows = []
    for bucket in grouped.values():
        bucket["model_count"] = len(bucket["models"])
        bucket["p_value"] = exact_two_sided_binomial_pvalue(
            bucket["improve"],
            bucket["regress"],
        )
        summary_rows.append(bucket)
    return summary_rows


def changed_item_ids(
    results: list[dict],
    path: Path,
    virtue: str,
    from_condition: str,
    to_condition: str,
    mode: str = "improve",
) -> set[Any]:
    """Return item ids that changed in the requested direction."""
    paired = paired_frame_result(results, path, virtue, from_condition, to_condition)
    if paired is None:
        return set()

    from_row = find_result_row(results, virtue, from_condition)
    to_row = find_result_row(results, virtue, to_condition)
    assert from_row is not None
    assert to_row is not None
    from_samples = sample_map(require_sample_details(from_row, path, virtue, from_condition))
    to_samples = sample_map(require_sample_details(to_row, path, virtue, to_condition))

    ids: set[Any] = set()
    for sample_id, from_detail in from_samples.items():
        from_correct = bool(from_detail.get("correct"))
        to_correct = bool(to_samples[sample_id].get("correct"))
        if mode == "improve" and (not from_correct) and to_correct:
            ids.add(sample_id)
        elif mode == "regress" and from_correct and (not to_correct):
            ids.add(sample_id)
    return ids


def incorrect_item_ids(
    results: list[dict],
    path: Path,
    virtue: str,
    condition: str,
) -> set[Any]:
    """Return item ids that are incorrect for a given condition."""
    row = find_result_row(results, virtue, condition)
    if row is None:
        return set()
    samples = require_sample_details(row, path, virtue, condition)
    return {detail["id"] for detail in samples if not bool(detail.get("correct"))}


def shared_item_ids_across_files(
    paths: list[Path],
    virtue: str,
    from_condition: str,
    to_condition: str,
    mode: str = "improve",
) -> tuple[set[Any], list[str]]:
    """Return shared changed item ids across files that support the comparison."""
    shared: set[Any] | None = None
    used_models: list[str] = []
    for path in paths:
        results = load_results_file(path)
        row = paired_frame_result(results, path, virtue, from_condition, to_condition)
        if row is None:
            continue
        used_models.append(row["model"])
        ids = changed_item_ids(results, path, virtue, from_condition, to_condition, mode=mode)
        shared = ids if shared is None else shared & ids
    return (shared or set()), used_models


def stable_failure_ids_across_files(
    paths: list[Path],
    virtue: str,
    condition: str,
) -> tuple[set[Any], list[str]]:
    """Return item ids that remain incorrect across all matching files."""
    shared: set[Any] | None = None
    used_models: list[str] = []
    for path in paths:
        results = load_results_file(path)
        row = find_result_row(results, virtue, condition)
        if row is None:
            continue
        used_models.append(row.get("model", ""))
        ids = incorrect_item_ids(results, path, virtue, condition)
        shared = ids if shared is None else shared & ids
    return (shared or set()), used_models


def representative_sample_map(
    paths: list[Path],
    virtue: str,
    preferred_condition: str,
) -> dict[Any, dict]:
    """Return a sample-detail map from the first file that has the requested condition."""
    for path in paths:
        results = load_results_file(path)
        row = find_result_row(results, virtue, preferred_condition)
        if row is None:
            continue
        return sample_map(require_sample_details(row, path, virtue, preferred_condition))
    raise ValueError(
        f"No result file contained virtue={virtue}, condition={preferred_condition}"
    )


def clean_prompt(prompt: str) -> str:
    """Trim the trailing question line from a prompt for reporting."""
    return prompt.split("\n\nWhat do you do?")[0].strip()


def summarize_item_details(
    sample_details_by_id: dict[Any, dict],
    item_ids: set[Any],
) -> list[dict]:
    """Return sorted item summaries for reporting and JSON output."""
    summaries = []
    for sample_id in sorted(item_ids):
        detail = sample_details_by_id[sample_id]
        summaries.append({
            "id": sample_id,
            "source": detail.get("metadata", {}).get("source", ""),
            "prompt": clean_prompt(detail.get("prompt", "")),
        })
    return summaries


def default_frame_log_paths() -> list[Path]:
    """Return the default frame-comparison log files in results/."""
    return sorted(RESULTS_DIR.glob("frames_*combined_logs.json"))


def print_paired_frame_table(rows: list[dict]) -> None:
    """Print paired movement rows in a compact table."""
    headers = ["Model", "From", "To", "Improve", "Regress", "Same Right", "Same Wrong", "p"]
    table_rows = [
        [
            row["model"],
            row["from_condition"],
            row["to_condition"],
            row["improve"],
            row["regress"],
            row["same_right"],
            row["same_wrong"],
            f"{row['p_value']:.3g}",
        ]
        for row in rows
    ]
    print("\nPaired Frame Results")
    print(tabulate(table_rows, headers=headers, tablefmt="github"))


def print_aggregate_paired_table(rows: list[dict]) -> None:
    """Print aggregate paired movement rows in a compact table."""
    headers = ["From", "To", "Models", "Improve", "Regress", "Same Right", "Same Wrong", "p"]
    table_rows = [
        [
            row["from_condition"],
            row["to_condition"],
            row["model_count"],
            row["improve"],
            row["regress"],
            row["same_right"],
            row["same_wrong"],
            f"{row['p_value']:.3g}",
        ]
        for row in rows
    ]
    print("\nAggregate Discordant Counts")
    print(tabulate(table_rows, headers=headers, tablefmt="github"))


def print_item_table(title: str, rows: list[dict]) -> None:
    """Print an item summary table."""
    headers = ["ID", "Source", "Prompt"]
    table_rows = [[row["id"], row["source"], row["prompt"]] for row in rows]
    print(f"\n{title}")
    print(tabulate(table_rows, headers=headers, tablefmt="github"))


def parse_comparisons(values: list[str] | None, default: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Parse repeated A:B comparison strings."""
    if not values:
        return default
    comparisons = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid comparison '{value}'. Use from:to")
        from_condition, to_condition = value.split(":", 1)
        comparisons.append((from_condition, to_condition))
    return comparisons


def build_frame_analysis_report(
    paths: list[Path],
    virtue: str,
    comparisons: list[tuple[str, str]],
    shared_flip_comparisons: list[tuple[str, str]],
    stable_failure_conditions: list[str],
) -> dict:
    """Build a machine-readable frame analysis report."""
    paired_rows = compute_paired_frame_results(paths, virtue, comparisons)
    aggregate_rows = aggregate_paired_results(paired_rows)

    report = {
        "virtue": virtue,
        "files": [path.name for path in paths],
        "paired_results": paired_rows,
        "aggregate_paired_results": aggregate_rows,
        "shared_flips": [],
        "stable_failures": [],
    }

    for from_condition, to_condition in shared_flip_comparisons:
        ids, models = shared_item_ids_across_files(
            paths,
            virtue,
            from_condition,
            to_condition,
            mode="improve",
        )
        sample_details_by_id = representative_sample_map(paths, virtue, to_condition)
        report["shared_flips"].append({
            "from_condition": from_condition,
            "to_condition": to_condition,
            "models": models,
            "count": len(ids),
            "items": summarize_item_details(sample_details_by_id, ids),
        })

    for condition in stable_failure_conditions:
        ids, models = stable_failure_ids_across_files(paths, virtue, condition)
        sample_details_by_id = representative_sample_map(paths, virtue, condition)
        report["stable_failures"].append({
            "condition": condition,
            "models": models,
            "count": len(ids),
            "items": summarize_item_details(sample_details_by_id, ids),
        })

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze VirtueBench frame logs with paired tests and shared flips."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        help="Paths to *_logs.json frame result files. Defaults to results/frames_*combined_logs.json",
    )
    parser.add_argument(
        "--virtue",
        default="courage",
        help="Virtue to analyze (default: courage)",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        help="Frame comparison in from:to form. Can be repeated.",
    )
    parser.add_argument(
        "--shared-flip",
        action="append",
        help="Shared flip comparison in from:to form. Can be repeated.",
    )
    parser.add_argument(
        "--stable-failure",
        action="append",
        help="Condition for stable-failure intersection. Can be repeated.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for the full report.",
    )

    args = parser.parse_args()

    paths = [Path(path) for path in args.logs] if args.logs else default_frame_log_paths()
    if not paths:
        raise SystemExit("No frame log files found. Pass explicit *_logs.json paths.")

    comparisons = parse_comparisons(args.comparison, DEFAULT_FRAME_COMPARISONS)
    shared_flip_comparisons = parse_comparisons(
        args.shared_flip,
        DEFAULT_SHARED_FLIP_COMPARISONS,
    )
    stable_failure_conditions = args.stable_failure or DEFAULT_STABLE_FAILURE_CONDITIONS

    report = build_frame_analysis_report(
        paths=paths,
        virtue=args.virtue,
        comparisons=comparisons,
        shared_flip_comparisons=shared_flip_comparisons,
        stable_failure_conditions=stable_failure_conditions,
    )

    print_paired_frame_table(report["paired_results"])
    print_aggregate_paired_table(report["aggregate_paired_results"])

    for shared in report["shared_flips"]:
        title = (
            f"Shared Flips: {shared['from_condition']} -> {shared['to_condition']} "
            f"({shared['count']} items across {len(shared['models'])} models)"
        )
        print_item_table(title, shared["items"])

    for stable in report["stable_failures"]:
        title = (
            f"Stable Failures: {stable['condition']} "
            f"({stable['count']} items across {len(stable['models'])} models)"
        )
        print_item_table(title, stable["items"])

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to: {output_path}")


if __name__ == "__main__":
    main()
