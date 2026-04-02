"""
Result artifact helpers.

Published summary files stay small. Per-sample debug detail lives in a
separate *_logs.json artifact when requested.
"""

import json
from pathlib import Path


def strip_sample_details(results: list[dict]) -> list[dict]:
    """Return result rows without per-sample debug detail."""
    return [
        {key: value for key, value in row.items() if key != "sample_details"}
        for row in results
    ]


def write_result_artifacts(
    results: list[dict],
    results_file: Path,
    write_logs: bool = False,
) -> tuple[list[dict], Path | None]:
    """Write summary results and optionally a separate debug log artifact."""
    summary_results = strip_sample_details(results)
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, default=str)

    logs_file = None
    if write_logs and any("sample_details" in row for row in results):
        logs_file = results_file.with_name(f"{results_file.stem}_logs.json")
        with open(logs_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

    return summary_results, logs_file
