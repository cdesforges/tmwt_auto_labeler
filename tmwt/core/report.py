"""
End-of-run report for label.py.

Writes two files to the output directory:
  labeling_report.csv — one row per video, for analysis in a spreadsheet.
  labeling_report.md  — a readable summary with counts and the same table.
"""

import csv
import os
from datetime import datetime

from tmwt.core.job import COURSE_M, REVIEW_APPROVED, REVIEW_REJECTED, STATUS_FAILED, STATUS_NO_BODY

# Overall results, in the order they're listed.
APPROVED = "approved"
UNREVIEWED = "auto (not reviewed)"
REJECTED = "rejected"
FAILED = "failed"
RESULTS = (APPROVED, UNREVIEWED, REJECTED, FAILED)

COLUMNS = [
    "file", "result", "reason", "endpoints", "timing", "start_method", "endpoint_behavior",
    "start_s", "end_s", "duration_s", "speed_mps", "outputs_saved",
]


def _fmt(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


def job_result(job):
    """
    (result, reason) summarising one video's outcome.

    result is one of RESULTS.
    """
    if job.review == REVIEW_REJECTED:
        return REJECTED, job.review_note
    if job.status in (STATUS_FAILED, STATUS_NO_BODY):
        return FAILED, job.error
    if job.review == REVIEW_APPROVED:
        return APPROVED, job.review_note
    if job.far_ep is None:
        if job.endpoint_problem:
            return FAILED, f"{job.endpoint_problem}; rope endpoints need clicking at review"
        return FAILED, "rope endpoints not set"
    if job.duration is None:
        missing = "walk start" if job.walk_start is None else "walk end"
        return FAILED, f"no {missing} detected"
    return UNREVIEWED, job.review_note


def _row(job):
    result, reason = job_result(job)
    return {
        "file": job.name,
        "result": result,
        "reason": reason,
        "endpoints": job.endpoint_source,
        "timing": job.timing_source,
        "start_method": job.timing_detail,
        "endpoint_behavior": job.endpoint_behavior,
        "start_s": _fmt(job.walk_start),
        "end_s": _fmt(job.walk_end),
        "duration_s": _fmt(job.duration),
        "speed_mps": _fmt(job.speed, 2),
        "outputs_saved": "yes" if job.saved else "no",
    }


def write_report(jobs, output_dir, pose_models):
    """Write the CSV and Markdown reports and print a summary. Returns (csv_path, md_path)."""
    rows = [_row(job) for job in jobs]
    csv_path = os.path.join(output_dir, "labeling_report.csv")
    md_path = os.path.join(output_dir, "labeling_report.md")
    _write_csv(csv_path, rows)
    _write_markdown(md_path, rows, pose_models)
    _print_summary(rows, csv_path, md_path)
    return csv_path, md_path


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path, rows, pose_models):
    counts = {result: sum(r["result"] == result for r in rows) for result in RESULTS}
    lines = [
        "# TMWT labeling report",
        "",
        f"- Run: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Pose model: {pose_models}",
        f"- Course length: {COURSE_M:g} m",
        f"- Videos: {len(rows)}",
    ]
    lines += [f"  - {result}: {count}" for result, count in counts.items() if count]
    lines += [
        "",
        "| File | Result | Reason | Endpoints | Timing | Start method | Endpoint behavior | Start (s) | End (s) | Duration (s) | Speed (m/s) | Saved |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(
            r[c].replace("|", "/") if r[c] else "—" for c in COLUMNS) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _print_summary(rows, csv_path, md_path):
    print(f"\n{'=' * 60}\nREPORT")
    for r in rows:
        extra = f"  {r['duration_s']}s" if r["duration_s"] else ""
        why = f"  ({r['reason']})" if r["reason"] else ""
        print(f"  {r['result']:<20} {r['file']}{extra}{why}")
    print(f"  Saved: {csv_path}\n         {md_path}")
