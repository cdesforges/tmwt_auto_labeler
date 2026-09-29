"""
End-of-run report for label.py.

Writes two files to the output directory:
  labeling_report.csv — one row per video, for analysis in a spreadsheet.
  labeling_report.md  — a readable summary with counts and the same table.
"""

import csv
import os
from datetime import datetime

COLUMNS = [
    "file", "result", "reason", "endpoints", "timing", "start_method",
    "start_s", "end_s", "duration_s", "speed_mps", "outputs_saved",
]


def _fmt(value, digits=3):
    return "" if value is None else f"{value:.{digits}f}"


def job_result(job):
    """
    (result, reason) summarising one video's outcome.

    result is one of: approved, auto (not reviewed), rejected, failed.
    """
    if job.review == "rejected":
        return "rejected", job.review_note
    if job.status == "failed":
        return "failed", job.error
    if job.far_ep is None:
        if job.endpoint_problem:
            return "failed", f"{job.endpoint_problem}; rope endpoints need clicking at review"
        return "failed", "rope endpoints not set"
    if job.walk_start is None or job.walk_end is None:
        missing = "walk start" if job.walk_start is None else "walk end"
        return "failed", f"no {missing} detected"
    if job.review == "approved":
        return "approved", job.review_note
    return "auto (not reviewed)", job.review_note


def _row(job, course_m):
    result, reason = job_result(job)
    duration = (job.walk_end - job.walk_start
                if job.walk_start is not None and job.walk_end is not None else None)
    return {
        "file": job.name,
        "result": result,
        "reason": reason,
        "endpoints": job.endpoint_source,
        "timing": job.timing_source,
        "start_method": job.timing_detail,
        "start_s": _fmt(job.walk_start),
        "end_s": _fmt(job.walk_end),
        "duration_s": _fmt(duration),
        "speed_mps": _fmt(course_m / duration if duration else None, 2),
        "outputs_saved": "yes" if job.saved else "no",
    }


def write_report(jobs, output_dir, course_m, backend_name):
    """Write the CSV and Markdown reports. Returns (csv_path, md_path)."""
    rows = [_row(job, course_m) for job in jobs]
    csv_path = os.path.join(output_dir, "labeling_report.csv")
    md_path = os.path.join(output_dir, "labeling_report.md")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["result"]] = counts.get(r["result"], 0) + 1

    lines = [
        "# TMWT labeling report",
        "",
        f"- Run: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Backend: {backend_name}",
        f"- Course length: {course_m:g} m",
        f"- Videos: {len(rows)}",
    ]
    for result in ("approved", "auto (not reviewed)", "rejected", "failed"):
        if counts.get(result):
            lines.append(f"  - {result}: {counts[result]}")
    lines += [
        "",
        "| File | Result | Reason | Endpoints | Timing | Start method | Start (s) | End (s) | Duration (s) | Speed (m/s) | Saved |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(
            r[c].replace("|", "/") if r[c] else "—" for c in COLUMNS) + " |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n{'=' * 60}\nREPORT")
    for r in rows:
        extra = f"  {r['duration_s']}s" if r["duration_s"] else ""
        why = f"  ({r['reason']})" if r["reason"] else ""
        print(f"  {r['result']:<20} {r['file']}{extra}{why}")
    print(f"  Saved: {csv_path}\n         {md_path}")
    return csv_path, md_path
