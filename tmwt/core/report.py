"""
Reports.

End of a review (write_report), in the output directory:
  labeling_report.csv — one row per video, for analysis in a spreadsheet.
  labeling_report.md  — a readable summary with counts and the same table.

End of processing (write_processing_report), in <videos>/tmwt_analysis/:
  processing_report.csv / .md — per video: whether processing worked, the pose
  model used in the end ("model_strength"), and the pose check's result,
  including videos whose anomalies remain for the reviewer to confirm.
"""

import csv
import os
from datetime import datetime

from tmwt.core import analysis_file

from tmwt.core.job import COURSE_M, REVIEW_APPROVED, REVIEW_REJECTED, STATUS_FAILED, STATUS_NO_BODY

# Overall results, in the order they're listed.
APPROVED = "approved"
UNREVIEWED = "auto (not reviewed)"
REJECTED = "rejected"
FAILED = "failed"
RESULTS = (APPROVED, UNREVIEWED, REJECTED, FAILED)

COLUMNS = [
    "file", "result", "reason", "endpoints", "timing", "start_method", "endpoint_behavior",
    "start_s", "end_s", "duration_s", "speed_mps", "model_strength", "pose_check", "outputs_saved",
]
# Processing report status of a video with no (usable) analysis file.
NOT_PROCESSED = "not processed"
PROCESSING_COLUMNS = ["file", "status", "error", "model_requested", "model_strength",
                      "pose_check", "flagged_frames_first", "flagged_frames_final", "examples"]


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
        "model_strength": job.model_strength if job.analysis_meta else "",
        "pose_check": pose_check_text(job),
        "outputs_saved": "yes" if job.saved else "no",
    }


def pose_check_text(job):
    """The pose check's outcome for one video, as the reports and timing JSON give it."""
    if not job.analysis_meta:
        return ""
    if not job.pose_flags:
        return "ok"
    text = f"{job.pose_flagged_frames} frame(s) flagged"
    if job.pose_edits:
        text += f"; {len(job.pose_edits)} point(s) smoothed at review"
    if job.pose_confirmed:
        return text + ("" if job.pose_edits else "; confirmed fine at review")
    if job.review == REVIEW_REJECTED:
        return text + "; video removed at review"
    return text + "; not confirmed"


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
        "| File | Result | Reason | Endpoints | Timing | Start method | Endpoint behavior | Start (s) | End (s) | Duration (s) | Speed (m/s) | Model strength | Pose check | Saved |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(
            r[c].replace("|", "/") if r[c] else "—" for c in COLUMNS) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_processing_report(videos):
    """
    Write processing_report.csv / .md in the videos' tmwt_analysis folder, from
    every video's analysis file (so it covers videos processed in earlier runs
    too). Returns (csv_path, md_path), or None if there are no videos.
    """
    if not videos:
        return None
    rows = [_processing_row(v) for v in videos]
    folder = os.path.dirname(analysis_file.analysis_path(videos[0]))
    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, "processing_report.csv")
    md_path = os.path.join(folder, "processing_report.md")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROCESSING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    remain = [r for r in rows if r["pose_check"].startswith("anomalies remain")]
    fixed = [r for r in rows if r["pose_check"] == "fixed by heavier model"]
    failed = [r for r in rows if r["status"] == STATUS_FAILED]
    unprocessed = [r for r in rows if r["status"] == NOT_PROCESSED]
    lines = [
        "# TMWT processing report", "",
        f"- Written: {datetime.now():%Y-%m-%d %H:%M}",
        f"- Videos: {len(rows)}",
        f"  - failed: {len(failed)}",
        f"  - not processed yet: {len(unprocessed)}",
        f"  - pose anomalies fixed by re-running with the heavier model: {len(fixed)}",
        f"  - pose anomalies remaining (the reviewer is asked to confirm or remove): {len(remain)}",
        "",
        "| " + " | ".join(PROCESSING_COLUMNS) + " |",
        "|" + "---|" * len(PROCESSING_COLUMNS),
    ]
    lines += ["| " + " | ".join(str(r[c]).replace("|", "/") or "—" for c in PROCESSING_COLUMNS) + " |"
              for r in rows]
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nProcessing report: {csv_path}")
    for r in remain:
        print(f"  Pose anomalies remain: {r['file']} ({r['flagged_frames_final']} frame(s): {r['examples']})")
    return csv_path, md_path


def _processing_row(video):
    meta = analysis_file.read_meta(video) or {}
    check = meta.get("pose_check") or {}
    runs = check.get("runs") or []
    final = runs[-1] if runs else {}
    return {
        "file": os.path.basename(video),
        "status": meta.get("status", NOT_PROCESSED),
        "error": meta.get("error") or "",
        "model_requested": meta.get("model", ""),
        "model_strength": meta.get("model_strength", meta.get("model", "")),
        "pose_check": check.get("result", "not checked" if meta else ""),
        "flagged_frames_first": runs[0]["flagged_frames"] if runs else "",
        "flagged_frames_final": final.get("flagged_frames", ""),
        "examples": "; ".join(f"{e['landmark']} {e['kind']} at {e['time_s']:.2f}s"
                              for e in final.get("examples", [])),
    }


def _print_summary(rows, csv_path, md_path):
    print(f"\n{'=' * 60}\nREPORT")
    for r in rows:
        extra = f"  {r['duration_s']}s" if r["duration_s"] else ""
        why = f"  ({r['reason']})" if r["reason"] else ""
        print(f"  {r['result']:<20} {r['file']}{extra}{why}")
    print(f"  Saved: {csv_path}\n         {md_path}")
