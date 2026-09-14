import argparse
import csv
import hashlib
import html
import io
import json
import math
import statistics
from pathlib import Path


IDENTITY_KEYS = ("task_id", "episode_index", "init_state_id", "seed")
EPISODE_COLUMNS = (
    "task_id", "task_name", "instruction", "episode_index", "init_state_id", "seed", "success",
    "policy_steps", "seconds", "terminated", "truncated", "peak_gpu_memory_bytes", "video", "record_path",
)
METRIC_COLUMNS = (
    "completed_episodes", "successes", "failures", "success_rate", "total_policy_steps", "mean_policy_steps",
    "total_episode_seconds", "mean_episode_seconds", "median_episode_seconds", "min_episode_seconds",
    "max_episode_seconds", "episodes_at_step_limit", "peak_gpu_memory_bytes", "recorded_videos",
)
TIMING_NOTE = (
    "Episode seconds include reset, inference, simulation, and recorded video-frame writes. Their sum is not "
    "total app time or GPU kernel time: model loading, between-episode persistence, video finalization, "
    "preparation, and downloads are excluded."
)
MEMORY_NOTE = (
    "Peak memory is the cumulative PyTorch allocated-memory high-water mark at episode completion, "
    "not an independently measured per-episode peak or total device memory usage."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def metrics(rows, max_steps):
    seconds = [row["seconds"] for row in rows]
    count = len(rows)
    successes = sum(row["success"] for row in rows)
    steps = sum(row["policy_steps"] for row in rows)
    return {
        "completed_episodes": count, "successes": successes, "failures": count - successes,
        "success_rate": successes / count, "total_policy_steps": steps, "mean_policy_steps": steps / count,
        "total_episode_seconds": sum(seconds), "mean_episode_seconds": sum(seconds) / count,
        "median_episode_seconds": statistics.median(seconds), "min_episode_seconds": min(seconds),
        "max_episode_seconds": max(seconds),
        "episodes_at_step_limit": sum(row["policy_steps"] == max_steps for row in rows),
        "peak_gpu_memory_bytes": max(row["peak_gpu_memory_bytes"] for row in rows),
        "recorded_videos": sum(bool(row.get("video")) for row in rows),
    }


def build_report_data(result_dir):
    root = Path(result_dir).resolve()
    hashes = {}

    def content(relative):
        path = root / relative
        require(path.resolve().is_relative_to(root), f"Artifact escapes result directory: {relative}")
        require(path.is_file() and path.stat().st_size > 0, f"Missing or empty result artifact: {relative}")
        data = path.read_bytes()
        hashes[relative] = hashlib.sha256(data).hexdigest()
        return data

    def read(relative):
        try:
            value = json.loads(content(relative))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON result artifact: {relative}") from exc
        require(isinstance(value, dict), f"Expected a JSON object: {relative}")
        return value

    settings = read("settings.json")
    summary = read("summary.json")
    status = read("status.json")
    require(status.get("state") == "completed" and summary.get("complete") is True, "Evaluation is not complete")
    try:
        config = settings["config"]
        protocol = config["protocol"]
        task_ids = protocol["task_ids"]
        per_task = protocol["episodes_per_task"]
        max_steps = protocol["max_policy_steps"]
        offset = protocol["initial_state_offset"]
        tasks = settings["checkpoint"]["tasks"]
        plan = settings["episode_plan"]
        require(config["source_run"] == settings["checkpoint"]["source_run"] and
                config["checkpoint_step"] == settings["checkpoint"]["checkpoint_step"], "Checkpoint identity differs")
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing or malformed evaluation settings") from exc
    require(isinstance(task_ids, list) and task_ids and all(integer(task) for task in task_ids), "Invalid task IDs")
    require(len(set(task_ids)) == len(task_ids), "Duplicate task IDs")
    require(integer(per_task, 1) and integer(max_steps, 1) and integer(offset), "Invalid episode protocol")
    require(isinstance(tasks, list) and all(isinstance(task, dict) and integer(task.get("task_id")) for task in tasks),
            "Invalid task metadata")
    task_map = {task["task_id"]: task for task in tasks}
    require(len(tasks) == len(task_map) == len(task_ids) and set(task_map) == set(task_ids), "Task metadata differs from protocol")
    for task in tasks:
        require(isinstance(task.get("instruction"), str) and task["instruction"] and
                isinstance(task.get("name"), str) and integer(task.get("initial_state_count"), offset + per_task),
                "Invalid task instruction or initial-state count")
    expected = {(task, episode) for task in task_ids for episode in range(per_task)}
    require(isinstance(plan, list) and len(plan) == len(expected), "Incomplete episode plan")
    require(all(isinstance(entry, dict) and all(integer(entry.get(key)) for key in IDENTITY_KEYS) for entry in plan),
            "Invalid episode plan identity")
    require({(entry["task_id"], entry["episode_index"]) for entry in plan} == expected, "Duplicate or missing planned episode")
    expected_paths = {f"episodes/task_{task:02d}/episode_{episode:03d}.json" for task, episode in expected}
    actual_paths = {path.relative_to(root).as_posix() for path in (root / "episodes").rglob("*.json")}
    require(actual_paths == expected_paths, "Missing or unexpected episode JSON files")
    rows = []
    for entry in sorted(plan, key=lambda item: (item["task_id"], item["episode_index"])):
        task_id, episode_index = entry["task_id"], entry["episode_index"]
        require(entry["init_state_id"] == offset + episode_index and entry["seed"] < 2**32, "Invalid planned state or seed")
        relative = f"episodes/task_{task_id:02d}/episode_{episode_index:03d}.json"
        row = read(relative)
        require(all(integer(row.get(key)) and row[key] == entry[key] for key in IDENTITY_KEYS), f"Episode identity differs: {relative}")
        require(row.get("instruction") == task_map[task_id]["instruction"], f"Episode instruction differs: {relative}")
        require(all(type(row.get(key)) is bool for key in ("success", "terminated", "truncated")), f"Invalid outcome: {relative}")
        require(row["success"] or row["terminated"] or row["truncated"], f"Missing episode end condition: {relative}")
        require(integer(row.get("policy_steps"), 1) and row["policy_steps"] <= max_steps, f"Invalid policy steps: {relative}")
        require(number(row.get("seconds")) and row["seconds"] > 0, f"Invalid episode time: {relative}")
        require(integer(row.get("peak_gpu_memory_bytes")), f"Invalid memory measurement: {relative}")
        if episode_index == 0:
            require(row.get("video") == f"videos/task_{task_id:02d}.mp4", f"Missing or unsafe video reference: {relative}")
            content(row["video"])
        else:
            require(not row.get("video"), f"Unexpected video reference: {relative}")
        rows.append({**row, "task_name": task_map[task_id]["name"], "record_path": relative})
    task_results = [{"task_id": task_id, "task_name": task_map[task_id]["name"],
                     "instruction": task_map[task_id]["instruction"],
                     **metrics([row for row in rows if row["task_id"] == task_id], max_steps)} for task_id in sorted(task_ids)]
    overall = metrics(rows, max_steps)
    macro = sum(task["success_rate"] for task in task_results) / len(task_results)
    for key, expected_value in (("expected_episodes", len(expected)), ("completed_episodes", len(rows)), ("successes", overall["successes"])):
        require(integer(summary.get(key)) and summary[key] == expected_value, f"Summary {key} differs from episode records")
    for key, expected_value in (("macro_success_rate", macro), ("success_rate_completed", overall["success_rate"])):
        require(number(summary.get(key)) and math.isclose(summary[key], expected_value, rel_tol=0, abs_tol=1e-12),
                f"Summary {key} differs from episode records")
    require(isinstance(summary.get("per_task"), dict) and set(summary["per_task"]) == {str(task) for task in task_ids},
            "Summary task IDs differ")
    for task in task_results:
        saved = summary["per_task"][str(task["task_id"])]
        require(isinstance(saved, dict), "Invalid per-task summary")
        for key in ("completed_episodes", "successes"):
            require(integer(saved.get(key)) and saved[key] == task[key], f"Per-task {key} differs")
        require(number(saved.get("success_rate")) and math.isclose(saved["success_rate"], task["success_rate"], rel_tol=0, abs_tol=1e-12),
                "Per-task success rate differs")
    return {"report_version": 1, "run_name": root.name, "settings": settings,
            "overall": {**overall, "task_count": len(task_ids), "episodes_per_task": per_task, "macro_success_rate": macro},
            "tasks": task_results, "episodes": rows, "source_sha256": hashes,
            "timing_note": TIMING_NOTE, "memory_note": MEMORY_NOTE}


def csv_text(rows, columns):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        values = {key: row.get(key, "") for key in columns}
        for key, value in values.items():
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
                values[key] = "'" + value
        writer.writerow(values)
    return output.getvalue()


def html_text(report):
    def escape(value):
        return html.escape(str(value), quote=True)

    overall = report["overall"]
    config = report["settings"]["config"]
    protocol = config["protocol"]
    task_rows = []
    for task in report["tasks"]:
        task_rows.append("<tr>" + "".join(f"<td>{escape(value)}</td>" for value in (
            task["task_id"], task["instruction"], task["completed_episodes"], task["successes"], task["failures"],
            f"{task['success_rate']:.1%}", f"{task['mean_policy_steps']:.1f}", f"{task['mean_episode_seconds']:.1f}",
            f"{task['total_episode_seconds'] / 60:.1f}", task["episodes_at_step_limit"],
        )) + "</tr>")
    episode_rows = []
    for row in report["episodes"]:
        cells = "".join(f"<td>{escape(value)}</td>" for value in (
            row["task_id"], row["episode_index"], row["instruction"], "Success" if row["success"] else "Failure",
            row["policy_steps"], f"{row['seconds']:.2f}", row["init_state_id"], row["seed"],
            row["terminated"], row["truncated"], f"{row['peak_gpu_memory_bytes'] / 2**30:.2f}",
        ))
        video = f'<a href="{escape(row["video"])}">Video</a>' if row.get("video") else "Not recorded"
        episode_rows.append(f'<tr data-success="{str(row["success"]).lower()}">{cells}<td>{video}</td>'
                            f'<td><a href="{escape(row["record_path"])}">JSON</a></td></tr>')
    provenance = {key: value for key, value in report["settings"].items() if key != "episode_plan"}
    title = escape(report["run_name"])
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — evaluation report</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ max-width: 1600px; margin: 32px auto; padding: 0 24px; line-height: 1.5; }}
h1 {{ overflow-wrap: anywhere; }} a {{ color: light-dark(#145bb4, #8abaff); }}
.cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 24px 0; }}
.card {{ border: 1px solid #8886; border-radius: 8px; padding: 16px 24px; }}
.card strong {{ display: block; font-size: 1.6rem; }}
.scroll {{ overflow-x: auto; max-height: 75vh; border: 1px solid #8886; }}
table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
th, td {{ padding: 8px 12px; border-bottom: 1px solid #8884; text-align: left; vertical-align: top; }}
th {{ position: sticky; top: 0; background: light-dark(#eee, #252525); white-space: nowrap; }}
tr[data-success="false"] {{ background: #b8505012; }}
input, select {{ padding: 8px; margin: 8px 12px 12px 0; font: inherit; }}
input {{ width: min(420px, 80vw); }} pre {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
.note {{ color: light-dark(#555, #bbb); }}
</style></head><body>
<h1>{title}</h1>
<p>Verified complete result set. Suite: <strong>{escape(protocol['suite'])}</strong> · Stage: {escape(config['stage'])}
 · Checkpoint: {escape(config['source_run'])}, step {escape(config['checkpoint_step'])}</p>
<div class="cards">
<div class="card">Success rate<strong>{overall['success_rate']:.1%}</strong>{overall['successes']} successes / {overall['completed_episodes']} episodes</div>
<div class="card">Coverage<strong>{overall['task_count']} tasks × {overall['episodes_per_task']} episodes</strong>{overall['completed_episodes']} episode records validated</div>
<div class="card">Summed episode time<strong>{overall['total_episode_seconds'] / 3600:.2f} hours</strong>{overall['mean_episode_seconds']:.1f} seconds/episode on average</div>
<div class="card">Policy actions<strong>{overall['total_policy_steps']:,}</strong>{overall['episodes_at_step_limit']} episodes reached the step limit</div>
</div>
<p><a href="episodes.csv">All episodes CSV</a> · <a href="tasks.csv">Per-task CSV</a> ·
<a href="report_data.json">Report data snapshot</a> · <a href="settings.json">Full settings and episode plan</a> ·
<a href="summary.json">Original summary</a> · <a href="status.json">Run status</a></p>
<p class="note">{escape(report['timing_note'])} {escape(report['memory_note'])}</p>
<p>{overall['recorded_videos']} videos were recorded: the first episode of each task. All {overall['completed_episodes']} episodes
are included below. Other videos and per-step action/observation trajectories were not recorded and cannot be reconstructed from these results.
Episode and task IDs are zero-based. CSV text that could be interpreted as a spreadsheet formula is prefixed with an apostrophe; JSON retains the original text.</p>
<h2>Per-task results</h2><div class="scroll"><table><thead><tr>
<th>Task ID</th><th>Instruction</th><th>Episodes</th><th>Successes</th><th>Failures</th><th>Success rate</th>
<th>Mean actions</th><th>Mean seconds</th><th>Total minutes</th><th>At step limit</th>
</tr></thead><tbody>{''.join(task_rows)}</tbody></table></div>
<h2>All episodes</h2>
<label>Search <input id="search" type="search" placeholder="Task, instruction, seed, episode…"></label>
<label>Outcome <select id="outcome"><option value="">All</option><option value="true">Success</option><option value="false">Failure</option></select></label>
<p><span id="shown">{overall['completed_episodes']}</span> / {overall['completed_episodes']} episodes shown</p>
<div class="scroll"><table id="episodes"><thead><tr>
<th>Task ID</th><th>Episode ID</th><th>Instruction</th><th>Outcome</th><th>Actions</th><th>Seconds</th>
<th>Initial state ID</th><th>Seed</th><th>Terminated</th><th>Truncated</th><th>Cumulative peak GiB</th><th>Video</th><th>Record</th>
</tr></thead><tbody>{''.join(episode_rows)}</tbody></table></div>
<details><summary><h2>Configuration and provenance</h2></summary><pre>{escape(json.dumps(provenance, indent=2, sort_keys=True))}</pre></details>
<script>
const search = document.getElementById('search');
const outcome = document.getElementById('outcome');
function filterRows() {{
  const query = search.value.toLowerCase();
  let shown = 0;
  document.querySelectorAll('#episodes tbody tr').forEach(row => {{
    row.hidden = !row.textContent.toLowerCase().includes(query) || (outcome.value !== '' && row.dataset.success !== outcome.value);
    if (!row.hidden) shown++;
  }});
  document.getElementById('shown').textContent = shown;
}}
search.addEventListener('input', filterRows);
outcome.addEventListener('change', filterRows);
</script></body></html>
'''


def render_reports(report, result_dir):
    root = Path(result_dir)
    outputs = {
        "report_data.json": json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        "episodes.csv": csv_text(report["episodes"], EPISODE_COLUMNS),
        "tasks.csv": csv_text(report["tasks"], ("task_id", "task_name", "instruction", *METRIC_COLUMNS)),
        "report.html": html_text(report),
    }
    for name, text in outputs.items():
        path = root / name
        if path.exists() and (not path.is_file() or path.read_bytes() != text.encode("utf-8")):
            raise FileExistsError(f"Refusing to overwrite existing report: {path}")
    for name, text in outputs.items():
        path = root / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(text.encode("utf-8"))


def create_report(result_dir):
    report = build_report_data(result_dir)
    render_reports(report, result_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate a complete LingBot evaluation and create offline HTML/CSV reports without inference.")
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = create_report(args.result_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Result reporting failed: {exc}\n")
    print(f"Validated {report['overall']['completed_episodes']} episodes; open {args.result_dir / 'report.html'}")


if __name__ == "__main__":
    main()
