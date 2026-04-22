"""Offline evaluation CLI for Scale SWE predictions.

Evaluates pre-collected agent patches against the Scale SWE benchmark
without running the agent. Reads a dataset JSONL and a predictions JSONL,
runs each prediction through ScaleSWEEvaluator, writes incremental progress,
and writes a final JSON report.

Usage:
    python recipes/scale_swe/eval_predictions.py \
        --data-file assets/scale-swe-batch1.jsonl \
        --predictions-file assets/predictions.jsonl \
        --output-file report.json \
        --docker-image-prefix harbor.zhejianglab.com/zj021 \
        --max-concurrent 8 \
        --timeout 600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
ALLOWED_PREDICTION_KEYS = {"instance_id", "model_name_or_path", "model_patch"}


class ProgressTracker:
    def __init__(self, total: int) -> None:
        self._total = total
        self._completed = 0
        self._resolved = 0
        self._unresolved = 0
        self._errored = 0
        self._lock = asyncio.Lock()

    async def advance(self, instance_id: str, status: str) -> None:
        async with self._lock:
            self._completed += 1
            if "failed" in status or "unresolved" in status:
                self._unresolved += 1
            elif "passed" in status or "resolved" in status:
                self._resolved += 1
            else:
                self._errored += 1
            width = 24
            filled = 0
            if self._total:
                filled = int(width * self._completed / self._total)
            bar = "#" * filled + "-" * (width - filled)
            counts = f"P:{self._resolved} F:{self._unresolved} E:{self._errored}"
            print(
                f"\r[{bar}] {self._completed}/{self._total} {counts} | {instance_id} {status}",
                end="",
                file=sys.stderr,
                flush=True,
            )
            if self._completed == self._total:
                print(file=sys.stderr, flush=True)


def resolve_image_url(image_url: str, prefix: str | None = None) -> str:
    if not prefix:
        return image_url
    _, _, image_name = image_url.rpartition("/")
    return f"{prefix.rstrip('/')}/{image_name}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Scale SWE predictions offline",
    )
    p.add_argument("--data-file", required=True, help="Path to dataset JSONL")
    p.add_argument("--predictions-file", required=True, help="Path to predictions JSONL")
    p.add_argument("--output-file", required=True, help="Path to write JSON report")
    p.add_argument(
        "--progress-file",
        default=None,
        help="Path to write append-only JSONL progress events (default: <output>.progress.jsonl)",
    )
    p.add_argument("--docker-image-prefix", default=None, help="Docker image registry prefix")
    p.add_argument(
        "--remove-image-after-eval",
        action="store_true",
        help="Remove the Docker image after each instance evaluation finishes",
    )
    p.add_argument("--max-concurrent", type=int, default=4, help="Max parallel evaluations")
    p.add_argument("--timeout", type=int, default=3600, help="Per-instance eval timeout (seconds)")
    p.add_argument(
        "--cleanup-interval",
        type=int,
        default=30,
        help="Minutes between periodic Docker cleanups (0 to disable, default: 30)",
    )
    p.add_argument(
        "--cleanup-min-age",
        type=int,
        default=30,
        help="Only remove dangling images older than this many minutes (default: 30)",
    )
    return p.parse_args()


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing_keys = ALLOWED_PREDICTION_KEYS - set(record.keys())
            if missing_keys:
                raise ValueError(
                    "Prediction schema violation: missing required keys "
                    f"{sorted(missing_keys)}. Required keys: {sorted(ALLOWED_PREDICTION_KEYS)}"
                )
            extra_keys = set(record.keys()) - ALLOWED_PREDICTION_KEYS
            if extra_keys:
                raise ValueError(
                    f"Prediction schema violation: unexpected keys {sorted(extra_keys)}. "
                    f"Allowed keys: {sorted(ALLOWED_PREDICTION_KEYS)}"
                )
            instance_id = record["instance_id"]
            if instance_id in predictions:
                raise ValueError(
                    f"Predictions contain duplicate instance_id: {instance_id!r}"
                )
            predictions[instance_id] = record
    return predictions


def load_progress_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            instance_id = record.get("instance_id")
            if instance_id:
                records[instance_id] = record
    return records


def classify_result_status(result: dict[str, Any]) -> str:
    status = result.get("status")
    if status in {"passed", "failed", "error"}:
        return status
    outcome = result.get("outcome")
    if outcome in {"passed", "failed", "error"}:
        return outcome
    if "error" in result:
        return "error"
    details = result.get("details")
    if isinstance(details, dict) and "error" in details:
        return "error"
    return "passed" if result.get("accepted") else "failed"


def is_terminal_progress_record(record: dict[str, Any]) -> bool:
    return classify_result_status(record) in {"passed", "failed"}


async def append_progress_record(
    path: Path,
    write_lock: asyncio.Lock,
    record: dict[str, Any],
) -> None:
    async with write_lock:
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")


def build_evaluation_context(
    instance: dict[str, Any],
    prediction: dict[str, Any],
    docker_image_prefix: str | None,
) -> dict[str, Any]:
    image_url = instance.get("image_url", "")
    image = resolve_image_url(image_url, docker_image_prefix)
    workdir = instance.get("workdir", "/testbed")
    patch = prediction.get("model_patch", "")
    return {
        "image": image,
        "workdir": workdir,
        "patch": patch,
    }


def build_report(
    dataset_instances: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
    evaluate_prediction: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    docker_image_prefix: str | None = None,
) -> dict[str, Any]:
    _ = docker_image_prefix
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    error_ids: list[str] = []
    empty_patch_ids: list[str] = []
    incomplete_ids: list[str] = []
    submitted_ids: list[str] = []

    for inst in dataset_instances:
        iid = inst["instance_id"]
        prediction = predictions_by_id.get(iid)

        if prediction is None:
            incomplete_ids.append(iid)
            continue

        submitted_ids.append(iid)
        patch = prediction.get("model_patch", "")

        if not patch or not patch.strip():
            empty_patch_ids.append(iid)
            continue

        try:
            result = evaluate_prediction(inst, prediction)
            outcome = classify_result_status(result)
            if outcome == "passed":
                resolved_ids.append(iid)
            elif outcome == "failed":
                unresolved_ids.append(iid)
            else:
                error_ids.append(iid)
        except Exception:
            logger.exception("Evaluation error for %s", iid)
            error_ids.append(iid)

    resolved_ids.sort()
    unresolved_ids.sort()
    error_ids.sort()
    empty_patch_ids.sort()
    incomplete_ids.sort()
    submitted_ids.sort()

    completed_ids = sorted(resolved_ids + unresolved_ids + error_ids)

    return {
        "completed_ids": completed_ids,
        "completed_instances": len(completed_ids),
        "empty_patch_ids": empty_patch_ids,
        "empty_patch_instances": len(empty_patch_ids),
        "error_ids": error_ids,
        "error_instances": len(error_ids),
        "incomplete_ids": incomplete_ids,
        "resolved_ids": resolved_ids,
        "resolved_instances": len(resolved_ids),
        "schema_version": SCHEMA_VERSION,
        "submitted_ids": submitted_ids,
        "submitted_instances": len(submitted_ids),
        "total_instances": len(dataset_instances),
        "unresolved_ids": unresolved_ids,
        "unresolved_instances": len(unresolved_ids),
    }


async def _evaluate_instance(
    instance: dict[str, Any],
    prediction: dict[str, Any],
    docker_image_prefix: str | None,
    timeout: int,
    remove_image_after_eval: bool,
) -> dict[str, Any]:
    from awe_agent.core.runtime import RuntimeConfig
    from awe_agent.core.runtime.docker import DockerRuntime
    from awe_agent.tasks.scale_swe.evaluator import ScaleSWEEvaluator
    from awe_agent.tasks.scale_swe.task import ScaleSWETask

    ctx = build_evaluation_context(instance, prediction, docker_image_prefix)
    task = ScaleSWETask(instances=[instance])
    inst_obj = task.get_instances()[0]
    inst_obj.image = ctx["image"]

    evaluator = ScaleSWEEvaluator(timeout=timeout)
    runtime = DockerRuntime(
        RuntimeConfig(
            backend="docker",
            image=ctx["image"],
            workdir=ctx["workdir"],
            docker={"remove_image_after_use": remove_image_after_eval},
        ),
    )
    eval_result = await evaluator.evaluate(inst_obj, ctx["patch"], runtime)
    outcome = classify_result_status(
        {"accepted": eval_result.accepted, "details": eval_result.details},
    )
    return {
        "accepted": eval_result.accepted,
        "status": outcome,
        "details": eval_result.details,
        "duration": eval_result.duration,
    }


async def _cleanup_dangling_images(min_age_minutes: int = 30) -> int:
    """Remove dangling Docker images older than *min_age_minutes*.

    Mirrors the logic in ``find-dangling-images.sh``: lists all dangling
    images, inspects each creation timestamp, and removes those older than
    the cutoff.  Returns the number of images removed.
    """
    script = f"""\
cutoff=$(date -d '{min_age_minutes} minutes ago' +%s)
removed=0
for img in $(docker images -f "dangling=true" -q); do
    created=$(docker inspect --format '{{{{.Created}}}}' "$img")
    created_epoch=$(date -d "$created" +%s)
    if [ "$created_epoch" -lt "$cutoff" ]; then
        docker image rm "$img" >/dev/null 2>&1 && removed=$((removed + 1))
    fi
done
echo "$removed"
"""
    proc = await asyncio.create_subprocess_shell(
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.warning("Dangling image cleanup failed: %s", stderr.decode().strip())
        return 0
    count = int(stdout.decode().strip() or "0")
    if count:
        logger.info("Removed %d dangling image(s)", count)
    return count


async def _periodic_cleanup_loop(
    interval_seconds: float,
    min_age_minutes: int,
) -> None:
    """Run Docker dangling-image cleanup on a fixed-period timer.

    The interval includes cleanup time so each cycle is exactly
    *interval_seconds* wall-clock seconds apart.
    """
    while True:
        await asyncio.sleep(interval_seconds)
        logger.info("Running periodic Docker cleanup …")
        await _cleanup_dangling_images(min_age_minutes)
        logger.info("Periodic cleanup complete")


async def _build_async_report(
    dataset_instances: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
    docker_image_prefix: str | None,
    max_concurrent: int,
    timeout: int,
    progress_file: Path,
    remove_image_after_eval: bool,
    cleanup_interval_minutes: int = 0,
    cleanup_min_age_minutes: int = 30,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(max_concurrent)
    write_lock = asyncio.Lock()
    results_by_id: dict[str, dict[str, Any]] = {}
    progress_records = load_progress_records(progress_file)
    total_to_track = sum(
        1
        for instance in dataset_instances
        if (prediction := predictions_by_id.get(instance["instance_id"])) is not None
        and prediction.get("model_patch", "").strip()
    )
    tracker = ProgressTracker(total_to_track)

    for instance_id, record in progress_records.items():
        if is_terminal_progress_record(record):
            results_by_id[instance_id] = record

    async def _evaluate_if_needed(instance: dict[str, Any]) -> None:
        instance_id = instance["instance_id"]
        prediction = predictions_by_id.get(instance_id)
        if prediction is None:
            return

        patch = prediction.get("model_patch", "")
        if not patch or not patch.strip():
            return

        cached = progress_records.get(instance_id)
        if cached is not None and is_terminal_progress_record(cached):
            status = f"cached-{classify_result_status(cached)}"
            await tracker.advance(instance_id, status)
            return

        async with semaphore:
            try:
                result = await _evaluate_instance(
                    instance=instance,
                    prediction=prediction,
                    docker_image_prefix=docker_image_prefix,
                    timeout=timeout,
                    remove_image_after_eval=remove_image_after_eval,
                )
                record = {
                    "instance_id": instance_id,
                    "accepted": result["accepted"],
                    "status": result["status"],
                    "duration": result.get("duration", 0.0),
                    "details": result.get("details", {}),
                }
                results_by_id[instance_id] = record
                await append_progress_record(progress_file, write_lock, record)
                await tracker.advance(instance_id, result["status"])
            except Exception as exc:
                record = {"instance_id": instance_id, "status": "error", "error": str(exc)}
                results_by_id[instance_id] = record
                await append_progress_record(progress_file, write_lock, record)
                await tracker.advance(instance_id, "error")

    cleanup_task = None
    if cleanup_interval_minutes > 0:
        cleanup_task = asyncio.create_task(
            _periodic_cleanup_loop(
                interval_seconds=cleanup_interval_minutes * 60,
                min_age_minutes=cleanup_min_age_minutes,
            )
        )

    try:
        await asyncio.gather(*(_evaluate_if_needed(instance) for instance in dataset_instances))
    finally:
        if cleanup_task is not None:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

    def evaluate_prediction(instance: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        result = results_by_id.get(instance["instance_id"])
        if result is None:
            raise ValueError(f"Missing evaluation result for {instance['instance_id']}")
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    return build_report(
        dataset_instances=dataset_instances,
        predictions_by_id=predictions_by_id,
        evaluate_prediction=evaluate_prediction,
        docker_image_prefix=docker_image_prefix,
    )


async def _run(args: argparse.Namespace) -> None:
    data_path = Path(args.data_file)
    predictions_path = Path(args.predictions_file)
    output_path = Path(args.output_file)
    progress_path = Path(args.progress_file) if args.progress_file else Path(f"{output_path}.progress.jsonl")

    from awe_agent.tasks.scale_swe.task import ScaleSWETask

    task = ScaleSWETask(data_file=str(data_path))
    dataset_instances = [instance.metadata["raw"] for instance in task.get_instances()]
    predictions_by_id = load_predictions(predictions_path)

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    report = await _build_async_report(
        dataset_instances=dataset_instances,
        predictions_by_id=predictions_by_id,
        docker_image_prefix=args.docker_image_prefix,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
        progress_file=progress_path,
        remove_image_after_eval=args.remove_image_after_eval,
        cleanup_interval_minutes=args.cleanup_interval,
        cleanup_min_age_minutes=args.cleanup_min_age,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Progress written to %s", progress_path)
    logger.info("Report written to %s", output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
