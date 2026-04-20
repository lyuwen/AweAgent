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
            if "unresolved" in status:
                self._unresolved += 1
            elif "resolved" in status:
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
            if result.get("accepted"):
                resolved_ids.append(iid)
            else:
                unresolved_ids.append(iid)
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
    return {
        "accepted": eval_result.accepted,
        "details": eval_result.details,
        "duration": eval_result.duration,
    }


async def _build_async_report(
    dataset_instances: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
    docker_image_prefix: str | None,
    max_concurrent: int,
    timeout: int,
    progress_file: Path,
    remove_image_after_eval: bool,
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
        if "accepted" in record or "error" in record:
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
        if cached is not None and ("accepted" in cached or "error" in cached):
            status = "cached-error" if "error" in cached else ("cached-resolved" if cached.get("accepted") else "cached-unresolved")
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
                    "duration": result.get("duration", 0.0),
                    "details": result.get("details", {}),
                }
                results_by_id[instance_id] = record
                await append_progress_record(progress_file, write_lock, record)
                status = "resolved" if result["accepted"] else "unresolved"
                await tracker.advance(instance_id, status)
            except Exception as exc:
                record = {"instance_id": instance_id, "error": str(exc)}
                results_by_id[instance_id] = record
                await append_progress_record(progress_file, write_lock, record)
                await tracker.advance(instance_id, "error")

    await asyncio.gather(*(_evaluate_if_needed(instance) for instance in dataset_instances))

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
