"""Offline evaluation CLI for Scale SWE predictions.

Evaluates pre-collected agent patches against the Scale SWE benchmark
without running the agent. Reads a dataset JSONL and a predictions JSONL,
runs each prediction through ScaleSWEEvaluator, and writes a JSON report.

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
import ast
import asyncio
import json
import logging
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
ALLOWED_PREDICTION_KEYS = {"instance_id", "model_name_or_path", "model_patch"}
RUN_INFER_ENV_VAR = "AWE_SCALE_SWE_RUN_INFER_PATH"


def _load_resolve_image_url_from_path(path: Path) -> Callable[[str, str | None], str]:
    source = path.read_text()
    module = ast.parse(source, filename=str(path))
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "resolve_image_url":
            function_module = ast.Module(body=[node], type_ignores=[])
            namespace: dict[str, Any] = {}
            exec(compile(function_module, str(path), "exec"), namespace)
            return namespace["resolve_image_url"]
    raise ValueError(f"resolve_image_url not found in {path}")


def _load_resolve_image_url() -> Callable[[str, str | None], str]:
    try:
        module = import_module("benchmarks.scaleswe.run_infer")
    except ImportError:
        env_path = os.getenv(RUN_INFER_ENV_VAR)
        if env_path:
            return _load_resolve_image_url_from_path(Path(env_path))
        raise ImportError(
            "Could not import benchmarks.scaleswe.run_infer. "
            f"Set {RUN_INFER_ENV_VAR} to the path of run_infer.py."
        )

    resolve = getattr(module, "resolve_image_url", None)
    if resolve is None:
        raise ValueError("benchmarks.scaleswe.run_infer does not define resolve_image_url")
    return resolve


resolve_image_url = _load_resolve_image_url()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Scale SWE predictions offline",
    )
    p.add_argument("--data-file", required=True, help="Path to dataset JSONL")
    p.add_argument("--predictions-file", required=True, help="Path to predictions JSONL")
    p.add_argument("--output-file", required=True, help="Path to write JSON report")
    p.add_argument("--docker-image-prefix", default=None, help="Docker image registry prefix")
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
        RuntimeConfig(backend="docker", image=ctx["image"], workdir=ctx["workdir"]),
    )
    eval_result = await evaluator.evaluate(inst_obj, ctx["patch"], runtime)
    return {"accepted": eval_result.accepted}


async def _build_async_report(
    dataset_instances: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
    docker_image_prefix: str | None,
    max_concurrent: int,
    timeout: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(max_concurrent)
    results_by_id: dict[str, dict[str, Any]] = {}

    async def _evaluate_if_needed(instance: dict[str, Any]) -> None:
        instance_id = instance["instance_id"]
        prediction = predictions_by_id.get(instance_id)
        if prediction is None:
            return

        patch = prediction.get("model_patch", "")
        if not patch or not patch.strip():
            return

        async with semaphore:
            try:
                results_by_id[instance_id] = await _evaluate_instance(
                    instance=instance,
                    prediction=prediction,
                    docker_image_prefix=docker_image_prefix,
                    timeout=timeout,
                )
            except Exception as exc:
                results_by_id[instance_id] = {"error": str(exc)}

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

    from awe_agent.tasks.scale_swe.task import ScaleSWETask

    task = ScaleSWETask(data_file=str(data_path))
    dataset_instances = [instance.metadata["raw"] for instance in task.get_instances()]
    predictions_by_id = load_predictions(predictions_path)

    report = await _build_async_report(
        dataset_instances=dataset_instances,
        predictions_by_id=predictions_by_id,
        docker_image_prefix=args.docker_image_prefix,
        max_concurrent=args.max_concurrent,
        timeout=args.timeout,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report written to %s", output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
