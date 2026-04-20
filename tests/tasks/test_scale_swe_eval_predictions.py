"""Tests for the Scale SWE offline evaluation CLI."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLI_PATH = REPO_ROOT / "recipes" / "scale_swe" / "eval_predictions.py"
CLI_PATH = Path(os.environ.get("AWE_SCALE_SWE_EVAL_CLI", DEFAULT_CLI_PATH))
TEST_ASSET_ROOT = REPO_ROOT
EXPECTED_REPORT_KEYS = {
    "completed_ids",
    "completed_instances",
    "empty_patch_ids",
    "empty_patch_instances",
    "error_ids",
    "error_instances",
    "incomplete_ids",
    "resolved_ids",
    "resolved_instances",
    "schema_version",
    "submitted_ids",
    "submitted_instances",
    "total_instances",
    "unresolved_ids",
    "unresolved_instances",
}


def _load_module():
    spec = importlib.util.spec_from_file_location("scale_swe_eval_predictions", CLI_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_predictions_cli_file_exists():
    assert CLI_PATH.exists()


def test_cli_path_defaults_to_same_checkout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AWE_SCALE_SWE_EVAL_CLI", raising=False)
    default_cli = REPO_ROOT / "recipes" / "scale_swe" / "eval_predictions.py"
    resolved_cli = Path(os.environ.get("AWE_SCALE_SWE_EVAL_CLI", default_cli))
    assert resolved_cli == default_cli


def test_parse_args_accepts_frozen_cli_contract(monkeypatch: pytest.MonkeyPatch):
    module = _load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_predictions.py",
            "--data-file",
            "assets/scale-swe-batch1.jsonl",
            "--predictions-file",
            "assets/predictions.jsonl",
            "--output-file",
            "report.json",
            "--docker-image-prefix",
            "harbor.zhejianglab.com/zj021",
            "--max-concurrent",
            "8",
            "--timeout",
            "600",
        ],
    )

    args = module.parse_args()

    assert args.data_file == "assets/scale-swe-batch1.jsonl"
    assert args.predictions_file == "assets/predictions.jsonl"
    assert args.output_file == "report.json"
    assert args.docker_image_prefix == "harbor.zhejianglab.com/zj021"
    assert args.max_concurrent == 8
    assert args.timeout == 600


def test_build_report_classifies_instances():
    module = _load_module()

    def evaluate_prediction(instance, *extra_args):
        assert extra_args is not None
        if instance["instance_id"] == "error-case":
            raise RuntimeError("boom")
        return {
            "resolved-case": {"accepted": True},
            "unresolved-case": {"accepted": False},
        }[instance["instance_id"]]

    report = module.build_report(
        dataset_instances=[
            {
                "instance_id": "resolved-case",
                "image_url": "aweaiteam/scaleswe/resolved:latest",
                "workdir": "/workspace",
            },
            {
                "instance_id": "unresolved-case",
                "image_url": "aweaiteam/scaleswe/unresolved:latest",
                "workdir": "/workspace",
            },
            {
                "instance_id": "error-case",
                "image_url": "aweaiteam/scaleswe/error:latest",
                "workdir": "/workspace",
            },
            {
                "instance_id": "empty-case",
                "image_url": "aweaiteam/scaleswe/empty:latest",
                "workdir": "/workspace",
            },
            {
                "instance_id": "missing-case",
                "image_url": "aweaiteam/scaleswe/missing:latest",
                "workdir": "/workspace",
            },
        ],
        predictions_by_id={
            "resolved-case": {
                "instance_id": "resolved-case",
                "model_name_or_path": "test-model",
                "model_patch": "diff --git a/a.py b/a.py\n",
            },
            "unresolved-case": {
                "instance_id": "unresolved-case",
                "model_name_or_path": "test-model",
                "model_patch": "diff --git a/b.py b/b.py\n",
            },
            "error-case": {
                "instance_id": "error-case",
                "model_name_or_path": "test-model",
                "model_patch": "diff --git a/c.py b/c.py\n",
            },
            "empty-case": {
                "instance_id": "empty-case",
                "model_name_or_path": "test-model",
                "model_patch": "   ",
            },
        },
        evaluate_prediction=evaluate_prediction,
        docker_image_prefix="harbor.zhejianglab.com/zj021",
    )

    assert set(report) == EXPECTED_REPORT_KEYS
    assert report["schema_version"]
    assert report["total_instances"] == 5
    assert report["submitted_ids"] == [
        "empty-case",
        "error-case",
        "resolved-case",
        "unresolved-case",
    ]
    assert report["submitted_instances"] == 4
    assert report["incomplete_ids"] == ["missing-case"]
    assert report["empty_patch_ids"] == ["empty-case"]
    assert report["resolved_ids"] == ["resolved-case"]
    assert report["unresolved_ids"] == ["unresolved-case"]
    assert report["error_ids"] == ["error-case"]
    assert report["completed_ids"] == ["error-case", "resolved-case", "unresolved-case"]
    assert report["completed_instances"] == 3
    assert report["resolved_instances"] == 1
    assert report["unresolved_instances"] == 1
    assert report["error_instances"] == 1
    assert report["empty_patch_instances"] == 1
    assert report["completed_ids"] == sorted(
        report["resolved_ids"] + report["unresolved_ids"] + report["error_ids"]
    )


def test_load_predictions_rejects_duplicate_instance_ids(tmp_path: Path):
    module = _load_module()

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instance_id": "dup-case",
                        "model_name_or_path": "model-a",
                        "model_patch": "diff --git a/a.py b/a.py\\n",
                    }
                ),
                json.dumps(
                    {
                        "instance_id": "dup-case",
                        "model_name_or_path": "model-b",
                        "model_patch": "diff --git a/b.py b/b.py\\n",
                    }
                ),
            ]
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="duplicate"):
        module.load_predictions(predictions_path)


def test_load_predictions_rejects_non_strict_schema(tmp_path: Path):
    module = _load_module()

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": "schema-case",
                "model_name_or_path": "test-model",
                "model_patch": "diff --git a/a.py b/a.py\\n",
                "extra_field": "not-allowed",
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="schema"):
        module.load_predictions(predictions_path)


def test_load_predictions_rejects_missing_required_keys(tmp_path: Path):
    module = _load_module()

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "model_name_or_path": "test-model",
                "model_patch": "diff --git a/a.py b/a.py\\n",
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="missing required keys"):
        module.load_predictions(predictions_path)


def test_build_report_treats_missing_model_patch_as_empty():
    module = _load_module()

    def evaluate_prediction(*extra_args):
        assert extra_args is not None
        return {"accepted": True}

    report = module.build_report(
        dataset_instances=[
            {
                "instance_id": "missing-patch-case",
                "image_url": "aweaiteam/scaleswe/missing-patch:latest",
                "workdir": "/workspace",
            }
        ],
        predictions_by_id={
            "missing-patch-case": {
                "instance_id": "missing-patch-case",
                "model_name_or_path": "test-model",
            }
        },
        evaluate_prediction=evaluate_prediction,
        docker_image_prefix=None,
    )

    assert report["submitted_ids"] == ["missing-patch-case"]
    assert report["empty_patch_ids"] == ["missing-patch-case"]
    assert report["completed_ids"] == []
    assert report["resolved_ids"] == []
    assert report["unresolved_ids"] == []
    assert report["error_ids"] == []


def test_run_loads_dataset_via_scale_swe_task_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_module()

    raw_instance = {
        "instance_id": "asset-case",
        "image_url": "aweaiteam/scaleswe:latest",
        "workdir": "/workspace",
    }
    captured: dict[str, object] = {}

    class FakeInstance:
        def __init__(self, raw: dict[str, str]):
            self.metadata = {"raw": raw}

    class FakeTask:
        def __init__(self, data_file: str):
            captured["data_file"] = data_file

        def get_instances(self):
            return [FakeInstance(raw_instance)]

    async def fake_build_async_report(
        dataset_instances,
        predictions_by_id,
        docker_image_prefix,
        max_concurrent,
        timeout,
    ):
        captured["dataset_instances"] = dataset_instances
        captured["predictions_by_id"] = predictions_by_id
        captured["docker_image_prefix"] = docker_image_prefix
        captured["max_concurrent"] = max_concurrent
        captured["timeout"] = timeout
        return {
            "completed_ids": [],
            "completed_instances": 0,
            "empty_patch_ids": [],
            "empty_patch_instances": 0,
            "error_ids": [],
            "error_instances": 0,
            "incomplete_ids": ["asset-case"],
            "resolved_ids": [],
            "resolved_instances": 0,
            "schema_version": "1.0",
            "submitted_ids": [],
            "submitted_instances": 0,
            "total_instances": 1,
            "unresolved_ids": [],
            "unresolved_instances": 0,
        }

    fake_task_module = types.ModuleType("awe_agent.tasks.scale_swe.task")
    setattr(fake_task_module, "ScaleSWETask", FakeTask)
    monkeypatch.setitem(sys.modules, "awe_agent.tasks.scale_swe.task", fake_task_module)
    monkeypatch.setattr(module, "_build_async_report", fake_build_async_report)

    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps(
            {
                "instance_id": "asset-case",
                "model_name_or_path": "test-model",
                "model_patch": "diff --git a/a.py b/a.py\\n",
            }
        )
        + "\n"
    )
    output_path = tmp_path / "report.json"
    data_path = TEST_ASSET_ROOT / "assets" / "scale-swe-batch1.jsonl"

    args = argparse.Namespace(
        data_file=str(data_path),
        predictions_file=str(predictions_path),
        output_file=str(output_path),
        docker_image_prefix="harbor.zhejianglab.com/zj021",
        max_concurrent=8,
        timeout=600,
    )

    asyncio.run(module._run(args))

    assert captured["data_file"] == str(data_path)
    assert captured["dataset_instances"] == [raw_instance]
    assert captured["docker_image_prefix"] == "harbor.zhejianglab.com/zj021"
    assert captured["max_concurrent"] == 8
    assert captured["timeout"] == 600
    assert json.loads(output_path.read_text())["incomplete_ids"] == ["asset-case"]


