# ScaleSWE

Run the ScaleSWE benchmark — a large-scale SWE-bench style coding evaluation. Uses the SearchSWE agent scaffold in non-search mode with CodeAct XML tool calling.

## Dataset

Download the ScaleSWE dataset from the [Hugging Face](https://huggingface.co/datasets/AweAI-Team/Scale-SWE):

```python
from datasets import load_dataset

# See the collection page for available splits
dataset = load_dataset("AweAI-Team/Scale-SWE")
```

Or download the JSONL file directly from the Hugging Face repo page.

## Run the benchmark with an agent

If you want to run the full agent workflow, use `recipes/scale_swe/run.py`.

```bash
python recipes/scale_swe/run.py \
    --data-file /path/to/scaleswe.jsonl --mode batch
```

## Offline evaluation of saved predictions

To score precomputed patches without running the agent, use `recipes/scale_swe/eval_predictions.py`.

This CLI reads:

- a ScaleSWE dataset JSONL via `--data-file`
- a predictions JSONL via `--predictions-file`
- writes a JSON summary report via `--output-file`

It evaluates only the submitted patches. It does not generate predictions and it does not run the agent loop.

### Required inputs

#### Dataset file: `--data-file`

The dataset file must be a ScaleSWE JSONL file. The evaluator loads instances from this file and matches them by `instance_id`.

#### Predictions file: `--predictions-file`

The predictions file must be JSONL. Each line must be a JSON object with exactly these keys:

```json
{
  "instance_id": "auth0__auth0-python-001",
  "model_name_or_path": "my-model-name",
  "model_patch": "diff --git a/file.py b/file.py\n..."
}
```

Current schema rules are strict:

- required keys: `instance_id`, `model_name_or_path`, `model_patch`
- no extra keys are allowed
- duplicate `instance_id` values are rejected
- empty or whitespace-only `model_patch` values are allowed, but they are reported as empty patches and are not evaluated

`model_name_or_path` is required by the schema, but the current evaluator does not use it for scoring.

### CLI arguments

```text
--data-file PATH             Path to the ScaleSWE dataset JSONL file (required)
--predictions-file PATH      Path to the predictions JSONL file (required)
--output-file PATH           Path to write the JSON report (required)
--docker-image-prefix TEXT   Optional Docker registry prefix override
--max-concurrent N           Max parallel evaluations (default: 4)
--timeout N                  Per-instance evaluation timeout in seconds (default: 3600)
```

Argument details:

- `--output-file` creates parent directories automatically if they do not already exist
- `--docker-image-prefix` overrides the registry prefix when resolving dataset image URLs
- `--max-concurrent` limits how many non-empty predictions are evaluated at once
- `--timeout` applies per instance

### Example usage

```bash
python recipes/scale_swe/eval_predictions.py \
    --data-file assets/scale-swe-batch1.jsonl \
    --predictions-file assets/predictions.jsonl \
    --output-file reports/scale_swe_eval.json
```

With a custom registry prefix, concurrency, and timeout:

```bash
python recipes/scale_swe/eval_predictions.py \
    --data-file assets/scale-swe-batch1.jsonl \
    --predictions-file assets/predictions.jsonl \
    --output-file reports/scale_swe_eval.json \
    --docker-image-prefix harbor.zhejianglab.com/zj021 \
    --max-concurrent 8 \
    --timeout 600
```

## Output report

The CLI writes one JSON object to `--output-file`.

Example:

```json
{
  "completed_ids": [
    "auth0__auth0-python-001",
    "fastapi__fastapi-002"
  ],
  "completed_instances": 2,
  "empty_patch_ids": [
    "pallets__flask-003"
  ],
  "empty_patch_instances": 1,
  "error_ids": [],
  "error_instances": 0,
  "incomplete_ids": [
    "django__django-004"
  ],
  "resolved_ids": [
    "auth0__auth0-python-001"
  ],
  "resolved_instances": 1,
  "schema_version": "1.0",
  "submitted_ids": [
    "auth0__auth0-python-001",
    "fastapi__fastapi-002",
    "pallets__flask-003"
  ],
  "submitted_instances": 3,
  "total_instances": 4,
  "unresolved_ids": [
    "fastapi__fastapi-002"
  ],
  "unresolved_instances": 1
}
```

### Exact report semantics

- `schema_version`: current report schema version. The current value is `"1.0"`.
- `total_instances`: total number of dataset instances in `--data-file`.
- `submitted_ids`: dataset instance IDs that have a matching prediction record in `--predictions-file`.
- `submitted_instances`: count of `submitted_ids`.
- `incomplete_ids`: dataset instance IDs that do not have any prediction record.
- `resolved_ids`: submitted instances with a non-empty patch whose evaluation returned `accepted=true`.
- `resolved_instances`: count of `resolved_ids`.
- `unresolved_ids`: submitted instances with a non-empty patch whose evaluation completed but did not resolve the task.
- `unresolved_instances`: count of `unresolved_ids`.
- `error_ids`: submitted instances with a non-empty patch whose evaluation raised an error.
- `error_instances`: count of `error_ids`.
- `empty_patch_ids`: submitted instances whose `model_patch` is empty or whitespace-only.
- `empty_patch_instances`: count of `empty_patch_ids`.
- `completed_ids`: only `resolved_ids + unresolved_ids + error_ids`, sorted.
- `completed_instances`: count of `completed_ids`.

The important distinctions are:

- `submitted_ids` means "a prediction record exists for this dataset instance." It includes empty-patch submissions.
- `incomplete_ids` means "this dataset instance has no prediction record at all."
- `completed_ids` includes only instances that reached an evaluation outcome: resolved, unresolved, or error.
- `completed_ids` does not include `empty_patch_ids`.
- `completed_ids` does not include `incomplete_ids`.

So the relationship is:

- `submitted_ids = resolved_ids + unresolved_ids + error_ids + empty_patch_ids`
- `completed_ids = resolved_ids + unresolved_ids + error_ids`

Predictions whose `instance_id` does not appear in the dataset are ignored by the report.

## Evaluation behavior

For each submitted instance with a non-empty patch, the CLI:

1. loads the ScaleSWE instance from `--data-file`
2. resolves the Docker image, optionally using `--docker-image-prefix`
3. runs `ScaleSWEEvaluator`
4. records whether the patch was accepted

The output report is summary-only. It does not include per-instance evaluator details beyond the resolved, unresolved, error, empty-patch, and incomplete buckets.

## Troubleshooting

- `Prediction schema violation: missing required keys ...`: each prediction record must contain exactly `instance_id`, `model_name_or_path`, and `model_patch`
- `Prediction schema violation: unexpected keys ...`: remove extra fields from the predictions JSONL
- `Predictions contain duplicate instance_id`: ensure each prediction appears only once
- Docker or image pull failures: verify Docker is running and the resolved images are accessible from your registry
- Long-running evaluations: lower `--max-concurrent`, raise `--timeout`, or both
