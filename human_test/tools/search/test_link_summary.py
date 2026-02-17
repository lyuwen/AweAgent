"""Debug: LinkSummaryTool — verify full pipeline (fetch + summarize) with real backends.

Uses real LinkReaderTool (backed by bytedance.bandai_mcp_host) and real LLM
(loaded from YAML config) instead of mocks.

Before running, set environment variables:
    export LINK_SUMMARY_CONFIG_PATH=configs/llm/link_summary/azure.yaml
Or:
    export LINK_SUMMARY_MODEL=gpt-4o-mini
    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=...
"""

import asyncio
import os
from pathlib import Path

from awe_agent.core.tool.search.constraints import SearchConstraints
from awe_agent.core.tool.search.link_reader_tool import LinkReaderTool
from awe_agent.core.tool.search.link_summary_tool import LinkSummaryTool


# ── Helpers ─────────────────────────────────────────────────────────────────

# Auto-detect config path relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "llm" / "link_summary" / "azure.yaml"


def ensure_config():
    """Set LINK_SUMMARY_CONFIG_PATH if not already set."""
    if not os.environ.get("LINK_SUMMARY_CONFIG_PATH") and not os.environ.get("LINK_SUMMARY_MODEL"):
        if _DEFAULT_CONFIG.exists():
            os.environ["LINK_SUMMARY_CONFIG_PATH"] = str(_DEFAULT_CONFIG)
            print(f"  Auto-detected config: {_DEFAULT_CONFIG}")
        else:
            print(f"  WARNING: No config found at {_DEFAULT_CONFIG}")
            print("  Set LINK_SUMMARY_CONFIG_PATH or LINK_SUMMARY_MODEL env var.")


# ── Test scenarios ──────────────────────────────────────────────────────────


async def test_full_pipeline():
    """Full pipeline: real LinkReaderTool fetch + real LLM summarize."""
    print("--- Full pipeline: real fetch + real LLM summarize ---")
    ensure_config()

    tool = LinkSummaryTool()

    result = await tool.execute({
        "url": "https://docs.djangoproject.com/en/5.0/ref/models/querysets/",
        "goal": "How does QuerySet lazy evaluation work?",
    })
    print(f"  Result length: {len(result)} chars")
    print(f"  First 500 chars:\n{result[:500]}")
    print()


async def test_custom_llm_params():
    """Verify custom LLM params work with real LLM backend."""
    print("--- Custom LLM params with real LLM ---")
    ensure_config()

    tool = LinkSummaryTool(
        llm_params={"max_completion_tokens": 2048},
    )

    result = await tool.execute({
        "url": "https://httpbin.org/html",
        "goal": "Summarize the content of this page.",
    })
    print(f"  Result length: {len(result)} chars")
    print(f"  First 300 chars:\n{result[:300]}")
    print()


async def test_url_blocked():
    """Blocked URLs should be rejected without making any request."""
    print("--- URL blocked ---")
    ensure_config()

    constraints = SearchConstraints.from_repo("django/django")
    tool = LinkSummaryTool(constraints=constraints)

    urls = [
        ("https://github.com/django/django/issues/100", "read issue"),
        ("https://gitlab.com/django/django/merge_requests/1", "read MR"),
        ("https://docs.djangoproject.com/en/5.0/", "read docs"),  # allowed
    ]
    for url, goal in urls:
        result = await tool.execute({"url": url, "goal": goal})
        tag = "BLOCKED" if "ACCESS DENIED" in result else "PASSED"
        print(f"  [{tag}] {url}")
    print()


async def test_no_llm_fallback():
    """Without LLM config, should return raw fetched content."""
    print("--- No LLM configured (raw content fallback) ---")
    # Temporarily clear env vars to force no-LLM path
    saved_model = os.environ.pop("LINK_SUMMARY_MODEL", None)
    saved_config = os.environ.pop("LINK_SUMMARY_CONFIG_PATH", None)

    try:
        reader = LinkReaderTool()
        # Explicitly pass no LLM config — forces raw content fallback
        tool = LinkSummaryTool(reader=reader)

        result = await tool.execute({
            "url": "https://httpbin.org/html",
            "goal": "What is on this page?",
        })
        print(f"  Contains 'no LLM configured': {'no LLM configured' in result}")
        print(f"  Result length: {len(result)} chars")
        print(f"  First 200 chars: {result[:200]}")
    finally:
        # Restore env vars
        if saved_model is not None:
            os.environ["LINK_SUMMARY_MODEL"] = saved_model
        if saved_config is not None:
            os.environ["LINK_SUMMARY_CONFIG_PATH"] = saved_config
    print()


async def test_real_doc_summary():
    """Summarize a real documentation page with real LLM."""
    print("--- Real documentation summary ---")
    ensure_config()

    tool = LinkSummaryTool()

    result = await tool.execute({
        "url": "https://pytorch.org/docs/stable/generated/torch.nn.Linear.html",
        "goal": "Extract the API signature, parameters, and a usage example for torch.nn.Linear.",
    })
    print(f"  Result length: {len(result)} chars")
    print(f"  First 500 chars:\n{result[:500]}")
    print()


async def test_empty_inputs():
    print("--- Empty inputs ---")
    tool = LinkSummaryTool()

    r1 = await tool.execute({"url": "", "goal": "test"})
    print(f"  Empty URL:  {r1}")

    r2 = await tool.execute({"url": "https://example.com", "goal": ""})
    print(f"  Empty goal: {r2}")
    print()


async def main():
    await test_empty_inputs()
    await test_url_blocked()
    await test_no_llm_fallback()
    await test_full_pipeline()
    await test_custom_llm_params()
    await test_real_doc_summary()
    print("All LinkSummaryTool tests done.")


if __name__ == "__main__":
    asyncio.run(main())
