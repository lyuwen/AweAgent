"""Debug: SearchTool — verify real search, constraint filtering, batch queries, error handling.

Uses real bandai_mcp_host Search backend for search tests.
If bandai_mcp_host is not installed, real search tests will return empty results.
"""

import asyncio

from awe_agent.core.tool.search.constraints import SearchConstraints
from awe_agent.core.tool.search.search_tool import SearchTool


# ── Tests with real search backend ──────────────────────────────────────────


async def test_real_single_query():
    """Single query via real bandai_mcp_host Search."""
    print("--- Real single query (no constraints) ---")
    tool = SearchTool()
    result = await tool.execute({"query": "django queryset lazy evaluation"})
    print(result)
    print()


async def test_real_single_query_with_constraints():
    """Real search with constraint filtering — github.com/django/* results should be filtered."""
    print("--- Real single query (with constraints) ---")
    constraints = SearchConstraints.from_repo("django/django")
    tool = SearchTool(constraints=constraints)
    result = await tool.execute({"query": "django queryset filter"})
    print(result)
    print()


async def test_real_batch_query():
    """Batch query via real search backend."""
    print("--- Real batch query ---")
    tool = SearchTool()
    result = await tool.execute({"query": ["python asyncio tutorial", "pytorch nn.Linear usage"], "num": 3})
    print(result)
    print()


async def test_real_num_and_start():
    """Verify num and start params with real backend."""
    print("--- Real search with num=3, start=0 ---")
    tool = SearchTool()
    result = await tool.execute({"query": "python logging best practices", "num": 3, "start": 0})
    print(result)
    print()


async def test_empty_query():
    """Empty query should return error without calling backend."""
    print("--- Empty query ---")
    tool = SearchTool()
    result = await tool.execute({"query": ""})
    print(f"  Result: {result}")
    print()


async def main():
    await test_empty_query()
    await test_real_single_query()
    await test_real_single_query_with_constraints()
    await test_real_batch_query()
    await test_real_num_and_start()
    print("All SearchTool tests done.")


if __name__ == "__main__":
    asyncio.run(main())
