"""SearchTool — web search with anti-hack constraint filtering."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool
from awe_agent.core.tool.search.constraints import SearchConstraints

logger = logging.getLogger(__name__)

# Default fields to include in formatted results
_DEFAULT_RESULT_SCHEME = ["position", "title", "description", "snippets", "url"]


class SearchTool(Tool):
    """Web search with anti-hack constraint filtering.

    Uses ``bytedance.bandai_mcp_host`` Search service. Falls back to a stub when
    unavailable.

    Args:
        engine: Search engine name. Defaults to env ``ENGINE`` or ``"google"``.
        constraints: Optional constraints for result filtering.
        max_attempts: Number of retry attempts for search calls.
        result_scheme: Fields to include in formatted output.
        search_fn: Optional callable to use instead of bandai_mcp_host.
            Useful for testing or custom backends. Signature:
            ``async (query, num, start, engine) -> str | dict | list``.
    """

    def __init__(
        self,
        engine: str | None = None,
        constraints: SearchConstraints | None = None,
        max_attempts: int = 1,
        result_scheme: list[str] | None = None,
        search_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._engine = engine or os.environ.get("ENGINE", "google")
        self._constraints = constraints or SearchConstraints()
        self._max_attempts = max_attempts
        self._result_scheme = result_scheme or list(_DEFAULT_RESULT_SCHEME)

        # Injected or lazy-loaded search function
        self._search_fn: Any = search_fn

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "Search the web for information. Use this when you need to look up "
            "external library documentation, debug unfamiliar errors, or research "
            "best practices. Do NOT use this for information that should be in the "
            "local codebase. Supports single or batch queries."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "oneOf": [
                        {
                            "type": "string",
                            "description": "A single search query.",
                        },
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Multiple search queries (batch).",
                        },
                    ],
                    "description": "The search query (string or list of strings).",
                },
                "num": {
                    "type": "integer",
                    "description": "Number of results to return (default 10).",
                },
                "start": {
                    "type": "integer",
                    "description": "Starting offset for results (pagination).",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        query = params.get("query", "")
        num = params.get("num", 10)
        start = params.get("start", 0)

        if not query:
            return "Error: empty search query."

        # Normalize to list of queries
        queries = query if isinstance(query, list) else [query]
        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        if not queries:
            return "Error: empty search query."

        parts: list[str] = []
        for q in queries:
            results = await self._search_single(q, num=num, start=start)
            filtered, filtered_count = self._constraints.filter_search_results(results)
            parts.append(self._format_results(q, filtered, filtered_count))

        return "\n\n".join(parts)

    async def _search_single(
        self, query: str, num: int, start: int,
    ) -> list[dict]:
        """Execute single search query via bandai_mcp_host.

        Lazy-loads the search function via ``bytedance.bandai_mcp_host.map_tools``.
        Retries up to ``max_attempts``.
        Falls back to empty results if ``bytedance.bandai_mcp_host`` is unavailable.
        """
        if self._search_fn is None:
            try:
                from bytedance.bandai_mcp_host import map_tools  # type: ignore[import-untyped]
                self._search_fn = map_tools("Search")
            except ImportError:
                logger.warning(
                    "bytedance.bandai_mcp_host not installed — search will return "
                    "empty results. Install it to enable web search."
                )
                self._search_fn = None
                return []

        if self._search_fn is None:
            return []

        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = await self._search_fn(
                    query=query,
                    num=num,
                    start=start,
                    engine=self._engine,
                )

                # Handle BandaiToolResponse (has .status and .result)
                if hasattr(response, "status") and hasattr(response, "result"):
                    if not response.status.is_succeeded():
                        error_msg = response.result or "Unknown error"
                        raise RuntimeError(f"Search failed: {error_msg}")
                    raw: Any = response.result
                else:
                    raw = response

                # Parse result — may be JSON string or dict/list
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        return [{"description": raw}]

                if isinstance(raw, dict):
                    return raw.get("results", raw.get("organic", [raw]))
                if isinstance(raw, list):
                    return raw
                return []
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Search attempt %d/%d failed for query %r: %s",
                    attempt + 1, self._max_attempts, query, exc,
                )

        logger.error("All search attempts failed for query %r: %s", query, last_error)
        return []

    def _format_results(
        self, query: str, results: list[dict], filtered_count: int,
    ) -> str:
        """Format results with optional filtered-count warning."""
        lines = [f"Search results for: {query}"]

        if filtered_count > 0:
            lines.append(
                f"WARNING: {filtered_count} result(s) filtered by security constraints."
            )

        if not results:
            lines.append("No results found.")
            return "\n".join(lines)

        lines.append("")
        for i, item in enumerate(results, 1):
            entry_parts: list[str] = []
            for field_name in self._result_scheme:
                value = item.get(field_name)
                if value is not None:
                    if field_name == "position":
                        continue  # use our own numbering
                    if isinstance(value, list):
                        value = " ".join(str(v) for v in value)
                    entry_parts.append(f"  {field_name}: {value}")
            if entry_parts:
                lines.append(f"[{i}]")
                lines.extend(entry_parts)
                lines.append("")

        return "\n".join(lines)
