"""LinkReaderTool — fetch raw content from URLs."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from awe_agent.core.runtime.protocol import RuntimeSession
from awe_agent.core.tool.protocol import Tool
from awe_agent.core.tool.search.constraints import SearchConstraints

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONTENT_TOKENS = 100000


class LinkReaderTool(Tool):
    """Fetch raw content from URLs (web pages or PDFs).

    Uses ``bytedance.bandai_mcp_host`` LinkReader service. Checks URLs against
    :class:`SearchConstraints` before fetching.

    Args:
        constraints: Optional constraints for URL blocking.
        max_content_tokens: Maximum tokens for content truncation.
        max_attempts: Number of retry attempts for fetch calls.
        reader_fn: Optional callable to use instead of bandai_mcp_host.
            Useful for testing or custom backends. Signature:
            ``async (url) -> str | dict``.
    """

    def __init__(
        self,
        constraints: SearchConstraints | None = None,
        max_content_tokens: int = _DEFAULT_MAX_CONTENT_TOKENS,
        max_attempts: int = 3,
        reader_fn: Callable[..., Any] | None = None,
    ) -> None:
        self._constraints = constraints or SearchConstraints()
        self._max_content_tokens = max_content_tokens
        self._max_attempts = max_attempts

        # Injected or lazy-loaded reader function
        self._reader_fn: Any = reader_fn
        # Lazy-loaded tiktoken encoding (avoid re-creating per call)
        self._tiktoken_enc: Any = None

    @property
    def name(self) -> str:
        return "link_reader"

    @property
    def description(self) -> str:
        return (
            "Fetch raw content from a URL. Returns the full text content of "
            "a web page or PDF. Use 'link_summary' instead if you need a "
            "concise summary of the content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch content from.",
                },
            },
            "required": ["url"],
        }

    async def execute(
        self,
        params: dict[str, Any],
        session: RuntimeSession | None = None,
    ) -> str:
        url = params.get("url", "").strip()
        if not url:
            return "Error: empty URL."

        # Check URL against constraints
        if self._constraints.is_url_blocked(url):
            return (
                f"ACCESS DENIED: The URL '{url}' is blocked by security constraints. "
                "This URL may point to the target repository and accessing it is "
                "not allowed during evaluation."
            )

        return await self._fetch(url)

    async def _fetch(self, url: str) -> str:
        """Fetch URL content via bandai_mcp_host LinkReader."""
        if self._reader_fn is None:
            try:
                from bytedance.bandai_mcp_host import map_tools  # type: ignore[import-untyped]
                self._reader_fn = map_tools("LinkReader")
            except ImportError:
                logger.warning(
                    "bytedance.bandai_mcp_host not installed — link_reader will be "
                    "unavailable. Install it to enable URL content fetching."
                )
                return (
                    "Error: bytedance.bandai_mcp_host is not installed. "
                    "Cannot fetch URL content without it."
                )

        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = await self._reader_fn(url=url)

                # Handle BandaiToolResponse (has .status and .result)
                if hasattr(response, "status") and hasattr(response, "result"):
                    if not response.status.is_succeeded():
                        error_msg = response.result or "Unknown error"
                        raise RuntimeError(f"LinkReader failed: {error_msg}")
                    raw = response.result
                else:
                    raw = response

                # Parse result — may be JSON string with 'content' key
                content: str
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, dict) and "content" in parsed:
                            content = parsed["content"]
                        else:
                            content = raw
                    except json.JSONDecodeError:
                        content = raw
                elif isinstance(raw, dict):
                    content = raw.get("content", str(raw))
                else:
                    content = str(raw)

                if not content:
                    return f"No content returned from {url}"

                return self._truncate_content(content, self._max_content_tokens)

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LinkReader attempt %d/%d failed for %r: %s",
                    attempt + 1, self._max_attempts, url, exc,
                )

        return f"Error: failed to fetch {url}: {last_error}"

    def _truncate_content(self, content: str, max_tokens: int) -> str:
        """Token-aware truncation using tiktoken (with char-based fallback)."""
        try:
            if self._tiktoken_enc is None:
                import tiktoken
                self._tiktoken_enc = tiktoken.get_encoding("o200k_base")
            tokens = self._tiktoken_enc.encode(content)
            if len(tokens) <= max_tokens:
                return content
            truncated = self._tiktoken_enc.decode(tokens[:max_tokens])
            return truncated + "\n\n... [content truncated]"
        except (ImportError, Exception):
            # Fallback: rough char-based estimate (~4 chars per token)
            max_chars = max_tokens * 4
            if len(content) <= max_chars:
                return content
            return content[:max_chars] + "\n\n... [content truncated]"
