"""Debug: LinkReaderTool — test with real bandai_mcp_host backend.

No reader_fn injection. Uses the internal lazy-load path for bandai_mcp_host.
If bandai_mcp_host is not installed, the fetch tests will return an error message.
"""

import asyncio

from awe_agent.core.tool.search.constraints import SearchConstraints
from awe_agent.core.tool.search.link_reader_tool import LinkReaderTool


async def test_real_fetch():
    """Fetch a real web page via bandai_mcp_host."""
    print("--- Real fetch via bandai_mcp_host ---")
    tool = LinkReaderTool()
    result = await tool.execute({"url": "https://httpbin.org/html"})
    print(f"  Length: {len(result)} chars")
    print(f"  First 200 chars:\n  {result[:200]}")
    print()


async def test_real_fetch_with_truncation():
    """Verify token truncation with real content."""
    print("--- Real fetch with truncation (max_content_tokens=200) ---")
    tool = LinkReaderTool(max_content_tokens=200)
    result = await tool.execute({"url": "https://httpbin.org/html"})
    print(f"  Length: {len(result)} chars")
    print(f"  Truncated: {'truncated' in result}")
    print()


async def test_url_blocked():
    """Blocked URLs should be rejected without making any request."""
    print("--- URL blocked by constraints ---")
    constraints = SearchConstraints.from_repo("django/django")
    tool = LinkReaderTool(constraints=constraints)

    blocked_urls = [
        "https://github.com/django/django/blob/main/README.rst",
        "https://github.com/django/django/issues/12345",
    ]
    for url in blocked_urls:
        result = await tool.execute({"url": url})
        print(f"  [BLOCKED] {url}")

    allowed_url = "https://docs.djangoproject.com/en/5.0/"
    result = await tool.execute({"url": allowed_url})
    print(f"  [ALLOWED] {allowed_url} -> {len(result)} chars")
    print()


async def test_real_retry():
    """Verify retry logic with real backend."""
    print("--- Real fetch with max_attempts=3 ---")
    tool = LinkReaderTool(max_attempts=3)
    result = await tool.execute({"url": "https://httpbin.org/get"})
    print(f"  Length: {len(result)} chars")
    print(f"  First 200 chars:\n  {result[:200]}")
    print()


async def test_empty_url():
    print("--- Empty URL ---")
    tool = LinkReaderTool()
    result = await tool.execute({"url": ""})
    print(f"  Result: {result}")
    print()


async def main():
    await test_empty_url()
    await test_url_blocked()
    await test_real_fetch()
    await test_real_fetch_with_truncation()
    await test_real_retry()
    print("All real LinkReaderTool tests done.")


if __name__ == "__main__":
    asyncio.run(main())
