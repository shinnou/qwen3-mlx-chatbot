"""Brave Search の MCP サーバー (@modelcontextprotocol/server-brave-search) を
呼び出すための薄いラッパー。

公式サーバーは Node.js 製で `npx -y @modelcontextprotocol/server-brave-search`
として stdio 経由で起動する。Streamlit は同期的にスクリプトを再実行する仕組み
なので、呼び出しごとに MCP セッションを開始・終了するシンプルな方式にしている。
"""

import asyncio
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BRAVE_TOOL_NAME = "brave_web_search"


def _server_params() -> StdioServerParameters:
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "BRAVE_API_KEY が設定されていません。.env に BRAVE_API_KEY=... を設定してください。"
        )
    return StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-brave-search"],
        env={**os.environ, "BRAVE_API_KEY": api_key},
    )


async def _search_async(query: str, count: int = 5) -> str:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                BRAVE_TOOL_NAME, {"query": query, "count": count}
            )
            texts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
            return "\n".join(texts) if texts else "(検索結果が空でした)"


def brave_web_search(query: str, count: int = 5) -> str:
    """同期コード (Streamlit) から呼び出すためのエントリポイント。

    呼び出しのたびに MCP サーバープロセスを起動して1回検索し、終了する。
    """
    try:
        return asyncio.run(_search_async(query, count))
    except Exception as e:  # noqa: BLE001 - チャットボットに文脈を返すため意図的に広く捕捉
        return f"[Brave Search MCP エラー] {e}"
