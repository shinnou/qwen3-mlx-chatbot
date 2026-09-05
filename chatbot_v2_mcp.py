"""Qwen3-32B (MLX 8bit) + Brave Search MCP を使った Streamlit チャットボット。

chatbot_v1_basic.py をベースに、最新情報が必要なときはモデル自身が
ツール呼び出し (Qwen の Hermes 形式 <tool_call> タグ) を発行し、
それを検出して Brave Search の MCP サーバーを呼び出し、検索結果を
会話に "tool" ロールとして差し戻してから最終回答を生成する。

v1 と同様に、KV prompt cache の再利用とストリーミング表示の間引きにより
体感速度を改善している（詳細は chatbot_v1_basic.py のモジュールdocstring参照）。
ツール呼び出しが発生したターンでは、1回目 (検索要否判定) と2回目 (最終回答) で
プロンプトの構成が変わる (tools定義の有無など) ため、キャッシュされたトークン列と
実際に送るプロンプトの接頭辞が一致しない場合は安全側に倒してキャッシュを作り直す。

起動:
    pixi run chat-mcp
    (または) pixi run streamlit run chatbot_v2_mcp.py

事前準備:
    - .env に BRAVE_API_KEY を設定 (.env.example 参照)
    - Node.js が使えること (npx で MCP サーバーを起動するため。pixi 環境に含めてある)
"""

import json
import os
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st
from dotenv import load_dotenv
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from mcp_brave_client import brave_web_search

load_dotenv()

MODEL_ID = os.environ.get("MODEL_ID", "mlx-community/Qwen3-32B-8bit")

UI_UPDATE_INTERVAL_SEC = 0.12  # ストリーミング表示の最小更新間隔

TIMEZONE = ZoneInfo("Asia/Tokyo")
WEEKDAY_JA = ["月", "火", "水", "木", "金", "土", "日"]


def current_date_line() -> str:
    """現在日付をシステムプロンプトに埋め込むための1行。

    モデルは「今日」の実際の日付を知らないため、これを与えないと
    「今日の天気」等の質問でも学習データ由来の古い/一般論の回答をしがちで、
    検索ツールを呼ぶ判断もできない。時刻(分単位)まで含めるとシステム
    プロンプトが毎ターン変わり KV キャッシュが再利用できなくなるため、
    日付単位(日が変わるまで安定)にとどめている。
    """
    now = datetime.now(TIMEZONE)
    weekday = WEEKDAY_JA[now.weekday()]
    return f"[現在の日付: {now.strftime('%Y年%m月%d日')}（{weekday}）, 日本時間(JST)]"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "brave_web_search",
            "description": (
                "Brave Search で Web を検索する。天気・気温・為替レート・株価・"
                "スポーツの結果・ニュース速報など日々変化する情報や、学習データに"
                "ない最新の出来事、日付・数値・固有名詞の確認が必要なときに使う。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索クエリ"},
                    "count": {
                        "type": "integer",
                        "description": "取得する検索結果の件数 (デフォルト5)",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

SYSTEM_PROMPT_DEFAULT = (
    "あなたは親切で誠実な日本語アシスタントです。\n"
    "天気・気温、為替レート・株価、スポーツの結果、ニュースなど日々変化する情報は、"
    "あなたの学習データからは正確に分かりません。"
    "「今日」「現在」「最新」「今の」といった言葉を含む質問や、"
    "日によって値が変わる可能性のある事実については、自分の知識だけで答えず、"
    "必ず brave_web_search ツールを使って確認してから回答してください。"
    "検索クエリには（システムプロンプト冒頭に示す）今日の日付や地名など"
    "具体的なキーワードを含めてください。"
    "検索結果を根拠にした場合はその旨を答えに含めてください。"
)

st.set_page_config(page_title="Qwen3-32B Chatbot (Brave MCP)", page_icon="🔎")
st.title("🔎 Qwen3-32B (MLX 8bit) + Brave Search MCP Chatbot")
st.caption(f"model: `{MODEL_ID}`")


@st.cache_resource(show_spinner="モデルを読み込んでいます（初回はダウンロードのため数分かかります）...")
def load_model(model_id: str):
    model, tokenizer = load(model_id)
    return model, tokenizer


model, tokenizer = load_model(MODEL_ID)

with st.sidebar:
    st.header("生成設定")
    max_tokens = st.slider("最大生成トークン数", 128, 4096, 1024, step=128)
    temperature = st.slider("temperature", 0.0, 1.5, 0.7, step=0.05)
    top_p = st.slider("top_p", 0.0, 1.0, 0.9, step=0.05)
    enable_thinking = st.checkbox("thinking モードを有効化", value=False)
    web_search_enabled = st.checkbox("Web検索 (Brave Search MCP) を有効化", value=True)
    search_count = st.slider("検索結果の取得件数", 1, 10, 5)
    system_prompt = st.text_area("システムプロンプト", value=SYSTEM_PROMPT_DEFAULT, height=140)
    show_perf = st.checkbox("生成速度 (tok/s) を表示", value=True)
    if st.button("会話をリセット", use_container_width=True):
        st.session_state.messages = []
        st.session_state.prompt_cache = make_prompt_cache(model)
        st.session_state.cached_token_ids = []
        st.rerun()

if not os.environ.get("BRAVE_API_KEY"):
    st.sidebar.warning("BRAVE_API_KEY が未設定です。.env を確認してください。")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "prompt_cache" not in st.session_state:
    st.session_state.prompt_cache = make_prompt_cache(model)
    st.session_state.cached_token_ids = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("search_query"):
            with st.expander(f"🔎 検索クエリ: {msg['search_query']}"):
                st.text(msg.get("search_result", ""))


def extract_tool_call(text: str):
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    return data.get("name"), data.get("arguments", {}) or {}


def build_prompt(chat_messages, use_tools: bool):
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    if use_tools:
        kwargs["tools"] = TOOLS
    return tokenizer.apply_chat_template(chat_messages, **kwargs)


def common_prefix_len(a: list, b: list) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def new_tokens_for(prompt_text: str) -> tuple[list, list]:
    """キャッシュとの接頭辞一致を確認し、新規に処理すべきトークン列を返す。

    戻り値: (今回モデルに渡す新規トークンID列, プロンプト全体のトークンID列)
    キャッシュ内容とプロンプトの接頭辞が一致しなければキャッシュを作り直す。
    """
    full_ids = tokenizer.encode(prompt_text)
    cached_ids = st.session_state.cached_token_ids
    prefix_len = common_prefix_len(cached_ids, full_ids)
    if prefix_len < len(cached_ids):
        st.session_state.prompt_cache = make_prompt_cache(model)
        prefix_len = 0
    return full_ids[prefix_len:], full_ids


def run_generation(prompt_text: str, placeholder, perf_placeholder, prefix: str = ""):
    new_ids, full_ids = new_tokens_for(prompt_text)
    sampler = make_sampler(temp=temperature, top_p=top_p)
    text = ""
    generated_ids = []
    last_render = 0.0
    last_chunk = None

    for chunk in stream_generate(
        model,
        tokenizer,
        prompt=new_ids,
        max_tokens=max_tokens,
        sampler=sampler,
        prompt_cache=st.session_state.prompt_cache,
    ):
        text += chunk.text
        generated_ids.append(chunk.token)
        last_chunk = chunk
        now = time.monotonic()
        if now - last_render >= UI_UPDATE_INTERVAL_SEC:
            if "<tool_call>" in text:
                placeholder.markdown(prefix + "🔍 情報を検索する必要があるか確認しています...")
            else:
                placeholder.markdown(prefix + text + "▌")
            last_render = now

    st.session_state.cached_token_ids = full_ids + generated_ids

    if show_perf and last_chunk is not None:
        perf_placeholder.caption(
            f"⚙ prefill: {last_chunk.prompt_tps:.1f} tok/s "
            f"({last_chunk.prompt_tokens} tok) / "
            f"generation: {last_chunk.generation_tps:.1f} tok/s "
            f"/ peak memory: {last_chunk.peak_memory:.2f} GB"
        )
    return text


if user_input := st.chat_input("メッセージを入力してください"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    system_content = f"{current_date_line()}\n{system_prompt}"
    base_messages = [{"role": "system", "content": system_content}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    with st.chat_message("assistant"):
        placeholder = st.empty()
        perf_placeholder = st.empty()

        prompt = build_prompt(base_messages, use_tools=web_search_enabled)
        first_pass_text = run_generation(prompt, placeholder, perf_placeholder)

        tool_call = extract_tool_call(first_pass_text) if web_search_enabled else None
        search_query = None
        search_result = None

        if tool_call and tool_call[0] == "brave_web_search":
            search_query = tool_call[1].get("query", user_input)
            count = int(tool_call[1].get("count", search_count) or search_count)
            with st.spinner(f"Brave Search で検索中: {search_query}"):
                search_result = brave_web_search(search_query, count=count)

            followup_messages = base_messages + [
                {"role": "assistant", "content": first_pass_text},
                {"role": "tool", "name": "brave_web_search", "content": search_result},
            ]
            followup_prompt = build_prompt(followup_messages, use_tools=False)
            final_text = run_generation(followup_prompt, placeholder, perf_placeholder)
        else:
            final_text = first_pass_text

        placeholder.markdown(final_text)
        if search_query:
            with st.expander(f"🔎 検索クエリ: {search_query}"):
                st.text(search_result or "")

    assistant_msg = {"role": "assistant", "content": final_text}
    if search_query:
        assistant_msg["search_query"] = search_query
        assistant_msg["search_result"] = search_result
    st.session_state.messages.append(assistant_msg)
