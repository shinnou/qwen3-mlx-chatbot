"""Qwen3-32B (MLX 8bit) を使ったシンプルな Streamlit チャットボット。

Hugging Face 上の MLX 8bit 量子化済みモデル (mlx-community/Qwen3-32B-8bit) を
mlx-lm でロードし、Apple Silicon の GPU (Metal) 上で推論する。
Ollama / LM Studio は使わず、HuggingFace Hub から直接モデルを取得する。

Apple Silicon は CPU/GPU が同一の統合メモリ (Unified Memory) を共有するため、
CUDA のような「モデルをVRAMへ転送する」という操作は存在しない。
mlx.core はデフォルトで GPU (Metal) デバイスを使うため、明示的な移動は不要。
体感速度を上げるため、ここでは以下の2点を実装している。

    1. 会話ターンごとに毎回全履歴を再計算しない (KV prompt cache の再利用)
    2. ストリーミング表示を毎トークンではなく間引いて更新する (UI再描画コスト削減)

起動:
    pixi run chat-basic
    (または) pixi run streamlit run chatbot_v1_basic.py
"""

import os
import time

import streamlit as st
from dotenv import load_dotenv
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

load_dotenv()

MODEL_ID = os.environ.get("MODEL_ID", "mlx-community/Qwen3-32B-8bit")

UI_UPDATE_INTERVAL_SEC = 0.12  # ストリーミング表示の最小更新間隔（これより短い間隔では再描画しない）

st.set_page_config(page_title="Qwen3-32B Chatbot", page_icon="🤖")
st.title("🤖 Qwen3-32B (MLX 8bit) Chatbot")
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
    system_prompt = st.text_area(
        "システムプロンプト",
        value="あなたは親切で誠実な日本語アシスタントです。",
    )
    show_perf = st.checkbox("生成速度 (tok/s) を表示", value=True)
    if st.button("会話をリセット", use_container_width=True):
        st.session_state.messages = []
        st.session_state.prompt_cache = make_prompt_cache(model)
        st.session_state.cached_token_ids = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "prompt_cache" not in st.session_state:
    # 会話履歴のKV（Key/Value）キャッシュ。ターンをまたいで使い回すことで、
    # 毎回すべての履歴をゼロから計算し直す無駄を省く。
    st.session_state.prompt_cache = make_prompt_cache(model)
    st.session_state.cached_token_ids = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def common_prefix_len(a: list, b: list) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


if user_input := st.chat_input("メッセージを入力してください"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    chat_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    prompt_text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_token_ids = tokenizer.encode(prompt_text)

    cached_token_ids = st.session_state.cached_token_ids
    prefix_len = common_prefix_len(cached_token_ids, full_token_ids)

    if prefix_len < len(cached_token_ids):
        # 過去のキャッシュが今回のプロンプトの接頭辞と一致しない
        # (システムプロンプト変更、thinkingタグの除去など) ので作り直す。
        st.session_state.prompt_cache = make_prompt_cache(model)
        prefix_len = 0

    new_token_ids = full_token_ids[prefix_len:]

    sampler = make_sampler(temp=temperature, top_p=top_p)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        perf_placeholder = st.empty()
        response_text = ""
        generated_token_ids = []
        last_render = 0.0
        last_chunk = None

        for chunk in stream_generate(
            model,
            tokenizer,
            prompt=new_token_ids,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_cache=st.session_state.prompt_cache,
        ):
            response_text += chunk.text
            generated_token_ids.append(chunk.token)
            last_chunk = chunk
            now = time.monotonic()
            if now - last_render >= UI_UPDATE_INTERVAL_SEC:
                placeholder.markdown(response_text + "▌")
                last_render = now
        placeholder.markdown(response_text)

        if show_perf and last_chunk is not None:
            perf_placeholder.caption(
                f"⚙ prefill: {last_chunk.prompt_tps:.1f} tok/s "
                f"({last_chunk.prompt_tokens} tok) / "
                f"generation: {last_chunk.generation_tps:.1f} tok/s "
                f"/ peak memory: {last_chunk.peak_memory:.2f} GB"
            )

    st.session_state.cached_token_ids = full_token_ids + generated_token_ids
    st.session_state.messages.append({"role": "assistant", "content": response_text})
