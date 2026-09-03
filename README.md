# Qwen3-32B (MLX 4bit) Streamlit Chatbot

Mac (Apple Silicon) 上で、Hugging Face から直接取得した **Qwen3-32B の4bit量子化モデル**を
[MLX](https://github.com/ml-explore/mlx) で実行する Streamlit チャットボットです。
Ollama や LM Studio は使わず、`mlx-lm` を介して Hugging Face Hub のモデルをそのままロードします。

2種類のバージョンを用意しています。

- **v1 (`chatbot_v1_basic.py`)**: MCP を使わないシンプルなチャットボット
- **v2 (`chatbot_v2_mcp.py`)**: v1 を拡張し、**Brave Search の MCP サーバー**を使って
  最新情報が必要な質問にはモデル自身が判断して Web 検索を行う版

---

## ファイル構成

| ファイル | 役割 |
|---|---|
| `chatbot_v1_basic.py` | MCP なしの基本チャットボット (Streamlit アプリ) |
| `chatbot_v2_mcp.py` | Brave Search MCP 対応チャットボット (Streamlit アプリ) |
| `mcp_brave_client.py` | Brave Search MCP サーバーを呼び出す薄いクライアントラッパー |
| `pixi.toml` / `pixi.lock` | 依存関係定義 ([pixi](https://pixi.sh) で管理) |
| `.env.example` | 環境変数のサンプル (`MODEL_ID`, `BRAVE_API_KEY`) |

---

## 必要要件

- Mac (Apple Silicon: M1/M2/M3/M4 系)、[pixi](https://pixi.sh) がインストール済みであること
- 統合メモリ 32GB 以上を推奨（Qwen3-32B の4bit量子化モデルを常駐させるため）
- インターネット接続（初回のモデルダウンロード、および v2 での Web 検索時）
- v2 を使う場合: [Brave Search API](https://brave.com/search/api/) のAPIキー

依存パッケージ（`streamlit`, `mlx`, `mlx-lm`, `mcp`, `python-dotenv`, `nodejs`）は
`pixi.toml` に定義済みで、`pixi install` で一括インストールされます。
`nodejs` は Brave Search MCP サーバー (`npx @modelcontextprotocol/server-brave-search`) の実行に使います。

---

## セットアップ

```bash
cd "/Volumes/Extened SDD/home2/ARAG"

# 依存関係のインストール
pixi install

# 環境変数ファイルを作成
cp .env.example .env
```

`.env` を編集し、必要に応じて以下を設定します。

```ini
# 使用するモデル (Hugging Face の MLX 4bit 量子化モデル)
MODEL_ID=mlx-community/Qwen3-32B-4bit

# v2 (MCP版) を使う場合のみ必須
BRAVE_API_KEY=your_brave_api_key_here
```

`MODEL_ID` を変更すれば、`mlx-community` が配布している他の量子化モデル
（例: `mlx-community/Qwen3-30B-A3B-4bit`, `mlx-community/Qwen3-8B-4bit` など）
に差し替えることもできます。

---

## 使い方

### v1: MCP なしの基本チャットボット

```bash
pixi run chat-basic
# 実体は: pixi run streamlit run chatbot_v1_basic.py
```

ブラウザで `http://localhost:8501` が開きます。初回起動時はモデルの
ダウンロード（数十GB）が走るため、数分〜数十分かかることがあります。
2回目以降は Hugging Face のローカルキャッシュ (`~/.cache/huggingface`) から
読み込むため高速に起動します。

サイドバーで以下を調整できます。

- 最大生成トークン数
- temperature / top_p
- thinking モードの有効化（Qwen3 の思考過程を出力するモード）
- システムプロンプト
- 「会話をリセット」ボタン

### v2: Brave Search MCP 対応チャットボット

```bash
pixi run chat-mcp
# 実体は: pixi run streamlit run chatbot_v2_mcp.py
```

v1 の機能に加えて、サイドバーに以下が追加されます。

- 「Web検索 (Brave Search MCP) を有効化」チェックボックス
- 検索結果の取得件数スライダー

**動作の流れ:**

1. ユーザーが質問を入力すると、モデルには「Brave Web検索」ツールの
   存在が伝えられます（Qwen3 のチャットテンプレートの `tools` 機構を使用）。
2. モデルが「最新情報の確認が必要」と判断した場合、回答の代わりに
   `<tool_call>{"name": "brave_web_search", "arguments": {...}}</tool_call>`
   という形式でツール呼び出しを出力します。
3. アプリ側がこれを検出し、`mcp_brave_client.py` 経由で
   `npx @modelcontextprotocol/server-brave-search` を起動して実際に検索を実行します。
4. 検索結果を会話履歴に `tool` ロールとして追加し、モデルに最終回答を再生成させます。
5. 検索が使われた場合、画面上に「🔎 検索クエリ: ...」という折りたたみ表示で
   検索クエリと生の検索結果を確認できます。

質問が最新情報を必要としない場合（雑談や一般知識の質問など）は、
モデルはツールを呼ばずに直接回答します。すべての質問で検索するわけではありません。

---

## トラブルシューティング

| 症状 | 対処 |
|---|---|
| `BRAVE_API_KEY が設定されていません` | `.env` に `BRAVE_API_KEY` を設定し、Streamlit を再起動する |
| モデルのダウンロードが遅い / 失敗する | ネットワークを確認。`huggingface-cli login` でHFアカウント認証が必要なモデルもある |
| メモリ不足でクラッシュする | サイドバーで「最大生成トークン数」を下げる、または `MODEL_ID` をより小さいモデル（例: `Qwen3-8B-4bit`）に変更する |
| `npx` が Brave Search サーバーを起動できない | `pixi run node --version` で Node.js (v22系) が入っているか確認する |
| モデルが検索すべき場面で検索しない/しすぎる | サイドバーの「システムプロンプト」を編集し、検索を使う条件を明示的に指示する |

---

## 補足: なぜ MLX / なぜこの構成か

- Mac (Apple Silicon) では `transformers` + `bitsandbytes` の4bit量子化はCUDA専用のため動作しません。
  そのため、Apple Silicon 向けにネイティブ最適化された **MLX** (`mlx-lm`) を採用し、
  Hugging Face 上で事前に4bit量子化済みの `mlx-community/Qwen3-32B-4bit` をそのままロードしています。
  （Ollama や LM Studio のような専用アプリは使わず、Pythonから直接 Hugging Face Hub のモデルを扱います。）
- v2 の検索連携は、常に検索するのではなく Qwen3 のネイティブな **tool calling**（Hermes形式の
  `<tool_call>` タグ）を利用し、モデル自身に検索の要否を判断させる設計にしています。
