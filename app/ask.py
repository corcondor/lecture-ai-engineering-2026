"""第1回: トークン表示と、履歴を保持する対話CLI。

uv run python -m app.ask "LoRAとは何ですか？"
uv run python -m app.ask --chat
uv run python -m app.ask --temperature 0.2 --model gemini-3.5-flash-lite
"""

import argparse
import math
import time
from dataclasses import dataclass

import httpx
from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


@dataclass(frozen=True)
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    thought_tokens: int | None = None
    cached_tokens: int | None = None

    @classmethod
    def from_response(cls, response):
        metadata = response.usage_metadata
        return cls(**{
            key: getattr(metadata, field, None)
            for key, field in {
                "input_tokens": "prompt_token_count",
                "output_tokens": "candidates_token_count",
                "total_tokens": "total_token_count",
                "thought_tokens": "thoughts_token_count",
                "cached_tokens": "cached_content_token_count",
            }.items()
        })


@dataclass(frozen=True)
class Result:
    text: str
    usage: Usage
    elapsed_seconds: float
    finish_reason: str | None
    model_version: str | None
    block_reason: str | None = None


def enum_value(value) -> str | None:
    return None if value is None else str(getattr(value, "value", value))


def make_client() -> genai.Client:
    """隠れた再試行を避ける。キーは環境変数からのみ読み込む。"""
    return genai.Client(
        api_key=get_api_key(),
        http_options=types.HttpOptions(
            timeout=120_000, retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )


def temperature_value(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 2:
        raise argparse.ArgumentTypeError("temperature は 0〜2 の有限数にしてください")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("1以上の整数にしてください")
    return number


class Conversation:
    """SDK の Content をそのまま保ち、署名等を壊さず次の質問に渡す。"""

    def __init__(self, client, *, model=MODEL, temperature=None, max_output_tokens=None):
        self.client = client
        self.model = model
        self.config = types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_output_tokens
        )
        self.history: list[types.Content] = []

    def reset(self):
        self.history.clear()

    def send(self, question: str) -> Result:
        if not question.strip():
            raise ValueError("空の質問は送信できません")
        user_content = types.Content(role="user", parts=[types.Part(text=question)])
        contents = [*self.history, user_content]
        start = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model, contents=contents, config=self.config
        )
        elapsed = time.perf_counter() - start
        text = response.text or ""
        candidate = response.candidates[0] if response.candidates else None
        content = candidate.content if candidate else None
        feedback = response.prompt_feedback
        finish_reason = enum_value(candidate.finish_reason) if candidate else None
        block_reason = enum_value(feedback.block_reason) if feedback else None
        # APIエラー・ブロック・空応答では履歴を変更しない。
        if (
            text and content is not None and content.role == "model"
            and not block_reason and finish_reason in (None, "STOP", "MAX_TOKENS")
        ):
            self.history.extend([user_content, content])
        return Result(
            text=text,
            usage=Usage.from_response(response),
            elapsed_seconds=elapsed,
            finish_reason=finish_reason,
            model_version=response.model_version,
            block_reason=block_reason,
        )


def format_usage(usage: Usage) -> str:
    def show(value):
        return "不明" if value is None else str(value)
    return (
        f"[tokens] 入力: {show(usage.input_tokens)} / 出力: {show(usage.output_tokens)}"
        f" / 合計: {show(usage.total_tokens)} / 思考: {show(usage.thought_tokens)}"
        f" / キャッシュ入力: {show(usage.cached_tokens)}"
    )


def display_result(result: Result):
    print(result.text or "[本文なし] 安全性によるブロック等の可能性があります。")
    print(format_usage(result.usage))
    print(f"[elapsed] {result.elapsed_seconds:.3f} 秒（API呼び出し全体）")
    if result.finish_reason and result.finish_reason != "STOP":
        print(f"[finish] {result.finish_reason}")
    if result.block_reason:
        print(f"[block] {result.block_reason}")


def ask(question: str, *, model=MODEL, temperature=None, max_output_tokens=None) -> str:
    """既存の ask(question) -> str を維持し、利用量も表示する。"""
    with make_client() as client:
        result = Conversation(
            client, model=model, temperature=temperature,
            max_output_tokens=max_output_tokens,
        ).send(question)
    print(format_usage(result.usage))
    return result.text


def run_chat(conversation: Conversation, initial_question=None):
    print("会話を開始します。/reset で履歴消去、/exit または Ctrl-D で終了。")
    print("履歴はこのプロセスのメモリ内のみ。長くなるほど入力トークンを消費します。")
    pending = initial_question
    while True:
        try:
            question = pending if pending is not None else input("あなた> ")
            pending = None
        except (EOFError, KeyboardInterrupt):
            print("\n終了しました。")
            return
        question = question.strip()
        if question in ("/exit", "/quit"):
            return
        if question == "/reset":
            conversation.reset()
            print("履歴を消去しました。")
            continue
        if not question:
            continue
        try:
            display_result(conversation.send(question))
        except errors.APIError as exc:
            # 生のAPIエラーはキーやリクエスト情報を含む可能性があるため出力しない。
            print(api_error_message(exc.code))
            if exc.code == 429:
                print("分単位だけでなく、日次やプロジェクトの上限も確認してください。")
        except httpx.HTTPError:
            print("通信に失敗しました。履歴は更新していません。接続を確認してください。")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する（--chat で履歴保持）")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--temperature", type=temperature_value)
    parser.add_argument("--max-output-tokens", type=positive_int)
    args = parser.parse_args()
    print(f"[model] {args.model}")
    try:
        with make_client() as client:
            conversation = Conversation(
                client, model=args.model, temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
            )
            if args.chat:
                run_chat(conversation, args.question)
            else:
                question = args.question if args.question is not None else DEFAULT_QUESTION
                print(f"[質問] {question}\n")
                display_result(conversation.send(question))
    except errors.APIError as exc:
        print(api_error_message(exc.code))
        raise SystemExit(1)
    except (httpx.HTTPError, ValueError):
        print("通信または入力に問題があります。接続と質問を確認してください。")
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\n中断しました。")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
