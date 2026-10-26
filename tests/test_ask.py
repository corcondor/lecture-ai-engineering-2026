"""第1回: Gemini を呼ばずに、利用量・会話履歴・CLI を検証する。"""

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from google.genai import errors, types

from app import ask


class FakeClient:
    """実際の SDK レスポンスを順に返す。ネットワーク接続は持たない。"""

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []
        self.models = self
        self.entered = False
        self.closed = False

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *_):
        self.closed = True


def response(text="回答", *, usage=None, finish_reason="STOP", parts=None):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(
                role="model", parts=parts if parts is not None else [types.Part(text=text)]
            ),
            finish_reason=finish_reason,
        )],
        usage_metadata=usage,
        model_version="offline-model-version",
    )


def api_error(code=429):
    # 意図的なダミー機密文字列。例外の生データが画面に出ないことを検証する。
    return errors.APIError(code, {"error": {
        "code": code, "message": "DUMMY_SECRET_MUST_NOT_APPEAR request details"
    }})


@pytest.fixture(autouse=True)
def prohibit_live_network(monkeypatch):
    """偽物の差し替え漏れがあっても、通信も本物のキーの利用も禁止する。"""
    def forbidden(*args, **kwargs):
        pytest.fail("第1回のテストからネットワークを利用してはいけません")

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)


def install_input(monkeypatch, *items):
    values = iter(items)

    def read(_prompt):
        value = next(values)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr("builtins.input", read)


def run_main(monkeypatch, client, *args):
    monkeypatch.setattr(ask, "make_client", lambda: client)
    monkeypatch.setattr(sys, "argv", ["app.ask", *args])
    ask.main()


def test_usage_reads_actual_sdk_metadata_including_thoughts_and_cache():
    metadata = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=12, candidates_token_count=7, total_token_count=24,
        thoughts_token_count=5, cached_content_token_count=3,
    )
    assert ask.Usage.from_response(response(usage=metadata)) == ask.Usage(12, 7, 24, 5, 3)


@pytest.mark.parametrize("metadata", [None, types.GenerateContentResponseUsageMetadata()])
def test_missing_usage_is_unknown_not_fabricated_zero(metadata):
    usage = ask.Usage.from_response(response(usage=metadata))
    assert usage == ask.Usage()
    assert ask.format_usage(usage).count("不明") == 5


def test_zero_usage_stays_zero_and_partial_metadata_stays_unknown():
    usage = ask.Usage.from_response(response(usage=types.GenerateContentResponseUsageMetadata(
        prompt_token_count=0, candidates_token_count=0, total_token_count=0,
    )))
    assert usage == ask.Usage(0, 0, 0, None, None)
    assert ask.format_usage(usage) == (
        "[tokens] 入力: 0 / 出力: 0 / 合計: 0 / 思考: 不明 / キャッシュ入力: 不明"
    )


def test_send_passes_model_config_and_measures_only_sdk_call(monkeypatch):
    client = FakeClient(response("こんにちは"))
    conversation = ask.Conversation(
        client, model="chosen-model", temperature=0.2, max_output_tokens=123
    )
    clock = iter([100.0, 101.25])
    monkeypatch.setattr(ask.time, "perf_counter", lambda: next(clock))

    result = conversation.send("  質問はそのまま  ")

    call = client.calls[0]
    assert call["model"] == "chosen-model"
    assert call["config"] is conversation.config
    assert call["config"].temperature == 0.2
    assert call["config"].max_output_tokens == 123
    assert call["contents"][0].role == "user"
    assert call["contents"][0].parts[0].text == "  質問はそのまま  "
    assert result.text == "こんにちは"
    assert result.elapsed_seconds == 1.25
    assert result.finish_reason == "STOP"
    assert result.model_version == "offline-model-version"
    assert result.block_reason is None


def test_default_generation_settings_are_left_for_model_defaults():
    conversation = ask.Conversation(FakeClient())
    assert conversation.model == ask.MODEL
    assert conversation.config.temperature is None
    assert conversation.config.max_output_tokens is None


def test_successive_turns_preserve_sdk_content_roles_and_thought_signature():
    first = response(parts=[
        types.Part(text="内部の思考", thought=True),
        types.Part(text="最初の回答", thought_signature=b"dummy-thought-signature"),
    ])
    second = response("次の回答")
    client = FakeClient(first, second)
    conversation = ask.Conversation(client)

    assert conversation.send("最初の質問").text == "最初の回答"
    preserved = first.candidates[0].content
    assert conversation.history[1] is preserved
    conversation.send("続けて質問")

    assert [content.role for content in conversation.history] == [
        "user", "model", "user", "model"
    ]
    assert client.calls[1]["contents"][1] is preserved
    assert client.calls[1]["contents"][1].parts[0].thought is True
    assert client.calls[1]["contents"][1].parts[1].thought_signature == b"dummy-thought-signature"
    assert client.calls[1]["contents"][2].parts[0].text == "続けて質問"
    assert conversation.history[3] is second.candidates[0].content
    # 過去のリクエストに、新しいターンを後から書き足してはいけない。
    assert len(client.calls[0]["contents"]) == 1
    assert len(client.calls[1]["contents"]) == 3


@pytest.mark.parametrize("question", ["", " ", "\n\t"])
def test_blank_questions_do_not_call_sdk_or_change_history(question):
    client = FakeClient(response())
    conversation = ask.Conversation(client)
    with pytest.raises(ValueError, match="空の質問"):
        conversation.send(question)
    assert client.calls == []
    assert conversation.history == []


@pytest.mark.parametrize("failure", [api_error(), httpx.ConnectError("dummy failure")])
def test_failed_turn_keeps_history_and_next_turn_does_not_include_failure(failure):
    client = FakeClient(response("回答1"), failure, response("回答3"))
    conversation = ask.Conversation(client)
    conversation.send("質問1")
    previous = conversation.history.copy()

    with pytest.raises(type(failure)):
        conversation.send("失敗する質問")
    assert conversation.history == previous
    assert all(current is old for current, old in zip(conversation.history, previous))
    conversation.send("質問3")
    assert [item.parts[0].text for item in client.calls[2]["contents"]] == [
        "質問1", "回答1", "質問3"
    ]


@pytest.mark.parametrize("empty_response", [
    types.GenerateContentResponse(),
    response(""),
    types.GenerateContentResponse(candidates=[types.Candidate(finish_reason="SAFETY")]),
    types.GenerateContentResponse(prompt_feedback=types.GenerateContentResponsePromptFeedback(
        block_reason=types.BlockedReason.SAFETY,
    )),
])
def test_empty_or_blocked_response_does_not_change_existing_history(empty_response):
    client = FakeClient(response(), empty_response)
    conversation = ask.Conversation(client)
    conversation.send("最初")
    previous = conversation.history.copy()
    result = conversation.send("本文なしになる質問")
    assert result.text == ""
    assert conversation.history == previous
    if empty_response.prompt_feedback:
        assert result.block_reason == "SAFETY"


@pytest.mark.parametrize("block_at_prompt", [False, True])
def test_nonempty_blocked_response_does_not_change_history(block_at_prompt):
    blocked = response("途中まで生成された本文", finish_reason="STOP" if block_at_prompt else "SAFETY")
    if block_at_prompt:
        blocked.prompt_feedback = types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.SAFETY,
        )
    client = FakeClient(response("以前の回答"), blocked, response("次の回答"))
    conversation = ask.Conversation(client)
    conversation.send("以前の質問")
    previous = conversation.history.copy()

    result = conversation.send("ブロックされる質問")

    assert result.text == "途中まで生成された本文"
    assert conversation.history == previous
    assert (result.block_reason if block_at_prompt else result.finish_reason) == "SAFETY"
    conversation.send("次の質問")
    assert [item.parts[0].text for item in client.calls[2]["contents"]] == [
        "以前の質問", "以前の回答", "次の質問"
    ]


@pytest.mark.parametrize("finish_reason", [None, "STOP", "MAX_TOKENS"])
def test_normal_and_length_limited_responses_remain_in_history(finish_reason):
    generated = response("保存する回答", finish_reason=finish_reason)
    conversation = ask.Conversation(FakeClient(generated))
    result = conversation.send("質問")
    assert result.finish_reason == finish_reason
    assert len(conversation.history) == 2
    assert conversation.history[1] is generated.candidates[0].content


def test_reset_removes_all_history_and_next_request_starts_fresh():
    client = FakeClient(response(), response())
    conversation = ask.Conversation(client)
    conversation.send("以前の質問")
    conversation.reset()
    conversation.reset()  # 空でも安全に実行できる。
    assert conversation.history == []
    conversation.send("新しい質問")
    assert len(client.calls[1]["contents"]) == 1
    assert client.calls[1]["contents"][0].parts[0].text == "新しい質問"


def test_make_client_uses_env_key_finite_timeout_and_one_attempt(monkeypatch):
    captured = {}
    sentinel = object()

    def construct(**kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setenv("GEMINI_API_KEY", "dummy-test-only-key")
    monkeypatch.setattr(ask.genai, "Client", construct)
    assert ask.make_client() is sentinel
    assert captured["api_key"] == "dummy-test-only-key"
    assert captured["http_options"].timeout == 120_000
    assert captured["http_options"].retry_options.attempts == 1


def test_missing_api_key_fails_before_client_creation(monkeypatch):
    def construct(**kwargs):
        pytest.fail("キーがなければ SDK クライアントを作成してはいけません")

    monkeypatch.setattr(ask.genai, "Client", construct)
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        ask.make_client()


def test_ask_compatibility_returns_text_and_prints_usage(monkeypatch, capsys):
    client = FakeClient(response("互換性のある回答"))
    monkeypatch.setattr(ask, "make_client", lambda: client)
    assert ask.ask("質問", model="specific", temperature=0.0, max_output_tokens=16) == "互換性のある回答"
    assert client.entered and client.closed
    assert client.calls[0]["model"] == "specific"
    assert client.calls[0]["config"].temperature == 0.0
    assert client.calls[0]["config"].max_output_tokens == 16
    output = capsys.readouterr().out
    assert "[tokens]" in output
    assert "互換性のある回答" not in output


def test_display_result_exposes_empty_blocked_finish_and_timing(capsys):
    ask.display_result(ask.Result("", ask.Usage(), 0.1254, "SAFETY", "test", "SAFETY"))
    output = capsys.readouterr().out
    assert "[本文なし]" in output
    assert "[elapsed] 0.125 秒" in output
    assert "[finish] SAFETY" in output
    assert "[block] SAFETY" in output


def test_normal_stop_does_not_print_extra_warning(capsys):
    ask.display_result(ask.Result("回答", ask.Usage(1, 2, 3), 1.0, "STOP", "test"))
    output = capsys.readouterr().out
    assert "回答" in output
    assert "[finish]" not in output
    assert "[block]" not in output


@pytest.mark.parametrize("command", ["/exit", "/quit", "  /exit  "])
def test_chat_exit_never_calls_sdk(monkeypatch, command):
    client = FakeClient()
    install_input(monkeypatch, command)
    ask.run_chat(ask.Conversation(client))
    assert client.calls == []


@pytest.mark.parametrize("ending", [EOFError(), KeyboardInterrupt()])
def test_chat_eof_and_keyboard_interrupt_exit_cleanly(monkeypatch, capsys, ending):
    install_input(monkeypatch, ending)
    ask.run_chat(ask.Conversation(FakeClient()))
    assert "終了しました" in capsys.readouterr().out


def test_chat_initial_question_blank_lines_and_reset(monkeypatch, capsys):
    client = FakeClient(response("初回回答"), response("リセット後の回答"))
    install_input(monkeypatch, "", "  ", "/reset", "  次の質問  ", "/exit")
    conversation = ask.Conversation(client)
    ask.run_chat(conversation, initial_question="最初の質問")
    assert len(client.calls) == 2
    assert client.calls[0]["contents"][0].parts[0].text == "最初の質問"
    assert len(client.calls[1]["contents"]) == 1
    assert client.calls[1]["contents"][0].parts[0].text == "次の質問"
    assert len(conversation.history) == 2
    output = capsys.readouterr().out
    assert "履歴を消去しました" in output
    assert "初回回答" in output and "リセット後の回答" in output


@pytest.mark.parametrize("failure", [api_error(429), api_error(403), httpx.ReadTimeout("DUMMY_SECRET_MUST_NOT_APPEAR")])
def test_chat_errors_are_sanitized_and_allow_another_turn(monkeypatch, capsys, caplog, failure):
    client = FakeClient(failure, response("復帰できました"))
    install_input(monkeypatch, "失敗する質問", "再試行する質問", "/exit")
    conversation = ask.Conversation(client)
    ask.run_chat(conversation)
    captured = capsys.readouterr()
    output = captured.out
    assert "DUMMY_SECRET_MUST_NOT_APPEAR" not in output + captured.err + caplog.text
    assert "request details" not in output + captured.err + caplog.text
    assert "復帰できました" in output
    assert len(client.calls) == 2  # 自動リトライではなく、次の入力で呼び出す。
    assert len(client.calls[1]["contents"]) == 1
    assert len(conversation.history) == 2
    if isinstance(failure, errors.APIError) and failure.code == 429:
        assert "日次" in output


@pytest.mark.parametrize("args,question", [([], ask.DEFAULT_QUESTION), (["明示した質問"], "明示した質問")])
def test_main_defaults_to_one_shot_and_preserves_default_question(monkeypatch, capsys, args, question):
    client = FakeClient(response("一度だけの回答"))
    run_main(monkeypatch, client, *args)
    assert client.entered and client.closed
    assert len(client.calls) == 1
    assert client.calls[0]["contents"][0].parts[0].text == question
    output = capsys.readouterr().out
    assert "一度だけの回答" in output
    assert "[tokens]" in output and "[elapsed]" in output


def test_main_passes_explicit_generation_options(monkeypatch):
    client = FakeClient(response())
    run_main(monkeypatch, client, "質問", "--model", "cli-model", "--temperature", "2", "--max-output-tokens", "42")
    assert client.calls[0]["model"] == "cli-model"
    assert client.calls[0]["config"].temperature == 2.0
    assert client.calls[0]["config"].max_output_tokens == 42


@pytest.mark.parametrize("initial", [None, "開始時の質問"])
def test_main_chat_mode_passes_optional_initial_question(monkeypatch, initial):
    client = FakeClient()
    received = []
    monkeypatch.setattr(ask, "run_chat", lambda conversation, question: received.append((conversation, question)))
    args = ["--chat"] + ([] if initial is None else [initial])
    run_main(monkeypatch, client, *args)
    assert received[0][1] == initial
    assert isinstance(received[0][0], ask.Conversation)
    assert client.calls == []
    assert client.closed


@pytest.mark.parametrize("option,value", [
    ("--temperature", "-0.1"), ("--temperature", "2.1"),
    ("--temperature", "nan"), ("--temperature", "inf"),
    ("--temperature", "not-a-number"),
    ("--max-output-tokens", "0"), ("--max-output-tokens", "-1"),
    ("--max-output-tokens", "1.5"),
])
def test_invalid_cli_values_fail_before_client_creation(monkeypatch, option, value):
    client = FakeClient()
    with pytest.raises(SystemExit) as caught:
        run_main(monkeypatch, client, option, value)
    assert caught.value.code == 2
    assert not client.entered
    assert client.calls == []


@pytest.mark.parametrize("failure", [api_error(429), api_error(403), api_error(503), httpx.ConnectError("DUMMY_SECRET_MUST_NOT_APPEAR")])
def test_main_errors_are_nonzero_sanitized_and_close_client(monkeypatch, capsys, caplog, failure):
    client = FakeClient(failure)
    with pytest.raises(SystemExit) as caught:
        run_main(monkeypatch, client, "質問")
    assert caught.value.code == 1
    assert client.closed
    assert len(client.calls) == 1
    output = capsys.readouterr()
    assert "DUMMY_SECRET_MUST_NOT_APPEAR" not in output.out + output.err + caplog.text
    assert "request details" not in output.out + output.err + caplog.text


def test_main_empty_question_is_input_error_without_calling_sdk(monkeypatch):
    client = FakeClient()
    with pytest.raises(SystemExit) as caught:
        run_main(monkeypatch, client, " ")
    assert caught.value.code == 1
    assert client.calls == []
    assert client.closed


def test_main_keyboard_interrupt_returns_130_and_closes_client(monkeypatch):
    client = FakeClient(KeyboardInterrupt())
    with pytest.raises(SystemExit) as caught:
        run_main(monkeypatch, client, "質問")
    assert caught.value.code == 130
    assert client.closed


@pytest.mark.parametrize("model", [None, "environment-selected-model"])
def test_model_env_is_read_on_fresh_import_without_touching_global_modules(model):
    """別プロセスの import だけを使い、既存テストのモジュール状態を変えない。"""
    env = os.environ.copy()
    env.pop("AIE_MODEL", None)
    if model is not None:
        env["AIE_MODEL"] = model
    result = subprocess.run(
        [sys.executable, "-c", "\n".join([
            "import json, socket",
            "def forbidden(*args, **kwargs): raise AssertionError('network forbidden')",
            "socket.socket.connect = forbidden",
            "socket.create_connection = forbidden",
            "from app import ask, config",
            "print(json.dumps([config.DEFAULT_MODEL, config.MODEL, ask.MODEL, ask.Conversation(None).model]))",
        ])],
        cwd=Path(__file__).resolve().parents[1], env=env,
        check=True, capture_output=True, text=True, timeout=15,
    )
    default, configured, imported, conversation_model = json.loads(result.stdout)
    assert default == "gemini-3.5-flash-lite"
    assert configured == imported == conversation_model == (model or default)


def test_cli_model_overrides_imported_environment_default(monkeypatch):
    monkeypatch.setattr(ask, "MODEL", "environment-selected-model")
    default_client = FakeClient(response())
    run_main(monkeypatch, default_client, "質問")
    explicit_client = FakeClient(response())
    run_main(monkeypatch, explicit_client, "質問", "--model", "explicit-cli-model")
    assert default_client.calls[0]["model"] == "environment-selected-model"
    assert explicit_client.calls[0]["model"] == "explicit-cli-model"
