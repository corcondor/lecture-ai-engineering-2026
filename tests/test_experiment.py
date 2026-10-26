"""第1回の比較実験を、APIキー・通信・料金なしで検証する。"""

import csv
import json
import random
import sys
from collections import Counter
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors, types

from app import ask, experiment


class FakeClient:
    """Record requests and return queued SDK-shaped responses or exceptions."""

    def __init__(self, *outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []
        self.models = self
        self.closed = False

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.closed = True


def response(text="回答", *, finish_reason="STOP", usage=None, block_reason=None):
    return SimpleNamespace(
        text=text,
        candidates=[SimpleNamespace(
            finish_reason=finish_reason,
            content=types.Content(role="model", parts=[types.Part(text=text or "")]),
        )],
        usage_metadata=usage,
        model_version="fake-model-version",
        prompt_feedback=(SimpleNamespace(block_reason=block_reason) if block_reason else None),
    )


def trial(**overrides):
    return {
        "trial_id": 1, "model": "model-a", "temperature": 0.2,
        "question": "LoRAとは？", "repeat": 1, **overrides,
    }


def successful_row(**overrides):
    return {
        **trial(), "status": "ok", "text": "回答", "elapsed_seconds": 2.0,
        **{field: None for field in experiment.TOKEN_FIELDS}, **overrides,
    }


@pytest.fixture(autouse=True)
def prohibit_live_clients(monkeypatch):
    """Fail closed even if a test accidentally uses an unmocked API path."""
    def forbidden(*_args, **_kwargs):
        pytest.fail("Offline experiment tests must never create a real API client")
    monkeypatch.setattr(experiment, "make_client", forbidden)
    monkeypatch.setattr(ask.genai, "Client", forbidden)
    monkeypatch.setattr(experiment, "MODEL", "model-a")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["app.experiment", *map(str, args)])
    return experiment.main()


def live_args(tmp_path, *, repeats=3):
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps(["独立した質問"]), encoding="utf-8")
    directory = tmp_path / "run"
    return directory, [
        "--confirm-free-tier", "--models", "model-a", "--temperatures", "0.2",
        "--questions-file", questions, "--repeats", str(repeats),
        "--interval", "0", "--output-dir", directory,
    ]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_default_plan_has_24_balanced_independent_trials():
    plan = experiment.build_plan(
        ["model-a", "model-b"], [0.2, 1.0], experiment.DEFAULT_QUESTIONS,
    )
    assert len(plan) == 24
    assert [row["trial_id"] for row in plan] == list(range(1, 25))
    counts = Counter((r["model"], r["temperature"], r["question"]) for r in plan)
    assert len(counts) == 8
    assert set(counts.values()) == {3}
    for group in counts:
        assert sorted(r["repeat"] for r in plan if
                      (r["model"], r["temperature"], r["question"]) == group) == [1, 2, 3]
    assert len({id(row) for row in plan}) == 24


def test_plan_shuffle_is_reproducible_without_modifying_global_rng():
    args = (["a", "b"], [0.2, 1.0], ["q1", "q2"])
    state = random.getstate()
    first = experiment.build_plan(*args, seed=23)
    assert first == experiment.build_plan(*args, seed=23)
    assert first != experiment.build_plan(*args, seed=24)
    assert random.getstate() == state
    canonical = lambda rows: Counter(
        tuple(row[key] for key in ("model", "temperature", "question", "repeat"))
        for row in rows
    )
    assert canonical(first) == canonical(experiment.build_plan(*args, seed=24))


def test_each_trial_uses_fresh_context_and_requested_configuration():
    client = FakeClient(response("最初の回答"), response("次の回答"))
    first = experiment.run_trial(client, trial(question="最初の質問"), 123)
    second = experiment.run_trial(client, trial(question="次の質問", trial_id=2), 321)
    assert first["status"] == second["status"] == "ok"
    assert [len(call["contents"]) for call in client.calls] == [1, 1]
    assert [call["contents"][0].parts[0].text for call in client.calls] == ["最初の質問", "次の質問"]
    assert all(call["contents"][0].role == "user" for call in client.calls)
    assert [call["config"].max_output_tokens for call in client.calls] == [123, 321]
    assert all(call["model"] == "model-a" and call["config"].temperature == 0.2 for call in client.calls)


def test_run_trial_flattens_real_usage_fields_and_records_duration(monkeypatch):
    usage = SimpleNamespace(
        prompt_token_count=11, candidates_token_count=7, total_token_count=23,
        thoughts_token_count=5, cached_content_token_count=3,
    )
    times = iter([10.0, 11.0, 13.5])
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: next(times))
    row = experiment.run_trial(FakeClient(response(usage=usage)), trial(), 128)
    assert row["status"] == "ok"
    assert row["elapsed_seconds"] == 2.5
    assert {key: row[key] for key in experiment.TOKEN_FIELDS} == {
        "input_tokens": 11, "output_tokens": 7, "total_tokens": 23,
        "thought_tokens": 5, "cached_tokens": 3,
    }
    assert "usage" not in row
    assert row["model_version"] == "fake-model-version"
    assert row["finish_reason"] == "STOP"
    assert row["started_at"].endswith("+00:00")


@pytest.mark.parametrize("text,reason,status", [
    ("完全な回答", "STOP", "ok"),
    ("途中まで", "MAX_TOKENS", "incomplete"),
    ("理由が不明", None, "incomplete"),
    ("", "STOP", "empty"),
    (None, "SAFETY", "empty"),
])
def test_trial_classifies_incomplete_and_empty_responses(text, reason, status):
    row = experiment.run_trial(FakeClient(response(text, finish_reason=reason)), trial(), 128)
    assert row["status"] == status
    assert all(row[field] is None for field in experiment.TOKEN_FIELDS)


@pytest.mark.parametrize("text,reason", [("", "SAFETY"), ("部分的な回答", "STOP")])
def test_block_reason_is_preserved_but_not_counted_as_success(text, reason):
    result = response(text, finish_reason=reason, block_reason="SAFETY")
    row = experiment.run_trial(FakeClient(result), trial(), 128)
    assert row["block_reason"] == "SAFETY"
    assert experiment.summarize([row], [trial()])[0]["successful"] == 0


@pytest.mark.parametrize("code", [429, 403, 503])
def test_api_errors_store_safe_status_not_raw_exception_or_secret(code):
    secret = "secret-key-must-not-appear"
    error = errors.ClientError(code, {"error": {"message": secret, "status": "FAILURE"}})
    row = experiment.run_trial(FakeClient(error), trial(), 128)
    assert row["status"] == "api_error"
    assert row["error_code"] == code
    assert row["error_message"]
    assert secret not in json.dumps(row)
    assert row["elapsed_seconds"] >= 0


def test_transport_error_hides_request_url_and_credentials():
    request = httpx.Request("POST", "https://example.invalid/?key=secret-key-must-not-appear")
    error = httpx.ConnectError("secret-key-must-not-appear", request=request)
    row = experiment.run_trial(FakeClient(error), trial(), 128)
    assert row["status"] == "network_error"
    assert "secret-key" not in json.dumps(row)
    assert "example.invalid" not in json.dumps(row)
    assert row["elapsed_seconds"] >= 0


def test_summary_excludes_failed_empty_and_truncated_results():
    plan = experiment.build_plan(["model-a"], [0.2], ["LoRAとは？"], repeats=5)
    rows = [
        successful_row(trial_id=1, text="回答1", elapsed_seconds=2, input_tokens=10),
        successful_row(trial_id=2, text="回答2", elapsed_seconds=4, input_tokens=20),
        successful_row(trial_id=3, status="incomplete", text="途中", elapsed_seconds=100, input_tokens=999),
        successful_row(trial_id=4, status="empty", text="", elapsed_seconds=100, input_tokens=999),
        {**trial(trial_id=5), "status": "api_error", "elapsed_seconds": 100},
    ]
    summary, = experiment.summarize(rows, plan)
    assert summary["planned"] == summary["attempted"] == 5
    assert summary["successful"] == 2
    assert summary["incomplete"] == 3
    assert summary["comparison_complete"] is False
    assert summary["distinct_answers"] == 2
    assert summary["mean_seconds"] == summary["median_seconds"] == 3
    assert summary["min_seconds"] == 2
    assert summary["max_seconds"] == 4
    assert summary["mean_input_tokens"] == 15
    assert summary["known_input_tokens"] == 2


def test_missing_metrics_remain_unknown_and_known_zero_stays_zero():
    plan = experiment.build_plan(["model-a"], [0.2], ["LoRAとは？"], repeats=3)
    rows = [successful_row(input_tokens=12, output_tokens=0), successful_row(trial_id=2)]
    summary, = experiment.summarize(rows, plan)
    assert summary["mean_input_tokens"] == 12
    assert summary["known_input_tokens"] == 1
    assert summary["mean_output_tokens"] == 0
    assert summary["known_output_tokens"] == 1
    assert summary["mean_total_tokens"] is None
    assert summary["known_total_tokens"] == 0
    assert summary["successful"] == 2
    assert summary["attempted"] == 2
    assert summary["planned"] == 3
    assert summary["comparison_complete"] is False


def test_summary_keeps_unattempted_groups_without_inventing_measurements():
    plan = experiment.build_plan(["model-a", "model-b"], [0.2], ["LoRAとは？"], repeats=3)
    summaries = experiment.summarize([successful_row()], plan)
    missing = next(row for row in summaries if row["model"] == "model-b")
    assert missing["planned"] == 3
    assert missing["attempted"] == missing["successful"] == missing["distinct_answers"] == 0
    assert missing["comparison_complete"] is False
    for metric in ("mean_seconds", "median_seconds", "min_seconds", "max_seconds"):
        assert missing[metric] is None
    for field in experiment.TOKEN_FIELDS:
        assert missing[f"mean_{field}"] is None
        assert missing[f"known_{field}"] == 0


def test_complete_comparison_requires_all_planned_successes():
    plan = experiment.build_plan(["model-a"], [0.2], ["LoRAとは？"], repeats=3)
    summary, = experiment.summarize([
        successful_row(trial_id=i, text="同じ回答") for i in range(1, 4)
    ], plan)
    assert summary["comparison_complete"] is True
    assert summary["distinct_answers"] == 1


@pytest.mark.parametrize("value", ["=1+1", "+SUM(A1:A2)", "-1+2", "@SUM(A1)", " \t=1+1", "\n+1"])
def test_csv_formula_cells_are_escaped(value):
    assert experiment.csv_safe(value) == "'" + value


@pytest.mark.parametrize("value", [None, 0, -5, 0.2, "普通の回答", "", "  normal"])
def test_csv_ordinary_data_is_unchanged(value):
    assert experiment.csv_safe(value) == value


def test_report_outputs_escape_csv_without_mutating_original_data(tmp_path):
    row = successful_row(question="=1+1", model="+model", text="@SUM(A1:A2)")
    plan = [trial(question=row["question"], model=row["model"])]
    experiment.write_reports(tmp_path, [row], plan, "complete")
    with (tmp_path / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        record, = csv.DictReader(stream)
    assert record["question"] == "'=1+1"
    assert record["model"] == "'+model"
    assert record["text"] == "'@SUM(A1:A2)"
    assert row["text"] == "@SUM(A1:A2)"
    assert (tmp_path / "results.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["groups"][0]["question"] == "=1+1"
    report = (tmp_path / "report.md").read_text()
    assert "    @SUM(A1:A2)" in report
    assert "TTFT" in report and "未測定" in report


def test_dry_run_default_is_24_requests_and_creates_no_api_client(monkeypatch, tmp_path, capsys):
    directory = tmp_path / "dry"
    invoke(monkeypatch, "--dry-run", "--output-dir", directory)
    plan = json.loads((directory / "plan.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    assert plan["dry_run"] is True
    assert plan["free_tier_user_confirmed"] is False
    assert plan["sdk_attempts"] == 1
    assert len(plan["trials"]) == 24
    assert summary["run_status"] == "dry_run"
    assert summary["planned"] == 24 and summary["attempted"] == 0
    assert all(group["comparison_complete"] is False for group in summary["groups"])
    assert (directory / "results.csv").exists() and (directory / "report.md").exists()
    assert not (directory / "results.jsonl").exists()
    assert "API呼び出しは0回" in capsys.readouterr().out


def test_live_run_without_consent_exits_two_before_creating_files(monkeypatch, tmp_path, capsys):
    directory = tmp_path / "denied"
    with pytest.raises(SystemExit) as caught:
        invoke(monkeypatch, "--output-dir", directory)
    assert caught.value.code == 2
    assert not directory.exists()
    assert "--confirm-free-tier" in capsys.readouterr().err


def test_request_cap_rejects_plan_before_any_side_effect(monkeypatch, tmp_path, capsys):
    directory = tmp_path / "over-cap"
    with pytest.raises(SystemExit) as caught:
        invoke(monkeypatch, "--dry-run", "--max-requests", "23", "--output-dir", directory)
    assert caught.value.code == 2
    assert not directory.exists()
    assert "24" in capsys.readouterr().err


def test_request_cap_includes_all_repeats(monkeypatch, tmp_path):
    directory, args = live_args(tmp_path, repeats=25)
    with pytest.raises(SystemExit) as caught:
        invoke(monkeypatch, *args)
    assert caught.value.code == 2
    assert not directory.exists()


def test_huge_repeat_cap_is_checked_before_materializing_plan(monkeypatch, tmp_path):
    def forbidden(*_args, **_kwargs):
        pytest.fail("An over-cap plan must not be materialized")
    monkeypatch.setattr(experiment, "build_plan", forbidden)
    directory = tmp_path / "huge-plan"
    with pytest.raises(SystemExit) as caught:
        invoke(monkeypatch, "--dry-run", "--repeats", "1000000000000",
               "--output-dir", directory)
    assert caught.value.code == 2
    assert not directory.exists()


def test_existing_output_directory_is_never_overwritten(monkeypatch, tmp_path, capsys):
    directory = tmp_path / "existing"
    directory.mkdir()
    sentinel = directory / "plan.json"
    sentinel.write_text("original-data")
    with pytest.raises(SystemExit) as caught:
        invoke(monkeypatch, "--dry-run", "--output-dir", directory)
    assert caught.value.code == 2
    assert sentinel.read_text() == "original-data"
    assert list(directory.iterdir()) == [sentinel]
    assert "上書き" in capsys.readouterr().err


def test_live_fake_run_writes_plan_raw_results_and_aggregates(monkeypatch, tmp_path):
    directory, args = live_args(tmp_path)
    client = FakeClient(*(response(f"回答{i}", usage=SimpleNamespace(prompt_token_count=i)) for i in (1, 2, 3)))
    monkeypatch.setattr(experiment, "make_client", lambda: client)
    sleeps = []
    monkeypatch.setattr(experiment.time, "sleep", sleeps.append)
    invoke(monkeypatch, *args)
    assert client.closed
    assert len(client.calls) == 3
    assert sleeps == [0.0, 0.0]
    assert {p.name for p in directory.iterdir()} == {
        "plan.json", "results.jsonl", "results.csv", "summary.json", "report.md",
    }
    rows = read_jsonl(directory / "results.jsonl")
    assert [row["text"] for row in rows] == ["回答1", "回答2", "回答3"]
    assert all(row["status"] == "ok" for row in rows)
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["run_status"] == "complete"
    assert summary["planned"] == summary["attempted"] == 3
    assert summary["groups"][0]["comparison_complete"] is True
    assert summary["groups"][0]["mean_input_tokens"] == 2
    assert json.loads((directory / "plan.json").read_text())["free_tier_user_confirmed"] is True


@pytest.mark.parametrize("failure,status", [
    (errors.ClientError(429, {"error": {"message": "never-log-this-secret"}}), "api_error"),
    (httpx.ReadTimeout("never-log-this-secret"), "network_error"),
])
def test_live_error_is_saved_and_stops_without_retry(monkeypatch, tmp_path, capsys, failure, status):
    directory, args = live_args(tmp_path)
    client = FakeClient(response(), failure, response("must not run"))
    monkeypatch.setattr(experiment, "make_client", lambda: client)
    with pytest.raises(SystemExit) as caught:
        invoke(monkeypatch, *args)
    assert caught.value.code == 1
    assert client.closed and len(client.calls) == 2
    rows = read_jsonl(directory / "results.jsonl")
    assert [row["status"] for row in rows] == ["ok", status]
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["run_status"] == status and summary["attempted"] == 2
    assert summary["groups"][0]["successful"] == 1
    assert summary["groups"][0]["comparison_complete"] is False
    output = capsys.readouterr()
    assert "never-log-this-secret" not in output.out + output.err
    assert all("never-log-this-secret" not in path.read_text(encoding="utf-8-sig") for path in directory.iterdir())
    if status == "api_error":
        assert "RPM" in output.out and "TPM" in output.out and "RPD" in output.out


def test_live_incomplete_responses_are_preserved_but_fail_completion(monkeypatch, tmp_path):
    directory, args = live_args(tmp_path)
    client = FakeClient(response(), response("途中", finish_reason="MAX_TOKENS"), response(""))
    monkeypatch.setattr(experiment, "make_client", lambda: client)
    with pytest.raises(SystemExit) as caught:
        invoke(monkeypatch, *args)
    assert caught.value.code == 1
    assert len(client.calls) == 3
    assert [r["status"] for r in read_jsonl(directory / "results.jsonl")] == ["ok", "incomplete", "empty"]
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["run_status"] == "complete_with_incomplete_responses"
    assert summary["groups"][0]["successful"] == 1


@pytest.mark.parametrize("interrupt_during", ["request", "interval"])
def test_keyboard_interrupt_preserves_previous_results(monkeypatch, tmp_path, interrupt_during):
    directory, args = live_args(tmp_path)
    client = FakeClient(response("保存すべき回答"), KeyboardInterrupt())
    monkeypatch.setattr(experiment, "make_client", lambda: client)
    if interrupt_during == "interval":
        def interrupt(_seconds):
            raise KeyboardInterrupt
        monkeypatch.setattr(experiment.time, "sleep", interrupt)
    with pytest.raises(SystemExit) as caught:
        invoke(monkeypatch, *args)
    assert caught.value.code == 130
    assert client.closed
    rows = read_jsonl(directory / "results.jsonl")
    assert len(rows) == 1 and rows[0]["text"] == "保存すべき回答"
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["run_status"] == "interrupted"
    assert summary["attempted"] == 1 and summary["planned"] == 3
    assert summary["groups"][0]["comparison_complete"] is False
    assert "保存すべき回答" in (directory / "report.md").read_text()
    assert len(client.calls) == (2 if interrupt_during == "request" else 1)


@pytest.mark.parametrize("contents", ["null", '"not an array"', "{}", "[]", '[""]', '["   "]', '[42]', '["ok", null]', '["ok", false]', "not json"])
def test_invalid_question_file_is_rejected_before_creating_output(monkeypatch, tmp_path, contents):
    questions = tmp_path / "questions.json"
    questions.write_text(contents, encoding="utf-8")
    directory = tmp_path / "invalid"
    with pytest.raises(SystemExit) as caught:
        invoke(monkeypatch, "--dry-run", "--questions-file", questions, "--output-dir", directory)
    assert caught.value.code == 2
    assert not directory.exists()


@pytest.mark.parametrize("kind", ["missing", "invalid_utf8"])
def test_unreadable_question_file_has_friendly_usage_error(monkeypatch, tmp_path, capsys, kind):
    questions = tmp_path / "questions.json"
    if kind == "invalid_utf8":
        questions.write_bytes(b"\xff\xfe")
    with pytest.raises(SystemExit) as caught:
        invoke(monkeypatch, "--dry-run", "--questions-file", questions, "--output-dir", tmp_path / "bad")
    assert caught.value.code == 2
    assert "UTF-8" in capsys.readouterr().err


def test_duplicate_conditions_are_deduplicated_before_applying_cap(monkeypatch, tmp_path):
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps(["同じ質問", "同じ質問"], ensure_ascii=False), encoding="utf-8")
    directory = tmp_path / "deduplicated"
    invoke(monkeypatch, "--dry-run", "--questions-file", questions,
           "--models", "model-a", "model-a", "--temperatures", "0.2", "0.2",
           "--max-requests", "3", "--output-dir", directory)
    plan = json.loads((directory / "plan.json").read_text())
    assert len(plan["trials"]) == 3
    assert all(t["question"] == "同じ質問" for t in plan["trials"])


@pytest.mark.parametrize("option,value", [
    ("--repeats", "0"), ("--max-requests", "0"), ("--max-output-tokens", "-1"),
    ("--interval", "-1"), ("--interval", "nan"), ("--interval", "inf"),
    ("--temperatures", "nan"), ("--temperatures", "inf"), ("--temperatures", "2.1"),
    ("--models", " "),
])
def test_invalid_numeric_options_and_blank_models_are_rejected(monkeypatch, tmp_path, option, value):
    directory = tmp_path / "bad-args"
    with pytest.raises(SystemExit) as caught:
        invoke(monkeypatch, "--dry-run", option, value, "--output-dir", directory)
    assert caught.value.code == 2
    assert not directory.exists()
