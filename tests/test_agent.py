"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

import pytest
from pydantic_ai import DeferredToolRequests, DeferredToolResults, models
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent, run_with_approval

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel(call_tools=sorted(BASE_TOOLS))
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


class LearningLogModel(TestModel):
    def gen_tool_args(self, tool_def):
        if tool_def.name == "save_learning_log":
            return {"lecture": 3, "content": "承認後だけ保存する。"}
        return super().gen_tool_args(tool_def)


@pytest.fixture(autouse=True)
def no_live_models(monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    path = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", path)
    return path


@pytest.mark.parametrize("approved", [True, False])
def test_save_waits_for_approval_and_resumes(log_dir, approved):
    model = LearningLogModel(call_tools=["save_learning_log"])
    result = agent.run_sync("学習記録を保存して", model=model)
    assert isinstance(result.output, DeferredToolRequests)
    assert not log_dir.exists()
    call, = result.output.approvals
    assert call.tool_name == "save_learning_log"
    result = agent.run_sync(model=model, message_history=result.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: approved}))
    assert isinstance(result.output, str)
    assert len(list(log_dir.glob("*.json"))) == int(approved)
    returns = [p for p in _parts(result, ToolReturnPart) if p.tool_name == "save_learning_log"]
    assert returns
    if approved:
        assert "保存しました" in str(returns[-1].content)
    else:
        assert "denied" in str(returns[-1].content).lower()


@pytest.mark.parametrize("reply,expected", [("y", 1), (" YES ", 1), ("n", 0), ("", 0), ("maybe", 0)])
def test_cli_approval(log_dir, monkeypatch, capsys, reply, expected):
    def confirm(prompt):
        assert not log_dir.exists()
        assert "承認後だけ保存する。" in capsys.readouterr().out
        return reply
    monkeypatch.setattr("builtins.input", confirm)
    result = run_with_approval(agent, "保存して", LearningLogModel(call_tools=["save_learning_log"]))
    assert isinstance(result.output, str)
    assert len(list(log_dir.glob("*.json"))) == expected


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_cli_interrupted_does_not_save(log_dir, monkeypatch, error):
    def interrupt(prompt):
        raise error()
    monkeypatch.setattr("builtins.input", interrupt)
    with pytest.raises(SystemExit):
        run_with_approval(agent, "保存して", LearningLogModel(call_tools=["save_learning_log"]))
    assert not log_dir.exists()


def test_quiz_registered_and_result_returned():
    result = agent.run_sync("確認問題", model=TestModel(call_tools=["make_quiz"], seed=2))
    assert any(p.tool_name == "make_quiz" for p in _parts(result, ToolReturnPart))


def test_cli_usage_includes_request_before_approval(log_dir, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    result = run_with_approval(agent, "保存して", LearningLogModel(call_tools=["save_learning_log"]))
    # TestModel の模擬リクエスト。保留前1回 + 再開後1回を合算する。
    assert result.usage.requests == 2
