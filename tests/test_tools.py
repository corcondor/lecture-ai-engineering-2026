"""ツールのテスト（LLMもAPIキーも使わない）。"""

import json

import pytest

from app import tools


def test_glossary_ids_are_unique_and_related_links_exist():
    terms = tools.load_glossary()
    ids = [t["id"] for t in terms]
    assert len(ids) == len(set(ids))
    for t in terms:
        for r in t["related"]:
            assert r in ids, f"{t['id']} の related に存在しない id: {r}"


def test_search_finds_term_by_name_and_alias():
    for query in ["LoRA", "lora", "Low-Rank Adaptation"]:
        hits = json.loads(tools.search_glossary(query))
        assert hits[0]["term"] == "LoRA", query


def test_search_finds_course_specific_term():
    hits = json.loads(tools.search_glossary("lecNN-start"))
    assert hits[0]["term"] == "lecNN-start タグ"


def test_search_limits_number_of_results():
    hits = json.loads(tools.search_glossary("エージェント"))
    assert 1 <= len(hits) <= tools.MAX_RESULTS


def test_search_reports_not_found():
    assert "見つかりませんでした" in tools.search_glossary("存在しない用語xyz")


def test_lecture_summary():
    assert "エージェント" in tools.get_lecture_summary(3)
    assert "まだありません" in tools.get_lecture_summary(99)


def test_default_question_of_lecture1_is_answerable_from_glossary():
    """第1回の既定の質問（app.ask）は、第3回のエージェントが用語集で正しく答えられること。"""
    from app.ask import DEFAULT_QUESTION

    assert "3レベル" in DEFAULT_QUESTION
    hits = json.loads(tools.search_glossary("3レベル"))
    assert hits[0]["term"] == "宿題の3レベル"


def test_make_quiz_uses_glossary():
    quiz = json.loads(tools.make_quiz(2))
    assert quiz["lecture"] == 2
    assert any(e["definition"] == quiz["answer"] and e["term"] in quiz["question"]
               for e in tools.load_glossary() if e["lecture"] == 2)
    assert "まだありません" in tools.make_quiz(99)


def test_save_json_and_no_overwrite(tmp_path, monkeypatch):
    target = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", target)
    content = "日本語の記録\n../../escape.json"
    tools.save_learning_log(3, content)
    tools.save_learning_log(3, content)
    files = list(target.glob("*.json"))
    assert len(files) == 2
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["content"] == content
        assert data["lecture"] == 3
        assert data["created_at"].endswith("+00:00")
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("lecture,content", [(0, "x"), (-1, "x"), (True, "x"),
    ("../escape", "x"), (3, ""), (3, " \n"), (3, "x" * 10001)])
def test_invalid_log_does_not_write(tmp_path, monkeypatch, lecture, content):
    target = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", target)
    with pytest.raises(ValueError):
        tools.save_learning_log(lecture, content)
    assert not target.exists()


def test_log_directory_symlink_rejected(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    target = tmp_path / "learning_log"
    target.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", target)
    with pytest.raises(ValueError):
        tools.save_learning_log(3, "test")
    assert not list(outside.iterdir())
