"""第3回: エージェントに持たせるツール。

ここに書いた関数の docstring と型ヒントが、そのままモデルへの「ツールの説明文」になる。
説明文を書き換えると、モデルがツールを使うかどうか・どんな引数で呼ぶかが変わる。
"""

import json
from datetime import datetime, timezone
from functools import cache
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from pydantic import Field

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_RESULTS = 3


@cache
def load_glossary() -> list[dict]:
    with open(DATA_DIR / "glossary.json", encoding="utf-8") as f:
        return json.load(f)["terms"]


def _normalize(text: str) -> str:
    return text.casefold().replace(" ", "").replace("　", "")


def _names(entry: dict) -> list[str]:
    return [_normalize(entry["term"]), _normalize(entry["id"])] + [
        _normalize(a) for a in entry["aliases"]
    ]


def search_glossary(query: str) -> str:
    """講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。

    AIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について
    答える前に使うこと。query には調べたい用語を1つだけ入れる（例: "LoRA"）。
    """
    q = _normalize(query)
    if not q:
        return "検索語が空です。調べたい用語を1つ指定してください。"
    glossary = load_glossary()

    exact = [e for e in glossary if q in _names(e)]
    # 「r」のような短い別名が何にでも当たらないよう、名前側が短いものは除く
    partial = [
        e for e in glossary
        if e not in exact and any(q in n or (len(n) >= 3 and n in q) for n in _names(e))
    ]
    in_definition = [
        e for e in glossary
        if e not in exact and e not in partial and q in _normalize(e["definition"])
    ]
    hits = (exact + partial + in_definition)[:MAX_RESULTS]

    if not hits:
        return f"用語集に「{query}」は見つかりませんでした。"
    return json.dumps(
        [
            {
                "term": e["term"],
                "lecture": e["lecture"],
                "definition": e["definition"],
                "related": e["related"],
            }
            for e in hits
        ],
        ensure_ascii=False,
    )


def get_lecture_summary(lecture: int) -> str:
    """指定した回（1始まりの整数）の講義の要点を返す。

    「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。
    """
    path = DATA_DIR / "lectures" / f"{lecture:02d}.md"
    if not path.exists():
        available = sorted(p.stem for p in (DATA_DIR / "lectures").glob("*.md"))
        return f"第{lecture}回の要点はまだありません。用意されている回: {', '.join(available)}"
    return path.read_text(encoding="utf-8")


LEARNING_LOG_DIR = Path(__file__).resolve().parent.parent / "learning_log"


def make_quiz(lecture: int) -> str:
    """指定した回の復習・理解確認をしたいとき、用語集に基づく問題と答えを1問返す。"""
    terms = [entry for entry in load_glossary() if entry["lecture"] == lecture]
    if not terms:
        return f"第{lecture}回の確認問題はまだありません。"
    term = terms[0]
    return json.dumps({"question": f"{term['term']}とは何ですか？",
                       "answer": term["definition"], "lecture": lecture}, ensure_ascii=False)


def save_learning_log(
    lecture: Annotated[int, Field(ge=1)],
    content: Annotated[str, Field(min_length=1, max_length=10000)],
) -> str:
    """利用者が学習内容を記録したいとき、確認を受けてから learning_log/ に保存する。

    lecture は1以上の講義番号、content は保存する学習内容そのもの。
    ファイル名は自動生成し、既存の記録を上書きしない。
    エージェントでは必ず requires_approval=True で登録する。
    """
    if isinstance(lecture, bool) or not isinstance(lecture, int) or lecture < 1:
        raise ValueError("lecture は1以上の整数にしてください。")
    if not isinstance(content, str) or not content.strip() or len(content) > 10000:
        raise ValueError("content は空白だけでない10000文字以内の文字列にしてください。")
    # モデルにはパスを渡させない。保存先自体のシンボリックリンクも拒否する。
    if LEARNING_LOG_DIR.is_symlink():
        raise ValueError("learning_log はシンボリックリンクにできません。")
    LEARNING_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    path = LEARNING_LOG_DIR / f"lecture-{lecture}-{uuid4().hex}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"lecture": lecture, "content": content, "created_at": timestamp},
                  stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return f"学習記録を保存しました: learning_log/{path.name}"
