"""第1回 基礎・発展: 独立した質問を3回ずつ比較し、実測値を保存する。

まず計画だけ確認: uv run python -m app.experiment --dry-run
無料枠とモデルを確認した後: uv run python -m app.experiment --confirm-free-tier
確認フラグは課金防止機構ではなく、実行者による確認の宣言。
"""

import argparse
import csv
import importlib.metadata
import itertools
import json
import math
import platform
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from google.genai import errors

from app.ask import Conversation, DEFAULT_QUESTION, make_client, positive_int, temperature_value
from app.config import MODEL, api_error_message

DEFAULT_QUESTIONS = ["LoRAとは何ですか？", DEFAULT_QUESTION]
TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens", "thought_tokens", "cached_tokens")
CSV_FIELDS = [
    "trial_id", "model", "temperature", "question", "repeat", "started_at", "status",
    "elapsed_seconds", *TOKEN_FIELDS, "finish_reason", "model_version", "block_reason",
    "error_code", "error_message", "text",
]


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("0以上の有限数にしてください")
    return number


def build_plan(models, temperatures, questions, repeats=3, seed=42):
    """実行順は再現可能にシャッフル。LLM出力の再現性を保証する seed ではない。"""
    trials = [
        {"model": model, "temperature": temperature, "question": question, "repeat": repeat}
        for model, temperature, question, repeat in itertools.product(
            models, temperatures, questions, range(1, repeats + 1)
        )
    ]
    random.Random(seed).shuffle(trials)
    return [{"trial_id": i, **trial} for i, trial in enumerate(trials, 1)]


def run_trial(client, trial, max_output_tokens):
    row = {**trial, "started_at": datetime.now(timezone.utc).isoformat()}
    start = time.perf_counter()
    try:
        # 毎回新しい履歴を使い、比較実験に前の回答が混入しないようにする。
        result = Conversation(
            client, model=trial["model"], temperature=trial["temperature"],
            max_output_tokens=max_output_tokens,
        ).send(trial["question"])
        row.update(asdict(result))
        row.update(row.pop("usage"))
        if not result.text:
            row["status"] = "empty"
        elif result.block_reason or result.finish_reason != "STOP":
            row["status"] = "incomplete"
        else:
            row["status"] = "ok"
    except errors.APIError as exc:
        row.update(status="api_error", error_code=exc.code,
                   error_message=api_error_message(exc.code),
                   elapsed_seconds=time.perf_counter() - start)
    except httpx.HTTPError:
        row.update(status="network_error", error_message="通信エラー（詳細は保存しない）",
                   elapsed_seconds=time.perf_counter() - start)
    return row


def summarize(rows, plan):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["temperature"], row["question"])].append(row)
    planned = defaultdict(int)
    for trial in plan:
        planned[(trial["model"], trial["temperature"], trial["question"])] += 1
    summary = []
    for key, count in planned.items():
        attempted = groups[key]
        success = [r for r in attempted if r["status"] == "ok"]
        latencies = [r["elapsed_seconds"] for r in success]
        entry = dict(zip(("model", "temperature", "question"), key))
        entry.update(planned=count, attempted=len(attempted), successful=len(success),
                     incomplete=len(attempted) - len(success),
                     comparison_complete=len(success) == count,
                     distinct_answers=len({r["text"] for r in success}))
        for metric, fn in (("mean_seconds", statistics.mean), ("median_seconds", statistics.median),
                           ("min_seconds", min), ("max_seconds", max)):
            entry[metric] = fn(latencies) if latencies else None
        for field in TOKEN_FIELDS:
            values = [r[field] for r in success if r.get(field) is not None]
            entry[f"mean_{field}"] = statistics.mean(values) if values else None
            entry[f"known_{field}"] = len(values)
        summary.append(entry)
    return summary


def csv_safe(value):
    # LLMの出力をスプレッドシートで開いても式として実行されないようにする。
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def write_reports(directory, rows, plan, run_status):
    summary = summarize(rows, plan)
    (directory / "summary.json").write_text(json.dumps({
        "run_status": run_status, "planned": len(plan), "attempted": len(rows),
        "groups": summary,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (directory / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows({field: csv_safe(row.get(field)) for field in CSV_FIELDS} for row in rows)
    lines = ["# 第1回 比較実験レポート", "", f"実行状態: {run_status}",
             f"計画 {len(plan)} 回 / 記録 {len(rows)} 回", "",
             "未実行・失敗・本文なし・打ち切りを成功として集計しない。欠測トークンは0ではなく不明。",
             "時間は非ストリーミングAPI呼び出し全体。接続・通信・待機・生成を含み、TTFTではない。",
             "SDK再試行は無効。リクエスト間の待機時間は測定に含まない。",
             "中断・通信失敗時はAPI側で処理済みでも利用量が不明なため、記録件数は消費枠と一致するとは限らない。",
             "3回程度の小標本で有意な優劣や精度を断定しない。完全一致率は事実性の評価ではない。", ""]
    for group in summary:
        lines += [f"## {group['model']} / temperature={group['temperature']}",
                  f"質問: {group['question']}",
                  f"成功 {group['successful']}/{group['planned']}（試行済み {group['attempted']}）",
                  f"比較完了: {group['comparison_complete']}"]
        for key in ("mean_seconds", "median_seconds", "min_seconds", "max_seconds", "distinct_answers"):
            lines.append(f"- {key}: {group[key] if group[key] is not None else '未測定'}")
        for field in TOKEN_FIELDS:
            value = group[f"mean_{field}"]
            lines.append(f"- 平均 {field}: {value if value is not None else '未測定'}"
                         f"（既知の件数 {group[f'known_{field}']}）")
        lines.append("")
    lines += ["## 回答本文（生データは results.jsonl）", ""]
    for row in rows:
        lines += [f"### 試行 {row['trial_id']} / {row['status']}",
                  f"モデル: {row['model']} / temperature={row['temperature']} / {row['repeat']}回目",
                  f"質問: {row['question']}", ""]
        # インデントで本文をデータとして表示し、Markdown見出し等の混入を防ぐ。
        lines += ["    " + line for line in row.get("text", row.get("error_message", "本文なし")).splitlines()]
        lines.append("")
    lines += ["## 読み取り時の注意", "",
              "- 同じ質問・モデルで temperature の条件間、同じ質問・temperature でモデル間を比較する。",
              "- モデルの回答と data/glossary.json の homework-levels を比較して、具体的な誤りを記録する。",
              "- 思考トークン等が別計上される場合、入力+出力と total_tokens は一致しないことがある。",
              "- RPM・入力TPM・RPDのいずれでも制限され得る。429発生は速度や品質の劣位とは別に扱う。",
              "- プロジェクト単位の上限のため、別のキーでも同じプロジェクトなら制限を共有する。",
              "- 短い待機で日次上限は解消しない。実際の上限・無料枠はAI Studioで確認する。",
              "", "出典: https://ai.google.dev/gemini-api/docs/rate-limits", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_models = list(dict.fromkeys([MODEL, "gemini-3.1-flash-lite"]))
    parser.add_argument("--models", nargs="+", default=default_models)
    parser.add_argument("--temperatures", nargs="+", type=temperature_value, default=[0.2, 1.0])
    parser.add_argument("--questions-file", type=Path)
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--seed", type=int, default=42, help="試行順だけを固定する seed")
    parser.add_argument("--interval", type=nonnegative_float, default=15.0,
                        help="応答後、次の試行まで待つ秒数。上限遵守を保証しない")
    parser.add_argument("--max-output-tokens", type=positive_int, default=1024)
    parser.add_argument("--max-requests", type=positive_int, default=24)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-free-tier", action="store_true",
                        help="選択モデル/プロジェクトの無料枠を実行者が確認済みと宣言する")
    args = parser.parse_args()
    questions = DEFAULT_QUESTIONS
    if args.questions_file:
        try:
            questions = json.loads(args.questions_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            parser.error("質問ファイルはUTF-8のJSON文字列配列にしてください")
    if not isinstance(questions, list) or not questions or any(
        not isinstance(q, str) or not q.strip() for q in questions
    ):
        parser.error("質問は空でない文字列の配列にしてください")
    if any(not model.strip() for model in args.models):
        parser.error("モデル名は空にできません")
    models = list(dict.fromkeys(args.models))
    temperatures = list(dict.fromkeys(args.temperatures))
    questions = list(dict.fromkeys(questions))
    count = len(models) * len(temperatures) * len(questions) * args.repeats
    if count > args.max_requests:
        parser.error(f"計画が {count} 回です。--max-requests {args.max_requests} を超えます")
    plan = build_plan(models, temperatures, questions, args.repeats, args.seed)
    if not args.dry_run and not args.confirm_free_tier:
        parser.error("まず --dry-run で確認し、無料枠確認後に --confirm-free-tier を指定してください")
    directory = args.output_dir or Path("results") / datetime.now(timezone.utc).strftime("lesson01_%Y%m%dT%H%M%S%fZ")
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("出力先が既に存在します。データを上書きしないよう新しいディレクトリを指定してください")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "dry_run": args.dry_run,
        "python": platform.python_version(), "google_genai": importlib.metadata.version("google-genai"),
        "max_output_tokens": args.max_output_tokens, "interval_seconds": args.interval,
        "order_seed": args.seed, "sdk_attempts": 1, "timeout_seconds": 120,
        "free_tier_user_confirmed": args.confirm_free_tier, "trials": plan,
        "note": "無料枠・残り利用枠・モデル利用可否はプログラムでは検証していない",
    }
    (directory / "plan.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"計画: {len(plan)} 回 / モデル {len(models)} / temperature {len(temperatures)} / 質問 {len(questions)}")
    print(f"保存先: {directory}")
    rows = []
    run_status = "dry_run" if args.dry_run else "not_started"
    if args.dry_run:
        write_reports(directory, rows, plan, run_status)
        print("計画だけを保存しました。API呼び出しは0回です。")
        return
    try:
        with make_client() as client, (directory / "results.jsonl").open("x", encoding="utf-8") as stream:
            run_status = "running"
            for index, trial in enumerate(plan):
                if index:
                    time.sleep(args.interval)
                row = run_trial(client, trial, args.max_output_tokens)
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"[{index + 1}/{len(plan)}] {trial['model']} {row['status']}")
                if row["status"] in ("api_error", "network_error"):
                    run_status = row["status"]
                    print("エラーを記録し停止しました。自動再試行はしません。")
                    if row.get("error_code") == 429:
                        print("RPM・TPM・RPDの上限を確認してください。日次上限には短い待機では対応できません。")
                    break
            else:
                run_status = "complete" if all(r["status"] == "ok" for r in rows) else "complete_with_incomplete_responses"
    except KeyboardInterrupt:
        run_status = "interrupted"
        print("中断しました。取得済みの結果は保存します。")
    finally:
        write_reports(directory, rows, plan, run_status)
    if run_status != "complete":
        raise SystemExit(130 if run_status == "interrupted" else 1)


if __name__ == "__main__":
    main()
