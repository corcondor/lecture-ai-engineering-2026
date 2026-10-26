# 第1回：実行ガイド

リポジトリのルートから実行する。実APIはまだ呼んでいないため、`answer.md` は実測前の状態。**下の実API用コマンドは、キーの安全な設定と無料枠の確認が終わるまで実行しない。**

## 1. APIを呼ばない確認

Python 3.12と講座の固定依存関係を使う。依存の更新は不要。

```bash
uv sync --frozen
uv run pytest -q
uv run python -m app.ask --help
uv run python -m app.experiment --help
uv run python -m app.experiment --dry-run
```

`--dry-run` は実験計画の確認だけで、APIキーもAPI呼び出しも不要。fixture/mockテストは実装の検証であり、回答品質・実際の応答時間・消費トークンを測る実験ではない。

## 2. 実APIを使う前に

1. 利用者の個人Googleアカウントで、対象プロジェクトとモデルの無料枠・残り枠を確認する。
2. リポジトリ指定の `gemini-3.5-flash-lite` と `gemini-3.1-flash-lite` が、そのアカウントで使用可能か確認する。使用不可なら勝手に有料モデルへ切り替えない。
3. `GEMINI_API_KEY` はCodespaces secretなどの安全な経路で設定する。キーをコマンド例、コード、チャット、コミットに書かない。今回キーの新規発行や登録は行っていない。
4. [公式のレート制限](https://ai.google.dev/gemini-api/docs/rate-limits) とAI Studioの現行上限を確認する。待機秒数や `--confirm-free-tier` の指定だけでは、無料であることをプログラムが保証できない。

## 3. 演習：単発質問とトークン表示

```bash
uv run python -m app.ask "LoRAとは何ですか？"
uv run python -m app.ask 'AIE2026の講座の宿題にある「3レベル」とは何ですか？'
uv run python -m app.ask 'AIE2026の講座で使う「lecNN-start タグ」とは何ですか？'
uv run python -m app.ask 'AIE2026の第2回では何を学びますか？'
```

表示された回答と入力・出力トークン数を記録し、`data/glossary.json` や講座資料と比較する。トークン数が「不明」のときに0を記入しない。

設定を変える例：

```bash
uv run python -m app.ask "LoRAとは何ですか？" \
  --model gemini-3.5-flash-lite --temperature 0.2 --max-output-tokens 512
AIE_MODEL=gemini-3.1-flash-lite uv run python -m app.ask "LoRAとは何ですか？"
```

## 4. 実装課題：会話履歴

```bash
uv run python -m app.ask --chat --temperature 0.2 --max-output-tokens 512
```

「LoRAとは何ですか？」、続いて「今の説明を小学生向けに言い換えて」と入力する。同じ会話の履歴に沿って答えられるか確認する。`/reset` で履歴を消去し、`/exit` またはCtrl-Dで終了する。新しいプロセスに履歴は引き継がれない。APIエラー・空応答の質問は履歴に追加しない。

## 5. 基礎・発展課題：24試行の比較

2モデル × 2温度（0.2、1.0）× 2問 × 3反復で24試行。会話履歴を毎回リセットした単発呼び出しで比較する。`AIE_MODEL` に影響されないよう、モデルを明示する。

```bash
# 計画だけを保存する。APIは呼ばない。
uv run python -m app.experiment --dry-run \
  --models gemini-3.5-flash-lite gemini-3.1-flash-lite \
  --temperatures 0.2 1.0 --repeats 3 \
  --questions-file experiments/lesson01_questions.json \
  --max-output-tokens 1024 --interval 15 --max-requests 24 --seed 42
```

計画を確認し、無料枠・残り枠・モデル利用可否も確認できたら、上の **`--dry-run` を `--confirm-free-tier` に置き換えて** 実行する。フラグは確認済みの宣言であり、課金を防ぐ機構ではない。

- 質問ファイルはUTF-8のJSON文字列配列。上の [サンプル](../../experiments/lesson01_questions.json) を使用できる。
- `--seed` はシャッフルした試行順の固定だけに使い、同じ回答の再現を保証しない。
- `--interval` は応答後から次の試行までの待機秒数。既定15秒。残りのRPM・TPM・RPDを見て決め、どの値でも制限回避や無料を保証しない。24試行の待機だけで計5分45秒、別にAPI応答時間が必要。
- `--max-requests` は計画の上限（既定24）。超える計画は実行を拒否する。支出上限ではない。
- 保存先は既定で `results/lesson01_<UTC日時>/`。`--output-dir` で変更できるが、既存のディレクトリは上書きしない。
- `plan.json` は設定、`results.jsonl` は実行時の生データ、`results.csv` は一覧、`summary.json` と `report.md` は集計。dry-runでは生データは作らず、観測0件の計画・集計を作る。回答の違いは全文から読み取り、数値だけで品質を決めない。
- タイムアウトは120秒。SDKの自動再試行は無効で、最初のAPI・通信エラーで記録して停止する。空の本文・未完了の回答は成功集計から除く。
- 429・503・途中終了があれば、成功・失敗・未実施の数を分ける。原因を確認せず全24回を何度も実行しない。
- 中断・通信失敗時には、API側で処理済みでも結果が保存されない場合がある。記録件数と利用枠の消費数が同じとは限らず、再開前に残り枠を確認する。
- 追加の単発質問と会話もAPI枠を使う。24回以外の試行を混ぜて集計しない。
- 実験出力には質問・回答が残るので、個人情報や秘密情報を入力しない。

## 6. 提出前

1. `lectures/01/answer.md` の未取得欄を、実際の回答・計測値・出力ファイルへの参照で更新する。予想を観測に置き換えない。
2. `uv run pytest -q` を再実行する。実API実験とは別の検証として記録する。
3. fork先と変更内容を確認し、利用者が提出操作を承認してから作業ブランチをpushし、**自分のforkのmain** 宛てにPRを作る。講座本体へのPRは作らない。
4. Codespacesを使った場合は、終了後に停止する。

アカウント設定、実測、PR提出が残っている間は「第1回をすべて完了」としない。
