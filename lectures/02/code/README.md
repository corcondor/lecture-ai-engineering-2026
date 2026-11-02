# 実行と再検証

[Kaggle用ノートブック](aie2026_day2_kaggle.ipynb)を非公開で取り込み、T4とインターネット接続を選んで実行しました。[実行スクリプト](kaggle_runner.py)が20エポックと10エポックを別プロセスで実行します。変えるCONFIGの項目は `num_train_epochs` だけです。配布モデルの固定版、FAQ、評価用質問、バッチ4、勾配累積2、学習率、LoRAの行列の次元、生成条件は共通です。

出力フォルダをKaggle用に変更し、割り当てられたGPUの先頭1台を使いました。CPUスレッド数は両設定とも2です。モデルの学習方式とデータは [公式ノートブック](../../../notebooks/lora_format_tuning.ipynb) の処理を使っています。

## 保存済み回答の再採点

リポジトリのルートから、Python標準ライブラリだけで実行できます。モデルの取得、推論、学習、認証操作は行いません。

```bash
python lectures/02/code/verify_actual.py lectures/02/results
```

再採点は公式ノートブックのGitハッシュを確認し、架空FAQと採点関数を読み取ります。保存済み84回答と知識質問16出力、設定差分、評価データの一致、最終lossと学習ログの一致を確認します。既定ではレポートを標準出力へ表示し、ファイルは変更しません。JSONを保存するときは `--report` に新しい出力先を指定してください。

## 学習の実行環境

Tesla T4、Python 3.13.15、PyTorch 2.11.0+cu128、transformers 5.18.0、peft 0.21.1、trl 1.14.2、accelerate 1.15.0、datasets 5.1.0を使用しました。実測の環境は [20エポック側](../results/epochs-20/runtime.json) と [10エポック側](../results/epochs-10/runtime.json) にあります。

学習の所要時間は `trainer.train()` の呼び出し全体です。学習中のチェックポイント保存を含み、呼び出し後のadapter保存、モデル取得、評価は含みません。最終lossは最後にログへ記録された学習区間の平均です。
