# B3：キャッサバの病害画像分類

教材は [Cassava_ResNet18_sample.ipynb](Cassava_ResNet18_sample.ipynb) です。画像の確認、データ分割、学習、評価、誤分類の分析、保存した重みでの分類を体験します。

モデルは **timmのResNet-18** を使います。事前学習なし・少量データ・短い学習から始める改善練習用のサンプルで、高精度を目指す初期設定ではありません。

## 環境の準備

既存の環境を使うか、専用の仮想環境を作成してください。Linuxの場合：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windowsでは `py -m venv .venv` で作成し、コマンドプロンプトで `.venv\Scripts\activate.bat` を実行します。

[PyTorch公式の手順](https://pytorch.org/get-started/locally/)に従い、環境に対応するPyTorchとTorchVisionを入れてください。その後、このREADMEのあるフォルダで実行します。

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

GPUが使えない場合はCPUで動きます。モデルにはtimm、画像の前処理にはTorchVisionを使います。
このワークスペースのDocker環境を使う場合は、[VS Code Notebook用の手順](docker/README.md)を参照してください。

## Dockerで実行する

PyTorch・TorchVision・timmを含む環境は `docker/` にあります。

```bash
cd docker
bash start.sh        # NVIDIA GPUを使用
# CPUのみの場合: bash start.sh --cpu
```

VS Codeの **Dev Containers: Attach to Running Container...** で
`b3cnnpart-notebook-1` に接続し、`/workspace/B3CNNpart` を開きます。
ノートブックのカーネルは **Python (B3CNNpart)** または `/opt/conda/bin/python` を選びます。
既存コンテナを使っている場合も、上のコマンドでtimmを含む環境に更新できます。
再作成時には実行中のカーネルが終了するため、先にノートブックを保存してください。

VS Codeでこの教材フォルダを開き、**Dev Containers: Reopen in Container** から起動することもできます（GPU構成）。
データと出力はホスト側の同じフォルダに残ります。ブラウザ用のポート公開はありません。
詳しくは[Dockerの手順](docker/README.md)を参照してください。

## データの配置

[Kaggle Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data)のデータをノートブックの **1-1節** からダウンロードできます。保存先は `data/cassava/` で、配置済みならスキップします。手動で取得・展開しても構いません。データは同梱しません。

初回はコンペページで規約に同意し、認証が必要ならセルの `KAGGLE_LOGIN = True` を設定してください。[Kaggle設定ページ](https://www.kaggle.com/settings/api)で生成したAPIトークンを入力欄に入れます。学習で使用する250枚だけでなくコンペ全体を取得します。詳しくはノートブックの説明と[kagglehub公式手順](https://github.com/Kaggle/kagglehub)を参照してください。

```text
B3CNNpart/
├── Cassava_ResNet18_sample.ipynb
├── README.md
├── requirements.txt
└── data/cassava/
    ├── train.csv
    ├── label_num_to_disease_map.json
    └── train_images/
        ├── 1000015157.jpg
        └── ...
```

`train.csv` の `image_id`・`label` と、JSONに記録された0〜4の病名対応を使います。別の場所に置く場合はノートブックの `DATA_DIR` を変更してください。

## 学習と評価

ノートブックを開き、上から順に **Shift + Enter** で実行します。

| 設定 | 初期値 |
| --- | --- |
| モデル | `MODEL_NAME = "resnet18"`、事前学習なし |
| データ | 各クラス最大50枚、合計250枚 |
| 分割 | 学習60%・検証20%・テスト20%（250枚なら150・50・50枚） |
| 前処理 | RGB化・64×64へリサイズ・0〜1のTensor化 |
| 学習 | 2 epoch、batch size 32、Adam、学習率0.0001 |
| 使用する重み | 最後のepoch |

クラス比率を保って分割します。Kaggleのラベルなしテスト画像ではなく、ラベル付き画像の一部を評価用に確保します。画像拡張・平均と標準偏差による正規化は行いません。

学習曲線、Accuracy、macro F1、クラス別指標、混同行列、誤分類画像を確認できます。設定の調整には検証データを使い、テスト結果を見ながら繰り返し調整しないでください。

`MODEL_NAME`を変更すると別のモデルを試せます。候補は `timm.list_models("resnet*")` などで確認できます。モデルごとの入力サイズや、事前学習を導入する場合の前処理は[timm公式ガイド](https://huggingface.co/docs/timm/quickstart)を確認してください。

## 保存した重みで50枚を分類する（9節）

9節は、学習済み重みを使う推論の演習です。

1. **9-1**で重み・前処理・クラス名を保存し、テスト画像を最大50枚 `inference_images/` へコピーします。初期設定では各クラス10枚です。正解は別のCSVに保存します。
2. **9-2**の `RUN_DIR` に、9-1で表示された保存先を設定します。
3. 9-2を実行すると、新しいモデルに保存済み重みを読み込み、画像フォルダを分類します。画像と予測病名を10枚ずつ表示し、最後に正解と照合します。

9-2はカーネル再起動後にも単独で実行でき、再学習は不要です。`INFERENCE_DIR`を変えると別の画像フォルダにも使えます。分類には正解ラベルを使いません。スコアはsoftmaxの値で、正しさを保証するものではありません。

この50枚は7節のテスト画像と同じであり、独立した追加評価ではありません。テストが50枚未満なら全枚、50枚を超える場合はクラス比率を保って50枚を選びます。

## 保存先

結果は `outputs/cassava_<モデル名>/<実行時刻>/` に保存します。

| ファイル・フォルダ | 内容 |
| --- | --- |
| `splits.csv` | 学習・検証・テストの画像一覧 |
| `history.csv` / `learning_curves.png` | 学習経過 |
| `classification_report.csv` / `confusion_matrix.png` | テスト評価 |
| `test_predictions.csv` | テスト画像ごとの予測 |
| `misclassified.png` | 誤分類例（誤分類がある場合） |
| `cassava_model.pth` / `summary.json` | 重みと実験条件 |
| `inference_images/` / `inference_answers.csv` | 分類演習用画像と正解 |
| `inference_predictions.csv` | 保存した重みからの予測 |
| `inference_gallery_*.png` | 予測病名付きの画像一覧 |
| `inference_checked.csv` | 正解と照合した結果 |

学習セルの再実行は最初からの学習です。同じ実行フォルダ内では生成物を上書きします。配布用ノートブックには実行結果を含めません。

## 改善の練習

epoch数、画像サイズ、学習時の画像拡張、事前学習、モデルの種類などを一つずつ変更し、検証結果を比べてください。`MAX_PER_CLASS = None` で全画像を使えますが、分割をやり直すと評価画像も変わります。比較する際は評価画像を固定してください。
