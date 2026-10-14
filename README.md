# B3：MNISTで画像分類の流れをつかむ

## キャッサバの病害分類サンプル

[Cassava_ResNet18_sample.ipynb](Cassava_ResNet18_sample.ipynb) を追加しました。
TorchVisionの事前学習済みResNet-18を使う、独立した最小サンプルです。
CSVと画像の読み込み、層化した学習／検証／テスト分割、学習曲線、クラス別指標、混同行列、誤分類画像、重み保存、1枚推論を含みます。

- データは [Kaggle Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data) の形式です。
- `data/cassava/` に `train.csv`、`label_num_to_disease_map.json`、`train_images/` を配置するか、Notebook冒頭の `DATA_DIR` を変更します。
- PyTorch・TorchVisionの準備後、`python -m pip install -r requirements-cassava.txt` で依存ライブラリを入れます。
- 初期設定は各クラス最大200枚、224×224、5 epoch、batch size 32、Adam、学習率0.0001。全画像を使う場合は `MAX_PER_CLASS = None` にします。
- 結果は `outputs/cassava_resnet18/<実行時刻>/` に保存します。実データは同梱しません。

合成画像50枚・1 epoch・CPUで全コードセルを順に実行し、学習、分析図、予測CSV、重み保存と再読み込みまで動作確認しました。キャッサバ実画像での学習・精度評価は未実施です。配布用Notebookは出力を含まない状態です。

以下は従来のMNIST教材の説明です。

教材は [MNIST_CNN_tutorial.ipynb](MNIST_CNN_tutorial.ipynb) です。Pythonが初めてで、機械学習の理論は学んだことのある学生を想定しています。課題や提出物はありません。

Notebookを上から順に実行すると、画像の確認 → 前処理と分割 → モデル定義 → 学習と検証 → テスト → 誤分類の確認 → 保存と推論を一巡できます。重要な文法や関数は初出の場所で、処理の意味はまとまりごとに説明しています。背景知識、検索語、公式資料へのリンクも本文に含めています。

## モデルと標準設定

- Notebook・全文版・実用版では、ResNet-10Tを `BasicBlock` と `ResNet10T` のクラスとして各ファイルにすべて記述します。畳み込み・残差接続・ダウンサンプリング・全結合層・初期化・順伝播を省略せず、timmや他の自作ファイルをimportしません。
- `MNIST_CNN_short.py` だけは `timm.create_model("resnet10t", pretrained=False, in_chans=1, num_classes=10)` を使います。明示版はtimm v1.0.24の標準構成を参照しており、短縮版もこのバージョンを指定します。
- ResNet-10Tは4段階にBasicBlockを1つずつ配置した浅い構成です。ResNet-18の各段階2ブロックより少ない構成を選びます。
- 元画像28×28を32×32にし、白黒1チャネルのまま使います。0〜1へ変換後、`(値 - 0.5) / 0.5` で正規化します。
- 学習1,000枚・検証200枚を公式学習データから分割し、テストは公式テストデータの先頭200枚を使います。初回ダウンロードではMNIST全体を取得します。
- バッチサイズ32、10エポック、Adam、学習率0.001。検証損失が最小の重みを採用します。
- 事前学習済み重みは使いません。精度の追求ではなく、深層学習の処理を体験するための設定です。NVIDIA GPUが利用できればGPU、それ以外はCPUで動きます。

参考：[timm ResNet-10T](https://huggingface.co/timm/resnet10t.c3_in1k)、[timmのモデル作成API](https://huggingface.co/docs/timm/quickstart)。

## 動作確認と以前の生成物

今回の明示版・短縮版（10エポック）の動作確認は未実施です。両実装の出力や重みの互換性、実行時間、精度も未検証です。

`outputs/` に残っているMiniAlexNet・AlexNet版の実行結果付きNotebook・HTML・図・重み・記録は旧版の生成物です。今回の重みと記録は `resnet10t_mnist.pt` と `resnet10t_run_summary.json` に保存します。

配布用Notebookは未実行の状態です。`outputs/` は生成物としてGitの管理対象から除外しています。

## 実行環境の準備

既にPyTorch・TorchVision・MatplotlibとJupyterが使える場合は、その環境でNotebookを開いてください。研究用の既存環境を更新する必要はありません。新規に準備する場合は専用の仮想環境を作ります。

Linuxの例（このREADMEのあるフォルダで実行）：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windowsの場合は `py -m venv .venv` で作成し、コマンドプロンプトで `.venv\Scripts\activate.bat` を実行します。以降のPythonコマンドは共通です。

次に [PyTorch公式のインストール選択画面](https://pytorch.org/get-started/locally/) で、OS・Pip・Python・GPUドライバに対応するCUDAビルドを選び、表示されるインストールコマンドを実行してください。PyTorchとTorchVisionは対応する組み合わせで入れます。GPUに合うビルドを選ぶ必要があるため、この2つは `requirements.txt` の自動インストール対象から外しています。Pythonの対応バージョンも同ページで確認してください。
インストールするPytorchのバージョンは以下のコマンドで確認

```bash
nvcc -V
```

その後、表示・Notebook用のライブラリを入れて起動します。Notebook・全文版・実用版にはtimmは不要です。

```bash
python -m pip install -r requirements.txt
```

短縮版を使う場合だけ、追加で次を実行します。

```bash
python -m pip install -r requirements-short.txt
```


## 使い方と生成物

Pythonスクリプトとして実行する場合は、次の3ファイルを使えます。

- [MNIST_CNN_full.py](MNIST_CNN_full.py)：Notebookの全20コードセルを元の順序・内容のまま収録した全文版です。モデルのクラス定義も含み、timmは使いません。画像やグラフの表示も含みます。図の表示で処理が止まる環境では、図のウィンドウを閉じると次へ進みます。
- [MNIST_CNN_practical.py](MNIST_CNN_practical.py)：教材用の確認出力、画像・グラフ表示、構造確認用の仮実行を省いた実用版です。モデルのクラス定義はすべて残し、timmは使いません。モデルの学習・検証・テスト・保存・再読み込みによる1枚推論を残しています。モデル構造、前処理、データ分割、学習条件、検証損失による重みの選択方法、保存形式は全文版と同じです。
- [MNIST_CNN_short.py](MNIST_CNN_short.py)：timmからResNet-10Tを作る短縮版です。学習と評価を `run_epoch()` にまとめ、他の自作Pythonファイルに依存せず実行できます。モデルの層構成・前処理・分割・学習条件を明示版に合わせています。記録には実装名とtimmのバージョンを残します。両実装の実行比較は未実施です。短縮版の保存先は常にスクリプトの隣の `outputs/`、データは隣の `data/` です。

`B3CNNpart` を作業フォルダにして、いずれかを実行します。

```bash
python MNIST_CNN_full.py
# または
python MNIST_CNN_practical.py
# または
python MNIST_CNN_short.py
```

3ファイルともモデルを最初から学習し、同じ `outputs/resnet10t_mnist.pt` と `outputs/resnet10t_run_summary.json` を上書きします。実用版・短縮版は推論専用ではありません。最後の1枚推論はNotebookと同じMNISTテストデータの先頭画像を使います。

1. 最上部から説明を読み、コードセルを **Shift(Ctrl) + Enter** で順に実行します。
2. 初回のみMNISTのダウンロードにインターネット接続が必要です。
3. 学習セルに各エポックの損失・正解率と所要時間が出ます。
4. 再開位置が分からなくなったら、カーネルを再起動して先頭から実行します。

生成物は `B3CNNpart` 内に保存します。通常はこのフォルダを作業フォルダとして起動してください。

```text
B3CNNpart/
├── MNIST_CNN_tutorial.ipynb    教材本体（未実行の状態で配布）
├── README.md                 環境準備と使い方
├── requirements.txt          表示・Notebook用の依存ライブラリ
├── requirements-short.txt    短縮版用（timmを追加）
├── data/MNIST/               初回実行時に作成
└── outputs/                  実行結果
    ├── resnet10t_mnist.pt          検証損失が最小の重み
    └── resnet10t_run_summary.json  条件・分割・学習曲線・テスト結果
```

学習セルの再実行はモデルを初期化して最初から学習します。保存セルの再実行は上記2ファイルを上書きします。別の実験結果を残す場合は、先にファイル名や保存先を変えてください。ダウンロードしたMNISTは再利用します。

## 教材の位置づけ

ResNetの残差接続と、画像分類の読み込み・学習・評価・保存・推論を学ぶ教材です。白黒1チャネル・10クラスのResNet-10Tを明示的に実装し、短縮版だけtimmで同じ構成を作ります。少ない画像とエポックで流れを体験することが目的で、モデル間の性能比較や原論文の再現実験ではありません。

参考：[ResNet原論文](https://arxiv.org/abs/1512.03385)、[timm ResNet-10T](https://huggingface.co/timm/resnet10t.c3_in1k)、[PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)。
