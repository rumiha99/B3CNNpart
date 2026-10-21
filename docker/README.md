# B3CNNpart：VS Code Notebook用Docker環境

`Cassava_ResNet18_sample.ipynb`をVS Code内で実行する環境です。
Dockerの設定はこのフォルダ、VS Codeの入口は
`../.devcontainer/devcontainer.json` にあります。
コンテナ内のPythonをNotebookカーネルとして使います。ポートの公開やJupyterLabサーバーの起動は行いません。

## VS Codeから使う（GPU）

1. ホストのVS Codeに **Dev Containers** (`ms-vscode-remote.remote-containers`) 拡張を入れます。
2. Gitから取得した `B3CNNpart` フォルダ（例：`/home/B3CNNpart`）をVS Codeで開きます。
3. コマンドパレットから **Dev Containers: Reopen in Container** を実行します。
4. `Cassava_ResNet18_sample.ipynb` を開きます。
5. 右上の **カーネルの選択** で **Python (B3CNNpart)** を選びます。
   表示されない場合は **Python Environments** から `/opt/conda/bin/python` を選びます。
6. データを配置し、上から順にセルを実行します。

PythonとJupyterのVS Code拡張はコンテナ側に自動インストールされます。
別PCからこのマシンを使う場合は、先にVS CodeのRemote - SSHで接続してから同じ手順を行います。

ノートブックの1-1節からデータを取得できます。Kaggle認証とコンペ規約への同意はノートブックの案内に従ってください。手動の場合は、ホスト側に次のように配置します。

```text
<B3CNNpart>/data/cassava/
├── train.csv
├── label_num_to_disease_map.json
└── train_images/
    └── *.jpg
```

コンテナ内では `/workspace/B3CNNpart` に同じファイルが見えます。
Notebookの初期設定の `DATA_DIR` をそのまま使えます。
Notebookの編集内容や `outputs/` の結果はホストに保存されます。実データはイメージに含めません。

## ターミナルから起動して接続する

```bash
cd /home/B3CNNpart/docker  # リポジトリを置いた場所に合わせて変更
bash start.sh
```

VS Codeの **Dev Containers: Attach to Running Container...** で
`b3cnnpart-notebook-1` に接続し、`/workspace/B3CNNpart` を開きます。
Attach方式では、コンテナ側にPython・Jupyter拡張がなければインストールしてください。

CPUだけで使う場合は次のコマンドで起動し、同じAttach方式で接続します。
GPU用の `Reopen in Container` 設定は使いません。

```bash
bash start.sh --cpu
```

停止する場合：

```bash
docker compose down
```

## 環境と確認方法

- PyTorch 2.5.1 / TorchVision 0.20.1 / CUDA 12.4
- kagglehub 1.0.2（Notebookからデータ取得）
- timm 1.0.30（モデル作成）、NumPy、pandas、scikit-learn、Matplotlib、Pillow
- ipykernel、nbformat、nbclient（ブラウザ用JupyterLabは追加しません）
- GPU利用時はホストのNVIDIAドライバーとNVIDIA Container Toolkitが必要です。
- 通常のNotebook作業は非rootの `student` ユーザーで行います。
- UID/GIDの既定値はこのホストに合わせて1001です。別環境では `bash start.sh` が実行ユーザーの値を渡します。
  VS CodeのReopen方式は `updateRemoteUserUID` でホストに合わせます。

起動中のコンテナでGPUを確認：

```bash
docker compose exec notebook python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

実データなしで全セルの動作を確認：

```bash
docker compose exec notebook python /opt/b3cnnpart/smoke_test.py
# CPUで確認する場合
docker compose exec notebook python /opt/b3cnnpart/smoke_test.py --cpu
```

合成画像250枚で初期設定の2 epochを学習し、評価・図の出力・重み保存・再読み込みと、独立した名前空間での50枚分類・予測一致を確認します。
合成データと結果は一時ディレクトリに作って終了時に削除するため、教材や実データを変更しません。
この確認はキャッサバ実画像での精度評価ではありません。

2026-10-21に、timm 1.0.30を含むイメージでRTX 4090とGPUを割り当てないCPU構成の両方を検証しました。合成画像250枚による全セル実行、重みの再読み込み、50枚の予測一致が成功し、`pip check`も通過しています。VS Codeの画面操作は未確認です。

Dockerfileやrequirements.txtを変更したら **Dev Containers: Rebuild Container**、
または `bash start.sh` で再ビルドします。

参考：
[VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/create-dev-container)、
[PyTorchのバージョン対応](https://pytorch.org/get-started/previous-versions/)、
[Docker ComposeのGPU設定](https://docs.docker.com/compose/how-tos/gpu-support/)。
