"""構造を明示したResNet-10Tで学習・評価・保存・推論を行う実用版。

実行: cd B3CNNpart && python MNIST_CNN_practical.py
学習1,000枚・検証200枚・テスト200枚・10エポック。事前学習なし。
outputs/resnet10t_mnist.pt と outputs/resnet10t_run_summary.json を上書きする。
"""

from pathlib import Path
from copy import deepcopy
import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
from torchvision import datasets, transforms


class BasicBlock(nn.Module):
    """2つの畳み込みと残差接続を持つブロック。"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

        # 足し合わせる2つの経路の縦横・チャネル数をそろえる。
        if stride != 1 or in_channels != out_channels:
            pool = nn.AvgPool2d(
                kernel_size=2, stride=stride, ceil_mode=True, count_include_pad=False,
            ) if stride != 1 else nn.Identity()
            self.downsample = nn.Sequential(
                pool,
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x = x + shortcut
        x = self.act2(x)
        return x


class ResNet10T(nn.Module):
    """ResNet-10Tの構造をPyTorchの層で明示した教材用モデル。

    白黒入力・10クラス分類。短縮版のresnet10tと同じ層構成を使う。
    構成の参照: timm v1.0.24 の resnet10t / deep_tiered / avg_down。
    https://github.com/huggingface/pytorch-image-models/blob/v1.0.24/timm/models/resnet.py
    """

    def __init__(self, in_chans=1, num_classes=10):
        super().__init__()
        # 入力部: チャネル数を 1 → 24 → 32 → 64 と増やす。
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4つの段階に残差ブロックを1つずつ配置する。
        self.layer1 = nn.Sequential(BasicBlock(64, 64, stride=1))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # 畳み込みはKaiming初期化、BatchNormは重み1・バイアス0。
        # 全結合層にはnn.Linearの標準初期化を使う。
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        # 各残差ブロックの最後のBNだけ重みを0にして学習を始める。
        for layer in self.modules():
            if isinstance(layer, BasicBlock):
                nn.init.zeros_(layer.bn2.weight)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            predictions = logits.argmax(dim=1)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def main():
    SEED = 42
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PROJECT_DIR = Path.cwd()
    if PROJECT_DIR.name != "B3CNNpart" and (PROJECT_DIR / "B3CNNpart").is_dir():
        PROJECT_DIR = PROJECT_DIR / "B3CNNpart"

    DATA_DIR = PROJECT_DIR / "data"
    OUTPUT_DIR = PROJECT_DIR / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])

    train_source = datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    test_source = datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )

    TRAIN_SIZE = 1000
    VAL_SIZE = 200
    TEST_SIZE = 200
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001

    test_dataset = Subset(test_source, range(TEST_SIZE))

    unused_size = len(train_source) - TRAIN_SIZE - VAL_SIZE
    split_generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset, unused_dataset = random_split(
        train_source,
        [TRAIN_SIZE, VAL_SIZE, unused_size],
        generator=split_generator,
    )

    train_generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    torch.manual_seed(SEED)
    train_generator.manual_seed(SEED)
    model = ResNet10T(in_chans=1, num_classes=10).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch

        elapsed = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.1%} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.1%} | "
            f"{elapsed:.1f} sec"
        )

    training_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    model.eval()
    print(f"学習・検証の合計時間: {training_seconds:.1f}秒")
    print(f"採用した重み: epoch {best_epoch}（検証損失が最小）")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"採用エポック: {best_epoch}")
    print(f"テスト損失: {test_loss:.4f}")
    print(f"テスト正解率: {test_acc:.2%}")

    model_path = OUTPUT_DIR / "resnet10t_mnist.pt"
    summary_path = OUTPUT_DIR / "resnet10t_run_summary.json"

    torch.save(model.state_dict(), model_path)

    summary = {
        "model": "ResNet10T", "implementation": "explicit_pytorch",
        "pretrained": False,
        "input_size": [1, 32, 32],
        "seed": SEED,
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "test_size": len(test_dataset),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "normalization": {"mean": [0.5], "std": [0.5]},
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "training_seconds": training_seconds,
        "parameter_count": parameter_count,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "device": str(device),
        "history": history,
        "train_indices": train_dataset.indices,
        "val_indices": val_dataset.indices,
        "test_indices": list(test_dataset.indices),
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("重み:", model_path)
    print("条件と結果:", summary_path)

    loaded_model = ResNet10T(in_chans=1, num_classes=10).to(device)
    saved_weights = torch.load(model_path, map_location=device, weights_only=True)
    loaded_model.load_state_dict(saved_weights)
    loaded_model.eval()

    image, true_label = test_dataset[0]
    input_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = loaded_model(input_batch)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()

    print("正解:", true_label)
    print("予測:", predicted_label)


if __name__ == "__main__":
    main()
