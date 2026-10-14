"""timmのResNet-10Tを使う短縮版。自作モジュールのimportは不要。

実行: cd B3CNNpart && python MNIST_CNN_short.py
学習・評価・保存・再読み込みによる推論まで行う。精度より処理の理解を優先。
"""

from copy import deepcopy
from pathlib import Path
import json
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
import torchvision
import timm
from torchvision import datasets, transforms

SEED = 42
TRAIN_SIZE = 1000
VAL_SIZE = 200
TEST_SIZE = 200
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


def run_epoch(model, loader, criterion, device, optimizer=None):
    """optimizerを渡したら学習、省略したら評価する。"""
    model.train(optimizer is not None)
    total_loss = total_correct = total_samples = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parent
    output = root / "outputs"
    output.mkdir(exist_ok=True)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    source = datasets.MNIST(root / "data", train=True, download=True, transform=transform)
    test_source = datasets.MNIST(root / "data", train=False, download=True, transform=transform)
    train, val, _ = random_split(
        source, [TRAIN_SIZE, VAL_SIZE, len(source) - TRAIN_SIZE - VAL_SIZE],
        generator=torch.Generator().manual_seed(SEED),
    )
    test = Subset(test_source, range(TEST_SIZE))
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE)

    # ライブラリからモデルを作成。事前学習なし、数字0〜9の10クラス。
    model = timm.create_model("resnet10t", pretrained=False, in_chans=1, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss, best_state, best_epoch = float("inf"), None, 0
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        for key, value in zip(history, (train_loss, val_loss, train_acc, val_acc)):
            history[key].append(value)
        if val_loss < best_loss:
            best_loss, best_state, best_epoch = val_loss, deepcopy(model.state_dict()), epoch
        print(f"Epoch {epoch}/{EPOCHS} | train loss {train_loss:.4f}, acc {train_acc:.1%} | "
              f"val loss {val_loss:.4f}, acc {val_acc:.1%}")
    training_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
    print(f"採用epoch: {best_epoch} | テスト正解率: {test_acc:.1%} | 学習・検証: {training_seconds:.1f}秒")

    model_path = output / "resnet10t_mnist.pt"
    torch.save(model.state_dict(), model_path)
    summary = {
        "model": "timm.resnet10t", "pretrained": False, "input_size": [1, 32, 32],
        "seed": SEED, "train_size": TRAIN_SIZE, "val_size": VAL_SIZE, "test_size": TEST_SIZE,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
        "normalization": {"mean": [0.5], "std": [0.5]},
        "best_epoch": best_epoch, "test_loss": test_loss, "test_accuracy": test_acc,
        "training_seconds": training_seconds,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "torch_version": str(torch.__version__), "torchvision_version": str(torchvision.__version__), "timm_version": str(timm.__version__),
        "device": str(device), "history": history,
        "train_indices": train.indices, "val_indices": val.indices, "test_indices": list(test.indices),
    }
    (output / "resnet10t_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    loaded_model = timm.create_model("resnet10t", pretrained=False, in_chans=1, num_classes=10).to(device)
    loaded_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    loaded_model.eval()
    image, label = test[0]
    with torch.no_grad():
        probabilities = loaded_model(image.unsqueeze(0).to(device)).softmax(dim=1)
        prediction = probabilities.argmax(dim=1).item()
    print(f"正解: {label} | 予測: {prediction} | 保存先: {model_path}")


if __name__ == "__main__":
    main()
