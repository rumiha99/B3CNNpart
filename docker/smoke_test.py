"""Execute every notebook cell on synthetic data without changing user files."""
import argparse
import json
from pathlib import Path
import tempfile

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; use the GPU Compose file or pass --cpu")
    notebook_path = Path("/workspace/B3CNNpart/Cassava_ResNet18_sample.ipynb")
    notebook = nbformat.read(notebook_path, as_version=4)
    with tempfile.TemporaryDirectory(prefix="b3cnnpart-smoke-") as temp:
        root = Path(temp)
        data = root / "data"
        (data / "train_images").mkdir(parents=True)
        rng = np.random.default_rng(17)
        rows = []
        for label in range(5):
            for index in range(50):
                name = f"{label}_{index}.jpg"
                pixels = rng.integers(0, 256, (72, 96, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(data / "train_images" / name)
                rows.append({"image_id": name, "label": label})
        pd.DataFrame(rows).to_csv(data / "train.csv", index=False)
        (data / "label_num_to_disease_map.json").write_text(
            json.dumps({str(i): f"Class {i}" for i in range(5)})
        )
        settings = next(cell for cell in notebook.cells if cell.get("id") == "settings")
        settings.source += (
            f"\nDATA_DIR = Path({str(data)!r})\n"
            f"OUTPUT_DIR = Path({str(root / 'outputs')!r})\n"
            "torch.set_num_threads(2)\n"
        )
        if args.cpu:
            settings.source += 'device = torch.device("cpu")\n'
        inference = next(cell for cell in notebook.cells if cell.get("id") == "inference")
        inference.source = inference.source.replace(
            'Path("outputs/cassava_resnet18/実行時刻")',
            f"Path({str(root / 'outputs')!r})",
        )
        if args.cpu:
            inference.source = inference.source.replace(
                'inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")',
                'inference_device = torch.device("cpu")',
            )
        # A separate namespace proves inference does not depend on training variables.
        inference.source = f"inference_namespace = {{}}\nexec({inference.source!r}, inference_namespace)"
        notebook.cells.append(nbformat.v4.new_code_cell('''
checkpoint = torch.load(OUTPUT_DIR / "cassava_model.pth", map_location="cpu", weights_only=True)
restored = timm.create_model(checkpoint["architecture"], pretrained=False, num_classes=num_classes)
restored.load_state_dict(checkpoint["model_state_dict"])
restored.to(device).eval()
batch, _ = next(iter(test_loader))
with torch.inference_mode():
    torch.testing.assert_close(restored(batch.to(device)), model(batch.to(device)))
assert len(history_df) == EPOCHS == 2
assert len(results) == len(test_df) == 50
predictions = pd.read_csv(OUTPUT_DIR / "inference_predictions.csv").set_index("image_id")
expected = results.set_index("image_id").loc[predictions.index]
assert len(predictions) == 50
assert (predictions.prediction == expected.prediction).all()
np.testing.assert_allclose(predictions.confidence, expected.confidence, rtol=1e-4, atol=1e-6)
assert len(list((OUTPUT_DIR / "inference_images").glob("*.jpg"))) == 50
assert len(list(OUTPUT_DIR.glob("inference_gallery_*.png"))) == 5
assert (OUTPUT_DIR / "confusion_matrix.png").is_file()
assert (OUTPUT_DIR / "test_predictions.csv").is_file()
'''))
        print("Executing notebook: synthetic 250 images / 2 epochs", flush=True)
        NotebookClient(notebook, timeout=300, kernel_name="b3cnnpart").execute(cwd=temp)
        print("PASS: all cells, training, evaluation, figures, checkpoint reload and predictions")
        print("Device:", "cpu" if args.cpu else torch.cuda.get_device_name(0))
        print("Synthetic scores are not cassava accuracy measurements.")


if __name__ == "__main__":
    main()
