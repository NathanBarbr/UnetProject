import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms
from segmentation_models_pytorch import Unet

from dataset import SignalMapDataset
from metrics import RegressionMetrics


def test_model(model_path, test_csv):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Dataset
    transform = transforms.Compose([])
    test_dataset = SignalMapDataset(test_csv, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Metrics
    metrics = RegressionMetrics()
    total_loss = 0.0
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            metrics.update(out, y)
            total_loss += loss.item()

    result = metrics.compute()
    print("\n[Test Results]")
    print(f"Loss: {total_loss:.4f}, MAE: {result['MAE']:.4f}, MSE: {result['MSE']:.4f}, SSIM: {result['SSIM']:.4f}")


if __name__ == "__main__":
    test_model(
        model_path="saved_models/best_model.pth",
        test_csv="/path/to/test.csv"
    )
