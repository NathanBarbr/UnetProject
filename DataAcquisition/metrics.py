import torch
import torch.nn.functional as F

class RegressionMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_mae = 0.0
        self.total_mse = 0.0
        self.total_ssim = 0.0
        self.count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.total_mae += torch.mean(torch.abs(preds - targets)).item()
        self.total_mse += torch.mean((preds - targets) ** 2).item()
        self.total_ssim += self._ssim(preds, targets)
        self.count += 1

    def compute(self):
        return {
            "MAE": self.total_mae / self.count,
            "MSE": self.total_mse / self.count,
            "SSIM": self.total_ssim / self.count,
        }

    def _ssim(self, x, y):
        var_x = torch.var(x)
        var_y = torch.var(y)
        mse = F.mse_loss(x, y)
        return (1 - mse / (var_x + var_y + 1e-8)).item()