import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

class RegressionTrainer:
    def __init__(self, model, optimizer, loss_fn, lr_scheduler, train_metric, val_metric,
                 train_data, val_data, device, num_epochs, training_save_dir,
                 batch_size=4, val_frequency=5,logger=None):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency
        self.logger = logger

        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        self.best_mse = float("inf")

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._train_epoch(epoch)
            if epoch % self.val_frequency == 0 or epoch == self.num_epochs:
                val_loss = self._val_epoch(epoch)
                if val_loss < self.best_mse:
                    self.best_mse = val_loss
                    torch.save(self.model.state_dict(), self.training_save_dir / "best_model.pth")

            if self.lr_scheduler:
                self.lr_scheduler.step()

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metric.reset()
        total_loss = 0.0

        for x, y in tqdm(self.train_loader, desc=f"Train Epoch {epoch}"):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.loss_fn(out, y)
            loss.backward()
            self.optimizer.step()
            self.train_metric.update(out, y)
            total_loss += loss.item()

        metrics = self.train_metric.compute()
        avg_loss=total_loss / len(self.train_loader)
        print(f"\n[Train] Epoch {epoch} - Loss: {total_loss:.4f}, MAE: {metrics['MAE']:.4f}, MSE: {metrics['MSE']:.4f}, SSIM: {metrics['SSIM']:.4f}")

        if self.logger:
            self.logger.log({
                "epoch": epoch,
                "train/loss": avg_loss,
                "train/mae": metrics["MAE"],
                "train/mse": metrics["MSE"],
                "train/ssim": metrics["SSIM"]
            }, step=epoch)

        return avg_loss

    def _val_epoch(self, epoch):
        self.model.eval()
        self.val_metric.reset()
        total_loss = 0.0

        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc=f"Val Epoch {epoch}"):
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.loss_fn(out, y)
                self.val_metric.update(out, y)
                total_loss += loss.item()

        metrics = self.val_metric.compute()
        print(f"\n[Val] Epoch {epoch} - Loss: {total_loss:.4f}, MAE: {metrics['MAE']:.4f}, MSE: {metrics['MSE']:.4f}, SSIM: {metrics['SSIM']:.4f}")

        if self.logger:
            self.logger.log({
                'val/loss': total_loss / len(self.val_loader),
                'val/mae': metrics['MAE'],
                'val/mse': metrics['MSE'],
                'val/ssim': metrics['SSIM'],
                'epoch': epoch
            })
        return total_loss / len(self.val_loader)