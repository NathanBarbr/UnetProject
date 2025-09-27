import os
import numpy as np
import pandas as pd
import logging
import torch
from torch.utils.data import Dataset
from scipy.interpolate import griddata

logging.basicConfig(level=logging.INFO)

class SpatialSignalDataset(Dataset):
    """Spatial dataset that:
    1. supprime les lignes sans position (lat/long manquants) ;
    2. supprime les métriques où >70 % des valeurs sont NaN ;
    3. impute les NaN restants par la moyenne de la colonne ;
    4. projette chaque métrique sur une grille régulière (d'abord interpolation linéaire, puis nearest pour combler les trous) ;
    5. découpe la grille en patches (on tolère encore quelques NaN : si <10 % du patch est NaN, on remplit ces pixels par la moyenne du patch) ;
       cela évite de jeter systématiquement des patches partiellement vides.
    """

    def __init__(self,
                 csv_path: str,
                 metrics: list | None = None,
                 grid_size: tuple[int, int] = (150, 150),
                 patch_size: int = 16,
                 stride: int | None = None,
                 transform=None,
                 patch_nan_threshold: float = 0.1):
        super().__init__()
        self.transform = transform
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.stride = stride or patch_size // 2  # par défaut 50 % de recouvrement
        self.nan_thr = patch_nan_threshold  # part max de NaN acceptable dans un patch

        if metrics is None:
            metrics = ["download_kbit", "upload_kbit", "ping_ms", "signal_strength"]
        self.metrics = metrics

        self.grids = self._build_grids(csv_path)
        self.patches = self._extract_patches()
        logging.info(f"Number of usable patches generated: {len(self.patches)}")

    # ------------------------------------------------------------------
    #  DATA CLEANING & GRID CONSTRUCTION
    # ------------------------------------------------------------------
    def _build_grids(self, csv_path: str):
        df = pd.read_csv(csv_path)
        # 1) drop rows without position
        before = len(df)
        df = df.dropna(subset=["lat", "long"])
        logging.info(f"Dropped {before-len(df)} rows without position.")

        # 2) ensure metrics exist & drop those with >70 % NaN
        valid_metrics: list[str] = []
        for m in self.metrics:
            if m not in df.columns:
                logging.warning(f"Metric {m} absent from CSV – skipped.")
                continue
            if df[m].isna().mean() > 0.7:
                logging.warning(f"Metric {m} has >70 % NaN – dropped.")
            else:
                valid_metrics.append(m)
        if len(valid_metrics) < 2:
            raise ValueError("Need at least two metrics with sufficient data.")
        self.metrics = valid_metrics

        # 3) impute remaining NaN per column mean
        df[self.metrics] = df[self.metrics].fillna(df[self.metrics].mean())

        # 4) create grid axes
        lat_min, lat_max = df["lat"].min(), df["lat"].max()
        lon_min, lon_max = df["long"].min(), df["long"].max()
        nx, ny = self.grid_size
        grid_lat = np.linspace(lat_min, lat_max, nx)
        grid_lon = np.linspace(lon_min, lon_max, ny)
        grid_x, grid_y = np.meshgrid(grid_lat, grid_lon)

        points = df[["lat", "long"]].values
        grids: dict[str, np.ndarray] = {}
        for met in self.metrics:
            values = df[met].values
            # première passe linéaire
            grid_lin = griddata(points, values, (grid_x, grid_y), method="linear", fill_value=np.nan)
            # deuxième passe nearest pour combler les NaN restants
            if np.isnan(grid_lin).any():
                grid_near = griddata(points, values, (grid_x, grid_y), method="nearest")
                grid_lin = np.where(np.isnan(grid_lin), grid_near, grid_lin)
            grids[met] = grid_lin.astype(np.float32)
        return grids

    # ------------------------------------------------------------------
    #  PATCH EXTRACTION
    # ------------------------------------------------------------------
    def _extract_patches(self):
        met_list = list(self.grids.keys())
        ny, nx = self.grids[met_list[0]].shape
        patches = []
        for i in range(0, ny - self.patch_size + 1, self.stride):
            for j in range(0, nx - self.patch_size + 1, self.stride):
                sub_arrays = [self.grids[m][i:i+self.patch_size, j:j+self.patch_size] for m in met_list]
                patch = np.stack(sub_arrays, axis=0)  # shape (C, H, W)
                nan_mask = np.isnan(patch)
                nan_ratio = nan_mask.mean()
                if nan_ratio > self.nan_thr:
                    continue  # trop de NaN
                if nan_ratio > 0:
                    # remplacer NaN par la moyenne du patch pour chaque canal
                    for c in range(patch.shape[0]):
                        channel = patch[c]
                        mean_val = channel[~nan_mask[c]].mean() if (~nan_mask[c]).any() else 0.0
                        channel[nan_mask[c]] = mean_val
                patches.append(patch)
        if not patches:
            logging.warning("No valid patches generated – consider relaxing thresholds or adjusting parameters.")
        return patches

    # ------------------------------------------------------------------
    #  Dataset API
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        # we will use the last metric as target
        image = torch.from_numpy(patch[:-1])
        target = torch.from_numpy(patch[-1:])
        if self.transform:
            image = self.transform(image)
        return image.float(), target.float()
