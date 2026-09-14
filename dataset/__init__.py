from pathlib import Path

import numpy as np
import torch


class ShardLoader:
    """Loads pretokenized uint16 hg38 shards written by dataset.prepare and serves
    random context-length windows for language-model training."""

    def __init__(self, data_dir: str, split: str, device: str = "cuda"):
        self.device = device
        shard_dir = Path(data_dir) / split
        self.shard_paths = sorted(shard_dir.glob(f"{split}_*.bin"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shards found in {shard_dir}. Run dataset.prepare first.")
        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.shard_paths]
        self.shard_lengths = [len(s) for s in self.shards]

    def get_batch(self, batch_size: int, block_size: int):
        xs, ys = [], []
        for _ in range(batch_size):
            shard_idx = np.random.randint(len(self.shards))
            shard = self.shards[shard_idx]
            max_start = len(shard) - block_size - 1
            if max_start <= 0:
                continue
            start = np.random.randint(max_start)
            chunk = shard[start : start + block_size + 1].astype(np.int64)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys))
        if self.device == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
