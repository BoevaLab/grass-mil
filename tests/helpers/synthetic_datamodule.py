from __future__ import annotations

import lightning as L
import torch

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
except Exception as exc:  # pragma: no cover
    raise ImportError("torch_geometric is required for synthetic test datamodule") from exc


def _make_graph(
    region_id: str,
    label: float,
    num_nodes: int = 6,
    input_dim: int = 8,
    patch_idx: int = 0,
    survival: bool = False,
) -> Data:
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0, 1, 2, 3, 4]],
        dtype=torch.long,
    )
    if survival:
        # [time, event]; order is load-bearing for the Cox loss. Times are
        # spread and events mixed so risk sets are non-degenerate -- if every
        # event sat at the largest time the partial likelihood would be
        # constant and carry no gradient.
        follow_up = 1.0 + float(patch_idx % 7) + 3.0 * label
        observed = float((patch_idx % 3) != 0)
        graph_y = torch.tensor([[follow_up, observed]], dtype=torch.float32)
    else:
        graph_y = torch.tensor([[label]], dtype=torch.float32)
    graph_w = torch.ones((1, 1), dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, graph_y=graph_y, graph_w=graph_w)
    data.sample_id = "sample_a"
    data.region_id = region_id
    data.patch_id = f"{region_id}_patch_{patch_idx}"
    data.graph_label_names = ["time", "event"] if survival else ["label"]
    return data


class SyntheticBagDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        input_dim: int = 8,
        graphs_per_region: int = 20,
        survival: bool = False,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.input_dim = input_dim
        self.graphs_per_region = graphs_per_region
        self.survival = survival
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None

    def setup(self, stage=None):
        data = []
        for i in range(self.graphs_per_region):
            data.append(
                _make_graph(
                    "region_0",
                    0.0,
                    input_dim=self.input_dim,
                    patch_idx=2 * i,
                    survival=self.survival,
                )
            )
            data.append(
                _make_graph(
                    "region_1",
                    1.0,
                    input_dim=self.input_dim,
                    patch_idx=2 * i + 1,
                    survival=self.survival,
                )
            )
        self.dataset_train = data
        self.dataset_val = data
        self.dataset_test = data

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.dataset_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
