import os
import torch
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf
from torch.utils.data import Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeCTDataset(Dataset):
    def __init__(self, paths, sample_ids):
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return torch.randn(1, 224, 224, 160), self.sample_ids[idx]


class _FakeTextDataset(Dataset):
    def __init__(self, paths, sample_ids):
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        return "mock report text", self.sample_ids[idx]


def _make_cfg(tmp_path):
    metadata = pd.DataFrame({
        "sample_id": ["S001", "S002", "S003"],
        "patient_id": ["P001", "P001", "P002"],
        "date": ["2020-01-15", "2021-03-20", "2019-06-10"],
        "ct_path": ["dummy.nii.gz"] * 3,
        "report_path": ["dummy.txt"] * 3,
    })
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    return OmegaConf.create({
        "paths": {
            "metadata_csv": str(tmp_path / "metadata.csv"),
            "out_dir": str(tmp_path / "out"),
        },
        "batch_size": 2,
        "huggingface_token": None,
        "trainer": {"accelerator": "cpu", "devices": 1},
    })


def test_extract_function():
    from compute_embeddings import extract

    model = MagicMock(side_effect=lambda x: torch.randn(x.shape[0], 64))
    model.eval = MagicMock()

    dataset = _FakeCTDataset([], ["S001", "S002", "S003"])
    result = extract(model, dataset, batch_size=2, accelerator="cpu")

    assert result["embeddings"].shape == (3, 64)
    assert result["sample_ids"] == ["S001", "S002", "S003"]


def test_embedding_computer_outputs(tmp_path):
    cfg = _make_cfg(tmp_path)

    with patch("compute_embeddings.CTDataset", _FakeCTDataset), \
         patch("compute_embeddings.TextDataset", _FakeTextDataset), \
         patch("compute_embeddings.MerlinEmbedder") as MockMerlin, \
         patch("compute_embeddings.Qwen3Embedder") as MockQwen3:

        MockMerlin.return_value = MagicMock(
            side_effect=lambda x: torch.randn(x.shape[0], 2048),
            eval=MagicMock(),
        )
        MockQwen3.return_value = MagicMock(
            side_effect=lambda x: torch.randn(len(x), 4096),
            eval=MagicMock(),
        )

        from compute_embeddings import EmbeddingComputer
        EmbeddingComputer(cfg).run()

    ct = torch.load(os.path.join(cfg.paths.out_dir, "ct_embeddings.pt"))
    assert ct["embeddings"].shape == (3, 2048)
    assert ct["sample_ids"] == ["S001", "S002", "S003"]

    rep = torch.load(os.path.join(cfg.paths.out_dir, "report_embeddings.pt"))
    assert rep["embeddings"].shape == (3, 4096)
    assert rep["sample_ids"] == ["S001", "S002", "S003"]
