import os
import torch
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch


def _make_dummy_weights(in_dim, n_concepts=16):
    return {
        "W_enc": torch.randn(in_dim, n_concepts),
        "b_enc": torch.zeros(n_concepts),
        "b_dec": torch.zeros(in_dim),
    }


def _patch_sae_weights(in_dim, n_concepts=16):
    weights = _make_dummy_weights(in_dim, n_concepts)
    return patch("models.sae_model._load_sae_weights", return_value=weights)


def test_apply_sae_output_shape():
    from models.sae_model import apply_sae
    with patch("models.sae_model._load_sae_weights", return_value=_make_dummy_weights(32, 64)):
        with patch("models.sae_model.TOP_K", 4):
            out = apply_sae(torch.randn(5, 32), "findings")
    assert out.shape == (5, 64)


def test_apply_sae_topk_sparsity():
    from models.sae_model import apply_sae
    TOP_K = 4
    with patch("models.sae_model._load_sae_weights", return_value=_make_dummy_weights(32, 64)):
        with patch("models.sae_model.TOP_K", TOP_K):
            out = apply_sae(torch.randn(5, 32), "findings")
    nonzero_per_row = (out > 0).sum(dim=1)
    assert (nonzero_per_row <= TOP_K).all(), f"Some rows have > {TOP_K} nonzero: {nonzero_per_row}"


def test_apply_sae_nonnegative():
    from models.sae_model import apply_sae
    with patch("models.sae_model._load_sae_weights", return_value=_make_dummy_weights(32, 64)):
        with patch("models.sae_model.TOP_K", 4):
            out = apply_sae(torch.randn(5, 32), "findings")
    assert (out >= 0).all()


def test_infer_sae_produces_csv(tmp_path):
    from omegaconf import OmegaConf
    from unittest.mock import patch, MagicMock

    n_samples = 6
    emb_dim = 32
    n_concepts = 16
    sample_ids = [f"S{i:03d}" for i in range(n_samples)]

    emb_path = tmp_path / "report_embeddings.pt"
    torch.save({"embeddings": torch.randn(n_samples, emb_dim), "sample_ids": sample_ids}, emb_path)

    cfg = OmegaConf.create({
        "paths": {
            "out_dir": str(tmp_path),
            "sae_cache_dir": str(tmp_path / "sae"),
        }
    })

    dummy_weights = _make_dummy_weights(emb_dim, n_concepts)

    with patch("models.sae_model._load_sae_weights", return_value=dummy_weights), \
         patch("models.sae_model.TOP_K", 4):
        from infer_sae import run_infer_sae
        run_infer_sae(cfg)

    out_csv = tmp_path / "concept_activations_findings.csv"
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert "sample_id" in df.columns
    assert df.shape == (n_samples, n_concepts + 1)
    assert list(df["sample_id"]) == sample_ids
