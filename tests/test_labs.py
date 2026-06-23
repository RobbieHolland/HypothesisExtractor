import os
import torch
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _make_labs_metadata():
    return pd.DataFrame([
        {"lab_id": 0, "name": "WBC (K/uL)", "is_numeric": True, "retained": True, "mean": 7.0, "std": 2.0, "categories": None},
        {"lab_id": 1, "name": "Hemoglobin (g/dL)", "is_numeric": True, "retained": True, "mean": 13.5, "std": 1.5, "categories": None},
    ])


def _make_loinc_map():
    return pd.DataFrame([
        {"loinc_code": "6690-2", "lab_id": 0, "name": "WBC (K/uL)"},
        {"loinc_code": "718-7", "lab_id": 1, "name": "Hemoglobin (g/dL)"},
    ])


def _make_labs_df():
    return pd.DataFrame([
        {"patient_id": "P001", "date": "2019-11-01", "loinc_code": "6690-2", "value": 6.5},
        {"patient_id": "P001", "date": "2019-12-15", "loinc_code": "718-7", "value": 14.0},
        {"patient_id": "P002", "date": "2019-03-01", "loinc_code": "6690-2", "value": 7.2},
    ])


def _make_metadata():
    # S001 (P001, 2020-01-15): both P001 labs are within 6 months → included
    # S002 (P001, 2021-03-20): no P001 labs within 6 months → skipped
    # S003 (P002, 2019-06-10): P002 lab (2019-03-01) within 6 months → included
    return pd.DataFrame([
        {"sample_id": "S001", "patient_id": "P001", "date": "2020-01-15"},
        {"sample_id": "S002", "patient_id": "P001", "date": "2021-03-20"},
        {"sample_id": "S003", "patient_id": "P002", "date": "2019-06-10"},
    ])


def test_labs_dataset_filtering():
    from models.labs_model import LabsDataset

    dataset = LabsDataset(_make_metadata(), _make_labs_df(), _make_loinc_map(), _make_labs_metadata())

    assert len(dataset) == 2
    sample_ids = [sid for _, sid in dataset.samples]
    assert "S001" in sample_ids
    assert "S003" in sample_ids
    assert "S002" not in sample_ids


def test_labs_dataset_item_format():
    from models.labs_model import LabsDataset

    dataset = LabsDataset(_make_metadata(), _make_labs_df(), _make_loinc_map(), _make_labs_metadata())

    lab_dicts, sample_id = dataset[0]
    assert isinstance(lab_dicts, list) and len(lab_dicts) > 0
    assert all(k in lab_dicts[0] for k in ("lab_id", "Value", "years_prior"))
    assert lab_dicts[0]["years_prior"] >= 0


def test_labs_dataset_unknown_loinc_ignored():
    from models.labs_model import LabsDataset

    labs_with_unknown = _make_labs_df().copy()
    labs_with_unknown = pd.concat([
        labs_with_unknown,
        pd.DataFrame([{"patient_id": "P001", "date": "2019-12-01", "loinc_code": "99999-9", "value": 1.0}])
    ], ignore_index=True)

    dataset = LabsDataset(_make_metadata(), labs_with_unknown, _make_loinc_map(), _make_labs_metadata())
    assert len(dataset) == 2  # same as without unknown LOINC


def test_labs_embedder_output_shape():
    from models.labs_model import LabsEmbedder

    embedder = LabsEmbedder(_make_labs_metadata())
    embedder.eval()

    batch = [
        [{"lab_id": 0, "Value": 6.5, "years_prior": 0.2}, {"lab_id": 1, "Value": 14.0, "years_prior": 0.1}],
        [{"lab_id": 0, "Value": 7.2, "years_prior": 0.3}],
    ]
    with torch.no_grad():
        out = embedder(batch)

    assert out.shape == (2, 768)


def test_labs_collate():
    from models.labs_model import labs_collate

    batch = [
        ([{"lab_id": 0, "Value": 1.0, "years_prior": 0.1}], "S001"),
        ([{"lab_id": 1, "Value": 2.0, "years_prior": 0.2}, {"lab_id": 0, "Value": 3.0, "years_prior": 0.05}], "S002"),
    ]
    lab_lists, ids = labs_collate(batch)
    assert ids == ["S001", "S002"]
    assert len(lab_lists[0]) == 1
    assert len(lab_lists[1]) == 2


def test_embedding_computer_with_labs(tmp_path):
    metadata = _make_metadata()
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    _make_labs_df().to_csv(tmp_path / "labs.csv", index=False)
    _make_labs_metadata().to_pickle(tmp_path / "labs_metadata.pkl")
    _make_loinc_map().to_csv(tmp_path / "loinc_map.csv", index=False)

    cfg = OmegaConf.create({
        "paths": {
            "metadata_csv": str(tmp_path / "metadata.csv"),
            "out_dir": str(tmp_path / "out"),
            "labs_csv": str(tmp_path / "labs.csv"),
            "labs_metadata": str(tmp_path / "labs_metadata.pkl"),
            "loinc_map": str(tmp_path / "loinc_map.csv"),
            "labs_checkpoint": None,
        },
        "batch_size": 2,
        "trainer": {"accelerator": "cpu", "devices": 1},
    })

    from models.labs_model import LabsDataset, LabsEmbedder, labs_collate

    class FakeLabsDataset(LabsDataset):
        pass

    fake_embedder = LabsEmbedder(_make_labs_metadata())

    with patch("compute_embeddings.LabsEmbedder", return_value=fake_embedder), \
         patch("compute_embeddings.LabsDataset", FakeLabsDataset):
        from compute_embeddings import EmbeddingComputer
        EmbeddingComputer(cfg).run()

    result = torch.load(os.path.join(cfg.paths.out_dir, "lab_embeddings.pt"))
    assert result["embeddings"].shape == (2, 768)
    assert set(result["sample_ids"]) == {"S001", "S003"}
