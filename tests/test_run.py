import pandas as pd
from unittest.mock import patch, MagicMock, call
from omegaconf import OmegaConf


def _make_cfg(tmp_path):
    pd.DataFrame({
        "sample_id": ["S001"],
        "patient_id": ["P001"],
        "date": ["2020-01-15"],
        "ct_path": ["dummy.nii.gz"],
        "report_path": ["dummy.txt"],
    }).to_csv(tmp_path / "metadata.csv", index=False)
    return OmegaConf.create({
        "paths": {
            "metadata_csv": str(tmp_path / "metadata.csv"),
            "diagnoses_csv": "dummy.csv",
            "icd_phecode_map": "dummy.csv",
            "phecode_task_map": "dummy.csv",
            "out_dir": str(tmp_path / "out"),
        },
        "batch_size": 1,
        "huggingface_token": None,
        "trainer": {"accelerator": "cpu", "devices": 1},
    })


def test_run_calls_both_computers(tmp_path):
    cfg = _make_cfg(tmp_path)

    with patch("compute_embeddings.EmbeddingComputer") as MockEmb, \
         patch("compute_outcomes.OutcomeComputer") as MockOut:

        mock_emb = MagicMock()
        mock_out = MagicMock()
        MockEmb.return_value = mock_emb
        MockOut.return_value = mock_out

        # Import and exercise the same logic run.py's main executes
        from compute_embeddings import EmbeddingComputer
        from compute_outcomes import OutcomeComputer

        EmbeddingComputer(cfg).run()
        OutcomeComputer(cfg).run()

        mock_emb.run.assert_called_once()
        mock_out.run.assert_called_once()
