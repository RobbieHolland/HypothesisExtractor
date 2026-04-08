import os
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from compute_outcomes import OutcomeComputer


def _make_mock_data(tmp_path):
    icd_map = pd.DataFrame({
        "ICD": ["I10", "E11.9"],
        "Flag": [10, 10],
        "ICDString": ["Essential hypertension", "Type 2 diabetes"],
        "Phecode": [401.1, 250.2],
        "PhecodeString": ["Hypertension", "Diabetes mellitus"],
        "PhecodeCategory": ["circulatory", "endocrine"],
    })
    task_map = pd.DataFrame({
        "phecodes": [401.1, 250.2],
        "task": ["hypertension", "diabetes"],
        "phenotype": ["Hypertension", "Diabetes mellitus"],
    })
    # S001 (P001, 2020-01-15): P001 has I10 +60d, E11.9 +503d
    # S002 (P002, 2020-01-15): P002 has I10 on 2019-01-01 (-380d), no diabetes
    metadata = pd.DataFrame({
        "sample_id": ["S001", "S002"],
        "patient_id": ["P001", "P002"],
        "date": ["2020-01-15", "2020-01-15"],
    })
    diagnoses = pd.DataFrame({
        "patient_id": ["P001", "P001", "P002"],
        "date": ["2020-03-15", "2021-06-02", "2019-01-01"],
        "icd9": [None, None, None],
        "icd10": ["I10", "E11.9", "I10"],
    })
    for name, df in [("icd_map", icd_map), ("task_map", task_map),
                     ("metadata", metadata), ("diagnoses", diagnoses)]:
        df.to_csv(tmp_path / f"{name}.csv", index=False)

    return OmegaConf.create({
        "paths": {
            "metadata_csv": str(tmp_path / "metadata.csv"),
            "diagnoses_csv": str(tmp_path / "diagnoses.csv"),
            "icd_phecode_map": str(tmp_path / "icd_map.csv"),
            "phecode_task_map": str(tmp_path / "task_map.csv"),
            "out_dir": str(tmp_path / "out"),
        }
    })


def test_event_flags(tmp_path):
    cfg = _make_mock_data(tmp_path)
    results = OutcomeComputer(cfg).run()

    # S001: P001 has both I10 and E11.9 → both events = 1
    assert results.loc["S001", "hypertension_event"] == 1
    assert results.loc["S001", "diabetes_event"] == 1

    # S002: P002 has I10 (past) but no diabetes → hypertension=1, diabetes=0
    assert results.loc["S002", "hypertension_event"] == 1
    assert results.loc["S002", "diabetes_event"] == 0


def test_event_times(tmp_path):
    cfg = _make_mock_data(tmp_path)
    results = OutcomeComputer(cfg).run()

    scan_s001 = pd.Timestamp("2020-01-15")
    expected_htn = (pd.Timestamp("2020-03-15") - scan_s001).days      # ~60
    expected_dm  = (pd.Timestamp("2021-06-02") - scan_s001).days      # ~503

    # Allow ±20 days to account for σ=3 Gaussian jitter
    assert abs(results.loc["S001", "hypertension_time"] - expected_htn) < 20
    assert abs(results.loc["S001", "diabetes_time"] - expected_dm) < 20


def test_censoring_time(tmp_path):
    cfg = _make_mock_data(tmp_path)
    results = OutcomeComputer(cfg).run()

    scan_s002 = pd.Timestamp("2020-01-15")
    last_p002 = pd.Timestamp("2019-01-01")  # last (and only) ICD record for P002
    expected_censor = (last_p002 - scan_s002).days  # negative (past)

    assert abs(results.loc["S002", "diabetes_time"] - expected_censor) < 20


def test_output_file_exists(tmp_path):
    cfg = _make_mock_data(tmp_path)
    OutcomeComputer(cfg).run()
    assert os.path.exists(os.path.join(cfg.paths.out_dir, "outcomes.pkl"))
