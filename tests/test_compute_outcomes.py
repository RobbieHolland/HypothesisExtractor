import os
import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from compute_outcomes import OutcomeComputer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_ICD_MAP = os.path.join(REPO_ROOT, "data", "mappings", "Phecode_map12_filtered.csv")
REAL_TASK_MAP = os.path.join(REPO_ROOT, "data", "mappings", "refined_mapped_phecodes.csv")


def _make_test_data(tmp_path):
    # Use real ICD and task maps
    icd_map = pd.read_csv(REAL_ICD_MAP)
    task_map = pd.read_csv(REAL_TASK_MAP)
    
    # Use real phecodes from mappings: 401.1 (hypertension), 250.2 (diabetes)
    test_phecodes = [401.1, 250.2]
    icd_map_filtered = icd_map[icd_map["Phecode"].isin(test_phecodes)].drop_duplicates("ICD")
    task_map_filtered = task_map[task_map["phecodes"].isin(test_phecodes)].copy()
    
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
    for name, df in [("icd_map", icd_map_filtered), ("task_map", task_map_filtered),
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
    cfg = _make_test_data(tmp_path)
    results = OutcomeComputer(cfg).run()

    # S001: P001 has both I10 and E11.9 → both events = 1
    assert results.loc["S001", "CVD_Hypertensive_Diseases_event"] == 1
    assert results.loc["S001", "diabetes_event"] == 1

    # S002: P002 has I10 (past) but no diabetes → CVD=1, diabetes=0
    assert results.loc["S002", "CVD_Hypertensive_Diseases_event"] == 1
    assert results.loc["S002", "diabetes_event"] == 0


def test_event_times(tmp_path):
    cfg = _make_test_data(tmp_path)
    results = OutcomeComputer(cfg).run()

    scan_s001 = pd.Timestamp("2020-01-15")
    expected_cvd = (pd.Timestamp("2020-03-15") - scan_s001).days      # ~60
    expected_dm  = (pd.Timestamp("2021-06-02") - scan_s001).days      # ~503

    # Allow ±20 days to account for σ=3 Gaussian jitter
    assert abs(results.loc["S001", "CVD_Hypertensive_Diseases_time"] - expected_cvd) < 20
    assert abs(results.loc["S001", "diabetes_time"] - expected_dm) < 20


def test_censoring_time(tmp_path):
    cfg = _make_test_data(tmp_path)
    results = OutcomeComputer(cfg).run()

    scan_s002 = pd.Timestamp("2020-01-15")
    last_p002 = pd.Timestamp("2019-01-01")  # last (and only) ICD record for P002
    expected_censor = (last_p002 - scan_s002).days  # negative (past)

    assert abs(results.loc["S002", "diabetes_time"] - expected_censor) < 20


def test_output_file_exists(tmp_path):
    cfg = _make_test_data(tmp_path)
    OutcomeComputer(cfg).run()
    assert os.path.exists(os.path.join(cfg.paths.out_dir, "outcomes.pkl"))
