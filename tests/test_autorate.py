import os
import re
import torch
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf


# --- helpers ---

def _make_metadata(tmp_path):
    return _make_metadata_n(tmp_path, [f"S{i:03d}" for i in range(1, 6)])


def _make_metadata_n(tmp_path, sample_ids):
    df = pd.DataFrame([
        {"sample_id": sid, "patient_id": f"P{sid[1:]}", "date": "2020-01-15"}
        for sid in sample_ids
    ])
    p = tmp_path / "metadata.csv"
    df.to_csv(p, index=False)
    return df, p


def _make_labs_metadata():
    return pd.DataFrame([
        {"lab_id": 0, "name": "WBC (K/uL)", "is_numeric": True, "retained": True,
         "mean": 7.0, "std": 2.0, "categories": None},
        {"lab_id": 1, "name": "Hemoglobin (g/dL)", "is_numeric": True, "retained": True,
         "mean": 13.5, "std": 1.5, "categories": None},
    ])


def _make_loinc_map():
    return pd.DataFrame([
        {"loinc_code": "6690-2", "lab_id": 0, "name": "WBC (K/uL)"},
        {"loinc_code": "718-7",  "lab_id": 1, "name": "Hemoglobin (g/dL)"},
    ])


def _make_labs_df():
    rows = []
    for pid in [f"P00{i}" for i in range(1, 6)]:
        rows += [
            {"patient_id": pid, "date": "2019-11-01", "loinc_code": "6690-2", "value": 7.0},
            {"patient_id": pid, "date": "2019-12-01", "loinc_code": "718-7",  "value": 13.5},
        ]
    return pd.DataFrame(rows)


def _make_activations(tmp_path, sample_ids, feature_name, n_concepts=8, n_active=5):
    data = {f"Concept_{i}": np.zeros(len(sample_ids)) for i in range(n_concepts)}
    for i in range(min(n_active, len(sample_ids))):
        data[feature_name][i] = 5.0 + i
    df = pd.DataFrame(data)
    df.insert(0, "sample_id", sample_ids)
    p = tmp_path / "concept_activations_findings.csv"
    df.to_csv(p, index=False)
    return p


def _make_interpretations(tmp_path):
    df = pd.DataFrame([{
        "inputs": "findings",
        "feature_name": "Concept_0",
        "top_k": 40,
        "matryoshka": "[128, 512, 2048, 8192]",
        "outcome": "cvd",
        "extracted_interpretation": "Calcified atherosclerotic plaques.",
        "test_interpretation_discrimination_accuracy": 0.75,
        "tri_valley_interpretation_discrimination_accuracy": 0.70,
    }])
    p = tmp_path / "interp.csv"
    df.to_csv(p, index=False)
    return p


# --- unit tests ---

def test_parse_predictions():
    from autorate import _parse_predictions
    response = "Some reasoning...\n* PREDICTIONS:\n1. 1\n2. 0\n3. 1\n4. 0"
    preds = _parse_predictions(response)
    assert preds == [1, 0, 1, 0]


def test_parse_predictions_no_header():
    from autorate import _parse_predictions
    response = "1. 0\n2. 1\n3. 1"
    preds = _parse_predictions(response)
    assert preds == [0, 1, 1]


def test_format_labs():
    from autorate import _format_labs
    labs_metadata = _make_labs_metadata()
    lab_dicts = [
        {"lab_id": 0, "Value": 9.0, "years_prior": 0.1},
        {"lab_id": 1, "Value": 10.5, "years_prior": 0.2},
    ]
    text = _format_labs(lab_dicts, labs_metadata)
    assert "WBC" in text
    assert "z_score" in text
    assert "0.1" in text


def test_format_labs_empty():
    from autorate import _format_labs
    assert _format_labs([], _make_labs_metadata()) == "(no labs)"


def test_autorate_end_to_end(tmp_path):
    sample_ids = [f"S{i:03d}" for i in range(1, 11)]
    metadata, meta_path = _make_metadata_n(tmp_path, sample_ids)
    interp_path = _make_interpretations(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _make_activations(out_dir, sample_ids, "Concept_0", n_concepts=8)

    # Write dummy report files so the LLM gets real text to reason about
    report_texts = {
        "S001": "CT findings: Large calcified atherosclerotic plaques along the aorta.",
        "S002": "CT findings: Severe calcification of the coronary arteries noted.",
        "S003": "CT findings: Atherosclerotic calcifications in the abdominal aorta and iliac vessels.",
        "S004": "CT findings: Normal liver. No focal lesions. Spleen unremarkable.",
        "S005": "CT findings: Mild hepatic steatosis. Gallbladder without stones.",
        "S006": "CT findings: No acute intra-abdominal pathology identified.",
        "S007": "CT findings: Bilateral renal cysts, otherwise unremarkable.",
        "S008": "CT findings: Normal bowel gas pattern. No free air.",
        "S009": "CT findings: Small hiatal hernia. No other significant findings.",
        "S010": "CT findings: Unremarkable abdominal CT.",
    }
    for sid, text in report_texts.items():
        rp = tmp_path / f"{sid}.txt"
        rp.write_text(text)
        metadata.loc[metadata["sample_id"] == sid, "report_path"] = str(rp)
    meta_path.unlink()
    metadata.to_csv(meta_path, index=False)

    cfg = OmegaConf.create({
        "paths": {
            "metadata_csv": str(meta_path),
            "out_dir": str(out_dir),
            "vertex_project": "som-nero-phi-sgatidis-ge-ai",
            "vertex_location": "us-central1",
            "vertex_model": "gemini-2.5-flash-lite",
            "labs_csv": None,
        }
    })

    with patch("autorate.INTERP_CSV", str(interp_path)), \
         patch("autorate.N_TEST", 2), \
         patch("autorate.ACTIVATION_QUANTILE", 0.0):
        from autorate import run_autorate
        run_autorate(cfg)

    result = pd.read_csv(out_dir / "autointerp_results.csv")
    assert len(result) == 1
    assert result.iloc[0]["inputs"] == "findings"
    assert result.iloc[0]["feature_name"] == "Concept_0"
    assert "outcome" not in result.columns
    assert "test_interpretation_discrimination_accuracy" not in result.columns
    acc = result.iloc[0]["mayo_interpretation_discrimination_accuracy"]
    assert 0.0 <= acc <= 1.0

    samples = pd.read_csv(out_dir / "autointerp_top_samples.csv")
    assert set(samples.columns) >= {"inputs", "feature_name", "rank", "sample_id", "activation", "text"}
    assert list(samples["rank"]) == sorted(samples["rank"].tolist())
    assert len(samples) <= 3
    assert (samples["activation"] > 0).all()
