"""
AutoRate: evaluate existing concept interpretations on new data (e.g. Mayo).

Steps:
  1. Load concept activations from infer_sae.py output (concept_activations_{modality}.csv)
  2. Load patient text (report files) or formatted labs
  3. For each concept in data/autointerp_interpretations.csv:
       - Sample high-activation patients + low-activation controls
       - Ask Vertex AI to discriminate based on the interpretation
       - Compute accuracy
  4. Save results to out/autointerp_results.csv
"""
import os
import re
import random
import numpy as np
import pandas as pd
from utilities.vertex_ai import VertexAIClient
from utilities.ollama_client import OllamaClient

INTERP_CSV = "data/autointerp_interpretations.csv"

# Modality → which field to show the LLM (matches original autointerp config)
INTERP_FIELD = {
    "findings": "findings",
    "image": "findings",   # image concepts are rated by report text
    "labs": "labs",
}

N_TEST = 20
ACTIVATION_QUANTILE = 0.85

DISCRIMINATE_PROMPT = """\
Task: Distinguish samples (randomly ordered) based on those that do and do not belong to the group characterized by the group characteristics.

Medical Findings Reports:
{patient_data}

Group characteristics:
{interpretation}

Analyze the findings reports one by one, reasoning on whether or not the characteristics apply to the report. Use the following format for each report:
{{i}}: 1 (if belongs to the group) or 0 (if does not belong to the group). We have designed the list so that exactly half of the reports belong to the group, and half do not. Therefore, you should use the characteristics to differentiate each of the reports into two distinct groups, rather than finding perfect matches.

You need to be as accurate as possible when assigning 1s and 0s. We will score the accuracy of your response at the end.

=== Format ===:
It is crucial that you present your final explanation in the following format, started with an asterisk:
* PREDICTIONS:
1. 1 (if the sample belongs to the group) or 0 (if it does not belong to the group)
2. 1 (if the sample belongs to the group) or 0 (if it does not belong to the group)
...
{n}. 1 (if the sample belongs to the group) or 0 (if it does not belong to the group)
"""


def _format_labs(lab_dicts, labs_metadata):
    meta = labs_metadata.set_index("lab_id")

    enriched = []
    for lab in lab_dicts:
        lab_id = lab["lab_id"]
        if lab_id not in meta.index:
            continue
        row = meta.loc[lab_id]
        if not row["is_numeric"] or row["std"] < 1e-6:
            continue
        val = float(lab["Value"])
        z = (val - row["mean"]) / row["std"]
        enriched.append({"name": row["name"], "value": val, "z_score": z, "years_prior": lab["years_prior"]})

    if not enriched:
        return "(no labs)"

    if len(enriched) > 30:
        sorted_e = sorted(enriched, key=lambda x: x["z_score"], reverse=True)
        selected = sorted_e[:15] + sorted_e[-15:]
    else:
        selected = sorted(enriched, key=lambda x: abs(x["z_score"]), reverse=True)

    return "\n".join(
        f"{e['name']} ({e['years_prior']:.1f} years ago): {e['value']} [z_score={e['z_score']:.1f}]"
        for e in selected
    )


def _get_report_text(sample_id, metadata):
    row = metadata[metadata["sample_id"] == sample_id]
    if row.empty or "report_path" not in row.columns:
        return ""
    path = row["report_path"].iloc[0]
    if not isinstance(path, str) or not os.path.exists(path):
        return ""
    with open(path) as f:
        return f.read().strip()


def _build_labs_for_sample(sample_id, metadata, labs_df, loinc_map, labs_metadata):
    meta_row = metadata[metadata["sample_id"] == sample_id]
    if meta_row.empty:
        return []
    patient_id = meta_row["patient_id"].iloc[0]
    scan_date = pd.to_datetime(meta_row["date"].iloc[0])
    window_start = scan_date - pd.DateOffset(months=6)

    patient_labs = labs_df[labs_df["patient_id"] == patient_id].copy()
    patient_labs = patient_labs[
        (patient_labs["date"] <= scan_date) & (patient_labs["date"] >= window_start)
    ]
    if patient_labs.empty:
        return []

    retained_ids = set(labs_metadata.loc[labs_metadata["retained"], "lab_id"])
    loinc_to_lab_id = loinc_map.dropna(subset=["loinc_code", "lab_id"]).set_index("loinc_code")["lab_id"].to_dict()
    patient_labs["lab_id"] = patient_labs["loinc_code"].map(loinc_to_lab_id)
    patient_labs = patient_labs.dropna(subset=["lab_id"])
    patient_labs["lab_id"] = patient_labs["lab_id"].astype(int)
    patient_labs = patient_labs[patient_labs["lab_id"].isin(retained_ids)]
    if patient_labs.empty:
        return []

    patient_labs = patient_labs.sort_values("date")
    patient_labs = (
        patient_labs.groupby(["lab_id", pd.Grouper(key="date", freq="QS")])
        .last().reset_index()
    )
    patient_labs["years_prior"] = (
        (scan_date - patient_labs["date"]).dt.total_seconds() / (365.25 * 24 * 3600)
    )

    return [
        {"lab_id": int(r["lab_id"]), "Value": r["value"], "years_prior": float(r["years_prior"])}
        for _, r in patient_labs.iterrows()
    ]


def _get_patient_text(sample_id, field, metadata, labs_df, loinc_map, labs_metadata):
    if field == "findings":
        return _get_report_text(sample_id, metadata)
    lab_dicts = _build_labs_for_sample(sample_id, metadata, labs_df, loinc_map, labs_metadata)
    return _format_labs(lab_dicts, labs_metadata)


def _parse_predictions(response_text):
    match = re.search(r"\*\s*PREDICTIONS:(.+?)(?=\*|$)", response_text, re.DOTALL | re.IGNORECASE)
    text = match.group(1) if match else response_text
    return [int(m) for m in re.findall(r"\d+\.\s*([01])", text)]


def _build_llm_client(config):
    backend = config.get("llm", "ollama")
    if backend == "vertex":
        return VertexAIClient(
            project=config.paths.vertex_project,
            location=config.paths.get("vertex_location", "us-central1"),
            model=config.paths.get("vertex_model", "gemini-2.5-pro"),
        )
    if backend == "ollama":
        return OllamaClient(
            model=config.get("ollama_model", "gemma4:31b"),
            host=config.get("ollama_host"),
            auto_start=config.get("ollama_auto_start", True),
        )
    raise ValueError(f"Unknown llm backend {backend!r} (expected 'vertex' or 'ollama')")


def run_autorate(config):
    out_dir = config.paths.out_dir
    os.makedirs(out_dir, exist_ok=True)

    interpretations = pd.read_csv(INTERP_CSV)
    metadata = pd.read_csv(config.paths.metadata_csv)

    llm_client = _build_llm_client(config)
    print(f"AutoRate LLM backend: {llm_client.name} "
          f"(host={getattr(llm_client, 'host', 'n/a')}, "
          f"auto_start={getattr(llm_client, 'auto_start', 'n/a')})")
    try:
        llm_client.ensure_ready()
    except Exception as e:
        raise SystemExit(f"AutoRate cannot start ({llm_client.name}): {e}") from None

    labs_df, loinc_map, labs_metadata = None, None, None
    if config.paths.get("labs_csv"):
        labs_df = pd.read_csv(config.paths.labs_csv)
        labs_df["date"] = pd.to_datetime(labs_df["date"])
        if "loinc_code" not in labs_df.columns and "code" in labs_df.columns:
            labs_df = labs_df.rename(columns={"code": "loinc_code"})
        if labs_df["loinc_code"].str.startswith("LOINC/").any():
            labs_df["loinc_code"] = labs_df["loinc_code"].str.replace("LOINC/", "", regex=False)
        loinc_map = pd.read_csv(config.paths.loinc_map)
        labs_metadata = pd.read_pickle(config.paths.labs_metadata)

    out_path = os.path.join(out_dir, "autointerp_results.csv")
    samples_path = os.path.join(out_dir, "autointerp_top_samples.csv")
    existing = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame()
    existing_samples = pd.read_csv(samples_path) if os.path.exists(samples_path) else pd.DataFrame()

    results = []
    top_sample_rows = []
    n_interp = len(interpretations)
    for i, (_, interp_row) in enumerate(interpretations.iterrows(), start=1):
        modality = interp_row["inputs"]
        feature_name = interp_row["feature_name"]
        interpretation = interp_row["extracted_interpretation"]
        progress = f"[{i}/{n_interp}]"
        if not isinstance(interpretation, str) or not interpretation.strip():
            continue

        acts_path = os.path.join(out_dir, f"concept_activations_{modality}.csv")
        if not os.path.exists(acts_path):
            print(f"{progress} Skipping {modality}/{feature_name}: no activations at {acts_path}")
            continue

        if not existing.empty and ((existing["inputs"] == modality) & (existing["feature_name"] == feature_name) & (existing["top_k"] == interp_row.get("top_k")) & (existing["matryoshka"] == interp_row.get("matryoshka"))).any():
            print(f"{progress} Skipping {modality}/{feature_name}: already rated")
            continue

        acts_df = pd.read_csv(acts_path, usecols=["sample_id", feature_name], dtype={feature_name: float})
        concept_acts = acts_df.set_index("sample_id")[feature_name]

        active = concept_acts[concept_acts > 0]
        if len(active) < 4:
            print(f"{progress} Skipping {modality}/{feature_name}: only {len(active)} active samples")
            continue

        threshold = active.quantile(ACTIVATION_QUANTILE)
        high_ids = active[active >= threshold].index.tolist()
        low_ids = concept_acts[concept_acts <= 0].index.tolist()

        n = min(N_TEST, len(high_ids), len(low_ids))
        field = INTERP_FIELD[modality]

        random.seed(42)
        high_sample = random.sample(high_ids, n)
        low_sample = random.sample(low_ids, n)

        combined = [(sid, 1) for sid in high_sample] + [(sid, 0) for sid in low_sample]
        random.shuffle(combined)
        groundtruth = np.array([label for _, label in combined])

        patient_data = "\n\n".join(
            f"--- Sample {i+1} ---\n{_get_patient_text(sid, field, metadata, labs_df, loinc_map, labs_metadata)}"
            for i, (sid, _) in enumerate(combined)
        )

        prompt = DISCRIMINATE_PROMPT.format(
            patient_data=patient_data,
            interpretation=interpretation,
            n=len(combined),
        )

        print(f"{progress} AutoRating {modality}/{feature_name} ({n} high + {n} low)...")
        try:
            response = llm_client.query(prompt, temperature=0.0)
            predicted = np.array(_parse_predictions(response))
            if len(predicted) == len(groundtruth):
                acc = float(np.mean(predicted == groundtruth))
            else:
                acc = float("nan")
                print(f"  WARNING: got {len(predicted)} predictions for {len(groundtruth)} samples")
        except Exception as e:
            print(f"  ERROR: {e}")
            response, predicted, acc = "", np.array([]), float("nan")

        print(f"  Discrimination accuracy: {acc:.2%}" if not np.isnan(acc) else "  Accuracy: N/A")

        # Top-3 activating samples
        top3_ids = active.nlargest(3).index.tolist()
        for rank, sid in enumerate(top3_ids, start=1):
            top_sample_rows.append({
                "inputs": modality,
                "feature_name": feature_name,
                "rank": rank,
                "sample_id": sid,
                "activation": float(active[sid]),
                "text": _get_patient_text(sid, field, metadata, labs_df, loinc_map, labs_metadata),
            })

        results.append({
            "inputs": modality,
            "feature_name": feature_name,
            "top_k": interp_row.get("top_k"),
            "matryoshka": interp_row.get("matryoshka"),
            "extracted_interpretation": interpretation,
            "mayo_n_tested": len(combined),
            "mayo_interpretation_discrimination_accuracy": acc,
        })

        all_results = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
        all_results.drop_duplicates(subset=["inputs", "feature_name", "top_k", "matryoshka"], keep="last", inplace=True)
        all_results.to_csv(out_path, index=False)

        all_samples = pd.concat([existing_samples, pd.DataFrame(top_sample_rows)], ignore_index=True)
        all_samples.drop_duplicates(subset=["inputs", "feature_name", "rank"], keep="last", inplace=True)
        all_samples.to_csv(samples_path, index=False)

    print(f"\nDone. Results saved to {out_path}  ({len(results)} new rows)")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="config", config_name="config", version_base=None)
    def main(cfg: DictConfig):
        run_autorate(cfg)

    main()
