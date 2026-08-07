import os
import numpy as np
import pandas as pd

from utilities.metadata_util import load_metadata


class OutcomeComputer:
    def __init__(self, config):
        self.config = config

    def run(self):
        os.makedirs(self.config.paths.out_dir, exist_ok=True)

        metadata = load_metadata(self.config.paths.metadata_csv)
        diagnoses = pd.read_csv(self.config.paths.diagnoses_csv)
        icd_map = pd.read_csv(self.config.paths.icd_phecode_map)
        task_map = pd.read_csv(self.config.paths.phecode_task_map)

        results = self._compute(metadata, diagnoses, icd_map, task_map)
        out_path = os.path.join(self.config.paths.out_dir, "outcomes.pkl")
        results.to_pickle(out_path)
        print(f"Saved outcomes: {results.shape} → {out_path}")
        return results

    def _compute(self, metadata, diagnoses, icd_map, task_map):
        # Parse dates
        diagnoses = diagnoses.copy()
        diagnoses["date"] = pd.to_datetime(diagnoses["date"])
        metadata = metadata.copy()
        metadata["date"] = pd.to_datetime(metadata["date"])

        # patient_id dtype mismatches (e.g. int vs str) silently zero out the joins below,
        # producing all-NaN outcome times rather than an error.
        meta_pid_dtype, diag_pid_dtype = metadata["patient_id"].dtype, diagnoses["patient_id"].dtype
        if meta_pid_dtype != diag_pid_dtype:
            print(f"WARNING: patient_id dtype mismatch — metadata={meta_pid_dtype}, diagnoses={diag_pid_dtype}. "
                  f"Casting both to str to avoid a silently empty join.")
            metadata["patient_id"] = metadata["patient_id"].astype(str)
            diagnoses["patient_id"] = diagnoses["patient_id"].astype(str)

        meta_pids = set(metadata["patient_id"])
        diag_pids = set(diagnoses["patient_id"])
        n_matched = len(meta_pids & diag_pids)
        print(f"Cohort coverage: diagnoses has {len(diag_pids)} unique patient_id(s) ({len(diagnoses)} rows); "
              f"{n_matched}/{len(meta_pids)} metadata patient_id(s) appear in diagnoses.")
        if n_matched == 0:
            print("  WARNING: NO metadata patient_id matches any diagnoses patient_id — every outcome will be NaN. "
                  "Check that the two files use the same patient identifier.")
        elif n_matched < 0.5 * len(meta_pids):
            missing = sorted(meta_pids - diag_pids)[:3]
            print(f"  WARNING: {len(meta_pids) - n_matched}/{len(meta_pids)} metadata patient_id(s) have NO diagnoses "
                  f"at all; their outcome times will be NaN. Examples: {missing}")

        # Last ICD code date per patient, computed before any task filtering (for censoring)
        last_date = diagnoses.groupby("patient_id")["date"].max()

        # Map ICD9 and ICD10 → phecode
        icd9_map = icd_map[icd_map["Flag"] == 9].drop_duplicates("ICD").set_index("ICD")["Phecode"]
        icd10_map = icd_map[icd_map["Flag"] == 10].drop_duplicates("ICD").set_index("ICD")["Phecode"]

        diagnoses["phecode"] = diagnoses["icd9"].map(icd9_map)
        icd10_mapped = diagnoses["icd10"].map(icd10_map)
        diagnoses.loc[icd10_mapped.notna(), "phecode"] = icd10_mapped[icd10_mapped.notna()]

        n_phecoded = int(diagnoses["phecode"].notna().sum())
        print(f"Diagnosis rows: {n_phecoded}/{len(diagnoses)} mapped to a phecode "
              f"({n_phecoded / max(len(diagnoses), 1):.1%})")
        if n_phecoded == 0:
            print("  WARNING: no ICD code mapped to a phecode — check the icd9/icd10 column formatting "
                  "(e.g. codes with dots vs without).")

        # Map phecode → task
        diag_mapped = diagnoses.merge(
            task_map[["phecodes", "task"]].rename(columns={"phecodes": "phecode"}),
            on="phecode",
            how="inner",
        )
        print(f"  {len(diag_mapped)} row(s) map to a modelled task, covering "
              f"{diag_mapped['patient_id'].nunique()} patient(s) and {diag_mapped['task'].nunique()} task(s)")

        extra_cols = [c for c in ["patient_sex", "patient_age", "recent_bmi", "race", "ethnicity", "smoking_status", "alcohol_use"] if c in metadata.columns]
        rename_map = {}
        for canonical, fallback in {"patient_sex": "sex", "patient_age": "age", "recent_bmi": "bmi"}.items():
            if canonical not in extra_cols and fallback in metadata.columns:
                extra_cols.append(fallback)
                rename_map[fallback] = canonical

        results = metadata.set_index("sample_id")[["patient_id", "date"] + extra_cols].copy()
        if rename_map:
            results.rename(columns=rename_map, inplace=True)

        for task in task_map["task"].unique():
            task_diag = diag_mapped[diag_mapped["task"] == task][["patient_id", "date"]].copy()

            # Find minimum time delta (days) to a task event for each (patient, sample) pair
            merged = task_diag.merge(
                results[["patient_id", "date"]].reset_index(),
                on="patient_id",
                suffixes=("_dx", "_scan"),
            )
            merged["delta"] = (merged["date_dx"] - merged["date_scan"]).dt.days
            first_event = merged.groupby("sample_id")["delta"].min()

            # Censoring time: days from scan to patient's last recorded ICD code
            censor_time = results.apply(
                lambda row: (last_date[row["patient_id"]] - row["date"]).days
                if row["patient_id"] in last_date.index else np.nan,
                axis=1,
            )

            results[f"{task}_event"] = 0
            results[f"{task}_time"] = censor_time.astype(float)
            results.loc[first_event.index, f"{task}_event"] = 1
            results.loc[first_event.index, f"{task}_time"] = first_event

            # Gaussian jitter (σ=3 days) for anonymization
            jitter = np.random.normal(0, 3, len(results))
            results[f"{task}_time"] += jitter

        time_cols = [c for c in results.columns if c.endswith("_time")]
        n_nan = int(results[time_cols[0]].isna().sum())
        print(f"Outcomes: {len(results) - n_nan}/{len(results)} sample(s) have a usable follow-up time; "
              f"{n_nan} ({n_nan / len(results):.1%}) are NaN because the patient has no diagnoses records.")

        # Drop internal columns
        return results.drop(columns=["patient_id", "date"])
