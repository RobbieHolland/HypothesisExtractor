import os
import numpy as np
import pandas as pd


class OutcomeComputer:
    def __init__(self, config):
        self.config = config

    def run(self):
        os.makedirs(self.config.paths.out_dir, exist_ok=True)

        metadata = pd.read_csv(self.config.paths.metadata_csv)
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

        # Last ICD code date per patient, computed before any task filtering (for censoring)
        last_date = diagnoses.groupby("patient_id")["date"].max()

        # Map ICD9 and ICD10 → phecode
        icd9_map = icd_map[icd_map["Flag"] == 9].drop_duplicates("ICD").set_index("ICD")["Phecode"]
        icd10_map = icd_map[icd_map["Flag"] == 10].drop_duplicates("ICD").set_index("ICD")["Phecode"]

        diagnoses["phecode"] = diagnoses["icd9"].map(icd9_map)
        icd10_mapped = diagnoses["icd10"].map(icd10_map)
        diagnoses.loc[icd10_mapped.notna(), "phecode"] = icd10_mapped[icd10_mapped.notna()]

        # Map phecode → task
        diag_mapped = diagnoses.merge(
            task_map[["phecodes", "task"]].rename(columns={"phecodes": "phecode"}),
            on="phecode",
            how="inner",
        )

        results = metadata.set_index("sample_id")[["patient_id", "date"]].copy()

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

        # Drop internal columns
        return results.drop(columns=["patient_id", "date"])
