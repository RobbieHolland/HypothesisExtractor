import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class FeatureTokenizer(nn.Module):
    def __init__(self, labs_metadata, embedding_dim):
        super().__init__()
        self.labs_metadata = labs_metadata.set_index("lab_id")
        self.embedding_dim = embedding_dim

        self.pad_embedding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, 1, embedding_dim)))
        self.mask_embedding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, embedding_dim)))

        self.time_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, embedding_dim),
        )

        self.pos_embeddings = nn.ParameterDict({
            str(row["lab_id"]): nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, embedding_dim)))
            for _, row in labs_metadata.iterrows() if row["retained"]
        })

        self.num_embeddings = nn.ParameterDict({
            str(row["lab_id"]): nn.Parameter(nn.init.kaiming_normal_(torch.empty(1, embedding_dim)))
            for _, row in labs_metadata.iterrows() if row["is_numeric"] and row["retained"]
        })

        self.cat_embeddings = nn.ModuleDict()
        for _, row in labs_metadata.iterrows():
            if not row["is_numeric"] and row["retained"]:
                lab_id = str(row["lab_id"])
                self.cat_embeddings[lab_id] = nn.ParameterDict()
                for category in row["categories"]:
                    self.cat_embeddings[lab_id][str(category).replace(".", "_")] = nn.Parameter(
                        nn.init.kaiming_normal_(torch.empty(1, embedding_dim))
                    )

    def forward(self, lab_ids, values, years_prior, truncate=128):
        max_length = min(max(len(v) for v in values), truncate)
        embeddings = self.pad_embedding.repeat(len(values), max_length, 1)
        attention_mask = torch.ones(len(values), max_length, dtype=torch.float32)

        for i, (batch_lab_ids, batch_values, batch_years_prior) in enumerate(zip(lab_ids, values, years_prior)):
            num_mask = [self.labs_metadata.loc[lid, "is_numeric"] for lid in batch_lab_ids]

            for j, (is_numeric, lab_id, value, yp) in enumerate(
                zip(num_mask, batch_lab_ids, batch_values, batch_years_prior)
            ):
                if j >= max_length:
                    break

                if is_numeric and isinstance(value, (int, float)):
                    std_val = self.labs_metadata.loc[lab_id, "std"]
                    adj = (value - self.labs_metadata.loc[lab_id, "mean"]) / std_val if std_val > 1e-6 else value
                    adj = torch.clamp(torch.tensor(adj, dtype=torch.float32), -3, 3)
                    embeddings[i, j] = adj * self.num_embeddings[str(lab_id)]
                    attention_mask[i, j] = 0
                elif not is_numeric:
                    key = str(value).replace(".", "_")
                    if key in self.cat_embeddings.get(str(lab_id), {}):
                        embeddings[i, j] = self.cat_embeddings[str(lab_id)][key]
                        attention_mask[i, j] = 0

                time_emb = self.time_embedding(
                    torch.tensor([float(yp)], dtype=torch.float32).to(embeddings.device)
                )
                embeddings[i, j] = embeddings[i, j] + time_emb + self.pos_embeddings[str(lab_id)]

        return embeddings, attention_mask


class LabsEmbedder(nn.Module):
    def __init__(self, labs_metadata, checkpoint_path=None):
        super().__init__()
        self.feature_tokenizer = FeatureTokenizer(labs_metadata, embedding_dim=768)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=4, dim_feedforward=3072, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=3)

        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        ckpts = glob.glob(os.path.join(glob.escape(checkpoint_path), "*.ckpt"))
        state = torch.load(ckpts[0], weights_only=False)["state_dict"]
        prefix = "encoder.inference_map.labs_with_mask->ft_transformer."
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        report = self.load_state_dict(state, strict=False)
        print(f"Loaded labs encoder from {ckpts[0]} ({len(state)} keys, {len(report.missing_keys)} missing, {len(report.unexpected_keys)} unexpected)")

    def forward(self, batch):
        # batch: list of lists of dicts, each dict has lab_id, Value, years_prior
        lab_ids = [[v["lab_id"] for v in sample] for sample in batch]
        values = [
            [float(v["Value"]) if isinstance(v["Value"], (int, float)) else v["Value"] for v in sample]
            for sample in batch
        ]
        years_prior = [[v["years_prior"] for v in sample] for sample in batch]

        embeddings, mask = self.feature_tokenizer(lab_ids, values, years_prior)
        mask = mask.to(embeddings.device)
        output = self.encoder(embeddings, src_key_padding_mask=mask)

        # Mean pool over non-padded tokens
        mask_exp = (1 - mask.unsqueeze(-1)).expand_as(output)
        return (output * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1e-6)


SENTINEL_VALUE_THRESHOLD = 999999  # |value| >= this is almost certainly a placeholder/error code, not a real result

LAB_COUNT_BUCKETS = [(0, 0), (1, 10), (11, 100), (101, 1000), (1001, float("inf"))]


def _print_lab_count_histogram(label, patient_ids, labs_subset):
    """Distribution of #labs-per-patient over `patient_ids`, using labs_subset (already
    linked to patients via patient_id, but with no other content filtering applied by the
    caller — the caller decides what "before" vs "after" filtering means)."""
    counts = labs_subset.groupby(labs_subset["patient_id"].astype(str)).size()
    counts = counts.reindex(patient_ids, fill_value=0)
    print(f"{label} (n={len(patient_ids)} patients):")
    for lo, hi in LAB_COUNT_BUCKETS:
        if hi == float("inf"):
            n = int((counts >= lo).sum())
            bucket_label = f"{lo}+"
        elif lo == hi == 0:
            n = int((counts == 0).sum())
            bucket_label = "0"
        else:
            n = int(((counts >= lo) & (counts <= hi)).sum())
            bucket_label = f"{lo}-{hi}"
        pct = 100 * n / len(patient_ids) if len(patient_ids) else 0.0
        print(f"    {bucket_label:>10s} labs: {n:>4d} patient(s)  ({pct:5.1f}%)")


class LabsDataset(Dataset):
    def __init__(self, metadata, labs_df, loinc_map, labs_metadata):
        retained_ids = set(labs_metadata.loc[labs_metadata["retained"], "lab_id"])
        loinc_to_lab_id = loinc_map.dropna(subset=["loinc_code", "lab_id"]).set_index("loinc_code")["lab_id"].to_dict()

        labs = labs_df.copy()

        if "date" not in labs.columns and "start" in labs.columns:
            labs = labs.rename(columns={"start": "date"})
        if "loinc_code" not in labs.columns and "code" in labs.columns:
            labs = labs.rename(columns={"code": "loinc_code"})
        if labs["loinc_code"].str.startswith("LOINC/").any():
            labs["loinc_code"] = labs["loinc_code"].str.replace("LOINC/", "", regex=False)

        labs["date"] = pd.to_datetime(labs["date"])
        labs = labs.dropna(subset=["value"])

        # --- Diagnostic: is labs_df scoped to this cohort, or a bigger population? ---
        # patient_id dtype mismatches (e.g. int vs str) silently zero out every join below,
        # which looks identical in the final counts to "no labs for this patient" — check first.
        meta_pid_dtype, labs_pid_dtype = metadata["patient_id"].dtype, labs["patient_id"].dtype
        if meta_pid_dtype != labs_pid_dtype:
            print(f"WARNING: patient_id dtype mismatch — metadata={meta_pid_dtype}, labs_df={labs_pid_dtype}. "
                  f"This can silently zero out every patient match below; comparing as strings.")
        meta_pids = set(metadata["patient_id"].astype(str))
        labs_pids_all = set(labs["patient_id"].astype(str))
        n_meta_pids_in_labs = len(meta_pids & labs_pids_all)
        print(f"Cohort coverage: labs_df has {len(labs_pids_all)} unique patient_id(s) total "
              f"({len(labs)} rows); {n_meta_pids_in_labs}/{len(meta_pids)} metadata patient_id(s) "
              f"appear anywhere in labs_df.")
        if len(labs_pids_all) > len(meta_pids) * 2:
            print(f"NOTE: labs_df has {len(labs_pids_all)} patients but metadata only has {len(meta_pids)} — "
                  f"labs_df looks like it was NOT pre-filtered to this cohort; row/patient counts below "
                  f"(e.g. 'Lab rows: X/Y') are population-wide, not per-case.")

        # Distribution BEFORE any content filtering — just linked to our cohort's patient_ids.
        _print_lab_count_histogram(
            "Raw labs per patient (linked to cohort, no LOINC/model filtering yet)",
            meta_pids, labs[labs["patient_id"].astype(str).isin(meta_pids)],
        )

        # --- Diagnostic: implausible sentinel values (e.g. 9999999) that aren't real results ---
        numeric_vals = pd.to_numeric(labs["value"], errors="coerce")
        sentinel_mask = numeric_vals.abs() >= SENTINEL_VALUE_THRESHOLD
        n_sentinel = int(sentinel_mask.sum())
        if n_sentinel:
            sentinel_by_code = labs.loc[sentinel_mask, "loinc_code"].value_counts()
            print(f"WARNING: dropping {n_sentinel} row(s) with |value| >= {SENTINEL_VALUE_THRESHOLD} "
                  f"(likely placeholder/error codes, not real results):")
            for code, n in sentinel_by_code.items():
                print(f"  {code}  ({n} rows)")
            labs = labs[~sentinel_mask]

        n_total_rows = len(labs)
        input_codes = set(labs["loinc_code"].dropna())
        matched_codes = input_codes.intersection(loinc_to_lab_id)
        unmatched_codes = input_codes - matched_codes
        rows_per_code = labs.groupby("loinc_code").size()

        labs["lab_id"] = labs["loinc_code"].map(loinc_to_lab_id)
        n_loinc_matched = labs["lab_id"].notna().sum()
        labs = labs.dropna(subset=["lab_id"])
        labs["lab_id"] = labs["lab_id"].astype(int)
        labs = labs[labs["lab_id"].isin(retained_ids)]
        n_retained = len(labs)

        loinc_name = loinc_map.groupby("loinc_code")["name"].apply(lambda x: ", ".join(x.unique()))
        print(f"LOINC mapping: {len(matched_codes)}/{len(input_codes)} unique codes mapped")
        print(f"Lab rows: {n_loinc_matched}/{n_total_rows} mapped to a known LOINC, {n_retained} retained after model filter")
        print("Matched LOINC codes:")
        for code in sorted(matched_codes, key=lambda c: -rows_per_code.get(c, 0)):
            print(f"  {code}  {loinc_name.get(code, '')}  ({rows_per_code.get(code, 0)} rows)")
        if unmatched_codes:
            print(f"Unmatched LOINC codes ({len(unmatched_codes)}):")
            for code in sorted(unmatched_codes):
                print(f"  {code}  ({rows_per_code.get(code, 0)} rows)")

        # Distribution AFTER LOINC/model filtering — same buckets, same patients, so you can see
        # directly how much each patient's count shrank once only recognized/retained labs count.
        _print_lab_count_histogram(
            "Labs per patient AFTER LOINC/model filtering (before the 6-month window)",
            meta_pids, labs[labs["patient_id"].astype(str).isin(meta_pids)],
        )

        # Warn about likely unit mismatches: labs where >50% of values are >5σ from training mean
        meta_idx = labs_metadata.set_index("lab_id")
        flagged = []
        for lab_id, group in labs.groupby("lab_id"):
            row_meta = meta_idx.loc[lab_id]
            if not row_meta["is_numeric"] or row_meta["std"] < 1e-6:
                continue
            z = (pd.to_numeric(group["value"], errors="coerce") - row_meta["mean"]) / row_meta["std"]
            frac_outlier = (z.abs() > 5).mean()
            if frac_outlier > 0.25:
                flagged.append((row_meta["name"], lab_id, frac_outlier, row_meta["mean"], row_meta["std"]))
        if flagged:
            print(f"WARNING: {len(flagged)} lab(s) have >25% of values >5σ from training distribution — likely unit mismatch:")
            for name, lid, frac, mean, std in flagged:
                print(f"  {name} (lab_id={lid}): {frac*100:.0f}% outliers  [training mean={mean:.3g}, std={std:.3g}]")

        self.samples = []
        # Track *why* each sample did or didn't get labs, instead of a single pass/fail count.
        reason_no_labs_ever = []      # patient has zero retained lab rows anywhere in labs_df
        reason_outside_window = []    # patient has retained labs, but none in the 6mo pre-scan window
        nearest_lab_days_outside = [] # for reason_outside_window: distance (days) to nearest retained lab

        for _, row in metadata.iterrows():
            scan_date = pd.to_datetime(row["date"])
            window_start = scan_date - pd.DateOffset(months=6)

            all_patient_labs = labs[labs["patient_id"] == row["patient_id"]]
            patient_labs = all_patient_labs[
                (all_patient_labs["date"] <= scan_date) & (all_patient_labs["date"] >= window_start)
            ].copy()

            if len(patient_labs) == 0:
                if len(all_patient_labs) == 0:
                    reason_no_labs_ever.append(row["sample_id"])
                else:
                    reason_outside_window.append(row["sample_id"])
                    nearest_gap_days = (all_patient_labs["date"] - scan_date).abs().dt.days.min()
                    nearest_lab_days_outside.append(int(nearest_gap_days))
                continue

            # Quarterly downsample: last lab per (lab_id, quarter)
            patient_labs = patient_labs.sort_values("date")
            patient_labs = (
                patient_labs.groupby(["lab_id", pd.Grouper(key="date", freq="QS")])
                .last()
                .reset_index()
            )

            patient_labs["years_prior"] = (
                (scan_date - patient_labs["date"]).dt.total_seconds() / (365.25 * 24 * 3600)
            )

            lab_dicts = [
                {"lab_id": int(r["lab_id"]), "Value": r["value"], "years_prior": float(r["years_prior"])}
                for _, r in patient_labs.iterrows()
            ]
            self.samples.append((lab_dicts, row["sample_id"]))

        n_total = len(metadata)
        n_with_labs = len(self.samples)
        avg_labs = np.mean([len(s[0]) for s in self.samples]) if self.samples else 0
        print(f"Samples with labs in 6-month window: {n_with_labs}/{n_total} (avg {avg_labs:.1f} labs per sample)")
        print(f"  -> {len(reason_no_labs_ever)}/{n_total} sample(s): patient has ZERO retained/matched "
              f"labs anywhere in labs_df (no data at all, not a window issue)")
        if reason_outside_window:
            days = np.array(nearest_lab_days_outside)
            print(f"  -> {len(reason_outside_window)}/{n_total} sample(s): patient has retained labs, but "
                  f"none within 6 months of the scan. Nearest lab is min={days.min()}d "
                  f"median={int(np.median(days))}d max={days.max()}d away — "
                  f"if these are clustered close to 180d, widening the window would recover them.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def labs_collate(batch):
    lab_lists, sample_ids = zip(*batch)
    return list(lab_lists), list(sample_ids)
