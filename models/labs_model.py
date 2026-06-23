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
        for _, row in metadata.iterrows():
            scan_date = pd.to_datetime(row["date"])
            window_start = scan_date - pd.DateOffset(months=6)

            patient_labs = labs[labs["patient_id"] == row["patient_id"]].copy()
            patient_labs = patient_labs[
                (patient_labs["date"] <= scan_date) & (patient_labs["date"] >= window_start)
            ]

            if len(patient_labs) == 0:
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def labs_collate(batch):
    lab_lists, sample_ids = zip(*batch)
    return list(lab_lists), list(sample_ids)
