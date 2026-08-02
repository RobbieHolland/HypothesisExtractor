"""
Apply pretrained SAE encoders to embeddings produced by run.py.
Outputs one CSV per modality: out/concept_activations_{modality}.csv
  columns: sample_id, Concept_0, Concept_1, ..., Concept_8191
"""
import os
import math
import torch
import yaml
import pandas as pd
from models.sae_model import apply_sae

MODALITY_EMBEDDING_FILE = {
    "findings": "report_embeddings.pt",
    "image": "ct_embeddings.pt",
    "labs": "lab_embeddings.pt",
}

# Tracked repo asset (Stanford train-split statistics), not part of the downloaded weight cache.
NORM_STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mappings", "sae_input_normalization.yaml")


def run_infer_sae(config):
    out_dir = config.paths.out_dir
    cache_dir = config.paths.get("sae_cache_dir", "data/mappings")

    with open(NORM_STATS_PATH) as f:
        norm_stats = yaml.safe_load(f)

    for modality, emb_file in MODALITY_EMBEDDING_FILE.items():
        emb_path = os.path.join(out_dir, emb_file)
        if not os.path.exists(emb_path):
            continue

        data = torch.load(emb_path, map_location="cpu", weights_only=True)
        embeddings = data["embeddings"]
        sample_ids = data["sample_ids"]

        s = norm_stats[modality]
        mean = torch.tensor(s["mean"], dtype=torch.float32)
        raw_norm = embeddings.float().norm(dim=1).mean().item()
        embeddings = ((embeddings.float() - mean) / s["std"]) * s["dim_scale"]

        # After normalization every modality should sit at ~sqrt(2048) regardless of its
        # dimensionality; a large deviation means these embeddings are off-distribution
        # relative to the data the SAE was trained on.
        target = math.sqrt(2048)
        norm_after = embeddings.norm(dim=1).mean().item()
        print(f"Normalized {modality} with Stanford train stats "
              f"(std={s['std']:.4g}, dim_scale={s['dim_scale']:.4g}): "
              f"mean row-norm {raw_norm:.1f} -> {norm_after:.1f} (expected ~{target:.1f}, "
              f"{norm_after / target - 1:+.0%})")
        if abs(norm_after / target - 1) > 0.35:
            print(f"  WARNING: {modality} embeddings are far from the SAE's training distribution — "
                  f"concept activations may be unreliable.")

        print(f"Applying SAE to {modality} embeddings {tuple(embeddings.shape)}...")
        acts = apply_sae(embeddings, modality, cache_dir=cache_dir)

        df = pd.DataFrame(acts.numpy(), columns=[f"Concept_{i}" for i in range(acts.shape[1])])
        df.insert(0, "sample_id", sample_ids)

        out_path = os.path.join(out_dir, f"concept_activations_{modality}.csv")
        df.to_csv(out_path, index=False)
        per_concept = (acts > 0).sum(dim=0)
        n_alive = int((per_concept > 0).sum())
        n_common = int((per_concept >= 25).sum())
        print(f"Saved {modality} concept activations: {df.shape}  "
              f"({n_alive} alive concepts, {n_common} firing in >=25 samples)  → {out_path}")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="config", config_name="config", version_base=None)
    def main(cfg: DictConfig):
        run_infer_sae(cfg)

    main()
