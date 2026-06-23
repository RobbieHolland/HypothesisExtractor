"""
Apply pretrained SAE encoders to embeddings produced by run.py.
Outputs one CSV per modality: out/concept_activations_{modality}.csv
  columns: sample_id, Concept_0, Concept_1, ..., Concept_8191
"""
import os
import torch
import pandas as pd
from models.sae_model import apply_sae

MODALITY_EMBEDDING_FILE = {
    "findings": "report_embeddings.pt",
    "image": "ct_embeddings.pt",
    "labs": "lab_embeddings.pt",
}


def run_infer_sae(config):
    out_dir = config.paths.out_dir
    cache_dir = config.paths.get("sae_cache_dir", "data/mappings")

    for modality, emb_file in MODALITY_EMBEDDING_FILE.items():
        emb_path = os.path.join(out_dir, emb_file)
        if not os.path.exists(emb_path):
            continue

        data = torch.load(emb_path, map_location="cpu", weights_only=True)
        embeddings = data["embeddings"]
        sample_ids = data["sample_ids"]

        print(f"Applying SAE to {modality} embeddings {tuple(embeddings.shape)}...")
        acts = apply_sae(embeddings, modality, cache_dir=cache_dir)

        df = pd.DataFrame(acts.numpy(), columns=[f"Concept_{i}" for i in range(acts.shape[1])])
        df.insert(0, "sample_id", sample_ids)

        out_path = os.path.join(out_dir, f"concept_activations_{modality}.csv")
        df.to_csv(out_path, index=False)
        n_alive = (acts > 0).any(dim=0).sum().item()
        print(f"Saved {modality} concept activations: {df.shape}  ({n_alive} alive concepts)  → {out_path}")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="config", config_name="config", version_base=None)
    def main(cfg: DictConfig):
        run_infer_sae(cfg)

    main()
