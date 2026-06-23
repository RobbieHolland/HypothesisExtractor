import os
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

HF_REPO = "RobbieHolland/HypothesisExtractor"
SAE_FILES = {
    "findings": "sae_findings.ckpt",
    "image": "sae_image.ckpt",
    "labs": "sae_labs.ckpt",
}
TOP_K = 40


def _load_sae_weights(modality, cache_dir="data/sae"):
    filename = SAE_FILES[modality]
    local_path = os.path.join(cache_dir, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from {HF_REPO}...")
        hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=cache_dir)
    return torch.load(local_path, map_location="cpu", weights_only=True)


def apply_sae(embeddings, modality, cache_dir="data/sae"):
    """
    Apply SAE encoder to embeddings and return sparse concept activations.
    embeddings: Tensor[N, D]
    returns: Tensor[N, 8192] with at most TOP_K non-zero values per row
    """
    weights = _load_sae_weights(modality, cache_dir)
    W_enc = weights["W_enc"].float()   # [D, 8192]
    b_enc = weights["b_enc"].float()   # [8192]
    b_dec = weights["b_dec"].float()   # [D]

    x = embeddings.float()
    acts = F.relu((x - b_dec) @ W_enc + b_enc)   # [N, 8192]

    # TopK sparsification
    topk_vals, topk_idx = acts.topk(TOP_K, dim=-1)
    sparse = torch.zeros_like(acts)
    sparse.scatter_(-1, topk_idx, topk_vals)
    return sparse
