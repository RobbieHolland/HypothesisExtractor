import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class TextDataset(Dataset):
    # Reads radiology report text files from paths.
    def __init__(self, paths, sample_ids):
        self.texts = [open(p).read() for p in paths]
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.sample_ids[idx]


class Qwen3Embedder(nn.Module):
    def __init__(self, hf_token=None, cache_dir=None):
        from transformers import AutoTokenizer, AutoModel
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-8B",
            padding_side="left",
            token=hf_token,
            cache_dir=cache_dir,
        )
        self.encoder = AutoModel.from_pretrained(
            "Qwen/Qwen3-Embedding-8B",
            token=hf_token,
            cache_dir=cache_dir,
        )
        self.max_length = 8192

    def forward(self, texts):
        # texts: list of strings
        texts_eos = [t + self.tokenizer.eos_token for t in texts]
        batch_dict = self.tokenizer(
            texts_eos,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.encoder.device)
        outputs = self.encoder(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        return F.normalize(embeddings, p=2, dim=1)
