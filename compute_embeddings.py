import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, Callback

from models.merlin_model import CTDataset, MerlinEmbedder
from models.qwen3_model import TextDataset, Qwen3Embedder


class EmbeddingExtractor(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def predict_step(self, batch, batch_idx):
        inputs, sample_ids = batch
        with torch.no_grad():
            embeddings = self.model(inputs)
        return {"embeddings": embeddings.cpu(), "sample_ids": list(sample_ids)}


class CollectCallback(Callback):
    def __init__(self):
        self.embeddings = []
        self.sample_ids = []

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.embeddings.append(outputs["embeddings"])
        self.sample_ids.extend(outputs["sample_ids"])


def extract(model, dataset, batch_size, accelerator="auto"):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    extractor = EmbeddingExtractor(model)
    collector = CollectCallback()
    trainer = Trainer(
        accelerator=accelerator,
        devices=1,
        callbacks=[collector],
        logger=False,
        enable_progress_bar=False,
    )
    trainer.predict(extractor, loader)
    embeddings = torch.cat(collector.embeddings, dim=0)
    return {"embeddings": embeddings, "sample_ids": collector.sample_ids}


class EmbeddingComputer:
    def __init__(self, config):
        self.config = config

    def run(self):
        os.makedirs(self.config.paths.out_dir, exist_ok=True)
        metadata = pd.read_csv(self.config.paths.metadata_csv)
        acc = self.config.trainer.accelerator

        if "ct_path" in metadata.columns:
            model = MerlinEmbedder()
            dataset = CTDataset(metadata["ct_path"].tolist(), metadata["sample_id"].tolist())
            result = extract(model, dataset, self.config.batch_size, acc)
            torch.save(result, os.path.join(self.config.paths.out_dir, "ct_embeddings.pt"))
            print(f"Saved CT embeddings: {result['embeddings'].shape}")

        if "report_path" in metadata.columns:
            model = Qwen3Embedder(hf_token=self.config.get("huggingface_token"))
            dataset = TextDataset(metadata["report_path"].tolist(), metadata["sample_id"].tolist())
            result = extract(model, dataset, self.config.batch_size, acc)
            torch.save(result, os.path.join(self.config.paths.out_dir, "report_embeddings.pt"))
            print(f"Saved report embeddings: {result['embeddings'].shape}")
