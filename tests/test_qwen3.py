import os
import torch
from unittest.mock import patch, MagicMock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_text_dataset():
    from models.qwen3_model import TextDataset

    report_path = os.path.join(REPO_ROOT, "data", "mock", "reports", "S001.txt")
    dataset = TextDataset([report_path], ["S001"])
    text, sid = dataset[0]
    assert isinstance(text, str) and len(text) > 0
    assert sid == "S001"


def test_qwen3_embedder_output_shape():
    # Patch AutoTokenizer and AutoModel at the source they're imported from inside __init__
    with patch("transformers.AutoTokenizer") as MockTok, \
         patch("transformers.AutoModel") as MockModel:

        mock_tok_instance = MagicMock()
        mock_tok_instance.eos_token = "<eos>"
        mock_tok_output = MagicMock()
        mock_tok_output.to.return_value = {
            "input_ids": torch.zeros(2, 10, dtype=torch.long),
            "attention_mask": torch.ones(2, 10, dtype=torch.long),
        }
        mock_tok_instance.return_value = mock_tok_output
        MockTok.from_pretrained.return_value = mock_tok_instance

        mock_enc_instance = MagicMock()
        mock_enc_instance.device = torch.device("cpu")
        mock_enc_output = MagicMock()
        mock_enc_output.last_hidden_state = torch.randn(2, 10, 4096)
        mock_enc_instance.return_value = mock_enc_output
        MockModel.from_pretrained.return_value = mock_enc_instance

        from models.qwen3_model import Qwen3Embedder

        embedder = Qwen3Embedder()
        out = embedder(["report one", "report two"])
        assert out.shape == (2, 4096)
