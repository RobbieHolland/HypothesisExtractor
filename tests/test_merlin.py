import torch
from unittest.mock import patch, MagicMock


def test_ct_dataset_shape(mock_nifti_path):
    from models.merlin_model import CTDataset

    dataset = CTDataset([mock_nifti_path], ["S001"])
    image, sid = dataset[0]
    assert image.shape == (1, 224, 224, 160)
    assert sid == "S001"


def test_merlin_embedder_output_shape():
    with patch("merlin.Merlin") as MockMerlin:
        mock_model = MagicMock(side_effect=lambda x: torch.randn(x.shape[0], 512))
        MockMerlin.return_value = mock_model

        from models.merlin_model import MerlinEmbedder

        embedder = MerlinEmbedder()
        x = torch.randn(2, 1, 224, 224, 160)
        out = embedder(x)
        assert out.shape == (2, 512)
