import torch
import torch.nn as nn
from torch.utils.data import Dataset
from monai.transforms import (
    CenterSpatialCropd, Compose, EnsureChannelFirstd,
    LoadImaged, Orientationd, ScaleIntensityRanged,
    Spacingd, SpatialPadd, ToTensord,
)

# Matches Merlin's preprocessing pipeline exactly
_ImageTransforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode="bilinear"),
    ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
    SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
    CenterSpatialCropd(keys=["image"], roi_size=[224, 224, 160]),
    ToTensord(keys=["image"]),
])


class CTDataset(Dataset):
    # Loads NIfTI CT scans from paths and applies Merlin's preprocessing transforms.
    def __init__(self, paths, sample_ids):
        self.paths = paths
        self.sample_ids = sample_ids
        self.transform = _ImageTransforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        item = {"image": self.paths[idx]}
        item = self.transform(item)
        return item["image"], self.sample_ids[idx]


class MerlinEmbedder(nn.Module):
    def __init__(self):
        from merlin import Merlin
        super().__init__()
        self.model = Merlin(ImageEmbedding=True)
        self.model.eval()

    def forward(self, images):
        # images: [B, 1, 224, 224, 160]
        return self.model(images)
