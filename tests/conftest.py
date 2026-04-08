import os
import pytest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOCK_CT_DIR = os.path.join(REPO_ROOT, "data", "mock", "ct")


@pytest.fixture(scope="session", autouse=True)
def create_mock_niftis():
    # Generate small NIfTI volumes for testing (32x32x32, HU range).
    # Merlin's transforms will resample and pad to 224x224x160.
    import nibabel as nib

    os.makedirs(MOCK_CT_DIR, exist_ok=True)
    for sid in ["S001", "S002", "S003", "S004", "S005"]:
        path = os.path.join(MOCK_CT_DIR, f"{sid}.nii.gz")
        if not os.path.exists(path):
            data = (np.random.rand(32, 32, 32).astype(np.float32) * 2000) - 1000
            nib.save(nib.Nifti1Image(data, np.eye(4)), path)


@pytest.fixture(scope="session")
def mock_nifti_path():
    return os.path.join(MOCK_CT_DIR, "S001.nii.gz")


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT
