# HypothesisExtractor

Extracts foundation model embeddings and longitudinal outcome labels from CT imaging and radiology report data.

## Requirements

- Python 3.10+
- GPU with ≥40 GB VRAM
- [Merlin](https://github.com/StanfordMIMI/Merlin): `pip install merlin-vlm`
- A HuggingFace token with access to `Qwen/Qwen3-Embedding-8B`

## Installation

```bash
git clone https://github.com/RobbieHolland/HypothesisExtractor
cd HypothesisExtractor
pip install -r requirements.txt
pip install merlin-vlm
```

## Configuration

Edit `config/config.yaml`:

- `paths.metadata_csv` / `paths.diagnoses_csv`: paths to your input CSVs (see below)
- `huggingface_token`: your HuggingFace API token
- `batch_size`: batch size for embedding extraction
- `trainer.accelerator`: `"auto"` to use GPU automatically

## Input CSVs

### metadata.csv
One row per sample:

| Column | Description |
|--------|-------------|
| `sample_id` | Unique string identifier for each sample |
| `patient_id` | Patient identifier (patients may have multiple samples) |
| `date` | Acquisition date (any parseable date string) |
| `ct_path` | Path to the NIfTI CT file for this sample |
| `report_path` | Path to the radiology report text file for this sample |

### diagnoses.csv
One row per ICD code event:

| Column | Description |
|--------|-------------|
| `patient_id` | Patient identifier, matching `metadata.csv` |
| `date` | Date the ICD code was recorded |
| `icd9` | ICD-9 code (leave blank if ICD-10) |
| `icd10` | ICD-10 code (leave blank if ICD-9) |

## Running

```bash
python run.py
```

## Outputs

All outputs are written to `out/` (configurable via `paths.out_dir`):

- **`ct_embeddings.pt`** — `{'embeddings': Tensor[N, M], 'sample_ids': [...]}`
- **`report_embeddings.pt`** — `{'embeddings': Tensor[N, M], 'sample_ids': [...]}`
- **`outcomes.pkl`** — DataFrame with one row per sample. For each task: `{task}_event` (0/1) and `{task}_time` (days from scan date to first event, or to last recorded ICD date if no event). Times include Gaussian anonymization jitter (σ=3 days).

ICD-to-task mappings are in `data/mappings/`.

## Testing

```bash
pytest tests/
```

Mock CT volumes are generated automatically in `data/mock/ct/` on first test run.
