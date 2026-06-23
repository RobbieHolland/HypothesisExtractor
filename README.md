# HypothesisExtractor

Extracts foundation model embeddings and longitudinal outcome labels from CT imaging, radiology report, and laboratory data.

## Requirements

- Python 3.10+
- If running batch size ≥24, a GPU with ≥40 GB VRAM is needed. Try reduced batch size otherwise.
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
| `ct_path` | Path to the NIfTI CT file for this sample *(optional)* |
| `report_path` | Path to the radiology report text file for this sample *(optional)* |

Each modality is only processed when its path column is present.

### diagnoses.csv
One row per ICD code event:

| Column | Description |
|--------|-------------|
| `patient_id` | Patient identifier, matching `metadata.csv` |
| `date` | Date the ICD code was recorded |
| `icd9` | ICD-9 code (leave blank if ICD-10) |
| `icd10` | ICD-10 code (leave blank if ICD-9) |

### labs.csv *(optional)*
One row per lab result:

| Column | Description |
|--------|-------------|
| `patient_id` | Patient identifier, matching `metadata.csv` |
| `date` | Date the lab was collected |
| `loinc_code` | LOINC code identifying the lab test |
| `value` | Numeric result value |

Set `paths.labs_csv` in `config/config.yaml` to enable lab embedding extraction. The LOINC mapping and pretrained encoder are bundled in `data/mappings/` and require no configuration. The column names `start` and `code` are accepted as aliases for `date` and `loinc_code`, and the `LOINC/` prefix on codes is stripped automatically.

Labs are filtered to the 6-month window prior to each scan date and quarterly-downsampled before embedding. Samples with no labs in that window are omitted from `lab_embeddings.pt`.

## Running

```bash
python run.py
```

By default this runs `embeddings` and `outcomes`. To run additional steps, override `steps` on the command line:

```bash
python run.py steps=[embeddings,outcomes,sae,autorate]   # full pipeline
python run.py steps=[sae,autorate]                       # SAE + AutoRate only (requires embeddings already computed)
```

**Steps:**
- `embeddings` — extract CT, report, and lab embeddings
- `outcomes` — compute survival outcome labels from ICD codes
- `sae` — apply pretrained Sparse Autoencoders to embeddings, producing sparse concept activations per sample. SAE weights are auto-downloaded from HuggingFace on first run.
- `autorate` — for each of 342 pre-generated concept interpretations, sample high- and low-activating patients, ask Vertex AI (`gemini-2.5-pro`) to discriminate between them using the interpretation text, and record the accuracy. This validates how well the interpretations transfer to your dataset.

For `autorate`, set `paths.vertex_project` in `config/config.yaml` to your GCP project ID.

## Outputs

All outputs are written to `out/` (configurable via `paths.out_dir`):

- **`ct_embeddings.pt`** — `{'embeddings': Tensor[N, 2048], 'sample_ids': [...]}`
- **`report_embeddings.pt`** — `{'embeddings': Tensor[N, 4096], 'sample_ids': [...]}`
- **`lab_embeddings.pt`** — `{'embeddings': Tensor[N, 768], 'sample_ids': [...]}` *(N may be less than total samples if some have no labs in window)*
- **`outcomes.pkl`** — DataFrame with one row per sample. For each task: `{task}_event` (0/1) and `{task}_time` (days from scan date to first event, or to last recorded ICD date if no event). Times include Gaussian anonymization jitter (σ=3 days).
- **`concept_activations_{modality}.csv`** — sparse SAE activations (`sample_id` + `Concept_0`…`Concept_8191`), one file per modality present. *(requires `sae` step)*
- **`autointerp_results.csv`** — one row per concept with `inputs`, `feature_name`, `top_k`, `matryoshka`, `extracted_interpretation`, and `mayo_interpretation_discrimination_accuracy`. *(requires `autorate` step)*
- **`autointerp_top_samples.csv`** — top-3 highest-activating samples per concept with their report text or formatted labs. *(requires `autorate` step)*

ICD-to-task mappings are in `data/mappings/`.

## Testing

```bash
pytest tests/
```

Mock CT volumes are generated automatically in `data/mock/ct/` on first test run.
