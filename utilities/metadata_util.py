import pandas as pd


def load_metadata(metadata_csv):
    return pd.read_csv(metadata_csv, dtype={"date": str}).drop_duplicates(subset="sample_id", keep="first")
