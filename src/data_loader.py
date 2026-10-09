from pathlib import Path
from typing import BinaryIO

import pandas as pd


SAMPLE_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "healthcare_dataset.csv"


def load_csv(source: str | Path | BinaryIO) -> pd.DataFrame:
    """Load a CSV from a path or file-like object."""
    return pd.read_csv(source)


def load_sample_data() -> pd.DataFrame:
    """Load the bundled demonstration dataset."""
    return load_csv(SAMPLE_DATA_PATH)
