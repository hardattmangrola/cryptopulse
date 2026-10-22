"""
CryptoPulse - Dataset Loader and Schema Validator
Provides functions to safely validate and inspect the historical Bitcoin 1-minute CSV dataset
without loading hundreds of megabytes into memory at once.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

from shared.config import HISTORICAL_DATA_PATH, EXPECTED_CSV_COLUMNS


def validate_csv_file(file_path: Optional[Path | str] = None) -> Dict[str, Any]:
    """
    Validates that the historical CSV dataset exists and matches the expected schema.

    Args:
        file_path: Optional path to the CSV file. Defaults to HISTORICAL_DATA_PATH from config.

    Returns:
        dict: Summary containing file existence status, columns, and sample preview.

    Raises:
        FileNotFoundError: If the CSV file does not exist at the specified path.
        ValueError: If required columns are missing or file is empty.
    """
    target_path = Path(file_path) if file_path else HISTORICAL_DATA_PATH

    # 1. Check file existence
    if not target_path.exists() or not target_path.is_file():
        # Only fallback if caller did not supply a specific explicit path
        if file_path is None:
            from shared.config import BASE_DIR
            candidates = [
                BASE_DIR / "btcusd_1-min_data.csv",
                Path("/app/btcusd_1-min_data.csv"),
                BASE_DIR / "app" / "btcusd_1-min_data.csv",
                Path("btcusd_1-min_data.csv").resolve(),
            ]
            for cand in candidates:
                if cand.exists() and cand.is_file():
                    target_path = cand
                    break
        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError(
                f"Historical dataset not found at: '{target_path.resolve()}'. "
                "Please ensure 'btcusd_1-min_data.csv' is placed in the project root."
            )

    # 2. Check file size
    file_size_bytes = target_path.stat().st_size
    if file_size_bytes == 0:
        raise ValueError(f"Historical dataset file at '{target_path.resolve()}' is empty (0 bytes).")

    # 3. Read only the header and first 5 rows to avoid loading huge file into RAM
    try:
        sample_df = pd.read_csv(target_path, nrows=5)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV at '{target_path.resolve()}': {exc}") from exc

    found_columns = list(sample_df.columns)

    # 4. Validate columns
    missing_columns = [col for col in EXPECTED_CSV_COLUMNS if col not in found_columns]
    if missing_columns:
        raise ValueError(
            f"Historical CSV schema mismatch! "
            f"Missing expected columns: {missing_columns}. "
            f"Found columns in file: {found_columns}. "
            f"Expected columns: {EXPECTED_CSV_COLUMNS}."
        )

    # 5. Check data types for sample rows
    if not pd.api.types.is_numeric_dtype(sample_df["Timestamp"]):
        raise ValueError("Column 'Timestamp' must contain numeric unix epoch timestamps.")

    for price_col in ["Open", "High", "Low", "Close", "Volume"]:
        if not pd.api.types.is_numeric_dtype(sample_df[price_col]):
            raise ValueError(f"Column '{price_col}' must contain numeric price/volume data.")

    return {
        "file_path": str(target_path.resolve()),
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "columns": found_columns,
        "sample_records": sample_df.to_dict(orient="records"),
        "status": "VALID"
    }


def inspect_dataset_summary(file_path: Optional[Path | str] = None, nrows: int = 1000) -> Dict[str, Any]:
    """
    Performs a lightweight inspection of the first and last records of the dataset.

    Args:
        file_path: Path to the CSV file.
        nrows: Number of rows to inspect from start.

    Returns:
        dict: Information on start/end timestamps, date range, and null value counts.
    """
    validation_info = validate_csv_file(file_path)
    target_path = Path(validation_info["file_path"])

    # Load head sample
    head_df = pd.read_csv(target_path, nrows=nrows)
    head_nulls = head_df.isnull().sum().to_dict()

    start_ts = int(head_df["Timestamp"].iloc[0])
    start_utc = pd.to_datetime(start_ts, unit="s", utc=True).isoformat()

    # Read last line efficiently by seeking from end of file
    tail_record = None
    with open(target_path, "rb") as f:
        # Seek near end of file (last 2KB is plenty for the final CSV lines)
        f.seek(max(0, target_path.stat().st_size - 2048))
        tail_lines = [line.decode("utf-8", errors="ignore").strip() for line in f.readlines()]
        non_empty = [line for line in tail_lines if line]
        if non_empty:
            last_line = non_empty[-1]
            parts = last_line.split(",")
            if len(parts) == len(EXPECTED_CSV_COLUMNS):
                try:
                    end_ts = int(float(parts[0]))
                    end_utc = pd.to_datetime(end_ts, unit="s", utc=True).isoformat()
                    tail_record = {
                        "timestamp": end_ts,
                        "timestamp_utc": end_utc,
                        "close": float(parts[4]),
                        "volume": float(parts[5])
                    }
                except ValueError:
                    tail_record = None

    return {
        "file_path": validation_info["file_path"],
        "file_size_mb": validation_info["file_size_mb"],
        "columns": validation_info["columns"],
        "start_timestamp": start_ts,
        "start_utc": start_utc,
        "tail_record": tail_record,
        "head_sample_null_counts": head_nulls
    }


def load_historical_dataframe(file_path: Optional[Path | str] = None, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Loads historical Bitcoin data from the CSV file into a pandas DataFrame.
    Converts 'Timestamp' to UTC datetime and sorts chronologically.

    Args:
        file_path: Path to CSV file.
        nrows: Optional row limit for fast loading or development testing.

    Returns:
        pd.DataFrame: Cleaned DataFrame with sorted datetime index or Timestamp column.
    """
    target_path = Path(file_path) if file_path else HISTORICAL_DATA_PATH
    validate_csv_file(target_path)

    df = pd.read_csv(target_path, nrows=nrows)

    # Ensure chronological order
    df = df.sort_values(by="Timestamp").reset_index(drop=True)

    # Drop any rows with NaN values if any exist
    df = df.dropna().reset_index(drop=True)

    return df
