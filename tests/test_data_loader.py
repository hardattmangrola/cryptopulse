"""
Unit tests for shared/data_loader.py
Validates CSV schema checks, error reporting on missing files, and data loading.
"""

import pytest
import pandas as pd
from pathlib import Path
from shared.data_loader import validate_csv_file, inspect_dataset_summary, load_historical_dataframe


def test_validate_csv_file_on_existing_dataset():
    """Verify that the project's historical dataset validates successfully."""
    result = validate_csv_file()
    assert result["status"] == "VALID"
    assert result["file_size_mb"] > 0
    assert "Timestamp" in result["columns"]
    assert "Close" in result["columns"]
    assert len(result["sample_records"]) > 0


def test_validate_csv_file_missing_file(tmp_path):
    """Verify FileNotFoundError is raised with helpful message when CSV is absent."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        validate_csv_file(non_existent)
    assert "Historical dataset not found" in str(exc_info.value)


def test_validate_csv_file_missing_columns(tmp_path):
    """Verify ValueError is raised when required columns are missing."""
    bad_csv = tmp_path / "bad_columns.csv"
    bad_csv.write_text("Timestamp,Open,Close\n1600000000,100,105\n")
    with pytest.raises(ValueError) as exc_info:
        validate_csv_file(bad_csv)
    assert "Missing expected columns" in str(exc_info.value)


def test_validate_csv_file_non_numeric_data(tmp_path):
    """Verify ValueError is raised when numeric columns contain invalid strings."""
    invalid_csv = tmp_path / "invalid_data.csv"
    invalid_csv.write_text("Timestamp,Open,High,Low,Close,Volume\nnot_a_ts,1,2,0.5,1.5,10\n")
    with pytest.raises(ValueError) as exc_info:
        validate_csv_file(invalid_csv)
    assert "must contain numeric" in str(exc_info.value)


def test_load_historical_dataframe_with_limit():
    """Verify historical dataframe can be loaded with nrows limit and correct types."""
    df = load_historical_dataframe(nrows=50)
    assert len(df) == 50
    assert list(df.columns) == ["Timestamp", "Open", "High", "Low", "Close", "Volume"]
    # Verify chronological sorting
    assert (df["Timestamp"].diff().dropna() >= 0).all()
