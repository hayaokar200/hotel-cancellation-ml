"""Everything about the raw data: load it, clean it, split it by time.

Flow:  load_raw → drop_unused_columns → add_dates → temporal_split → split_xy
"""
import pandas as pd

from hotel_cancel import config


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the hotel bookings CSV.  (notebook cell 2)"""
    ...


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leaky and unused columns listed in config.  (cells 12, 20)"""
    ...


def add_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Add arrival_date (from year/month/day) and booking_date = arrival_date - lead_time.  (cell 33)"""
    ...


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by booking_date and return (train, val, test) with no time overlap.  (cell 34)"""
    ...


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features X from the target y, dropping the helper date columns.  (cell 35)"""
    ...
