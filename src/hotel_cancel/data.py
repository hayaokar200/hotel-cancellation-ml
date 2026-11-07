"""Everything about the raw data: load it, clean it, split it by time.

Flow:  load_raw → drop_unused_columns → add_dates → temporal_split → split_xy
"""
import pandas as pd

from hotel_cancel import config


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    """Read the hotel bookings CSV.  (notebook cell 2)"""
    df = pd.read_csv(path)
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove leaky and unused columns listed in config.  (cells 12, 20)"""
    df = df.drop(columns=config.LEAKY_COLUMNS + config.OTHER_DROPPED_COLUMNS )
    return df


def add_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Add arrival_date (from year/month/day) and booking_date = arrival_date - lead_time.  (cell 33)"""
    
    df["arrival_date"] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + "-" +
        df["arrival_date_month"].astype(str) + "-" +
        df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d"
    )
    df['booking_date'] = (df['arrival_date'] - pd.to_timedelta(df["lead_time"], unit="D"))
    return df


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sort by booking_date and return (train, val, test) with no time overlap.  (cell 34)"""
    df = df.sort_values("booking_date").reset_index(drop=True)
    n = len(df)

    train_end = int(n*config.TRAIN_FRACTION)
    val_end = int(n*(config.TRAIN_FRACTION+config.VAL_FRACTION))

    train_cutoff = df.loc[train_end,"booking_date"]
    val_cutoff = df.loc[val_end,"booking_date"]

    train_df = df[df['booking_date'] < train_cutoff].copy()
    val_df = df[
        (df["booking_date"] >= train_cutoff) &
        (df["booking_date"] < val_cutoff)
    ].copy()
    test_df = df[df['booking_date'] >= val_cutoff].copy()
    return train_df,val_df,test_df



def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features X from the target y, dropping the helper date columns.  (cell 35)"""
    df = df.drop(columns=config.DATE_COLUMNS)
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    return X,y