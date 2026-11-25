import numpy as np
import pandas as pd


def load_psd_excel_series(
    filepath: str,
    sheet_name: str = "1 Type of sale",
    metric_code: str = "1.1",
    start_col: int = 12,
) -> pd.Series:
    raw = pd.read_excel(filepath, sheet_name=sheet_name, header=None)
    code_col = raw.iloc[:, 0].astype(str).str.strip()
    target_rows = raw.index[code_col == str(metric_code)].tolist()
    if not target_rows:
        raise ValueError(f"Показник '{metric_code}' не знайдено на аркуші '{sheet_name}'.")
    row_idx = target_rows[0]

    period_row = None
    for r in range(row_idx):
        sample = raw.iloc[r, start_col:].dropna().astype(str)
        if any("-Q" in val for val in sample):
            period_row = r
            break
    if period_row is None:
        raise ValueError("Не вдалося визначити рядок квартальних періодів.")

    valid = raw.iloc[period_row, start_col:].notna() & raw.iloc[row_idx, start_col:].notna()
    periods = raw.iloc[period_row, start_col:][valid].values.astype(str)
    values = raw.iloc[row_idx, start_col:][valid].values.astype(float)
    label = str(raw.iloc[row_idx, 1]).strip()
    return pd.Series(values, index=periods, name=label)