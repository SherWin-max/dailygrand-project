# ------------------------------------------------------------
# Daily Grand Descriptive Window Report (NO prediction)
# ------------------------------------------------------------
# This script generates a purely descriptive statistical report
# for a selected date window of Daily Grand results.
#
# IMPORTANT:
# - This is NOT a prediction system.
# - Lottery draws are independent and random.
# - Outputs show historical counts/distributions only.
#
# Input columns required:
# DATE, NUMBER1..NUMBER5, GRAND NUMBER
#
# Outputs:
# - Excel file with multiple sheets (raw window data + analytics)
# - Terminal prints analytics sheets (raw window data is not printed)
# ------------------------------------------------------------

import os
import sys
import math
import itertools
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np

# -----------------------------
# Constants / Column names
# -----------------------------
DATE_COL = "DATE"
N1 = "NUMBER1"
N2 = "NUMBER2"
N3 = "NUMBER3"
N4 = "NUMBER4"
N5 = "NUMBER5"
GRAND_COL = "GRAND NUMBER"

MAIN_COLS = [N1, N2, N3, N4, N5]

# High/Low split definition (you can change if you want)
LOW_MAX = 24  # 1..24 low, 25..49 high

# Excel sheet names (kept stable for future comparison)
SHEET_META = "00_META"
SHEET_VALIDATION = "01_VALIDATION"
SHEET_WINDOW_DATA = "02_WINDOW_DATA"
SHEET_MAIN_STATS = "03_MAIN_STATS_1_49"
SHEET_MAIN_STATS_SORTED = "03B_MAIN_STATS_SORTED_BY_FREQ"
SHEET_HOT_TOP10 = "04_HOT_TOP10"
SHEET_COLD_BOTTOM10 = "05_COLD_BOTTOM10"
SHEET_GRAND_STATS = "07_GRAND_STATS_1_7"
SHEET_SUM_SPREAD_SUMMARY = "09_SUM_SPREAD_SUMMARY"
SHEET_PARITY_SIMPLE = "10_PARITY_HIGHLOW_SIMPLE"
SHEET_COMPOSITION_MATRIX = "11_COMPOSITION_MATRIX"
SHEET_OVERLAP_HIST = "12_CONSEC_OVERLAP_HIST"
SHEET_PAIRS = "13_PAIRS_THRESHOLD"


# ------------------------------------------------------------
# Utility: print full dataframe to terminal (not truncated)
# ------------------------------------------------------------
def print_df_full(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    if df is None or len(df) == 0:
        print("[EMPTY]")
        return

    # Print without truncation
    with pd.option_context(
        "display.max_rows", None,
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", None,
    ):
        print(df)


# ------------------------------------------------------------
# Core Data Pipeline
# ------------------------------------------------------------
def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw dataset from Excel.
    Expected columns:
    DATE, NUMBER1..NUMBER5, GRAND NUMBER
    """
    df = pd.read_excel(filepath)

    # Normalize column names (strip spaces)
    df.columns = [str(c).strip() for c in df.columns]

    return df


def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans dataset and validates ranges.
    Returns:
      clean_df: cleaned dataframe sorted by date
      validation_report: counts of ok / dropped reasons
    """
    required = [DATE_COL] + MAIN_COLS + [GRAND_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert date
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # Convert numbers to numeric
    for c in MAIN_COLS + [GRAND_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Validation checks
    reasons = []

    for idx, row in df.iterrows():
        reason_list = []

        if pd.isna(row[DATE_COL]):
            reason_list.append("invalid_date")

        # Main numbers must be integers 1..49
        main_vals = [row[c] for c in MAIN_COLS]
        if any(pd.isna(x) for x in main_vals):
            reason_list.append("null_main")
        else:
            # Must be in range
            if any((x < 1 or x > 49) for x in main_vals):
                reason_list.append("main_out_of_range")

        # Grand number must be integer 1..7
        g = row[GRAND_COL]
        if pd.isna(g):
            reason_list.append("null_grand")
        else:
            if g < 1 or g > 7:
                reason_list.append("grand_out_of_range")

        # If any validation errors, mark dropped
        if reason_list:
            reasons.append((idx, "dropped", ",".join(reason_list)))
        else:
            reasons.append((idx, "ok", ""))

    validation_df = pd.DataFrame(reasons, columns=["row_index", "status", "reason"])

    clean_df = df[validation_df["status"] == "ok"].copy()
    clean_df = clean_df.sort_values(DATE_COL).reset_index(drop=True)

    # Report counts
    report = (
        validation_df.groupby(["status", "reason"])
        .size()
        .reset_index(name="count")
        .sort_values(["status", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Add a dropped column for quick summary
    report["dropped"] = np.where(report["status"] == "dropped", report["count"], 0)

    return clean_df, report


def generate_analysis_window(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Filters df to a date window inclusive.
    """
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    window_df = df[(df[DATE_COL] >= start_dt) & (df[DATE_COL] <= end_dt)].copy()
    window_df = window_df.sort_values(DATE_COL).reset_index(drop=True)

    return window_df


def get_main_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts NUMBER1..NUMBER5 as an Nx5 integer matrix.
    """
    return df[MAIN_COLS].astype(int).to_numpy()


def get_grand_series(df: pd.DataFrame) -> pd.Series:
    """
    Extracts GRAND NUMBER as int series.
    """
    return df[GRAND_COL].astype(int)


# ------------------------------------------------------------
# Analytics Computation
# ------------------------------------------------------------

# Main number stats (1..49) for the selected window:
# - frequency: how many times the number appears in the window
# - freq_rate: frequency / (5 * number_of_draws)
# - last_seen_draw_index: last draw position (1..N) where it appeared
# - last_seen_date: last draw date where it appeared
# - current_gap_draws: draws since last appearance (N - last_seen_draw_index)
def compute_main_stats_1_49(main_matrix: np.ndarray, dates: pd.Series) -> pd.DataFrame:
    n_draws = main_matrix.shape[0]
    total_slots = n_draws * 5
    freq = {i: 0 for i in range(1, 50)}

    first_seen_index = {i: None for i in range(1, 50)}
    first_seen_date  = {i: None for i in range(1, 50)}

    last_seen_index = {i: None for i in range(1, 50)}
    last_seen_date  = {i: None for i in range(1, 50)}

    for i in range(n_draws):
        nums = main_matrix[i, :]
        draw_date = dates.iloc[i].date()

        for x in nums:
            x = int(x)
            freq[x] += 1

            # first seen
            if first_seen_index[x] is None:
                first_seen_index[x] = i + 1
                first_seen_date[x] = draw_date

            # last seen
            last_seen_index[x] = i + 1
            last_seen_date[x] = draw_date

    rows = []
    for num in range(1, 50):
        f = freq[num]
        rate = f / total_slots if total_slots > 0 else 0.0

        fsi = first_seen_index[num]
        fsd = first_seen_date[num]
        lsi = last_seen_index[num]
        lsd = last_seen_date[num]

        gap = n_draws if lsi is None else (n_draws - lsi)

        rows.append(
            {
                "number": num,
                "frequency": f,
                "freq_rate": round(rate, 4),

                "first_seen_draw_index": fsi if fsi is not None else 0,
                "first_seen_date": str(fsd) if fsd is not None else "",

                "last_seen_draw_index": lsi if lsi is not None else 0,
                "last_seen_date": str(lsd) if lsd is not None else "",

                "current_gap_draws": gap,
            }
        )

    df = pd.DataFrame(rows)
    return df



def compute_hot_top10(main_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Top 10 by frequency (historical frequency ranking).
    """
    df = main_stats.sort_values(["frequency", "number"], ascending=[False, True]).head(10).copy()
    return df[["number", "frequency", "freq_rate"]]


def compute_cold_bottom10(main_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Bottom 10 by frequency (historical frequency ranking).
    """
    df = main_stats.sort_values(["frequency", "number"], ascending=[True, True]).head(10).copy()
    return df[["number", "frequency", "freq_rate"]]


def compute_grand_stats_1_7(grand_series: pd.Series) -> pd.DataFrame:
    """
    Frequency + last seen + gap for Grand Number domain (1..7).
    """
    n_draws = len(grand_series)
    total = n_draws

    rows = []
    for g in range(1, 8):
        freq = int((grand_series == g).sum())
        rate = freq / total if total > 0 else 0.0

        # last seen index
        seen_idx = np.where(grand_series.to_numpy() == g)[0]
        if len(seen_idx) == 0:
            last_idx = 0
            gap = n_draws
        else:
            last_idx = int(seen_idx[-1]) + 1  # 1-indexed
            gap = n_draws - last_idx

        rows.append(
            {
                "grand_number": g,
                "frequency": freq,
                "freq_rate": round(rate, 3),
                "current_gap_draws": gap,
            }
        )

    df = pd.DataFrame(rows).sort_values("grand_number").reset_index(drop=True)
    return df


def compute_sum_spread_summary(main_matrix: np.ndarray) -> pd.DataFrame:
    """
    Aggregate stats for:
      - Sum of main numbers per draw
      - Spread (max-min) per draw
    """
    sums = main_matrix.sum(axis=1)
    spreads = main_matrix.max(axis=1) - main_matrix.min(axis=1)

    def summarize(arr: np.ndarray, label: str) -> Dict[str, object]:
        values = arr.tolist()
        mean = float(np.mean(values))
        median = float(np.median(values))
        # mode can be multiple; we choose smallest most frequent
        vals, counts = np.unique(values, return_counts=True)
        mode = int(vals[np.argmax(counts)]) if len(vals) else 0
        std_dev = float(np.std(values, ddof=0))
        minv = int(np.min(values)) if len(values) else 0
        maxv = int(np.max(values)) if len(values) else 0
        return {
            "metric": label,
            "mean": round(mean, 3),
            "median": round(median, 3),
            "mode": mode,
            "std_dev": round(std_dev, 6),
            "min_value": minv,
            "max_value": maxv,
        }

    rows = [
        summarize(sums, "Sum of Main Numbers"),
        summarize(spreads, "Spread (Max-Min)"),
    ]

    return pd.DataFrame(rows)


def compute_parity_highlow_simple(main_matrix: np.ndarray) -> pd.DataFrame:
    """
    Simple counts & proportions for odd/even and high/low categories.
    """
    n_draws = main_matrix.shape[0]
    total_slots = n_draws * 5

    flat = main_matrix.flatten()

    odd_count = int((flat % 2 == 1).sum())
    even_count = int((flat % 2 == 0).sum())

    low_count = int((flat <= LOW_MAX).sum())
    high_count = int((flat > LOW_MAX).sum())

    rows = [
        {"category": "Odd", "count": odd_count, "historical_proportion": round(odd_count / total_slots, 4)},
        {"category": "Even", "count": even_count, "historical_proportion": round(even_count / total_slots, 4)},
        {"category": f"Low (1-{LOW_MAX})", "count": low_count, "historical_proportion": round(low_count / total_slots, 4)},
        {"category": f"High ({LOW_MAX+1}-49)", "count": high_count, "historical_proportion": round(high_count / total_slots, 4)},
    ]

    return pd.DataFrame(rows)


def compute_composition_matrix(main_matrix: np.ndarray) -> pd.DataFrame:
    """
    Aggregate counts of compositions:
      - odd/even pattern per draw
      - high/low pattern per draw
      - combined matrix counts
    Example row:
      oe_pattern="3 Odd, 2 Even", hl_pattern="2 High, 3 Low", count=..., proportion=...
    """
    n_draws = main_matrix.shape[0]

    rows = []
    for i in range(n_draws):
        nums = main_matrix[i, :]
        odd = int((nums % 2 == 1).sum())
        even = 5 - odd
        high = int((nums > LOW_MAX).sum())
        low = 5 - high

        rows.append((odd, even, high, low))

    df = pd.DataFrame(rows, columns=["odd", "even", "high", "low"])

    grouped = (
        df.groupby(["odd", "even", "high", "low"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    grouped["oe_pattern"] = grouped.apply(lambda r: f"{int(r['odd'])} Odd, {int(r['even'])} Even", axis=1)
    grouped["hl_pattern"] = grouped.apply(lambda r: f"{int(r['high'])} High, {int(r['low'])} Low", axis=1)
    grouped["historical_proportion"] = (grouped["count"] / n_draws).round(4)

    return grouped[["oe_pattern", "hl_pattern", "count", "historical_proportion"]]


def compute_consecutive_overlap_hist(main_matrix: np.ndarray) -> pd.DataFrame:
    """
    Overlap histogram for consecutive draws only:
      overlap = how many main numbers are shared between draw i and draw i+1
    Outputs distribution 0..5
    """
    n = main_matrix.shape[0]
    if n < 2:
        return pd.DataFrame(
            [{"numbers_in_common": k, "count_of_draw_pairs": 0, "proportion_of_pairs": 0.0} for k in range(6)]
        )

    overlaps = []
    for i in range(n - 1):
        a = set(main_matrix[i, :].tolist())
        b = set(main_matrix[i + 1, :].tolist())
        overlaps.append(len(a.intersection(b)))

    total_pairs = len(overlaps)
    rows = []
    for k in range(6):
        c = overlaps.count(k)
        p = c / total_pairs if total_pairs > 0 else 0.0
        rows.append({"numbers_in_common": k, "count_of_draw_pairs": c, "proportion_of_pairs": round(p, 6)})

    return pd.DataFrame(rows)


def compute_pairs_threshold(
    main_matrix: np.ndarray,
    dates: pd.Series,
    min_count: int = 4
) -> pd.DataFrame:
    """
    Pair co-occurrence counts (threshold-based, NOT Top-N).
    This avoids ranking noise for small samples.

    Output columns:
      pair, count, pair_rate, first_seen_date, last_seen_date

    pair_rate here is count / number_of_draws (simple, window-level rate).
    """
    n_draws = main_matrix.shape[0]
    pair_counts: Dict[Tuple[int, int], int] = {}
    first_seen: Dict[Tuple[int, int], str] = {}
    last_seen: Dict[Tuple[int, int], str] = {}

    for i in range(n_draws):
        nums = sorted(main_matrix[i, :].tolist())
        draw_date = str(dates.iloc[i].date())

        for a, b in itertools.combinations(nums, 2):
            pair = (a, b)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
            if pair not in first_seen:
                first_seen[pair] = draw_date
            last_seen[pair] = draw_date

    rows = []
    for pair, cnt in pair_counts.items():
        if cnt >= min_count:
            a, b = pair
            rows.append(
                {
                    "pair": f"{a}-{b}",
                    "count": cnt,
                    "pair_rate": round(cnt / n_draws, 3) if n_draws > 0 else 0.0,
                    "first_seen_date": first_seen[pair],
                    "last_seen_date": last_seen[pair],
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    df = df.sort_values(["count", "pair"], ascending=[False, True]).reset_index(drop=True)
    return df

def make_sorted_main_stats(main_stats: pd.DataFrame, sort_by: str = "number", ascending: bool = True) -> pd.DataFrame:
       """
       Returns a sorted copy of the main_stats table.
       sort_by options: number, frequency, current_gap_draws, last_seen_draw_index
       """
       allowed = {"number", "frequency", "current_gap_draws", "last_seen_draw_index"}
       if sort_by not in allowed:
        raise ValueError(f"Invalid sort_by={sort_by}. Allowed: {sorted(list(allowed))}")

       # tie-breaker: number
       df = main_stats.sort_values([sort_by, "number"], ascending=[ascending, True]).reset_index(drop=True).copy()
       return df

# ------------------------------------------------------------
# Reporting / Export
# ------------------------------------------------------------
def write_excel_report(out_path: str, sheets: Dict[str, pd.DataFrame]) -> None:
    """
    Writes all outputs into a single Excel file, one sheet per output.
    """
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            if df is None:
                pd.DataFrame().to_excel(writer, sheet_name=name, index=False)
            else:
                df.to_excel(writer, sheet_name=name, index=False)


# ------------------------------------------------------------
# Terminal Output Controls
# ------------------------------------------------------------
def print_window_summary(window_df: pd.DataFrame, validation_report: pd.DataFrame, input_file: str, out_path: str) -> None:
    """
    Prints a compact summary for the selected window.
    This replaces printing the full raw dataset to terminal.
    """
    n = len(window_df)
    if n == 0:
        print("\n==============================")
        print("Daily Grand Window Report")
        print("==============================")
        print("No draws found in the requested window.")
        print("==============================")
        return

    start = window_df[DATE_COL].min().date()
    end = window_df[DATE_COL].max().date()

    dropped_total = 0
    if isinstance(validation_report, pd.DataFrame) and "dropped" in validation_report.columns:
        dropped_total = int(validation_report["dropped"].fillna(0).sum())

    print("\n==============================")
    print("Daily Grand Window Report")
    print("==============================")
    print(f"Input file: {os.path.basename(input_file)}")
    print(f"Window: {start} -> {end} | Draws: {n} | Total main slots: {n*5}")
    print(f"Excel output: {out_path}")
    print(f"Validation: dropped_rows={dropped_total}")


def print_selected_sheets_to_terminal(sheets: Dict[str, pd.DataFrame], include_sheet_names: list) -> None:
    """
    Terminal output is intentionally limited to analysis sheets only.
    Raw draw rows remain in the Excel export ("02_WINDOW_DATA").
    """
    for sheet_name in include_sheet_names:
        if sheet_name not in sheets:
            continue
        df = sheets[sheet_name]
        rows = 0 if df is None else (len(df) if isinstance(df, pd.DataFrame) else 0)
        print_df_full(f"SHEET: {sheet_name} | rows={rows}", df)


# ------------------------------------------------------------
# Main runner
# ------------------------------------------------------------
def main():
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python3 D2016-17.py <input_excel_path> <start_date> <end_date>")
        print("Example:")
        print('  python3 D2016-17.py "Db.Daily2016-17.xlsx" 2016-10-20 2017-12-28')
        sys.exit(1)

    input_file = sys.argv[1]
    start_date = sys.argv[2]
    end_date = sys.argv[3]

    # Load + validate
    raw_df = load_raw_data(input_file)
    clean_df, validation_report = validate_and_clean(raw_df)

    # Window
    window_df = generate_analysis_window(clean_df, start_date, end_date)

    # NOTE:
    # Raw window rows ("02_WINDOW_DATA") are exported to Excel for reference,
    # but intentionally NOT printed to terminal to keep logs clean and readable.

    if len(window_df) == 0:
        print("\nNo draws found in the requested window.")
        sys.exit(0)

    main_matrix = get_main_matrix(window_df)
    grand_series = get_grand_series(window_df)
    dates = window_df[DATE_COL]

    # Compute outputs
    main_stats = compute_main_stats_1_49(main_matrix, dates)
    main_stats_sorted = make_sorted_main_stats(main_stats, sort_by="frequency", ascending=False)
    hot_top10 = compute_hot_top10(main_stats)
    cold_bottom10 = compute_cold_bottom10(main_stats)

    grand_stats = compute_grand_stats_1_7(grand_series)

    sum_spread_summary = compute_sum_spread_summary(main_matrix)
    parity_simple = compute_parity_highlow_simple(main_matrix)
    composition_matrix = compute_composition_matrix(main_matrix)

    overlap_hist = compute_consecutive_overlap_hist(main_matrix)

    # Pair analysis: threshold-based (default min_count=4)
    pairs_threshold = compute_pairs_threshold(main_matrix, dates, min_count=4)

    # Build sheet dictionary
    meta_df = pd.DataFrame(
        [
            {"key": "analysis_type", "value": "descriptive_only"},
            {"key": "input_file", "value": os.path.basename(input_file)},
            {"key": "window_start", "value": start_date},
            {"key": "window_end", "value": end_date},
            {"key": "draw_count", "value": len(window_df)},
            {"key": "disclaimer", "value": "Historical descriptive summary only. No prediction. Draws are independent."},
        ]
    )

    sheets: Dict[str, pd.DataFrame] = {
        SHEET_META: meta_df,
        SHEET_VALIDATION: validation_report,
        SHEET_WINDOW_DATA: window_df,  # stored in Excel only
        SHEET_MAIN_STATS: main_stats,
        SHEET_MAIN_STATS_SORTED: main_stats_sorted,
        SHEET_HOT_TOP10: hot_top10,
        SHEET_COLD_BOTTOM10: cold_bottom10,
        SHEET_GRAND_STATS: grand_stats,
        SHEET_SUM_SPREAD_SUMMARY: sum_spread_summary,
        SHEET_PARITY_SIMPLE: parity_simple,
        SHEET_COMPOSITION_MATRIX: composition_matrix,
        SHEET_OVERLAP_HIST: overlap_hist,
        SHEET_PAIRS: pairs_threshold,
    }

    # Output path
    out_name = f"DailyGrand_Report_{start_date.replace('-','')}_{end_date.replace('-','')}.xlsx"
    out_path = os.path.join(os.path.dirname(os.path.abspath(input_file)), out_name)

    # Write Excel
    write_excel_report(out_path, sheets)

    # Terminal summary + analysis sheets (skip raw data)
    print_window_summary(window_df, validation_report, input_file, out_path)

    terminal_sheets = [
        SHEET_META,
        SHEET_VALIDATION,
        # SHEET_WINDOW_DATA,  # intentionally not printed
        SHEET_HOT_TOP10,
        SHEET_COLD_BOTTOM10,
        SHEET_GRAND_STATS,
        SHEET_SUM_SPREAD_SUMMARY,
        SHEET_PARITY_SIMPLE,
        SHEET_COMPOSITION_MATRIX,
        SHEET_OVERLAP_HIST,
        SHEET_PAIRS,
        SHEET_MAIN_STATS,  # keep this last because it's long (1..49)
        SHEET_MAIN_STATS_SORTED,

        
    ]

    print_selected_sheets_to_terminal(sheets, terminal_sheets)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
