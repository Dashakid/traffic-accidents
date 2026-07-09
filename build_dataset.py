#!/usr/bin/env python3
"""
Build a slim, deployable motorcycle-fatalities dataset for Streamlit Cloud.

Combines all available NHTSA FARS years (National + Puerto Rico), filters to
motorcyclists via vehicle BODY_TYP codes, keeps all injury severities, and
saves ONLY the columns the app needs. The result is small enough for GitHub
(< a few MB) so it deploys to Streamlit Community Cloud directly.

Run:  python build_dataset.py
Output: traffic-accidents/motorcycle_data.csv
"""

import os
import pandas as pd

DOWNLOADS = os.path.expanduser("~/Downloads")
OUT_DIR = os.path.join(os.getcwd(), "traffic-accidents")
os.makedirs(OUT_DIR, exist_ok=True)

# FARS motorcycle body types (from vehicle.csv BODY_TYP)
MOTORCYCLE_BODY_TYPES = [80, 82, 83, 85, 87, 88]

# (year, region_label, folder_name) — only combos that exist locally
SOURCES = [
    (2022, "National", "FARS2022NationalCSV"),
    (2023, "National", "FARS2023NationalCSV"),
    (2023, "PuertoRico", "FARS2023PuertoRicoCSV"),
    (2024, "National", "FARS2024NationalCSV"),
    (2024, "PuertoRico", "FARS2024PuertoRicoCSV"),
]

STATE_CODE_TO_NAME = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "District of Columbia",
    12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois",
    18: "Indiana", 19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana",
    23: "Maine", 24: "Maryland", 25: "Massachusetts", 26: "Michigan",
    27: "Minnesota", 28: "Mississippi", 29: "Missouri", 30: "Montana",
    31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey",
    35: "New Mexico", 36: "New York", 37: "North Carolina", 38: "North Dakota",
    39: "Ohio", 40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania",
    44: "Rhode Island", 45: "South Carolina", 46: "South Dakota",
    47: "Tennessee", 48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia",
    53: "Washington", 54: "West Virginia", 55: "Wisconsin", 56: "Wyoming",
    43: "Puerto Rico", 72: "Puerto Rico",
}

DAY_NAMES = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday",
             5: "Thursday", 6: "Friday", 7: "Saturday"}


def read_csv_safe(path: str) -> pd.DataFrame:
    """Read a FARS CSV, handling latin-1 encoding used by some files."""
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, low_memory=False, encoding=enc)
            df.columns = [c.upper() for c in df.columns]
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not read {path}")


def process_source(year: int, region: str, folder: str) -> pd.DataFrame | None:
    base = os.path.join(DOWNLOADS, folder, folder)
    acc_path = os.path.join(base, "accident.csv")
    per_path = os.path.join(base, "person.csv")
    veh_path = os.path.join(base, "vehicle.csv")

    if not all(os.path.exists(p) for p in (acc_path, per_path, veh_path)):
        print(f"  ⚠️  Skipping {year} {region}: files not found")
        return None

    acc = read_csv_safe(acc_path)
    per = read_csv_safe(per_path)
    veh = read_csv_safe(veh_path)

    # Motorcycles only
    veh_moto = veh[veh["BODY_TYP"].isin(MOTORCYCLE_BODY_TYPES)]

    # Person -> accident (for STATE + DAY_WEEK), then -> motorcycle vehicles
    merged = per.merge(
        acc[["ST_CASE", "STATE", "DAY_WEEK"]],
        on="ST_CASE", how="inner", suffixes=("", "_ACC"),
    )
    merged = merged.merge(
        veh_moto[["ST_CASE", "VEH_NO"]],
        on=["ST_CASE", "VEH_NO"], how="inner",
    )

    # Prefer person STATE; fall back to accident STATE
    if "STATE" not in merged.columns and "STATE_ACC" in merged.columns:
        merged["STATE"] = merged["STATE_ACC"]

    out = pd.DataFrame({
        "YEAR": year,
        "STATE": merged["STATE"],
        "DAY_WEEK": merged["DAY_WEEK"],
        "INJ_SEV": merged["INJ_SEV"],
    })
    out["State"] = out["STATE"].map(STATE_CODE_TO_NAME)
    out["Day"] = out["DAY_WEEK"].map(DAY_NAMES)
    out["Region"] = out["STATE"].apply(
        lambda x: "Puerto Rico" if x in (43, 72) else "US Mainland"
    )
    out["Time Window"] = out["DAY_WEEK"].apply(
        lambda x: "Weekend (Fri–Sun)" if x in (1, 6, 7) else "Weekday (Mon–Thu)"
    )

    fatal = int((out["INJ_SEV"] == 4).sum())
    print(f"  ✅ {year} {region:11s}: {len(out):5d} riders, {fatal:4d} fatalities")
    return out


def main():
    print("Building slim motorcycle dataset (National + Puerto Rico)...\n")
    frames = []
    for year, region, folder in SOURCES:
        df = process_source(year, region, folder)
        if df is not None:
            frames.append(df)

    if not frames:
        raise SystemExit("❌ No source data found in ~/Downloads")

    combined = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(OUT_DIR, "motorcycle_data.csv")
    combined.to_csv(out_path, index=False)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print("\n" + "=" * 60)
    print(f"✅ Saved {out_path}")
    print(f"   Rows: {len(combined):,}  |  Size: {size_mb:.2f} MB")
    print(f"   Years: {sorted(combined['YEAR'].unique().tolist())}")
    print(f"   Puerto Rico riders: {(combined['Region'] == 'Puerto Rico').sum():,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
