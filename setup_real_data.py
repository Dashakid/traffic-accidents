import pandas as pd
import os

# Paths
base_path = os.path.expanduser("~/Downloads")
project_path = os.path.expanduser("~/Projects/traffic-accidents/traffic-accidents")

# Ensure output directory exists
os.makedirs(project_path, exist_ok=True)

# 2024: Combine National + Puerto Rico
print("📊 Loading 2024 FARS data...")
acc_2024_nat = pd.read_csv(f"{base_path}/FARS2024NationalCSV/FARS2024NationalCSV/accident.csv")
acc_2024_pr = pd.read_csv(f"{base_path}/FARS2024PuertoRicoCSV/FARS2024PuertoRicoCSV/accident.csv")
per_2024_nat = pd.read_csv(f"{base_path}/FARS2024NationalCSV/FARS2024NationalCSV/person.csv")
per_2024_pr = pd.read_csv(f"{base_path}/FARS2024PuertoRicoCSV/FARS2024PuertoRicoCSV/person.csv")

# Combine
accident_combined = pd.concat([acc_2024_nat, acc_2024_pr], ignore_index=True)
person_combined = pd.concat([per_2024_nat, per_2024_pr], ignore_index=True)

print(f"  → Accident records: {len(accident_combined):,}")
print(f"  → Person records: {len(person_combined):,}")

# Save
accident_combined.to_csv(f"{project_path}/accident.csv", index=False)
person_combined.to_csv(f"{project_path}/person.csv", index=False)

print("✅ Saved to ~/Projects/traffic-accidents/traffic-accidents/")
print(f"   - accident.csv: {len(accident_combined):,} rows")
print(f"   - person.csv: {len(person_combined):,} rows")

# Verify
print("\n🔍 Verifying data...")
print(f"   State codes present: {accident_combined['STATE'].nunique()}")
print(f"   Years present: {accident_combined['YEAR'].unique()}")
