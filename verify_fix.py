#!/usr/bin/env python3
import pandas as pd
import os

# Recreate vehicle.csv if needed
print("Building vehicle data...")
base_path = os.path.expanduser("~/Downloads")
veh_2024_nat = pd.read_csv(f"{base_path}/FARS2024NationalCSV/FARS2024NationalCSV/vehicle.csv", low_memory=False)
veh_2024_pr = pd.read_csv(f"{base_path}/FARS2024PuertoRicoCSV/FARS2024PuertoRicoCSV/vehicle.csv", low_memory=False)
veh_combined = pd.concat([veh_2024_nat, veh_2024_pr], ignore_index=True)
veh_combined.to_csv("traffic-accidents/vehicle.csv", index=False)

# Load and process
per = pd.read_csv("traffic-accidents/person.csv", low_memory=False)
acc = pd.read_csv("traffic-accidents/accident.csv", low_memory=False)
veh = pd.read_csv("traffic-accidents/vehicle.csv", low_memory=False)

STATE_CODE_TO_NAME = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
    8: "Colorado", 9: "Connecticut", 10: "Delaware", 12: "Florida", 13: "Georgia",
    15: "Hawaii", 16: "Idaho", 17: "Illinois", 18: "Indiana", 19: "Iowa",
    20: "Kansas", 21: "Kentucky", 22: "Louisiana", 23: "Maine", 24: "Maryland",
    25: "Massachusetts", 26: "Michigan", 27: "Minnesota", 28: "Mississippi",
    29: "Missouri", 30: "Montana", 31: "Nebraska", 32: "Nevada", 33: "New Hampshire",
    34: "New Jersey", 35: "New Mexico", 36: "New York", 37: "North Carolina",
    38: "North Dakota", 39: "Ohio", 40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania",
    44: "Rhode Island", 45: "South Carolina", 46: "South Dakota", 47: "Tennessee",
    48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia", 53: "Washington",
    54: "West Virginia", 55: "Wisconsin", 56: "Wyoming", 72: "Puerto Rico"
}

# Merge and filter
motorcycle_body_types = [80, 82, 83, 85, 87, 88]
veh_moto = veh[veh['BODY_TYP'].isin(motorcycle_body_types)]

per_upper = per.copy()
per_upper.columns = [c.upper() for c in per_upper.columns]
acc_upper = acc.copy()
acc_upper.columns = [c.upper() for c in acc_upper.columns]
veh_upper = veh_moto.copy()
veh_upper.columns = [c.upper() for c in veh_upper.columns]

merged = per_upper.merge(acc_upper[['ST_CASE', 'STATE']], on='ST_CASE', how='inner', suffixes=('', '_from_acc'))
merged = merged.merge(veh_upper[['ST_CASE', 'VEH_NO']], on=['ST_CASE', 'VEH_NO'], how='inner')

# Use STATE from acc if it got shadowed
if 'STATE_from_acc' in merged.columns:
    merged['STATE'] = merged['STATE_from_acc']

fatal = merged[merged['INJ_SEV'] == 4]

print("\n" + "="*60)
print("✅ MOTORCYCLE FATALITY DATA (CORRECTED)")
print("="*60)
print(f"Total motorcycle fatalities: {len(fatal):,}")
print(f"\nTop 10 states:")

for state_code, count in fatal['STATE'].value_counts().head(10).items():
    name = STATE_CODE_TO_NAME.get(state_code, f"State {state_code}")
    print(f"  {name:20s}: {count:4d}")

print("="*60)
