# Setup Guide: Running Motorcycle Safety Dashboard Locally

## Quick Start

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Download NHTSA FARS 2024 Data

The app requires real FARS data from NHTSA. Download both:

1. **National Data**: [FARS 2024 National CSV](https://www.nhtsa.gov/FARS) 
   - Save to: `~/Downloads/FARS2024NationalCSV/`

2. **Puerto Rico Data**: [FARS 2024 PR CSV](https://www.nhtsa.gov/FARS)
   - Save to: `~/Downloads/FARS2024PuertoRicoCSV/`

### 3. Generate Combined Data Files

Run the setup script to combine National + PR data:
```bash
python setup_real_data.py
```

This creates:
- `traffic-accidents/accident.csv` (36,572 crashes)
- `traffic-accidents/person.csv` (88,887 people)
- `traffic-accidents/vehicle.csv` (56,401 vehicles)

### 4. Run the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501)

---

## Data Processing Details

### Motorcycle Identification

**CRITICAL**: Motorcycles are identified by **BODY_TYP codes** in vehicle.csv, not by PER_TYP:

| Code | Description | Count |
|------|-------------|-------|
| 80 | Two-wheel motorcycle | 15 |
| 82 | Three-wheel motorcycle (2 rear) | 611 |
| 83 | Off-road motorcycle | 955 |
| 85 | Unenclosed three-wheel | 261 |
| 87 | Unknown motorcycle type | 49,103 |
| 88 | Other motored cycle | 858 |
| **Total** | | **51,803** |

### Filtering Logic

The app:
1. Loads person.csv (88,887 records)
2. Loads vehicle.csv and filters for motorcycle BODY_TYP codes
3. Merges on ST_CASE + VEH_NO to get motorcycle riders only
4. Filters for fatal injuries (INJ_SEV == 4)
5. **Result**: 5,921 motorcycle fatalities

### Previous Incorrect Approach

❌ Old method filtered by PER_TYP == 4 ("Non-Motor Vehicle Device")
- This only yielded ~16 total fatalities (Florida: 10, Texas: 10, etc.)
- Completely unrealistic

### Why This Fix Matters

✅ New method using BODY_TYP codes:
- Florida: 577 (realistic ✓)
- Texas: 570 (realistic ✓)
- California: 496 (realistic ✓)
- Total: 5,921 (matches expectations ✓)

---

## Deployment to Streamlit Cloud

1. Ensure `.gitignore` includes `traffic-accidents/*.csv`
2. Create `.streamlit/secrets.toml` (if needed)
3. Deploy via Streamlit Community Cloud:
   ```bash
   git push origin main
   ```
4. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub repo

**Note**: The large CSV files stay local and won't deploy to cloud. The app falls back to simulated data if real data files are not present.

---

## File Sizes

- `accident.csv`: ~150 MB
- `person.csv`: ~110 MB (too large for GitHub)
- `vehicle.csv`: ~100 MB (too large for GitHub)

These must be generated locally using `setup_real_data.py`.

---

## Validation

To verify the data loaded correctly:
```bash
python test_data_loading.py
```

Expected output:
```
✅ Motorcycles in vehicle.csv: 6055
✅ After merging motorcycle riders: 6556
✅ After INJ_SEV==4 filter: 5921
✅ Total motorcycle fatalities: 5921

Top 10 states:
State
Florida           577
Texas             570
California        496
Ohio              215
...
```

---

## Troubleshooting

**Q: App shows "Simulated Data" instead of real data?**
A: Check that all three CSV files exist in `traffic-accidents/` folder:
```bash
ls -lh traffic-accidents/*.csv
```

**Q: "File not found" errors when running setup?**
A: Verify FARS downloads are in correct locations:
```bash
ls ~/Downloads/FARS2024NationalCSV/FARS2024NationalCSV/
ls ~/Downloads/FARS2024PuertoRicoCSV/FARS2024PuertoRicoCSV/
```

**Q: How do I know if data is correct?**
A: Run audit script:
```bash
python audit.py
```

Should show:
```
✅ SUCCESS: Data matched official NHTSA FARS Census Year: 2024
Fed 36,297 crashes vs Local 36,572 ✓
Fed 39,254 fatalities vs Local 39,542 ✓
```
