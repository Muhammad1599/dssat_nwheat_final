# Initial Layers Calculation for 2015 Wheat Data

This folder contains all the work done to calculate initial soil conditions (water and nitrogen) for the 2015 wheat experiment data.

## Contents

### Scripts and Documentation
- **calculate_initial_layers_2015_wheat.R** - Main script to calculate initial conditions
- **test_initial_layers_calculation.R** - Verification/test script
- **README_initial_layers_calculation.md** - Detailed documentation and calculations
- **README.md** - This file (quick start guide)

### Results Files (ACTUAL CALCULATED VALUES)
- **initial_conditions_2015_wheat_ACTUAL.csv** - ✅ **ACTUAL calculated results** (layer-by-layer format)
- **initial_conditions_2015_wheat_dssat_format.txt** - ✅ **ACTUAL results** in DSSAT-readable format
- **CALCULATION_SUMMARY.txt** - ✅ **ACTUAL results** with detailed summary

### Reference Files
- **initial_conditions_2015_wheat_expected.csv** - Expected/verification values (for comparison only)

## Quick Start

1. Ensure you have the `csmTools` package or source the function from `R/calculate_initial_layers.R`
2. Ensure the data folder structure exists:
   - `data/1_icasa/SOIL_PROFILE_LAYERS.csv` (contains soil profile data)
3. Run the main script:
   ```r
   source("calculate_initial_layers_2015_wheat.R")
   ```

## Parameters Used

- **Available Water**: 90%
- **Total Nitrogen**: 100 kg/ha
- **Soil Profile**: 6 layers (5, 15, 30, 60, 100, 200 cm)

## Output

The script generates:
1. Layer-by-layer CSV format (for verification)
2. DSSAT format with list columns (ready for DSSAT .WHX file)

## Dependencies

- R packages: `dplyr`, `readr`
- `csmTools` package (or source `R/calculate_initial_layers.R` and `R/utils_dataframes.R`)
- Soil profile data from: `data/1_icasa/SOIL_PROFILE_LAYERS.csv`

## Data Source

The soil profile data comes from the DüRNast Long-Term Fertilization Experiment dataset, converted to ICASA format and stored in `data/1_icasa/SOIL_PROFILE_LAYERS.csv`.

## Notes

- No soil profile normalization was needed (depths already standard DSSAT format)
- Calculations follow DSSAT XBuild logic
- Nitrogen split: 90% NO₃⁻, 10% NH₄⁺ (DSSAT standard)

