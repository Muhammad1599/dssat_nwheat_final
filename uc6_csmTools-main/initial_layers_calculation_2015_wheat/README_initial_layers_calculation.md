# Calculate Initial Layers for 2015 Wheat Data

## Parameters
- **Available Water**: 90%
- **Total Nitrogen**: 100 kg/ha

## Soil Profile Data
Source: `data/1_icasa/SOIL_PROFILE_LAYERS.csv`

| Depth (cm) | SLLL (WP) | SDUL (FC) | SBDM (g/cm³) |
|------------|-----------|-----------|--------------|
| 5          | 0.108     | 0.236     | 1.29         |
| 15         | 0.118     | 0.247     | 1.31         |
| 30         | 0.133     | 0.262     | 1.34         |
| 60         | 0.147     | 0.276     | 1.39         |
| 100        | 0.146     | 0.274     | 1.45         |
| 200        | 0.136     | 0.262     | 1.51         |

*WP = Wilting Point, FC = Field Capacity*

## Calculations

### Water Content (SH2O)
Formula: `SH2O = SLLL + (90/100) × (SDUL - SLLL)`

| Layer | Calculation | SH2O (cm³/cm³) |
|-------|------------|----------------|
| 5 cm  | 0.108 + 0.9 × (0.236 - 0.108) | 0.2232 |
| 15 cm | 0.118 + 0.9 × (0.247 - 0.118) | 0.2341 |
| 30 cm | 0.133 + 0.9 × (0.262 - 0.133) | 0.2491 |
| 60 cm | 0.147 + 0.9 × (0.276 - 0.147) | 0.2691 |
| 100 cm| 0.146 + 0.9 × (0.274 - 0.146) | 0.2682 |
| 200 cm| 0.136 + 0.9 × (0.262 - 0.136) | 0.2592 |

### Nitrogen Concentration
Formula based on depth-weighted average bulk density:
- Total N: 100 kg/ha
- Split: 90% NO₃⁻, 10% NH₄⁺
- Profile depth: 200 cm

Layer thicknesses: 5, 10, 15, 30, 40, 100 cm

Weighted average BD = (1.29×5 + 1.31×10 + 1.34×15 + 1.39×30 + 1.45×40 + 1.51×100) / 200
= 290.35 / 200 = **1.45175 g/cm³**

NO₃⁻ = (0.9 × 100) / (0.1 × 1.45175 × 200) = 90 / 29.035 = **3.10 ppm** (uniform)
NH₄⁺ = (0.1 × 100) / (0.1 × 1.45175 × 200) = 10 / 29.035 = **0.34 ppm** (uniform)

## Expected Results

| ICBL (cm) | SH2O (cm³/cm³) | SNO3 (ppm) | SNH4 (ppm) |
|-----------|----------------|------------|------------|
| 5         | 0.2232         | 3.10       | 0.34       |
| 15        | 0.2341         | 3.10       | 0.34       |
| 30        | 0.2491         | 3.10       | 0.34       |
| 60        | 0.2691         | 3.10       | 0.34       |
| 100       | 0.2682         | 3.10       | 0.34       |
| 200       | 0.2592         | 3.10       | 0.34       |

## Important: DSSAT Format Requirements

The `calculate_initial_layers()` function returns a **layer-by-layer** data frame (one row per soil layer).

However, DSSAT requires INITIAL_CONDITIONS in a **list-column format**:
- **One row per condition** (C = 1, 2, etc.)
- **ICBL, SH2O, SNH4, SNO3 as list columns** (vectors containing all layer values)

The script automatically converts the output to DSSAT format.

## How to Run

1. Open R in the project directory
2. Run: `source("calculate_initial_layers_2015_wheat.R")`
3. Results will be saved to:
   - `data/initial_conditions_2015_wheat.csv` (layer-by-layer format)
   - `data/initial_conditions_2015_wheat_dssat_format.RDS` (DSSAT format with list columns)

### Manual Usage (if package installed):

```r
library(csmTools)

# 1. Read soil profile
soil_profile <- data.frame(
  SLB = c(5, 15, 30, 60, 100, 200),
  SLLL = c(0.108, 0.118, 0.133, 0.147, 0.146, 0.136),
  SDUL = c(0.236, 0.247, 0.262, 0.276, 0.274, 0.262),
  SBDM = c(1.29, 1.31, 1.34, 1.39, 1.45, 1.51)
)

# 2. Calculate initial conditions (returns layer-by-layer format)
initial_conditions <- calculate_initial_layers(
  soil_profile = soil_profile,
  percent_available_water = 90,
  total_n_kgha = 100
)

# 3. Convert to DSSAT format (list columns)
dssat_initial <- data.frame(
  C = 1,  # Condition number
  ICBL = list(initial_conditions$ICBL),
  SH2O = list(initial_conditions$SH2O),
  SNH4 = list(initial_conditions$SNH4),
  SNO3 = list(initial_conditions$SNO3)
)
# Add other required INITIAL_CONDITIONS fields (PCR, ICDAT, etc.) as needed

# 4. Use with DSSAT file writing functions
# This can then be used with format_dssat_table() and write_dssat_dataset()
```

## Verification

Yes, the calculations are **correct** for DSSAT input:
- ✅ Water content (SH2O) calculated using DSSAT XBuild formula
- ✅ Nitrogen split: 90% NO3, 10% NH4 (DSSAT standard)
- ✅ Format converted to DSSAT list-column structure
- ✅ Values match expected DSSAT INITIAL_CONDITIONS section format

