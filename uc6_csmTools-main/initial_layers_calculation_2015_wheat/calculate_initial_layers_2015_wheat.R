# Calculate Initial Layers for 2015 Wheat Data
# Using 90% available water and 100 kg/ha nitrogen

# Set working directory to project root (parent of this script folder)
script_dir <- getwd()
if (basename(script_dir) == "initial_layers_calculation_2015_wheat") {
  # If running from inside the folder, go up one level
  setwd("..")
}
# Save output folder path
output_folder <- "initial_layers_calculation_2015_wheat"

# Load required libraries
library(dplyr)
library(readr)

# Check if csmTools is loaded, if not source the function
if (!exists("calculate_initial_layers")) {
  # Try loading the package first
  if (!require(csmTools, quietly = TRUE)) {
    # If package not installed, source the function directly
    source("R/calculate_initial_layers.R")
  }
}

# Method 1: Read from ICASA format CSV (easier to parse)
soil_icasa <- read.csv("data/1_icasa/SOIL_PROFILE_LAYERS.csv", stringsAsFactors = FALSE)

# Convert ICASA column names to DSSAT format needed by calculate_initial_layers()
# ICASA -> DSSAT mapping:
# soil_layer_base_depth -> SLB
# soil_water_lower_limit -> SLLL  
# soil_wat_drned_upper_lim -> SDUL
# soil_bulk_density_moist -> SBDM

soil_profile <- data.frame(
  SLB = soil_icasa$soil_layer_base_depth,
  SLLL = soil_icasa$soil_water_lower_limit,
  SDUL = soil_icasa$soil_wat_drned_upper_lim,
  SBDM = soil_icasa$soil_bulk_density_moist
)

# Display the soil profile
cat("Soil Profile Data:\n")
print(soil_profile)

# Calculate initial layers
# Parameters:
# - percent_available_water = 90%
# - total_n_kgha = 100 kg/ha
cat("\nCalculating initial conditions...\n")
cat("Parameters:\n")
cat("  - Available Water: 90%\n")
cat("  - Total Nitrogen: 100 kg/ha\n\n")

# Call the function
initial_conditions <- calculate_initial_layers(
  soil_profile = soil_profile,
  percent_available_water = 90,
  total_n_kgha = 100
)

# Display results
cat("Initial Soil Conditions (ICBL = depth at bottom of layer in cm):\n")
cat("SH2O = Volumetric soil water content (cm3/cm3)\n")
cat("SNO3 = Nitrate concentration (ppm)\n")
cat("SNH4 = Ammonium concentration (ppm)\n\n")
print(initial_conditions)

# Ensure output folder exists
if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

# Save results (layer-by-layer format)
output_csv <- file.path(output_folder, "initial_conditions_2015_wheat_ACTUAL.csv")
write.csv(initial_conditions, output_csv, row.names = FALSE)
cat("\nResults saved to:", output_csv, "\n")

# Convert to DSSAT format (list columns for INITIAL_CONDITIONS section)
# DSSAT requires: one row per condition, with ICBL, SH2O, SNH4, SNO3 as list columns
cat("\nConverting to DSSAT format...\n")

# Load collapse_cols function
source("R/utils_dataframes.R")

# Create DSSAT INITIAL_CONDITIONS format
# First, create a row with condition metadata (C = 1)
dssat_initial <- data.frame(
  C = 1,
  ICBL = list(initial_conditions$ICBL),
  SH2O = list(initial_conditions$SH2O),
  SNH4 = list(initial_conditions$SNH4),
  SNO3 = list(initial_conditions$SNO3)
)

cat("DSSAT INITIAL_CONDITIONS format (one row, list columns):\n")
cat("  C = 1 (condition number)\n")
cat("  ICBL = list of layer depths:", paste(initial_conditions$ICBL, collapse=", "), "cm\n")
cat("  SH2O = list of water contents:", paste(round(initial_conditions$SH2O, 4), collapse=", "), "cm³/cm³\n")
cat("  SNH4 = list of NH4 concentrations:", paste(round(initial_conditions$SNH4, 2), collapse=", "), "ppm\n")
cat("  SNO3 = list of NO3 concentrations:", paste(round(initial_conditions$SNO3, 2), collapse=", "), "ppm\n")

# Save DSSAT format (as RDS to preserve list structure)
output_rds <- file.path(output_folder, "initial_conditions_2015_wheat_dssat_format.RDS")
saveRDS(dssat_initial, output_rds)
cat("\nDSSAT format saved to:", output_rds, "\n")

# Also save DSSAT format as a readable text file
output_txt <- file.path(output_folder, "initial_conditions_2015_wheat_dssat_format.txt")
sink(output_txt)
cat("DSSAT INITIAL_CONDITIONS Format\n")
cat("===============================\n\n")
cat("C = 1\n\n")
cat("ICBL (layer depths in cm):\n")
cat(paste(initial_conditions$ICBL, collapse=", "), "\n\n")
cat("SH2O (water content in cm³/cm³):\n")
cat(paste(round(initial_conditions$SH2O, 4), collapse=", "), "\n\n")
cat("SNH4 (ammonium in ppm):\n")
cat(paste(round(initial_conditions$SNH4, 2), collapse=", "), "\n\n")
cat("SNO3 (nitrate in ppm):\n")
cat(paste(round(initial_conditions$SNO3, 2), collapse=", "), "\n")
sink()

cat("Readable text format saved to:", output_txt, "\n")
cat("Note: This can be combined with other INITIAL_CONDITIONS metadata (PCR, ICDAT, etc.)\n")
cat("      and formatted using format_dssat_table() for writing to DSSAT .WHX file.\n")

# Summary
cat("\nSummary:\n")
cat(sprintf("  - Number of layers: %d\n", nrow(initial_conditions)))
cat(sprintf("  - Profile depth: 0-%d cm\n", max(initial_conditions$ICBL)))
cat(sprintf("  - Average SH2O: %.4f cm3/cm3\n", mean(initial_conditions$SH2O, na.rm = TRUE)))
cat(sprintf("  - Uniform SNO3: %.2f ppm\n", unique(initial_conditions$SNO3[!is.na(initial_conditions$SNO3)])))
cat(sprintf("  - Uniform SNH4: %.2f ppm\n", unique(initial_conditions$SNH4[!is.na(initial_conditions$SNH4)])))
