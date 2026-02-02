# Quick test script to verify calculations
# Manual calculation check

# Soil data from SOIL_PROFILE_LAYERS.csv
soil_data <- data.frame(
  SLB = c(5, 15, 30, 60, 100, 200),
  SLLL = c(0.108, 0.118, 0.133, 0.147, 0.146, 0.136),
  SDUL = c(0.236, 0.247, 0.262, 0.276, 0.274, 0.262),
  SBDM = c(1.29, 1.31, 1.34, 1.39, 1.45, 1.51)
)

# Calculate water content (90% available water)
percent_aw <- 90
SH2O <- soil_data$SLLL + (percent_aw / 100) * (soil_data$SDUL - soil_data$SLLL)
cat("SH2O values (cm³/cm³):\n")
print(round(SH2O, 4))

# Calculate nitrogen (100 kg/ha total)
total_n <- 100
layer_thicknesses <- c(soil_data$SLB[1], diff(soil_data$SLB))
total_depth <- max(soil_data$SLB)
weighted_bd <- sum(soil_data$SBDM * layer_thicknesses) / total_depth
cat("\nWeighted average bulk density:", weighted_bd, "g/cm³\n")
cat("Total profile depth:", total_depth, "cm\n")

# Nitrogen concentrations (uniform across all layers)
SNO3 <- (0.9 * total_n) / (0.1 * weighted_bd * total_depth)
SNH4 <- (0.1 * total_n) / (0.1 * weighted_bd * total_depth)
cat("\nSNO3 (uniform):", round(SNO3, 2), "ppm\n")
cat("SNH4 (uniform):", round(SNH4, 2), "ppm\n")

# Create final table
results <- data.frame(
  ICBL = soil_data$SLB,
  SH2O = round(SH2O, 4),
  SNO3 = round(SNO3, 2),
  SNH4 = round(SNH4, 2)
)

cat("\nFinal Initial Conditions:\n")
print(results)


