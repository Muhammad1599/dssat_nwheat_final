#!/usr/bin/env python3
"""
Generalized Duernast Comprehensive Visualization Package

Purpose: Creates 12-panel visualization for N-Wheat model results showing seasonal
         progression, stress factors, yield components, and validation against
         observed data for selected treatments (1, 8, 15).

Note: Only treatments 1 (Medium N), 8 (High N), and 15 (Control) are displayed.
      Supports both 2015 (Spring Wheat) and 2017 (Winter Wheat) experiments.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import re
from collections import Counter
from datetime import datetime, timedelta

# Set style for publication-quality visualization
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define which treatments to display (only 3 treatments)
SELECTED_TREATMENTS = [1, 8, 15]  # Treatment 1: Medium N, Treatment 8: High N, Treatment 15: Control

# Get configuration from environment variables (set by MASTER_WORKFLOW.py)
EXPERIMENT_YEAR = int(os.environ.get('DUERNAST_YEAR', 2015))
NORMALIZE_DAS = os.environ.get('DUERNAST_NORMALIZE_DAS', 'False').lower() == 'true'
FILE_PREFIX = os.environ.get('DUERNAST_FILE_PREFIX', 'TUDU1501')
OUTPUT_PREFIX = os.environ.get('DUERNAST_OUTPUT_PREFIX', 'duernast_2015')

print(f"[INFO] Visualization configuration:")
print(f"  Year: {EXPERIMENT_YEAR}")
print(f"  File Prefix: {FILE_PREFIX}")
print(f"  Output Prefix: {OUTPUT_PREFIX}")
print(f"  Normalize DAS: {NORMALIZE_DAS}")

def parse_summary_phenology():
    """Parse phenology stages and nitrogen levels from Summary.OUT for selected treatments"""
    
    stages = {}
    n_levels = {}
    
    if not Path('Summary.OUT').exists():
        print("[ERROR] Summary.OUT not found!")
        return None, None
    
    try:
        with open('Summary.OUT', 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if not lines:
            print("[ERROR] Summary.OUT is empty!")
            return None, None
        
        # Detect model type from Summary.OUT
        model_type = 'UNKNOWN'
        for line in lines[:10]:
            if 'WHAPS' in line:
                model_type = 'NWHEAT'
                break
        
        print(f"[INFO] Summary.OUT model type: {model_type}")
        
        if model_type != 'NWHEAT':
            print("[ERROR] Only N-Wheat (WHAPS) model is supported!")
            return None, None
        
        # Find data section - look for lines starting with numbers
        data_start = -1
        for i, line in enumerate(lines):
            # Skip header lines and look for data lines
            if line.strip() and line.strip()[0].isdigit() and not line.startswith('!'):
                data_start = i
                break
        
        if data_start == -1:
            print("[ERROR] Could not find data section in Summary.OUT")
            return None, None
        
        # Parse each treatment
        for i in range(data_start, min(data_start + 20, len(lines))):
            line = lines[i].strip()
            if not line or line.startswith('*'):
                continue
            
            parts = line.split()
            if len(parts) >= 51:  # Need at least 51 columns to get NICM
                try:
                    treatment = int(parts[1])
                    
                    # Only process selected treatments
                    if treatment not in SELECTED_TREATMENTS:
                        continue
                    
                    sdat = int(parts[16])  # Sowing date (SDAT)
                    pdat = int(parts[17])  # Planting date (PDAT)
                    edat = int(parts[18])  # Emergence date (EDAT)
                    adat = int(parts[19])  # Anthesis date (ADAT)
                    mdat = int(parts[20])  # Maturity date (MDAT)
                    hdat = int(parts[21])  # Harvest date (HDAT)
                    nicm = int(parts[50])   # Nitrogen applied (NICM)
                    
                    # Convert to days after sowing
                    def date_to_das(date, sdate):
                        if date == -99 or sdate == -99:
                            return -99
                        date_doy = date % 1000
                        sdate_doy = sdate % 1000
                        return date_doy - sdate_doy if date_doy >= sdate_doy else date_doy + 365 - sdate_doy
                    
                    stages[treatment] = {
                        'emergence_das': date_to_das(edat, pdat),
                        'anthesis_das': date_to_das(adat, pdat),
                        'maturity_das': date_to_das(mdat, pdat),
                        'harvest_das': date_to_das(hdat, pdat),
                        'sowing_date': sdat,
                        'planting_date': pdat
                    }
                    
                    # Store nitrogen level for this treatment
                    n_levels[treatment] = nicm
                    
                except (ValueError, IndexError) as e:
                    if len(stages) == 0:  # Only warn for first parse attempts
                        print(f"[WARNING] Skipped line {i}: {str(e)[:50]}")
                    continue
        
        if not stages or not n_levels:
            print("[ERROR] No phenology data parsed from Summary.OUT")
            return None, None
        
        return stages, n_levels
        
    except Exception as e:
        print(f"[ERROR] Could not parse Summary.OUT: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def get_consensus_stages(detailed_stages):
    """Get consensus phenology stages (most treatments have same timing)"""
    
    if not detailed_stages or len(detailed_stages) == 0:
        print("[WARNING] No phenology stages provided, using defaults")
        return {'emergence_das': 12, 'anthesis_das': 97, 'maturity_das': 144, 'harvest_das': 160}
    
    try:
        emergence_vals = [s['emergence_das'] for s in detailed_stages.values() if s.get('emergence_das', -99) != -99]
        anthesis_vals = [s['anthesis_das'] for s in detailed_stages.values() if s.get('anthesis_das', -99) != -99]
        maturity_vals = [s['maturity_das'] for s in detailed_stages.values() if s.get('maturity_das', -99) != -99]
        harvest_vals = [s['harvest_das'] for s in detailed_stages.values() if s.get('harvest_das', -99) != -99]
        
        print(f"[DEBUG] Emergence values: {set(emergence_vals) if emergence_vals else 'None'}")
        print(f"[DEBUG] Anthesis values: {set(anthesis_vals) if anthesis_vals else 'None'}")
        print(f"[DEBUG] Maturity values: {set(maturity_vals) if maturity_vals else 'None'}")
        print(f"[DEBUG] Harvest values: {set(harvest_vals) if harvest_vals else 'None'}")
        
        if not emergence_vals or not anthesis_vals or not maturity_vals:
            print("[WARNING] Missing phenology values, using defaults")
            return {'emergence_das': 12, 'anthesis_das': 97, 'maturity_das': 144, 'harvest_das': 160}
        
        consensus = {
            'emergence_das': Counter(emergence_vals).most_common(1)[0][0],
            'anthesis_das': Counter(anthesis_vals).most_common(1)[0][0],
            'maturity_das': Counter(maturity_vals).most_common(1)[0][0]
        }
        
        # Add harvest_das if available
        if harvest_vals:
            consensus['harvest_das'] = Counter(harvest_vals).most_common(1)[0][0]
        else:
            consensus['harvest_das'] = 160  # Default fallback
        
        print(f"[DEBUG] Consensus: {consensus}")
        
        return consensus
    
    except Exception as e:
        print(f"[WARNING] Error calculating consensus stages: {e}, using defaults")
        return {'emergence_das': 12, 'anthesis_das': 97, 'maturity_das': 144, 'harvest_das': 160}

def detect_model_type():
    """Detect if PlantGro.OUT is from N-Wheat model"""
    
    try:
        with open('PlantGro.OUT', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(500)  # Read first 500 chars
        
        if 'WHAPS' in content or 'N-Wheat' in content:
            return 'NWHEAT'
        else:
            print("[WARNING] Model type not detected as N-Wheat")
            return 'UNKNOWN'
    except Exception as e:
        print(f"[WARNING] Could not detect model type: {e}")
        return 'UNKNOWN'

def parse_temperature_data():
    """Parse Weather.OUT for temperature data (returns dictionary)"""
    
    if not Path('Weather.OUT').exists():
        print("[WARNING] Weather.OUT not found, temperature data unavailable")
        return {}
    
    try:
        weather_by_das = {}
        with open('Weather.OUT', 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if not lines:
            print("[WARNING] Weather.OUT is empty")
            return {}
        
        data_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('@YEAR DOY'):
                data_start = i + 1
                break
        
        if data_start == -1:
            print("[WARNING] Could not find data section in Weather.OUT")
            return {}
        
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith(('*', '@', '!')):
                continue
            
            parts = line.split()
            if len(parts) >= 12:
                try:
                    das = int(parts[2])
                    tavd = float(parts[11])  # TAVD = average daily temperature
                    if -50 <= tavd <= 60:  # Sanity check for temperature
                        weather_by_das[das] = tavd
                except (ValueError, IndexError):
                    continue
        
        if weather_by_das:
            print(f"[INFO] Loaded temperature data for {len(weather_by_das)} days")
        else:
            print("[WARNING] No temperature data parsed from Weather.OUT")
        
        return weather_by_das
    except Exception as e:
        print(f"[WARNING] Error parsing Weather.OUT: {e}")
        return {}

def parse_plantgro_data(n_levels=None):
    """Parse PlantGro.OUT for selected treatments (1, 8, 15) - N-Wheat model
    
    Args:
        n_levels: Dictionary mapping treatment number to N applied (kg/ha)
    """
    
    if not Path('PlantGro.OUT').exists():
        print("[ERROR] PlantGro.OUT not found!")
        return None
    
    # Detect model type
    model_type = detect_model_type()
    print(f"[INFO] Detected model type: {model_type}")
    
    if model_type != 'NWHEAT':
        print("[ERROR] Only N-Wheat model is supported!")
        return None
    
    # Load temperature data (shared across all treatments)
    weather_data = parse_temperature_data()
    
    try:
        with open('PlantGro.OUT', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content:
            print("[ERROR] PlantGro.OUT is empty!")
            return None
        
        # Split by RUN sections
        run_sections = re.split(r'\*RUN\s+\d+', content)
        
        treatments_data = {}
        
        # Generate treatment names using actual N levels from data
        fertilizer_types = {
            1: "Harnstoff", 2: "Harnstoff", 3: "Mixed",
            4: "AmmonSulf", 5: "Kalkamm", 6: "Kalkamm",
            7: "Kalkamm", 8: "Kalkamm", 9: "UAN",
            10: "Mixed", 11: "Kalkstick", 12: "Mixed",
            13: "UAN", 14: "UAN", 15: "Control"
        }
        
        treatment_names = {}
        for trt in range(1, 16):
            fert_type = fertilizer_types.get(trt, f"Trt{trt}")
            if n_levels and trt in n_levels:
                n_kg = n_levels[trt]
                if trt == 15:
                    treatment_names[trt] = f"Trt{trt}:Control-{n_kg}N"
                elif trt in [3, 10]:  # Problem treatments - mark with asterisk
                    treatment_names[trt] = f"Trt{trt}:{fert_type}-{n_kg}N*"
                else:
                    treatment_names[trt] = f"Trt{trt}:{fert_type}-{n_kg}N"
            else:
                # Fallback if N levels not provided
                treatment_names[trt] = f"Trt{trt}:{fert_type}"
        
        # Only process selected treatments (1, 8, 15)
        for run_num in range(1, min(16, len(run_sections))):
            if run_num not in SELECTED_TREATMENTS:
                continue  # Skip treatments not in selected list
                
            section = run_sections[run_num]
            treatment_name = treatment_names.get(run_num, f"Treatment{run_num}")
            
            lines = section.split('\n')
            data_start = -1
            
            for i, line in enumerate(lines):
                if line.strip().startswith('@YEAR DOY'):
                    data_start = i + 1
                    break
            
            if data_start == -1:
                continue
            
            data_rows = []
            for i in range(data_start, len(lines)):
                line = lines[i].strip()
                
                if not line or line.startswith('*RUN') or line.startswith('$'):
                    break
                
                if line.startswith('@') or line.startswith('*'):
                    continue
                
                parts = line.split()
                
                try:
                    # N-Wheat column positions (WHAPS048 model)
                    if len(parts) >= 35:
                        das = int(parts[2])
                        # Get temperature from Weather.OUT
                        tmean = float(weather_data.get(das, 15.0)) if len(weather_data) > 0 else 15.0
                        
                        # N-Wheat water stress variables (matching professor's approach)
                        wspd = max(0.0, float(parts[18]))  # Water stress photosynthesis (cumulative, can be > 1)
                        wsgd = max(0.0, float(parts[19]))  # Water stress grain filling (daily 0-1 factor)
                        slft = max(0.0, min(1.0, float(parts[20])))  # Soil water factor for leaves (0-1)
                        nstd = max(0.0, float(parts[21]))  # N stress (can be > 1)
                        
                        # N-Wheat doesn't have daily N stress factor, so derive from cumulative
                        if nstd == 0:
                            nftd = 1.0
                        else:
                            nftd = max(0.0, min(1.0, 1.0 - (nstd / 100.0)))
                        
                        # Water stress factor: use minimum of photosynthesis and leaf factors
                        if wspd > 0 or slft > 0:
                            wftd = min(wspd, slft)
                        else:
                            wftd = 1.0
                        
                        # Extract values with bounds checking
                        cwad = max(0.0, float(parts[12]))  # Total biomass (non-negative)
                        hwad = max(0.0, float(parts[9]))   # Grain weight at 0% moisture (dry matter)
                        hiad = max(0.0, min(1.0, float(parts[16])))  # Harvest index (0-1)
                        h_ad = max(0.0, float(parts[13]))  # Grain number (non-negative)
                        gwgd = max(0.0, float(parts[15]))  # Grain weight per grain (non-negative)
                        rdpd = max(0.0, float(parts[34]))  # Root depth (non-negative)
                        
                        data_rows.append({
                            'DAS': das,
                            'TMEAN': tmean,
                            'CWAD': cwad,
                            'HWAD': hwad,  # Grain yield at 0% moisture (dry weight)
                            'HIAD': hiad,
                            'H#AD': h_ad,
                            'GWGD': gwgd,
                            'RDPD': rdpd,
                            'WFTD': wftd,
                            'WFPD': wspd,
                            'WFGD': wsgd,
                            'NFTD': nftd,
                            'NSTD': nstd,
                        })
                except (ValueError, IndexError) as e:
                    if run_num == 1 and len(data_rows) < 5:  # Debug first treatment only
                        print(f"[DEBUG] Skipped line (treatment {run_num}): {str(e)[:100]}")
                    continue
            
            if data_rows:
                df = pd.DataFrame(data_rows)
                
                if df.empty:
                    print(f"[WARNING] Empty dataframe for {treatment_name}")
                    continue
                
                # CRITICAL: Normalize DAS to start from 0 if configured (for multi-year simulations)
                if NORMALIZE_DAS and 'DAS' in df.columns and len(df) > 0:
                    min_das = df['DAS'].min()
                    df['DAS'] = df['DAS'] - min_das  # Normalize to start at 0
                    print(f"[INFO] Normalized DAS for {treatment_name} (min was {min_das})")
                
                # Calculate derived variables with safe operations
                df['grain_size_mg'] = df['GWGD'].clip(lower=0)
                
                # Stress calculations
                df['daily_water_stress'] = df['WFTD']  # 1=optimal, 0=stressed
                # True cumulative water stress (cumulative sum of WFGD)
                df['cumulative_water_stress'] = df['WFGD'].cumsum()
                df['daily_nitrogen_stress'] = df['NFTD']  # 1=optimal, 0=stressed
                # True cumulative nitrogen stress (cumulative sum of NSTD)
                df['cumulative_nitrogen_stress'] = df['NSTD'].cumsum()
                
                treatments_data[treatment_name] = df
        
        if not treatments_data:
            print("[ERROR] No treatment data was parsed successfully!")
            return None
        
        print(f"[INFO] Successfully parsed {len(treatments_data)} treatments")
        return treatments_data
        
    except Exception as e:
        print(f"[ERROR] Error parsing PlantGro.OUT: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_nitrogen_data(n_levels=None):
    """Parse PlantN.OUT for nitrogen dynamics
    
    Args:
        n_levels: Dictionary mapping treatment number to N applied (kg/ha)
    """
    
    if not Path('PlantN.OUT').exists():
        print("[WARNING] PlantN.OUT not found, skipping nitrogen data")
        return None
    
    try:
        with open('PlantN.OUT', 'r') as f:
            content = f.read()
        
        run_sections = re.split(r'\*RUN\s+\d+', content)
        
        nitrogen_data = {}
        
        # Generate treatment names using actual N levels from data
        fertilizer_types = {
            1: "Harnstoff", 2: "Harnstoff", 3: "Mixed",
            4: "AmmonSulf", 5: "Kalkamm", 6: "Kalkamm",
            7: "Kalkamm", 8: "Kalkamm", 9: "UAN",
            10: "Mixed", 11: "Kalkstick", 12: "Mixed",
            13: "UAN", 14: "UAN", 15: "Control"
        }
        
        treatment_names = {}
        for trt in range(1, 16):
            fert_type = fertilizer_types.get(trt, f"Trt{trt}")
            if n_levels and trt in n_levels:
                n_kg = n_levels[trt]
                if trt == 15:
                    treatment_names[trt] = f"Trt{trt}:Control-{n_kg}N"
                elif trt in [3, 10]:
                    treatment_names[trt] = f"Trt{trt}:{fert_type}-{n_kg}N*"
                else:
                    treatment_names[trt] = f"Trt{trt}:{fert_type}-{n_kg}N"
            else:
                treatment_names[trt] = f"Trt{trt}:{fert_type}"
        
        # Only process selected treatments (1, 8, 15)
        for run_num in range(1, min(16, len(run_sections))):
            if run_num not in SELECTED_TREATMENTS:
                continue  # Skip treatments not in selected list
                
            section = run_sections[run_num]
            treatment_name = treatment_names.get(run_num, f"Treatment{run_num}")
            
            lines = section.split('\n')
            data_start = -1
            
            for i, line in enumerate(lines):
                if line.strip().startswith('@YEAR DOY'):
                    data_start = i + 1
                    break
            
            if data_start == -1:
                continue
            
            data_rows = []
            for i in range(data_start, len(lines)):
                line = lines[i].strip()
                if not line or line.startswith('*') or line.startswith('@'):
                    break
                
                parts = line.split()
                if len(parts) >= 6:  # Need at least columns 0-5 for GNAD
                    try:
                        data_rows.append({
                            'DAS': int(parts[2]),
                            'CNAD': float(parts[4]),  # Crop N (total: leaves+stems+grains)
                            'GNAD': float(parts[5]),  # Grain N (only grains) - CORRECT INDEX!
                            'nitrogen_uptake': float(parts[5]),  # Use GNAD for comparison with observed
                        })
                    except (ValueError, IndexError):
                        continue
            
            if data_rows:
                df = pd.DataFrame(data_rows)
                
                # CRITICAL: Normalize DAS to start from 0 if configured
                if NORMALIZE_DAS and 'DAS' in df.columns and len(df) > 0:
                    min_das = df['DAS'].min()
                    df['DAS'] = df['DAS'] - min_das  # Normalize to start at 0
                
                nitrogen_data[treatment_name] = df
        
        return nitrogen_data
        
    except Exception as e:
        print(f"[WARNING] Could not parse nitrogen data: {e}")
        return None

def parse_weather_data():
    """Parse Weather.OUT for environmental variables"""
    
    if not Path('Weather.OUT').exists():
        print("[WARNING] Weather.OUT not found")
        return None
    
    try:
        with open('Weather.OUT', 'r') as f:
            content = f.read()
        
        # Split by RUN sections - just get first one (weather same for all)
        run_sections = re.split(r'\*RUN\s+\d+', content)
        
        if len(run_sections) < 2:
            return None
        
        section = run_sections[1]
        lines = section.split('\n')
        data_start = -1
        
        for i, line in enumerate(lines):
            if line.strip().startswith('@YEAR DOY'):
                data_start = i + 1
                break
        
        if data_start == -1:
            return None
        
        data_rows = []
        for i in range(data_start, len(lines)):
            line = lines[i].strip()
            if not line or line.startswith('*') or line.startswith('@'):
                break
            
            parts = line.split()
            if len(parts) >= 12:
                try:
                    data_rows.append({
                        'DAS': int(parts[2]),
                        'PRED': float(parts[3]),  # Precipitation
                        'SRAD': float(parts[6]),  # Solar radiation (SRAD column 6)
                        'TMAX': float(parts[9]),  # Max temp (TMXD column 9)
                        'TMIN': float(parts[10]),  # Min temp (TMND column 10)
                    })
                except (ValueError, IndexError):
                    continue
        
        if data_rows:
            df = pd.DataFrame(data_rows)
            
            # Calculate TMEAN from TMAX and TMIN
            if 'TMAX' in df.columns and 'TMIN' in df.columns:
                df['TMEAN'] = (df['TMAX'] + df['TMIN']) / 2.0
            
            # CRITICAL: Normalize DAS to start from 0 if configured
            if NORMALIZE_DAS and 'DAS' in df.columns and len(df) > 0:
                min_das = df['DAS'].min()
                df['DAS'] = df['DAS'] - min_das  # Normalize to start at 0
            
            return df
        
    except Exception as e:
        print(f"[WARNING] Could not parse weather: {e}")
        return None

def parse_observed_data(n_levels=None):
    """Parse observed data (yield, grain weight, grain nitrogen) for validation
    
    Args:
        n_levels: Dictionary mapping treatment number to N applied (kg/ha)
    
    Returns:
        Dictionary with treatment names as keys, containing:
        - yield: mean harvest yield (kg/ha)
        - grain_weight: mean grain weight (mg) if available
        - grain_nitrogen: mean grain N (kg/ha) if available
        - std_yield, std_grain_weight, std_grain_nitrogen: standard deviations
        - n: number of replications
    """
    
    try:
        # Try .WHT file first (has more data: yield + grain weight + grain nitrogen)
        wht_file = Path(f'{FILE_PREFIX}.WHT')
        wha_file = Path(f'{FILE_PREFIX}.WHA')
        
        observed_raw = {}
        has_grain_data = False
        
        if wht_file.exists():
            print(f"  [OK] Using {FILE_PREFIX}.WHT (detailed data: yield + grain weight + grain N + grain number)")
            with open(f'{FILE_PREFIX}.WHT', 'r') as f:
                lines = f.readlines()
            
            # Parse .WHT format: TRNO DATE HWAD GWGD GNAD G#AD
            observed_harvest_date = None
            parse_errors = []
            for line_num, line in enumerate(lines, 1):
                if line.strip() and not line.startswith('@') and not line.startswith('*') and not line.startswith('!'):
                    parts = line.split()
                    if len(parts) >= 5:  # Need at least 5 columns (G#AD is optional)
                        try:
                            treatment = int(parts[0])
                            harvest_date = int(parts[1])  # Extract observed harvest date
                            
                            # Validate treatment number
                            if treatment < 1 or treatment > 15:
                                parse_errors.append(f"Line {line_num}: Invalid treatment number {treatment}")
                                continue
                            
                            # Store observed harvest date (should be same for all treatments)
                            if observed_harvest_date is None:
                                observed_harvest_date = harvest_date
                            elif observed_harvest_date != harvest_date:
                                print(f"  [WARNING] Line {line_num}: Harvest date mismatch ({harvest_date} vs {observed_harvest_date})")
                            
                            # Only process selected treatments
                            if treatment not in SELECTED_TREATMENTS:
                                continue
                            
                            hwad_obs = int(parts[2])      # Harvest weight (kg/ha)
                            gwgd_obs = int(parts[3])      # Grain weight (mg/grain)
                            gnad_obs = int(parts[4])      # Grain nitrogen (kg N/ha)
                            gnum_obs = int(parts[5]) if len(parts) > 5 else None  # Grain number (G#AD) if available
                            
                            # Validate data ranges
                            if hwad_obs < 0 or hwad_obs > 20000:
                                parse_errors.append(f"Line {line_num}: Suspicious yield value {hwad_obs} kg/ha")
                            if gwgd_obs < 0 or gwgd_obs > 100:
                                parse_errors.append(f"Line {line_num}: Suspicious grain weight {gwgd_obs} mg/grain")
                            if gnad_obs < 0 or gnad_obs > 500:
                                parse_errors.append(f"Line {line_num}: Suspicious grain N {gnad_obs} kg N/ha")
                            if gnum_obs is not None and (gnum_obs < 0 or gnum_obs > 50000):
                                parse_errors.append(f"Line {line_num}: Suspicious grain number {gnum_obs}")
                            
                            if treatment not in observed_raw:
                                observed_raw[treatment] = {
                                    'yield': [],
                                    'grain_weight': [],
                                    'grain_nitrogen': [],
                                    'grain_number': []
                                }
                            observed_raw[treatment]['yield'].append(hwad_obs)
                            observed_raw[treatment]['grain_weight'].append(gwgd_obs)
                            observed_raw[treatment]['grain_nitrogen'].append(gnad_obs)
                            if gnum_obs is not None:
                                observed_raw[treatment]['grain_number'].append(gnum_obs)
                            has_grain_data = True
                        except (ValueError, IndexError) as e:
                            parse_errors.append(f"Line {line_num}: Parse error - {str(e)}")
                            continue
            
            # Report parsing errors if any
            if parse_errors:
                print(f"  [WARNING] {len(parse_errors)} parsing issues found:")
                for error in parse_errors[:5]:  # Show first 5 errors
                    print(f"    - {error}")
                if len(parse_errors) > 5:
                    print(f"    ... and {len(parse_errors) - 5} more")
        
        elif wha_file.exists():
            print(f"  [OK] Using {FILE_PREFIX}.WHA (yield only)")
            with open(f'{FILE_PREFIX}.WHA', 'r') as f:
                lines = f.readlines()
            
            # Parse .WHA format: TRNO HDAT HWAM
            for line in lines:
                if line.strip() and not line.startswith('@') and not line.startswith('*') and not line.startswith('!'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            treatment = int(parts[0])
                            
                            # Only process selected treatments
                            if treatment not in SELECTED_TREATMENTS:
                                continue
                            
                            hwam_obs = int(parts[2])
                            
                            if treatment not in observed_raw:
                                observed_raw[treatment] = {
                                    'yield': [],
                                    'grain_weight': [],
                                    'grain_nitrogen': []
                                }
                            observed_raw[treatment]['yield'].append(hwam_obs)
                        except (ValueError, IndexError):
                            continue
        
        else:
            print(f"  [WARNING] No observed data file found ({FILE_PREFIX}.WHT or .WHA)")
            return None
        
        # Generate treatment names using actual N levels from data
        fertilizer_types = {
            1: "Harnstoff", 2: "Harnstoff", 3: "Mixed",
            4: "AmmonSulf", 5: "Kalkamm", 6: "Kalkamm",
            7: "Kalkamm", 8: "Kalkamm", 9: "UAN",
            10: "Mixed", 11: "Kalkstick", 12: "Mixed",
            13: "UAN", 14: "UAN", 15: "Control"
        }
        
        # Calculate means and standard deviations
        observed_means = {}
        print(f"\n  [VERIFY] Observed data summary for selected treatments:")
        for treatment in sorted(observed_raw.keys()):
            data = observed_raw[treatment]
            fert_type = fertilizer_types.get(treatment, f"Trt{treatment}")
            
            if n_levels and treatment in n_levels:
                n_kg = n_levels[treatment]
                if treatment == 15:
                    trt_name = f"Trt{treatment}:Control-{n_kg}N"
                elif treatment in [3, 10]:
                    trt_name = f"Trt{treatment}:{fert_type}-{n_kg}N*"
                else:
                    trt_name = f"Trt{treatment}:{fert_type}-{n_kg}N"
            else:
                trt_name = f"Trt{treatment}:{fert_type}"
            
            yield_mean = np.mean(data['yield'])
            yield_std = np.std(data['yield'])
            yield_n = len(data['yield'])
            
            observed_means[trt_name] = {
                'yield': yield_mean,
                'std_yield': yield_std,
                'n': yield_n
            }
            
            # Print verification for each treatment
            print(f"    Treatment {treatment}: Yield = {yield_mean:.0f} ± {yield_std:.0f} kg/ha (n={yield_n})")
            
            # Add grain weight data if available
            if has_grain_data and len(data['grain_weight']) > 0:
                grain_wt_mean = np.mean(data['grain_weight'])
                grain_wt_std = np.std(data['grain_weight'])
                observed_means[trt_name]['grain_weight'] = grain_wt_mean
                observed_means[trt_name]['std_grain_weight'] = grain_wt_std
                print(f"      Grain weight = {grain_wt_mean:.1f} ± {grain_wt_std:.1f} mg/grain")
            
            # Add grain nitrogen data if available
            if has_grain_data and len(data['grain_nitrogen']) > 0:
                grain_n_mean = np.mean(data['grain_nitrogen'])
                grain_n_std = np.std(data['grain_nitrogen'])
                observed_means[trt_name]['grain_nitrogen'] = grain_n_mean
                observed_means[trt_name]['std_grain_nitrogen'] = grain_n_std
                print(f"      Grain N = {grain_n_mean:.1f} ± {grain_n_std:.1f} kg N/ha")
            
            # Add grain number data if available
            if has_grain_data and len(data.get('grain_number', [])) > 0:
                grain_num_mean = np.mean(data['grain_number'])
                grain_num_std = np.std(data['grain_number'])
                observed_means[trt_name]['grain_number'] = grain_num_mean
                observed_means[trt_name]['std_grain_number'] = grain_num_std
                print(f"      Grain number = {grain_num_mean:.0f} ± {grain_num_std:.0f} grains/m²")
        
        # Maintain backward compatibility with 'std' key for yield
        for trt_name in observed_means:
            observed_means[trt_name]['std'] = observed_means[trt_name]['std_yield']
        
        # Add observed harvest date to the returned dictionary
        if observed_harvest_date is not None:
            observed_means['_observed_harvest_date'] = observed_harvest_date
            print(f"  [OK] Observed harvest date: {observed_harvest_date}")
        
        if has_grain_data:
            grain_num_info = " + grain number" if any('grain_number' in data and len(data.get('grain_number', [])) > 0 for data in observed_raw.values()) else ""
            print(f"  [OK] Loaded yield + grain weight + grain nitrogen{grain_num_info} for {len(observed_means)} treatments")
        else:
            print(f"  [OK] Loaded yield data for {len(observed_means)} treatments")
        
        return observed_means
        
    except Exception as e:
        print(f"[WARNING] Could not parse observed data: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_treatment_names(n_levels):
    """Generate treatment names using actual N levels from data
    
    Args:
        n_levels: Dictionary mapping treatment number to N applied (kg/ha)
    
    Returns:
        Dictionary mapping treatment number to display name
    """
    fertilizer_types = {
        1: "Harnstoff", 2: "Harnstoff", 3: "Mixed",
        4: "AmmonSulf", 5: "Kalkamm", 6: "Kalkamm",
        7: "Kalkamm", 8: "Kalkamm", 9: "UAN",
        10: "Mixed", 11: "Kalkstick", 12: "Mixed",
        13: "UAN", 14: "UAN", 15: "Control"
    }
    
    treatment_names = {}
    for trt in range(1, 16):
        fert_type = fertilizer_types.get(trt, f"Trt{trt}")
        if n_levels and trt in n_levels:
            n_kg = n_levels[trt]
            if trt == 15:
                treatment_names[trt] = f"Trt{trt}:Control-{n_kg}N"
            elif trt in [3, 10]:
                treatment_names[trt] = f"Trt{trt}:{fert_type}-{n_kg}N*"
            else:
                treatment_names[trt] = f"Trt{trt}:{fert_type}-{n_kg}N"
        else:
            treatment_names[trt] = f"Trt{trt}:{fert_type}"
    
    return treatment_names

def get_treatment_styles(treatment_names_dict):
    """Define visual styles for selected treatments (1, 8, 15)
    
    Args:
        treatment_names_dict: Dictionary mapping treatment numbers to names
    """
    
    # Define distinct colors for the 3 selected treatments
    treatment_colors = {
        1: '#1f77b4',  # Blue - Medium N
        8: '#ff7f0e',  # Orange - High N
        15: '#000000'  # Black - Control
    }
    
    styles = {}
    
    for trt_num, trt_name in treatment_names_dict.items():
        # Control treatment (15) - distinct style (thicker, dotted)
        if trt_num == 15:
            styles[trt_name] = {
                'color': treatment_colors[15],
                'linestyle': ':',
                'linewidth': 2.0,
                'alpha': 0.85,
                'marker': 'o'
            }
        # Treatment 1 (Medium N) - solid blue line
        elif trt_num == 1:
            styles[trt_name] = {
                'color': treatment_colors[1],
                'linestyle': '-',
                'linewidth': 1.5,
                'alpha': 0.8,
                'marker': None
            }
        # Treatment 8 (High N) - solid orange line
        elif trt_num == 8:
            styles[trt_name] = {
                'color': treatment_colors[8],
                'linestyle': '-',
                'linewidth': 1.5,
                'alpha': 0.8,
                'marker': None
            }
    
    return styles

def create_comprehensive_visualization(treatments_data, phenology_stages, consensus_stages, 
                                     weather_data=None, nitrogen_data=None, observed_data=None, n_levels=None):
    """Create comprehensive 12-panel visualization for Duernast (3 selected treatments)"""
    
    if not treatments_data:
        print("[ERROR] No treatment data available!")
        return None
    
    print(f"\n[INFO] Creating visualization for {len(treatments_data)} treatments...")
    
    # Filter treatments_data to only include selected treatments
    filtered_treatments_data = {}
    for trt_name, df in treatments_data.items():
        trt_num = int(trt_name.split('Trt')[1].split(':')[0])
        if trt_num in SELECTED_TREATMENTS:
            filtered_treatments_data[trt_name] = df
    treatments_data = filtered_treatments_data
    
    # Filter phenology_stages to only include selected treatments
    if phenology_stages:
        filtered_phenology_stages = {trt: stages for trt, stages in phenology_stages.items() 
                                     if trt in SELECTED_TREATMENTS}
        phenology_stages = filtered_phenology_stages
    
    # Filter nitrogen_data to only include selected treatments
    if nitrogen_data:
        filtered_nitrogen_data = {}
        for trt_name, df in nitrogen_data.items():
            trt_num = int(trt_name.split('Trt')[1].split(':')[0])
            if trt_num in SELECTED_TREATMENTS:
                filtered_nitrogen_data[trt_name] = df
        nitrogen_data = filtered_nitrogen_data
    
    # Filter observed_data to only include selected treatments
    observed_harvest_date = None
    if observed_data:
        filtered_observed_data = {}
        for trt_name, data in observed_data.items():
            if trt_name == '_observed_harvest_date':
                observed_harvest_date = data
                continue
            trt_num = int(trt_name.split('Trt')[1].split(':')[0])
            if trt_num in SELECTED_TREATMENTS:
                filtered_observed_data[trt_name] = data
        observed_data = filtered_observed_data
        if observed_harvest_date is not None:
            observed_data['_observed_harvest_date'] = observed_harvest_date
    
    # Generate treatment names from N levels (only for selected treatments)
    if n_levels:
        treatment_names_dict = generate_treatment_names(n_levels)
        # Filter to only selected treatments
        treatment_names_dict = {trt: name for trt, name in treatment_names_dict.items() 
                               if trt in SELECTED_TREATMENTS}
    else:
        # Fallback to extracting from treatment keys
        treatment_names_dict = {int(k.split('Trt')[1].split(':')[0]): k 
                               for k in treatments_data.keys() if 'Trt' in k}
    
    treatment_styles = get_treatment_styles(treatment_names_dict)
    
    # Create figure with 12 vertical panels (weather, stress, growth, validation)
    fig, axes = plt.subplots(12, 1, figsize=(18, 36))
    fig.suptitle(f'Duernast {EXPERIMENT_YEAR} - Comprehensive Seasonal Analysis\n' + 
                 '3 Selected Treatments (Control, Medium N, High N) × Multiple Variables', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # Get consensus stages for vertical lines
    if consensus_stages:
        emergence_das = consensus_stages['emergence_das']
        anthesis_das = consensus_stages['anthesis_das']
        maturity_das = consensus_stages['maturity_das']
        harvest_das = consensus_stages.get('harvest_das', 160)
        print(f"[INFO] Phenology: Emergence={emergence_das}, Anthesis={anthesis_das}, Maturity={maturity_das}, Harvest={harvest_das}")
    else:
        emergence_das, anthesis_das, maturity_das, harvest_das = 7, 101, 144, 160
    
    # Override harvest_das with observed harvest date if available AND it's after maturity
    if observed_data and '_observed_harvest_date' in observed_data:
        observed_harvest_date_val = observed_data['_observed_harvest_date']
        # Get planting date from phenology_stages
        if phenology_stages and len(phenology_stages) > 0:
            first_stage = list(phenology_stages.values())[0]
            planting_date = first_stage.get('planting_date')
            if planting_date and planting_date != -99:
                # Calculate DAS from planting date
                def date_to_das(date, pdate):
                    if date == -99 or pdate == -99:
                        return -99
                    date_doy = date % 1000
                    pdate_doy = pdate % 1000
                    return date_doy - pdate_doy if date_doy >= pdate_doy else date_doy + 365 - pdate_doy
                
                observed_harvest_das = date_to_das(observed_harvest_date_val, planting_date)
                # Only use observed harvest date if it's after maturity (makes biological sense)
                if observed_harvest_das != -99 and observed_harvest_das >= maturity_das:
                    harvest_das = observed_harvest_das
                    print(f"[INFO] Using observed harvest date: {observed_harvest_date_val} = {harvest_das} DAS (after maturity at {maturity_das} DAS)")
                elif observed_harvest_das != -99:
                    print(f"[WARNING] Observed harvest date {observed_harvest_date_val} = {observed_harvest_das} DAS is before maturity ({maturity_das} DAS). Using simulated harvest date instead.")
    
    # STEP 1: Calculate consistent x-axis range and ticks for ALL panels
    # Find maximum DAS across all treatment data
    if treatments_data:
        max_das = max([df['DAS'].max() for df in treatments_data.values()])
    else:
        max_das = 150  # Fallback
    
    # Calculate consistent x-axis ticks for all panels (reduced density for readability)
    import numpy as np
    # Create ticks: 0, key phenology stages, and every 40 days (less dense)
    consistent_xticks = [0]
    # Add key phenology stages
    if emergence_das > 0 and emergence_das <= max_das:
        consistent_xticks.append(emergence_das)
    if anthesis_das not in consistent_xticks and anthesis_das <= max_das:
        consistent_xticks.append(anthesis_das)
    if maturity_das not in consistent_xticks and maturity_das <= max_das:
        consistent_xticks.append(maturity_das)
    if harvest_das not in consistent_xticks and harvest_das <= max_das:
        consistent_xticks.append(harvest_das)
    # Add ticks every 40 days (less dense than 20 days)
    for tick in range(40, int(max_das) + 1, 40):
        if tick not in consistent_xticks:
            consistent_xticks.append(tick)
    # Ensure max_das is included if not already
    if max_das not in consistent_xticks:
        consistent_xticks.append(int(max_das))
    
    consistent_xticks = sorted(list(set(consistent_xticks)))  # Remove duplicates and sort
    
    # Filter out ticks that are too close together (within 3 days) to prevent overlapping labels
    filtered_xticks = [consistent_xticks[0]]  # Always keep first tick (0)
    for tick in consistent_xticks[1:]:
        if tick - filtered_xticks[-1] >= 3:  # Only add if at least 3 days apart
            filtered_xticks.append(tick)
    consistent_xticks = filtered_xticks
    
    consistent_xlim = (-5, max_das + 5)  # Consistent x-axis limits for all panels
    
    print(f"[INFO] Consistent x-axis: range={consistent_xlim}, ticks={consistent_xticks}")
    
    # Define plot configurations
    plot_configs = [
        # SECTION 1: Environmental Drivers
        {'idx': 0, 'var': 'weather', 'title': 'a) Weather Pattern (Tmax, Tmin, Tmean, Rain, Solar Rad)', 'ylabel': 'Temperature (°C), Rain (mm), Rad (MJ/m²)'},
        {'idx': 1, 'var': 'cumulative_water_stress', 'title': 'b) Cumulative Water Stress (cumulative WSGD)', 'ylabel': 'Cumulative WSGD\n(0=optimal, higher=more stress)'},
        {'idx': 2, 'var': 'cumulative_nitrogen_stress', 'title': 'c) Cumulative Nitrogen Stress (cumulative NSTD)', 'ylabel': 'Cumulative NSTD\n(0=optimal, higher=more stress)'},
        
        # SECTION 2: Crop Growth Responses
        {'idx': 3, 'var': 'HWAD', 'title': 'd) Grain Yield at 0% moisture (dry weight) (lines=simulated, circles=observed)', 'ylabel': 'Grain Yield (kg/ha @ 0% moisture)'},
        {'idx': 4, 'var': 'CWAD', 'title': 'e) Total Biomass Development', 'ylabel': 'Biomass (kg/ha)'},
        {'idx': 5, 'var': 'HIAD', 'title': 'f) Harvest Index', 'ylabel': 'Harvest Index (0-1)'},
        {'idx': 6, 'var': 'H#AD', 'title': 'g) Grain Number (lines=simulated, circles=observed)', 'ylabel': 'Grains/m²'},
        {'idx': 7, 'var': 'grain_size_mg', 'title': 'h) Grain Weight (lines=simulated, circles=observed)', 'ylabel': 'mg/grain'},
        {'idx': 8, 'var': 'RDPD', 'title': 'i) Root Depth', 'ylabel': 'Root Depth (m)'},
        {'idx': 9, 'var': 'nitrogen_uptake', 'title': 'j) Grain Nitrogen (lines=simulated, circles=observed)', 'ylabel': 'Grain N (kg/ha)'},
        {'idx': 10, 'var': 'grain_protein', 'title': 'k) Grain Protein (lines=simulated, circles=observed)', 'ylabel': 'Grain Protein (%)'},
        
        # SECTION 3: Summary & Validation
        {'idx': 11, 'var': 'yield_comparison', 'title': 'l) Simulated vs Observed Yields (0% moisture)', 'ylabel': 'Yield (kg/ha @ 0% moisture)'},
    ]
    
    # Create each panel
    for config in plot_configs:
        ax = axes[config['idx']]
        var = config['var']
        
        # Special panels
        if var == 'weather':
            # Combined weather plot with Tmax, Tmin, Tmean, Rain, and Solar Rad
            if weather_data is not None:
                ax2 = ax.twinx()
                ax3 = ax.twinx()
                ax3.spines['right'].set_position(('outward', 60))
                
                # Filter weather data to only show up to harvest date
                weather_filtered = weather_data[weather_data['DAS'] <= harvest_das].copy()
                
                ax.plot(weather_filtered['DAS'], weather_filtered['TMAX'], 
                       color='red', linewidth=1.2, label='Tmax', alpha=0.8)
                ax.plot(weather_filtered['DAS'], weather_filtered['TMIN'],
                       color='blue', linewidth=1.2, label='Tmin', alpha=0.8)
                if 'TMEAN' in weather_filtered.columns:
                    ax.plot(weather_filtered['DAS'], weather_filtered['TMEAN'],
                           color='green', linewidth=1.5, label='Tmean', alpha=0.9, linestyle='--')
                ax2.bar(weather_filtered['DAS'], weather_filtered['PRED'],
                       color='skyblue', alpha=0.3, label='Rain', width=1.0)
                ax3.plot(weather_filtered['DAS'], weather_filtered['SRAD'],
                        color='orange', linewidth=1.0, label='Solar Rad', alpha=0.7)
                
                ax.set_ylabel('Temperature (°C)', color='red')
                ax2.set_ylabel('Precipitation (mm)', color='blue')
                ax3.set_ylabel('Solar Rad (MJ/m²)', color='orange')
                ax.legend(loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3, linewidth=0.8)
            
        elif var == 'nitrogen_uptake':
            # Grain nitrogen uptake from PlantN data (GNAD - grain N only, matches observed)
            if nitrogen_data:
                for trt_name, df in nitrogen_data.items():
                    style = treatment_styles.get(trt_name, {})
                    # Filter nitrogen data to only show up to harvest date
                    df_filtered = df[df['DAS'] <= harvest_das].copy()
                    ax.plot(df_filtered['DAS'], df_filtered['GNAD'],
                           color=style.get('color', 'black'),
                           linestyle=style.get('linestyle', '-'),
                           linewidth=style.get('linewidth', 0.9),
                           alpha=style.get('alpha', 0.7),
                           label=trt_name if len(nitrogen_data) <= 6 else '')
                
                # Add observed grain nitrogen points at harvest
                if observed_data and consensus_stages:
                    harvest_das = consensus_stages.get('harvest_das', 160)
                    for trt_name in nitrogen_data.keys():
                        if trt_name in observed_data and 'grain_nitrogen' in observed_data[trt_name]:
                            style = treatment_styles.get(trt_name, {})
                            obs_grain_n = observed_data[trt_name]['grain_nitrogen']
                            obs_std_n = observed_data[trt_name].get('std_grain_nitrogen', 0)
                            
                            ax.scatter([harvest_das], [obs_grain_n], 
                                     color=style.get('color', 'black'),
                                     marker='o', s=80, alpha=0.9,
                                     edgecolors='white', linewidth=1.5,
                                     zorder=10)
                            
                            if obs_std_n > 0:
                                ax.errorbar([harvest_das], [obs_grain_n], 
                                          yerr=[obs_std_n],
                                          color=style.get('color', 'black'),
                                          fmt='none', capsize=4, alpha=0.6,
                                          linewidth=1.5, zorder=9)
                    
                    ax.scatter([], [], color='none', marker='o', s=80, 
                             edgecolors='black', linewidth=1.5,
                             label='Observed grain N (circles)', alpha=0.9)
        
        elif var == 'grain_protein':
            # Grain protein calculated as (GNAD / HWAD) * 100 * 5.75
            if nitrogen_data and treatments_data:
                for trt_name in nitrogen_data.keys():
                    if trt_name in treatments_data:
                        style = treatment_styles.get(trt_name, {})
                        
                        df_n = nitrogen_data[trt_name].copy()
                        df_g = treatments_data[trt_name].copy()
                        
                        df_merged = pd.merge(df_n[['DAS', 'GNAD']], df_g[['DAS', 'HWAD']], on='DAS', how='inner')
                        
                        df_merged['grain_protein'] = df_merged.apply(
                            lambda row: (row['GNAD'] / row['HWAD'] * 100 * 5.75) if row['HWAD'] > 0 else 0,
                            axis=1
                        )
                        
                        df_filtered = df_merged[df_merged['DAS'] <= harvest_das].copy()
                        
                        ax.plot(df_filtered['DAS'], df_filtered['grain_protein'],
                               color=style.get('color', 'black'),
                               linestyle=style.get('linestyle', '-'),
                               linewidth=style.get('linewidth', 0.9),
                               alpha=style.get('alpha', 0.7),
                               label=trt_name if len(nitrogen_data) <= 6 else '')
                
                # Add observed grain protein points at harvest
                if observed_data and consensus_stages:
                    harvest_das = consensus_stages.get('harvest_das', 160)
                    for trt_name in nitrogen_data.keys():
                        if trt_name in observed_data and 'grain_nitrogen' in observed_data[trt_name]:
                            if trt_name in treatments_data:
                                style = treatment_styles.get(trt_name, {})
                                
                                obs_grain_n = observed_data[trt_name]['grain_nitrogen']
                                sim_grain_yield = treatments_data[trt_name][treatments_data[trt_name]['DAS'] <= harvest_das]['HWAD'].iloc[-1] if len(treatments_data[trt_name][treatments_data[trt_name]['DAS'] <= harvest_das]) > 0 else None
                                
                                if sim_grain_yield and sim_grain_yield > 0:
                                    obs_grain_protein = (obs_grain_n / sim_grain_yield) * 100 * 5.75
                                    
                                    ax.scatter([harvest_das], [obs_grain_protein], 
                                             color=style.get('color', 'black'),
                                             marker='o', s=80, alpha=0.9,
                                             edgecolors='white', linewidth=1.5,
                                             zorder=10)
                                    
                                    obs_std_n = observed_data[trt_name].get('std_grain_nitrogen', 0)
                                    if obs_std_n > 0:
                                        obs_std_protein = (obs_std_n / sim_grain_yield) * 100 * 5.75
                                        ax.errorbar([harvest_das], [obs_grain_protein], 
                                                  yerr=[obs_std_protein],
                                                  color=style.get('color', 'black'),
                                                  fmt='none', capsize=4, alpha=0.6,
                                                  linewidth=1.5, zorder=9)
                    
                    ax.scatter([], [], color='none', marker='o', s=80, 
                             edgecolors='black', linewidth=1.5,
                             label='Observed grain protein (circles)', alpha=0.9)
        
        elif var == 'yield_comparison':
            # Simulated vs Observed comparison
            if observed_data:
                sim_yields = []
                obs_yields = []
                trt_labels = []
                colors_list = []
                
                for trt_name in sorted(treatments_data.keys()):
                    if trt_name in observed_data:
                        sim_yield = treatments_data[trt_name]['HWAD'].iloc[-1]
                        obs_yield = observed_data[trt_name]['yield']
                        
                        sim_yields.append(sim_yield)
                        obs_yields.append(obs_yield)
                        trt_labels.append(trt_name.split(':')[0])
                        
                        style = treatment_styles.get(trt_name, {})
                        colors_list.append(style.get('color', 'black'))
                
                x_pos = range(len(trt_labels))
                
                ax.bar([x - 0.2 for x in x_pos], obs_yields, width=0.4, 
                      color=colors_list, alpha=0.6, label='Observed')
                ax.bar([x + 0.2 for x in x_pos], sim_yields, width=0.4,
                      color=colors_list, alpha=0.9, label='Simulated')
                
                for i, (obs, sim) in enumerate(zip(obs_yields, sim_yields)):
                    error_pct = ((sim - obs) / obs * 100)
                    ax.text(i, max(obs, sim) + 200, f'{error_pct:+.0f}%',
                           ha='center', fontsize=7, fontweight='bold')
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(trt_labels, rotation=45, ha='right', fontsize=8)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
        
        else:
            # Regular variable plots
            for trt_name, df in treatments_data.items():
                if var in df.columns:
                    style = treatment_styles.get(trt_name, {})
                    
                    df_filtered = df[df['DAS'] <= harvest_das].copy()
                    
                    trt_num = int(trt_name.split('Trt')[1].split(':')[0])
                    show_label = trt_num in SELECTED_TREATMENTS
                    
                    ax.plot(df_filtered['DAS'], df_filtered[var],
                           color=style.get('color', 'black'),
                           linestyle=style.get('linestyle', '-'),
                           linewidth=style.get('linewidth', 0.9),
                           alpha=style.get('alpha', 0.7),
                           label=trt_name if show_label else '')
                
                if var == 'daily_nitrogen_stress':
                    ax.set_ylim(-0.05, 1.05)
                    ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.3, linewidth=0.8)
                    ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.3, linewidth=0.8)
                    ax.text(0.02, 1.02, 'Optimal', transform=ax.get_yaxis_transform(), 
                           fontsize=7, color='green', alpha=0.7)
                    ax.text(0.02, 0.52, 'Moderate', transform=ax.get_yaxis_transform(), 
                           fontsize=7, color='orange', alpha=0.7)
            
            # Add observed data points at harvest for grain yield (HWAD)
            if var == 'HWAD' and observed_data and consensus_stages:
                harvest_das = consensus_stages.get('harvest_das', 160)
                points_plotted = 0
                for trt_name in treatments_data.keys():
                    if trt_name in observed_data:
                        style = treatment_styles.get(trt_name, {})
                        obs_yield = observed_data[trt_name]['yield']
                        obs_std = observed_data[trt_name].get('std', observed_data[trt_name].get('std_yield', 0))
                        
                        # Validate data before plotting
                        if obs_yield is not None and obs_yield > 0 and obs_yield < 20000:
                            ax.scatter([harvest_das], [obs_yield], 
                                     color=style.get('color', 'black'),
                                     marker='o', s=80, alpha=0.9,
                                     edgecolors='white', linewidth=1.5,
                                     zorder=10)
                            points_plotted += 1
                            
                            if obs_std > 0:
                                ax.errorbar([harvest_das], [obs_yield], 
                                          yerr=[obs_std],
                                          color=style.get('color', 'black'),
                                          fmt='none', capsize=4, alpha=0.6,
                                          linewidth=1.5, zorder=9)
                        else:
                            print(f"  [WARNING] Skipping invalid observed yield for {trt_name}: {obs_yield}")
                
                if points_plotted > 0:
                    ax.scatter([], [], color='none', marker='o', s=80, 
                             edgecolors='black', linewidth=1.5,
                             label='Observed (colored circles)', alpha=0.9)
                else:
                    print(f"  [WARNING] No valid observed yield points plotted for HWAD panel")
            
            # Add observed data points for grain number (H#AD)
            if var == 'H#AD' and observed_data and consensus_stages:
                harvest_das = consensus_stages.get('harvest_das', 160)
                observed_points_added = False
                for trt_name in treatments_data.keys():
                    if trt_name in observed_data and 'grain_number' in observed_data[trt_name]:
                        style = treatment_styles.get(trt_name, {})
                        obs_grain_num = observed_data[trt_name]['grain_number']
                        obs_std_grain_num = observed_data[trt_name].get('std_grain_number', 0)
                        
                        # Validate and plot if we have valid observed grain number data
                        if obs_grain_num is not None and obs_grain_num > 0 and obs_grain_num < 50000:
                            ax.scatter([harvest_das], [obs_grain_num], 
                                     color=style.get('color', 'black'),
                                     marker='o', s=100, alpha=0.9,
                                     edgecolors='white', linewidth=2.0,
                                     zorder=10)
                            observed_points_added = True
                            
                            if obs_std_grain_num > 0:
                                ax.errorbar([harvest_das], [obs_grain_num], 
                                          yerr=[obs_std_grain_num],
                                          color=style.get('color', 'black'),
                                          fmt='none', capsize=5, alpha=0.7,
                                          linewidth=2.0, zorder=9)
                        elif obs_grain_num is not None:
                            print(f"  [WARNING] Skipping invalid observed grain number for {trt_name}: {obs_grain_num}")
                
                # Add legend entry for observed points if any were added
                if observed_points_added:
                    ax.scatter([], [], color='none', marker='o', s=100, 
                             edgecolors='black', linewidth=2.0,
                             label='Observed grain number (circles)', alpha=0.9)
            
            # Add observed data points for grain weight (grain_size_mg)
            if var == 'grain_size_mg' and observed_data and consensus_stages:
                harvest_das = consensus_stages.get('harvest_das', 160)
                observed_points_added = False
                for trt_name in treatments_data.keys():
                    if trt_name in observed_data and 'grain_weight' in observed_data[trt_name]:
                        style = treatment_styles.get(trt_name, {})
                        obs_grain_wt = observed_data[trt_name]['grain_weight']
                        obs_std_grain = observed_data[trt_name].get('std_grain_weight', 0)
                        
                        # Validate and plot if we have valid observed grain weight data
                        if obs_grain_wt is not None and obs_grain_wt > 0 and obs_grain_wt < 100:
                            ax.scatter([harvest_das], [obs_grain_wt], 
                                     color=style.get('color', 'black'),
                                     marker='o', s=100, alpha=0.9,
                                     edgecolors='white', linewidth=2.0,
                                     zorder=10, label=f'{trt_name.split(":")[0]} obs' if not observed_points_added else '')
                            observed_points_added = True
                            
                            if obs_std_grain > 0:
                                ax.errorbar([harvest_das], [obs_grain_wt], 
                                          yerr=[obs_std_grain],
                                          color=style.get('color', 'black'),
                                          fmt='none', capsize=5, alpha=0.7,
                                          linewidth=2.0, zorder=9)
                        elif obs_grain_wt is not None:
                            print(f"  [WARNING] Skipping invalid observed grain weight for {trt_name}: {obs_grain_wt}")
                
                # Add legend entry for observed points if any were added
                if observed_points_added:
                    ax.scatter([], [], color='none', marker='o', s=100, 
                             edgecolors='black', linewidth=2.0,
                             label='Observed grain weight (circles)', alpha=0.9)
        
        # Standard formatting
        ax.set_title(config['title'], fontsize=11, fontweight='bold', pad=6)
        if var != 'weather':
            ax.set_ylabel(config['ylabel'], fontsize=10)
        ax.grid(True, alpha=0.3, linewidth=0.8)
        
        # Set consistent x-axis limits and ticks for ALL panels (except special ones)
        if var not in ['yield_comparison']:
            ax.set_xlim(consistent_xlim[0], consistent_xlim[1])
            ax.set_xticks(consistent_xticks)
            
            # Add date labels to ALL panels x-axis (after ticks are set)
            if phenology_stages and len(phenology_stages) > 0:
                first_stage = list(phenology_stages.values())[0]
                planting_date_dssat = first_stage.get('planting_date')
                
                if planting_date_dssat and planting_date_dssat != -99:
                    # Convert DSSAT date format (YYYYDDD) to datetime
                    year = planting_date_dssat // 1000
                    doy = planting_date_dssat % 1000
                    planting_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
                    
                    # Create date labels for each tick
                    date_labels = []
                    for tick in consistent_xticks:
                        if tick >= 0:  # Only for valid DAS values
                            date = planting_date + timedelta(days=int(tick))
                            # Format: "MMM DD" (e.g., "Mar 11")
                            date_labels.append(date.strftime('%b %d'))
                        else:
                            date_labels.append('')
                    
                    # Set x-axis labels with dates below DAS for ALL panels
                    ax.set_xticklabels([f'{int(t)}\n{date}' if date else f'{int(t)}' 
                                       for t, date in zip(consistent_xticks, date_labels)], fontsize=7)
                    
                    # Add a note about the planting date only on weather panel
                    if var == 'weather':
                        ax.text(0.02, 0.02, f'Planting: {planting_date.strftime("%b %d, %Y")}', 
                               transform=ax.transAxes, fontsize=8, 
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add phenology markers to ALL panels (except special ones)
        if var not in ['yield_comparison']:
            if config['idx'] == 0:
                if treatments_data:
                    first_trt = list(treatments_data.values())[0]
                    print(f"[DEBUG] X-axis (DAS) range in data: {first_trt['DAS'].min()} to {first_trt['DAS'].max()}")
                print(f"[DEBUG] Plotting vertical lines at: Emergence={emergence_das}, Anthesis={anthesis_das}, Maturity={maturity_das}, Harvest={harvest_das}")
            
            ax.axvline(x=emergence_das, color='green', linestyle='--', alpha=0.6, linewidth=1.2, label=f'Emergence ({emergence_das})')
            ax.axvline(x=anthesis_das, color='deeppink', linestyle='--', alpha=0.6, linewidth=1.2, label=f'Anthesis ({anthesis_das})')
            ax.axvline(x=maturity_das, color='darkorange', linestyle='--', alpha=0.6, linewidth=1.2, label=f'Maturity ({maturity_das})')
            ax.axvline(x=harvest_das, color='red', linestyle='--', alpha=0.6, linewidth=1.2, label=f'Harvest ({harvest_das})')
            
            if config['idx'] == 0:
                y_max = ax.get_ylim()[1]
                ax.text(emergence_das, y_max * 0.95, f'E\n{emergence_das}', ha='center', fontsize=8, color='green', fontweight='bold')
                ax.text(anthesis_das, y_max * 0.95, f'A\n{anthesis_das}', ha='center', fontsize=8, color='deeppink', fontweight='bold')
                ax.text(maturity_das, y_max * 0.95, f'M\n{maturity_das}', ha='center', fontsize=8, color='darkorange', fontweight='bold')
                ax.text(harvest_das, y_max * 0.95, f'H\n{harvest_das}', ha='center', fontsize=8, color='red', fontweight='bold')
        
        # Show x-axis labels on ALL panels with DAP and dates
        if var not in ['yield_comparison']:
            ax.set_xlabel('Days After Planting (DAP) / Date', fontsize=9)
        
        if config['idx'] == len(plot_configs) - 1:
            ax.set_xlabel('Days After Planting (DAS)', fontsize=12, fontweight='bold')
        
        # Add legend for select panels
        if config['idx'] in [0, 6, 13]:
            if ax.get_legend_handles_labels()[0]:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7)
    
    # Add overall phenology legend
    fig.text(0.02, 0.985, 'Phenology Markers:', fontsize=10, fontweight='bold')
    fig.text(0.02, 0.980, f'Green: Emergence ({emergence_das} DAS)', fontsize=9, color='green')
    fig.text(0.02, 0.975, f'Pink: Anthesis ({anthesis_das} DAS)', fontsize=9, color='deeppink')
    fig.text(0.02, 0.970, f'Orange: Maturity ({maturity_das} DAS)', fontsize=9, color='orange')
    fig.text(0.02, 0.965, f'Red: Harvest ({harvest_das} DAS)', fontsize=9, color='red')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, right=0.88, left=0.08, bottom=0.02, hspace=0.35)
    
    return fig

def main():
    """Main execution function"""
    
    print("="*80)
    print(f"DUERNAST {EXPERIMENT_YEAR} - COMPREHENSIVE VISUALIZATION")
    print("="*80)
    print()
    print("Creating 12-panel vertical layout with:")
    print("  - 3 selected treatments (1: Medium N, 8: High N, 15: Control)")
    print("  - Grain yield, biomass, harvest index")
    print("  - Root development, grain components")
    print("  - Nitrogen and water stress")
    print("  - Weather patterns")
    print("  - Phenology timeline")
    print("  - Simulated vs observed comparison")
    print()
    
    # Check required files
    required_files = ['PlantGro.OUT', 'Summary.OUT']
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        print(f"[ERROR] Missing required files: {missing}")
        print(f"Please run simulation first: DSCSM048.EXE A {FILE_PREFIX}.WHX")
        return 1
    
    # Check for observed data (prefer .WHT, fallback to .WHA)
    if not (Path(f'{FILE_PREFIX}.WHT').exists() or Path(f'{FILE_PREFIX}.WHA').exists()):
        print(f"[WARNING] No observed data file found ({FILE_PREFIX}.WHT or .WHA)")
        print("Visualization will proceed without observed data points.")
    
    # Parse all data
    print("[1/6] Parsing phenology stages and nitrogen levels...")
    phenology_stages, n_levels = parse_summary_phenology()
    if not phenology_stages:
        print("[ERROR] Failed to parse phenology!")
        return 1
    
    consensus_stages = get_consensus_stages(phenology_stages)
    print(f"[OK] Loaded phenology for {len(phenology_stages)} treatments")
    print(f"[OK] Loaded nitrogen levels for {len(n_levels)} treatments")
    
    print("[2/6] Parsing plant growth data...")
    treatments_data = parse_plantgro_data(n_levels)
    if not treatments_data:
        print("[ERROR] Failed to parse PlantGro.OUT!")
        return 1
    print(f"[OK] Loaded growth data for {len(treatments_data)} treatments")
    
    print("[3/6] Parsing nitrogen data...")
    nitrogen_data = parse_nitrogen_data(n_levels)
    if nitrogen_data:
        print(f"[OK] Loaded nitrogen data for {len(nitrogen_data)} treatments")
    
    print("[4/6] Parsing weather data...")
    weather_data = parse_weather_data()
    if weather_data is not None:
        print(f"[OK] Loaded weather data ({len(weather_data)} days)")
    
    print("[5/6] Parsing observed data...")
    observed_data = parse_observed_data(n_levels)
    if observed_data:
        # Count treatments (excluding metadata like _observed_harvest_date)
        treatment_count = len([k for k in observed_data.keys() if not k.startswith('_')])
        print(f"[OK] Loaded observed data for {treatment_count} treatments")
        
        # Verify all selected treatments have observed data
        missing_treatments = []
        for trt_num in SELECTED_TREATMENTS:
            found = False
            for trt_name in observed_data.keys():
                if trt_name.startswith(f"Trt{trt_num}:"):
                    found = True
                    break
            if not found:
                missing_treatments.append(trt_num)
        
        if missing_treatments:
            print(f"  [WARNING] Missing observed data for treatments: {missing_treatments}")
        else:
            print(f"  [OK] All selected treatments ({SELECTED_TREATMENTS}) have observed data")
    
    print("[6/6] Creating comprehensive visualization...")
    fig = create_comprehensive_visualization(
        treatments_data, phenology_stages, consensus_stages,
        weather_data, nitrogen_data, observed_data, n_levels
    )
    
    if fig is None:
        print("[ERROR] Failed to create visualization!")
        return 1
    
    # Save outputs using dynamic output prefix
    output_png = f'{OUTPUT_PREFIX}_comprehensive_analysis.png'
    output_pdf = f'{OUTPUT_PREFIX}_comprehensive_analysis.pdf'
    
    try:
        print(f"\nSaving outputs...")
        
        # Save PNG
        try:
            fig.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
            png_size = Path(output_png).stat().st_size if Path(output_png).exists() else 0
            print(f"[OK] Saved: {output_png} ({png_size:,} bytes)")
        except Exception as e:
            print(f"[ERROR] Failed to save PNG: {e}")
        
        # Save PDF
        try:
            fig.savefig(output_pdf, dpi=300, bbox_inches='tight', facecolor='white')
            pdf_size = Path(output_pdf).stat().st_size if Path(output_pdf).exists() else 0
            print(f"[OK] Saved: {output_pdf} ({pdf_size:,} bytes)")
        except Exception as e:
            print(f"[ERROR] Failed to save PDF: {e}")
        
    except Exception as e:
        print(f"[ERROR] Unexpected error during save: {e}")
        return 1
    
    print("\n" + "="*80)
    print("[SUCCESS] Comprehensive visualization created!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {output_png} (high-resolution PNG)")
    print(f"  - {output_pdf} (vector PDF for publications)")
    print(f"\nVisualization includes:")
    print(f"  - 12 panels covering all major crop processes")
    print(f"  - 3 selected treatments (Control, Medium N, High N)")
    print(f"  - Phenology markers (emergence, anthesis, maturity)")
    print(f"  - Simulated vs observed comparison")
    print(f"  - Ready for publication!")
    
    plt.close(fig)  # Close figure to free memory
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)


