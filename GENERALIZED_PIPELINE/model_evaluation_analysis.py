#!/usr/bin/env python3
"""
Model Evaluation and Statistical Analysis Pipeline

Purpose: Extracts observed and simulated data for ALL 15 treatments, calculates
         model performance metrics, and generates statistical visualizations
         showing model accuracy.

Output:
    - CSV files in output/Model_analysis/ folder
    - Statistical visualizations showing model performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys
import re
try:
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
except ImportError:
    # Fallback if sklearn not available - implement manually
    def mean_squared_error(y_true, y_pred):
        return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    def r2_score(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

# Get configuration from environment variables
EXPERIMENT_YEAR = int(os.environ.get('DUERNAST_YEAR', 2015))
FILE_PREFIX = os.environ.get('DUERNAST_FILE_PREFIX', 'TUDU1501')
OUTPUT_PREFIX = os.environ.get('DUERNAST_OUTPUT_PREFIX', 'duernast_2015')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# All 15 treatments for complete analysis
ALL_TREATMENTS = list(range(1, 16))

# Selected treatments for visualization (user requested: 1, 8, 15 - where 15 is control)
SELECTED_TREATMENTS_FOR_VISUALIZATION = [1, 8, 15]

def parse_summary_all_treatments():
    """Parse Summary.OUT for ALL 15 treatments"""
    
    if not Path('Summary.OUT').exists():
        print("[ERROR] Summary.OUT not found!")
        return None, None
    
    stages = {}
    n_levels = {}
    summary_data = {}
    
    try:
        with open('Summary.OUT', 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if not lines:
            print("[ERROR] Summary.OUT is empty!")
            return None, None, None
        
        # Find data section - look for header line first, then data
        data_start = -1
        header_found = False
        for i, line in enumerate(lines):
            # Look for header line with column names
            if '@' in line and ('RUN' in line.upper() or 'TRNO' in line.upper() or 'TREATMENT' in line.upper()):
                header_found = True
                # Data should start a few lines after header
                continue
            # Look for data lines (starting with numbers)
            if header_found and line.strip() and line.strip()[0].isdigit() and not line.startswith('!'):
                data_start = i
                break
        
        # Fallback: if no header found, just look for first line starting with digit
        if data_start == -1:
            for i, line in enumerate(lines):
                if line.strip() and line.strip()[0].isdigit() and not line.startswith('!'):
                    data_start = i
                    break
        
        if data_start == -1:
            print("[ERROR] Could not find data section in Summary.OUT")
            print(f"[DEBUG] First 20 lines of Summary.OUT:")
            for i, line in enumerate(lines[:20]):
                print(f"  Line {i}: {line[:80]}")
            return None, None, None
        
        print(f"[DEBUG] Found data section starting at line {data_start}")
        
        # Parse each treatment (all 15) - check more lines
        for i in range(data_start, min(data_start + 20, len(lines))):
            line = lines[i].strip()
            if not line or line.startswith('*') or line.startswith('@'):
                continue
            
            parts = line.split()
            
            # Debug: print first few lines to see format
            if len(summary_data) == 0 and i < data_start + 3:
                print(f"[DEBUG] Line {i}: {len(parts)} columns, first few: {parts[:5] if len(parts) >= 5 else parts}")
            
            if len(parts) >= 51:
                try:
                    treatment = int(parts[1])
                    
                    # Process ALL treatments (no filtering)
                    # Column positions based on Summary.OUT format:
                    # 0=RUNNO, 1=TRNO, 2=R#, 3=O#, 4=P#, 5=CR, 6=MODEL (string), 7=EXNAME (string), 
                    # 8=TNAM (string), 9=FNAM (string), 10=WSTA (string), 11=WYEAR, 12=SOIL_ID (string),
                    # 13=XLAT, 14=LONG, 15=ELEV, 16=SDAT, 17=PDAT, 18=EDAT, 19=ADAT, 20=MDAT, 21=HDAT,
                    # 22=HYEAR, 23=DWAP, 24=CWAM (biomass), 25=HWAM (yield), 50=NICM (nitrogen)
                    sdat = int(parts[16])  # Sowing date
                    pdat = int(parts[17])  # Planting date
                    edat = int(parts[18])  # Emergence date
                    adat = int(parts[19])  # Anthesis date
                    mdat = int(parts[20])  # Maturity date
                    hdat = int(parts[21])  # Harvest date
                    nicm = int(parts[50])  # Nitrogen applied
                    
                    # Extract yield and biomass - correct column positions
                    try:
                        cwam = float(parts[24]) if len(parts) > 24 else -99  # Total biomass (kg/ha) - CWAM
                        hwam = float(parts[25]) if len(parts) > 25 else -99  # Grain yield (kg/ha) - HWAM
                    except (ValueError, IndexError):
                        cwam = -99
                        hwam = -99
                    
                    # Convert dates to DAS
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
                    
                    n_levels[treatment] = nicm
                    
                    summary_data[treatment] = {
                        'treatment': treatment,
                        'nitrogen_applied': nicm,
                        'yield_simulated': hwam,
                        'biomass_simulated': cwam,
                        'emergence_das': date_to_das(edat, pdat),
                        'anthesis_das': date_to_das(adat, pdat),
                        'maturity_das': date_to_das(mdat, pdat),
                        'harvest_das': date_to_das(hdat, pdat)
                    }
                    
                except (ValueError, IndexError) as e:
                    if len(summary_data) == 0:  # Only print first error for debugging
                        print(f"[DEBUG] Error parsing line {i}: {str(e)[:50]}, parts={len(parts)}")
                    continue
            elif len(parts) > 0 and len(summary_data) == 0:
                # Debug: show what we're skipping
                print(f"[DEBUG] Skipping line {i}: only {len(parts)} columns (need >=51)")
        
        print(f"[OK] Parsed Summary.OUT for {len(summary_data)} treatments")
        return summary_data, stages, n_levels
        
    except Exception as e:
        print(f"[ERROR] Could not parse Summary.OUT: {e}")
        return None, None, None

def parse_plantgro_all_treatments():
    """Parse PlantGro.OUT for ALL 15 treatments - extract final values"""
    
    if not Path('PlantGro.OUT').exists():
        print("[ERROR] PlantGro.OUT not found!")
        return None
    
    try:
        with open('PlantGro.OUT', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content:
            print("[ERROR] PlantGro.OUT is empty!")
            return None
        
        # Split by RUN sections
        run_sections = re.split(r'\*RUN\s+\d+', content)
        
        plantgro_data = {}
        
        # Process all 15 treatments
        for run_num in range(1, min(16, len(run_sections))):
            section = run_sections[run_num]
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
                if len(parts) >= 35:
                    try:
                        das = int(parts[2])
                        hwad = max(0.0, float(parts[9]))   # Grain yield
                        cwad = max(0.0, float(parts[12]))  # Total biomass
                        hiad = max(0.0, min(1.0, float(parts[16])))  # Harvest index
                        gwgd = max(0.0, float(parts[15]))  # Grain weight
                        h_ad = max(0.0, float(parts[13]))  # Grain number
                        
                        data_rows.append({
                            'DAS': das,
                            'HWAD': hwad,
                            'CWAD': cwad,
                            'HIAD': hiad,
                            'GWGD': gwgd,
                            'GrainNumber': h_ad
                        })
                    except (ValueError, IndexError):
                        continue
            
            if data_rows:
                df = pd.DataFrame(data_rows)
                # Get final values (last row)
                plantgro_data[run_num] = {
                    'yield_final': df['HWAD'].iloc[-1] if len(df) > 0 else -99,
                    'biomass_final': df['CWAD'].iloc[-1] if len(df) > 0 else -99,
                    'harvest_index': df['HIAD'].iloc[-1] if len(df) > 0 else -99,
                    'grain_weight': df['GWGD'].iloc[-1] if len(df) > 0 else -99,
                    'grain_number': df['GrainNumber'].iloc[-1] if len(df) > 0 else -99,
                    'max_das': df['DAS'].max() if len(df) > 0 else -99
                }
        
        print(f"[OK] Parsed PlantGro.OUT for {len(plantgro_data)} treatments")
        return plantgro_data
        
    except Exception as e:
        print(f"[ERROR] Could not parse PlantGro.OUT: {e}")
        return None

def parse_plantn_all_treatments():
    """Parse PlantN.OUT for ALL 15 treatments - extract final grain nitrogen"""
    
    if not Path('PlantN.OUT').exists():
        print("[WARNING] PlantN.OUT not found!")
        return None
    
    try:
        with open('PlantN.OUT', 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if not content:
            print("[WARNING] PlantN.OUT is empty!")
            return None
        
        # Split by RUN sections
        run_sections = re.split(r'\*RUN\s+\d+', content)
        
        plantn_data = {}
        
        # Process all 15 treatments
        for run_num in range(1, min(16, len(run_sections))):
            section = run_sections[run_num]
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
                if len(parts) >= 6:
                    try:
                        das = int(parts[2])
                        gnad = float(parts[5])  # Grain nitrogen (kg N/ha)
                        
                        data_rows.append({
                            'DAS': das,
                            'GNAD': gnad
                        })
                    except (ValueError, IndexError):
                        continue
            
            if data_rows:
                df = pd.DataFrame(data_rows)
                # Get final grain nitrogen
                plantn_data[run_num] = {
                    'grain_nitrogen_final': df['GNAD'].iloc[-1] if len(df) > 0 else -99
                }
        
        print(f"[OK] Parsed PlantN.OUT for {len(plantn_data)} treatments")
        return plantn_data
        
    except Exception as e:
        print(f"[WARNING] Could not parse PlantN.OUT: {e}")
        return None

def parse_observed_all_treatments():
    """Parse observed data for ALL 15 treatments"""
    
    wht_file = Path(f'{FILE_PREFIX}.WHT')
    wha_file = Path(f'{FILE_PREFIX}.WHA')
    
    observed_data = {}
    
    if wht_file.exists():
        print(f"[OK] Using {FILE_PREFIX}.WHT (detailed data)")
        with open(wht_file, 'r') as f:
            lines = f.readlines()
        
        # Parse .WHT format: TRNO DATE HWAD GWGD GNAD
        for line in lines:
            if line.strip() and not line.startswith(('@', '*', '!')):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        treatment = int(parts[0])
                        harvest_date = int(parts[1])
                        hwad_obs = int(parts[2])      # Yield (kg/ha)
                        gwgd_obs = int(parts[3])      # Grain weight (mg)
                        gnad_obs = int(parts[4])      # Grain nitrogen (kg N/ha)
                        
                        if treatment not in observed_data:
                            observed_data[treatment] = {
                                'yield': [],
                                'grain_weight': [],
                                'grain_nitrogen': [],
                                'harvest_date': harvest_date
                            }
                        
                        observed_data[treatment]['yield'].append(hwad_obs)
                        observed_data[treatment]['grain_weight'].append(gwgd_obs)
                        observed_data[treatment]['grain_nitrogen'].append(gnad_obs)
                    except (ValueError, IndexError):
                        continue
    
    elif wha_file.exists():
        print(f"[OK] Using {FILE_PREFIX}.WHA (yield only)")
        with open(wha_file, 'r') as f:
            lines = f.readlines()
        
        # Parse .WHA format: TRNO HDAT HWAM
        for line in lines:
            if line.strip() and not line.startswith(('@', '*', '!')):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        treatment = int(parts[0])
                        harvest_date = int(parts[1])
                        hwam_obs = int(parts[2])
                        
                        if treatment not in observed_data:
                            observed_data[treatment] = {
                                'yield': [],
                                'grain_weight': [],
                                'grain_nitrogen': [],
                                'harvest_date': harvest_date
                            }
                        
                        observed_data[treatment]['yield'].append(hwam_obs)
                    except (ValueError, IndexError):
                        continue
    
    else:
        print(f"[WARNING] No observed data file found ({FILE_PREFIX}.WHT or .WHA)")
        return None
    
    # Calculate means and standard deviations for all treatments
    observed_summary = {}
    for treatment, data in observed_data.items():
        observed_summary[treatment] = {
            'treatment': treatment,
            'yield_mean': np.mean(data['yield']) if data['yield'] else -99,
            'yield_std': np.std(data['yield']) if len(data['yield']) > 1 else 0,
            'yield_n': len(data['yield']),
            'grain_weight_mean': np.mean(data['grain_weight']) if data['grain_weight'] else -99,
            'grain_weight_std': np.std(data['grain_weight']) if len(data['grain_weight']) > 1 else 0,
            'grain_nitrogen_mean': np.mean(data['grain_nitrogen']) if data['grain_nitrogen'] else -99,
            'grain_nitrogen_std': np.std(data['grain_nitrogen']) if len(data['grain_nitrogen']) > 1 else 0,
            'harvest_date': data.get('harvest_date', -99)
        }
    
    print(f"[OK] Parsed observed data for {len(observed_summary)} treatments")
    return observed_summary

def calculate_model_metrics(observed, simulated):
    """Calculate model performance metrics"""
    
    # Remove missing values
    mask = (observed != -99) & (simulated != -99) & ~np.isnan(observed) & ~np.isnan(simulated)
    obs_clean = observed[mask]
    sim_clean = simulated[mask]
    
    if len(obs_clean) == 0 or len(sim_clean) == 0:
        return {
            'RMSE': -99,
            'MAE': -99,
            'R2': -99,
            'ModelEfficiency': -99,
            'Bias': -99,
            'n': 0
        }
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(obs_clean, sim_clean))
    mae = mean_absolute_error(obs_clean, sim_clean)
    r2 = r2_score(obs_clean, sim_clean)
    
    # Model Efficiency (Nash-Sutcliffe)
    obs_mean = np.mean(obs_clean)
    numerator = np.sum((obs_clean - sim_clean) ** 2)
    denominator = np.sum((obs_clean - obs_mean) ** 2)
    model_efficiency = 1 - (numerator / denominator) if denominator != 0 else -99
    
    # Bias (mean error)
    bias = np.mean(sim_clean - obs_clean)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'ModelEfficiency': model_efficiency,
        'Bias': bias,
        'n': len(obs_clean)
    }

def create_comparison_dataframe(summary_data, plantgro_data, plantn_data, observed_data):
    """Create comprehensive comparison dataframe for all 15 treatments"""
    
    comparison_data = []
    
    for treatment in ALL_TREATMENTS:
        row = {
            'Treatment': treatment,
            'Nitrogen_Applied_kg_ha': summary_data.get(treatment, {}).get('nitrogen_applied', -99) if summary_data else -99
        }
        
        # Simulated data
        if summary_data and treatment in summary_data:
            row['Yield_Simulated_kg_ha'] = summary_data[treatment]['yield_simulated']
            row['Biomass_Simulated_kg_ha'] = summary_data[treatment]['biomass_simulated']
        elif plantgro_data and treatment in plantgro_data:
            row['Yield_Simulated_kg_ha'] = plantgro_data[treatment]['yield_final']
            row['Biomass_Simulated_kg_ha'] = plantgro_data[treatment]['biomass_final']
        else:
            row['Yield_Simulated_kg_ha'] = -99
            row['Biomass_Simulated_kg_ha'] = -99
        
        if plantgro_data and treatment in plantgro_data:
            row['Harvest_Index_Simulated'] = plantgro_data[treatment]['harvest_index']
            row['Grain_Weight_Simulated_mg'] = plantgro_data[treatment]['grain_weight']
            row['Grain_Number_Simulated'] = plantgro_data[treatment]['grain_number']
        else:
            row['Harvest_Index_Simulated'] = -99
            row['Grain_Weight_Simulated_mg'] = -99
            row['Grain_Number_Simulated'] = -99
        
        if plantn_data and treatment in plantn_data:
            row['Grain_Nitrogen_Simulated_kg_ha'] = plantn_data[treatment]['grain_nitrogen_final']
        else:
            row['Grain_Nitrogen_Simulated_kg_ha'] = -99
        
        # Observed data
        if observed_data and treatment in observed_data:
            row['Yield_Observed_kg_ha'] = observed_data[treatment]['yield_mean']
            row['Yield_Observed_Std'] = observed_data[treatment]['yield_std']
            row['Yield_Observed_n'] = observed_data[treatment]['yield_n']
            row['Grain_Weight_Observed_mg'] = observed_data[treatment]['grain_weight_mean']
            row['Grain_Weight_Observed_Std'] = observed_data[treatment]['grain_weight_std']
            row['Grain_Nitrogen_Observed_kg_ha'] = observed_data[treatment]['grain_nitrogen_mean']
            row['Grain_Nitrogen_Observed_Std'] = observed_data[treatment]['grain_nitrogen_std']
            row['Harvest_Date_Observed'] = observed_data[treatment]['harvest_date']
        else:
            row['Yield_Observed_kg_ha'] = -99
            row['Yield_Observed_Std'] = -99
            row['Yield_Observed_n'] = 0
            row['Grain_Weight_Observed_mg'] = -99
            row['Grain_Weight_Observed_Std'] = -99
            row['Grain_Nitrogen_Observed_kg_ha'] = -99
            row['Grain_Nitrogen_Observed_Std'] = -99
            row['Harvest_Date_Observed'] = -99
        
        # Calculate differences and errors
        if row['Yield_Simulated_kg_ha'] != -99 and row['Yield_Observed_kg_ha'] != -99:
            row['Yield_Difference_kg_ha'] = row['Yield_Simulated_kg_ha'] - row['Yield_Observed_kg_ha']
            row['Yield_Error_Percent'] = ((row['Yield_Simulated_kg_ha'] - row['Yield_Observed_kg_ha']) / row['Yield_Observed_kg_ha']) * 100 if row['Yield_Observed_kg_ha'] > 0 else -99
        else:
            row['Yield_Difference_kg_ha'] = -99
            row['Yield_Error_Percent'] = -99
        
        if row['Grain_Weight_Simulated_mg'] != -99 and row['Grain_Weight_Observed_mg'] != -99:
            row['Grain_Weight_Difference_mg'] = row['Grain_Weight_Simulated_mg'] - row['Grain_Weight_Observed_mg']
            row['Grain_Weight_Error_Percent'] = ((row['Grain_Weight_Simulated_mg'] - row['Grain_Weight_Observed_mg']) / row['Grain_Weight_Observed_mg']) * 100 if row['Grain_Weight_Observed_mg'] > 0 else -99
        else:
            row['Grain_Weight_Difference_mg'] = -99
            row['Grain_Weight_Error_Percent'] = -99
        
        if row['Grain_Nitrogen_Simulated_kg_ha'] != -99 and row['Grain_Nitrogen_Observed_kg_ha'] != -99:
            row['Grain_Nitrogen_Difference_kg_ha'] = row['Grain_Nitrogen_Simulated_kg_ha'] - row['Grain_Nitrogen_Observed_kg_ha']
            row['Grain_Nitrogen_Error_Percent'] = ((row['Grain_Nitrogen_Simulated_kg_ha'] - row['Grain_Nitrogen_Observed_kg_ha']) / row['Grain_Nitrogen_Observed_kg_ha']) * 100 if row['Grain_Nitrogen_Observed_kg_ha'] > 0 else -99
        else:
            row['Grain_Nitrogen_Difference_kg_ha'] = -99
            row['Grain_Nitrogen_Error_Percent'] = -99
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    return df

def calculate_overall_metrics(df):
    """Calculate overall model performance metrics"""
    
    metrics = {}
    
    # Yield metrics
    yield_obs = df['Yield_Observed_kg_ha'].values
    yield_sim = df['Yield_Simulated_kg_ha'].values
    yield_metrics = calculate_model_metrics(yield_obs, yield_sim)
    metrics['Yield'] = yield_metrics
    
    # Grain weight metrics
    gw_obs = df['Grain_Weight_Observed_mg'].values
    gw_sim = df['Grain_Weight_Simulated_mg'].values
    gw_metrics = calculate_model_metrics(gw_obs, gw_sim)
    metrics['GrainWeight'] = gw_metrics
    
    # Grain nitrogen metrics
    gn_obs = df['Grain_Nitrogen_Observed_kg_ha'].values
    gn_sim = df['Grain_Nitrogen_Simulated_kg_ha'].values
    gn_metrics = calculate_model_metrics(gn_obs, gn_sim)
    metrics['GrainNitrogen'] = gn_metrics
    
    return metrics

def create_statistical_visualizations(df, overall_metrics, output_dir):
    """Create statistical visualizations showing model accuracy
    
    Note: df should already be filtered to SELECTED_TREATMENTS_FOR_VISUALIZATION
    """
    
    # Ensure dataframe is sorted by Treatment for consistency
    df = df.sort_values('Treatment').copy()
    
    # Create helper functions for labels
    def get_treatment_label_with_n(treatment, n_applied):
        """Generate treatment label with nitrogen amount - for one plot only"""
        if pd.isna(n_applied) or n_applied == -99:
            return f"T{treatment}"
        return f"T{treatment}\n({int(n_applied)}N)"
    
    def get_treatment_label_simple(treatment):
        """Generate simple treatment label - just number"""
        return f"T{treatment}"
    
    # Create figure with multiple subplots (2 rows x 4 columns = 8 panels)
    # Add extra space at top for treatment-N legend
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.4, top=0.92, bottom=0.08)
    
    # Create treatment-N mapping for legend (from all valid treatments)
    treatment_n_map = {}
    for _, row in df.iterrows():
        trt = int(row['Treatment'])
        n_val = row.get('Nitrogen_Applied_kg_ha', -99)
        if n_val != -99:
            treatment_n_map[trt] = int(n_val)
    
    # Add legend at top of entire figure
    legend_text = "Treatment N (kg/ha): "
    sorted_treatments = sorted(treatment_n_map.keys())
    legend_parts = [f"T{trt}: {treatment_n_map[trt]}" for trt in sorted_treatments]
    legend_text += " | ".join(legend_parts)
    
    fig.text(0.5, 0.96, legend_text, transform=fig.transFigure,
            fontsize=8, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
            family='monospace')
    
    # 1. Yield: Simulated vs Observed (scatter plot with 1:1 line) - WITH treatment labels on points
    ax1 = fig.add_subplot(gs[0, 0])
    yield_obs = df['Yield_Observed_kg_ha'].values
    yield_sim = df['Yield_Simulated_kg_ha'].values
    treatments = df['Treatment'].values
    mask = (yield_obs != -99) & (yield_sim != -99)
    
    if np.sum(mask) > 0:
        # Plot points
        ax1.scatter(yield_obs[mask], yield_sim[mask], s=100, alpha=0.7, edgecolors='black', linewidth=1.5)
        
        # Add treatment labels on points
        for i in range(len(yield_obs)):
            if mask[i]:
                label = f"T{int(treatments[i])}"
                ax1.annotate(label, (yield_obs[i], yield_sim[i]), 
                           fontsize=8, fontweight='bold', alpha=0.9,
                           xytext=(0, 0), textcoords='offset points',
                           ha='center', va='center', color='white')
        
        # Add 1:1 line
        min_val = min(np.min(yield_obs[mask]), np.min(yield_sim[mask]))
        max_val = max(np.max(yield_obs[mask]), np.max(yield_sim[mask]))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
        
        # Add regression line
        if np.sum(mask) > 1:
            z = np.polyfit(yield_obs[mask], yield_sim[mask], 1)
            p = np.poly1d(z)
            ax1.plot(yield_obs[mask], p(yield_obs[mask]), 'b-', alpha=0.5, linewidth=1.5, label='Regression')
        
        ax1.set_xlabel('Observed Yield (kg/ha)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Simulated Yield (kg/ha)', fontsize=11, fontweight='bold')
        ax1.set_title('a) Yield: Simulated vs Observed', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Add metrics text
        if overall_metrics['Yield']['n'] > 0:
            metrics_text = f"R² = {overall_metrics['Yield']['R2']:.3f}\n"
            metrics_text += f"RMSE = {overall_metrics['Yield']['RMSE']:.0f} kg/ha\n"
            metrics_text += f"MAE = {overall_metrics['Yield']['MAE']:.0f} kg/ha\n"
            metrics_text += f"ME = {overall_metrics['Yield']['ModelEfficiency']:.3f}"
            ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Yield Error by Treatment - Treatment numbers on x-axis only
    ax2 = fig.add_subplot(gs[0, 1])
    treatments = df['Treatment'].values
    yield_error = df['Yield_Error_Percent'].values
    mask_error = yield_error != -99
    
    if np.sum(mask_error) > 0:
        # Get indices for valid treatments
        valid_indices = np.where(mask_error)[0]
        valid_treatments = treatments[valid_indices].astype(int)  # Ensure integer
        valid_errors = yield_error[valid_indices]
        
        colors = ['red' if x < 0 else 'green' for x in valid_errors]
        ax2.bar(valid_treatments, valid_errors, color=colors, alpha=0.6, edgecolor='black', width=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Treatment', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Yield Error (%)', fontsize=11, fontweight='bold')
        ax2.set_title('b) Yield Prediction Error by Treatment', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # X-axis: Just treatment numbers (1, 2, 3...)
        ax2.set_xticks(valid_treatments)
        ax2.set_xticklabels([str(int(t)) for t in valid_treatments], fontsize=10)
    
    # 3. Treatment Performance Ranking - Simple labels only
    ax3 = fig.add_subplot(gs[0, 2])
    treatment_performance = []
    for _, row in df.iterrows():
        if row['Yield_Error_Percent'] != -99:
            treatment_performance.append({
                'Treatment': int(row['Treatment']),  # Ensure integer
                'AbsError': abs(row['Yield_Error_Percent'])
            })
    
    if treatment_performance:
        perf_df = pd.DataFrame(treatment_performance)
        perf_df = perf_df.sort_values('AbsError')
        
        colors_perf = ['green' if x < 10 else 'orange' if x < 20 else 'red' for x in perf_df['AbsError']]
        ax3.barh(perf_df['Treatment'], perf_df['AbsError'], color=colors_perf, alpha=0.6, edgecolor='black')
        ax3.set_xlabel('Absolute Yield Error (%)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Treatment', fontsize=11, fontweight='bold')
        ax3.set_title('c) Treatment Performance Ranking', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Simple labels - just treatment numbers
        treatment_labels = [get_treatment_label_simple(int(t)) for t in perf_df['Treatment']]
        ax3.set_yticks(perf_df['Treatment'])
        ax3.set_yticklabels(treatment_labels, fontsize=10)
    
    # 4. Model Efficiency by Treatment - Just numbers on x-axis (1, 2, 3...)
    ax4 = fig.add_subplot(gs[0, 3])
    treatment_efficiency = []
    mask_res = (yield_obs != -99) & (yield_sim != -99)
    for _, row in df.iterrows():
        if row['Yield_Observed_kg_ha'] != -99 and row['Yield_Simulated_kg_ha'] != -99:
            obs_val = row['Yield_Observed_kg_ha']
            sim_val = row['Yield_Simulated_kg_ha']
            error = sim_val - obs_val
            efficiency = 1 - (error**2) / ((obs_val - np.mean(yield_obs[mask_res]))**2) if np.sum(mask_res) > 0 else -99
            treatment_efficiency.append({
                'Treatment': int(row['Treatment']),  # Ensure integer
                'Efficiency': efficiency
            })
    
    if treatment_efficiency:
        eff_df = pd.DataFrame(treatment_efficiency)
        eff_df = eff_df.sort_values('Treatment')
        
        colors_eff = ['green' if x > 0.5 else 'orange' if x > 0 else 'red' for x in eff_df['Efficiency']]
        ax4.bar(eff_df['Treatment'], eff_df['Efficiency'], color=colors_eff, alpha=0.6, edgecolor='black', width=0.8)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_xlabel('Treatment', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Model Efficiency', fontsize=11, fontweight='bold')
        ax4.set_title('d) Model Efficiency by Treatment', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Just numbers on x-axis (1, 2, 3...) - no "T" prefix
        ax4.set_xticks(eff_df['Treatment'])
        ax4.set_xticklabels([str(int(t)) for t in eff_df['Treatment']], fontsize=10)
    
    # 5. Grain Weight: Simulated vs Observed - WITH treatment labels on points
    ax5 = fig.add_subplot(gs[1, 0])
    gw_obs = df['Grain_Weight_Observed_mg'].values
    gw_sim = df['Grain_Weight_Simulated_mg'].values
    treatments_gw = df['Treatment'].values
    mask_gw = (gw_obs != -99) & (gw_sim != -99)
    
    if np.sum(mask_gw) > 0:
        # Plot points
        ax5.scatter(gw_obs[mask_gw], gw_sim[mask_gw], s=100, alpha=0.7, edgecolors='black', linewidth=1.5, color='green')
        
        # Add treatment labels on points
        for i in range(len(gw_obs)):
            if mask_gw[i]:
                label = f"T{int(treatments_gw[i])}"
                ax5.annotate(label, (gw_obs[i], gw_sim[i]), 
                           fontsize=8, fontweight='bold', alpha=0.9,
                           xytext=(0, 0), textcoords='offset points',
                           ha='center', va='center', color='white')
        
        min_val = min(np.min(gw_obs[mask_gw]), np.min(gw_sim[mask_gw]))
        max_val = max(np.max(gw_obs[mask_gw]), np.max(gw_sim[mask_gw]))
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
        ax5.set_xlabel('Observed Grain Weight (mg)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Simulated Grain Weight (mg)', fontsize=11, fontweight='bold')
        ax5.set_title('e) Grain Weight: Simulated vs Observed', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        if overall_metrics['GrainWeight']['n'] > 0:
            metrics_text = f"R² = {overall_metrics['GrainWeight']['R2']:.3f}\n"
            metrics_text += f"RMSE = {overall_metrics['GrainWeight']['RMSE']:.2f} mg\n"
            metrics_text += f"MAE = {overall_metrics['GrainWeight']['MAE']:.2f} mg"
            ax5.text(0.05, 0.95, metrics_text, transform=ax5.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Grain Nitrogen: Simulated vs Observed - WITH treatment labels on points
    ax6 = fig.add_subplot(gs[1, 1])
    gn_obs = df['Grain_Nitrogen_Observed_kg_ha'].values
    gn_sim = df['Grain_Nitrogen_Simulated_kg_ha'].values
    treatments_gn = df['Treatment'].values
    mask_gn = (gn_obs != -99) & (gn_sim != -99)
    
    if np.sum(mask_gn) > 0:
        # Plot points
        ax6.scatter(gn_obs[mask_gn], gn_sim[mask_gn], s=100, alpha=0.7, edgecolors='black', linewidth=1.5, color='orange')
        
        # Add treatment labels on points
        for i in range(len(gn_obs)):
            if mask_gn[i]:
                label = f"T{int(treatments_gn[i])}"
                ax6.annotate(label, (gn_obs[i], gn_sim[i]), 
                           fontsize=8, fontweight='bold', alpha=0.9,
                           xytext=(0, 0), textcoords='offset points',
                           ha='center', va='center', color='white')
        
        min_val = min(np.min(gn_obs[mask_gn]), np.min(gn_sim[mask_gn]))
        max_val = max(np.max(gn_obs[mask_gn]), np.max(gn_sim[mask_gn]))
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')
        ax6.set_xlabel('Observed Grain N (kg/ha)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Simulated Grain N (kg/ha)', fontsize=11, fontweight='bold')
        ax6.set_title('f) Grain Nitrogen: Simulated vs Observed', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)
        
        if overall_metrics['GrainNitrogen']['n'] > 0:
            metrics_text = f"R² = {overall_metrics['GrainNitrogen']['R2']:.3f}\n"
            metrics_text += f"RMSE = {overall_metrics['GrainNitrogen']['RMSE']:.2f} kg/ha\n"
            metrics_text += f"MAE = {overall_metrics['GrainNitrogen']['MAE']:.2f} kg/ha"
            ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 7. Model Performance Metrics Summary
    ax7 = fig.add_subplot(gs[1, 2])
    metrics_names = ['R²', 'RMSE', 'MAE', 'Model Efficiency']
    yield_vals = [
        overall_metrics['Yield']['R2'] if overall_metrics['Yield']['R2'] != -99 else 0,
        overall_metrics['Yield']['RMSE'] / 1000 if overall_metrics['Yield']['RMSE'] != -99 else 0,  # Scale for visibility
        overall_metrics['Yield']['MAE'] / 1000 if overall_metrics['Yield']['MAE'] != -99 else 0,
        overall_metrics['Yield']['ModelEfficiency'] if overall_metrics['Yield']['ModelEfficiency'] != -99 else 0
    ]
    
    x_pos = np.arange(len(metrics_names))
    bars = ax7.bar(x_pos, yield_vals, alpha=0.7, color=['blue', 'red', 'orange', 'green'])
    ax7.set_xlabel('Metric', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Value (scaled)', fontsize=11, fontweight='bold')
    ax7.set_title('g) Model Performance Metrics (Yield)', fontsize=12, fontweight='bold')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, yield_vals)):
        if i < 2:  # R² and scaled RMSE/MAE
            label = f'{val:.3f}' if i == 0 else f'{val*1000:.0f}'
        else:
            label = f'{val:.3f}'
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 8. Summary Statistics Table
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')
    
    # Create summary table
    summary_data = []
    if overall_metrics['Yield']['n'] > 0:
        summary_data.append(['Variable', 'R²', 'RMSE', 'MAE', 'ME'])
        summary_data.append(['Yield', 
                            f"{overall_metrics['Yield']['R2']:.3f}" if overall_metrics['Yield']['R2'] != -99 else 'N/A',
                            f"{overall_metrics['Yield']['RMSE']:.0f}" if overall_metrics['Yield']['RMSE'] != -99 else 'N/A',
                            f"{overall_metrics['Yield']['MAE']:.0f}" if overall_metrics['Yield']['MAE'] != -99 else 'N/A',
                            f"{overall_metrics['Yield']['ModelEfficiency']:.3f}" if overall_metrics['Yield']['ModelEfficiency'] != -99 else 'N/A'])
    
    if overall_metrics['GrainWeight']['n'] > 0:
        summary_data.append(['Grain Weight',
                            f"{overall_metrics['GrainWeight']['R2']:.3f}" if overall_metrics['GrainWeight']['R2'] != -99 else 'N/A',
                            f"{overall_metrics['GrainWeight']['RMSE']:.2f}" if overall_metrics['GrainWeight']['RMSE'] != -99 else 'N/A',
                            f"{overall_metrics['GrainWeight']['MAE']:.2f}" if overall_metrics['GrainWeight']['MAE'] != -99 else 'N/A',
                            f"{overall_metrics['GrainWeight']['ModelEfficiency']:.3f}" if overall_metrics['GrainWeight']['ModelEfficiency'] != -99 else 'N/A'])
    
    if overall_metrics['GrainNitrogen']['n'] > 0:
        summary_data.append(['Grain N',
                            f"{overall_metrics['GrainNitrogen']['R2']:.3f}" if overall_metrics['GrainNitrogen']['R2'] != -99 else 'N/A',
                            f"{overall_metrics['GrainNitrogen']['RMSE']:.2f}" if overall_metrics['GrainNitrogen']['RMSE'] != -99 else 'N/A',
                            f"{overall_metrics['GrainNitrogen']['MAE']:.2f}" if overall_metrics['GrainNitrogen']['MAE'] != -99 else 'N/A',
                            f"{overall_metrics['GrainNitrogen']['ModelEfficiency']:.3f}" if overall_metrics['GrainNitrogen']['ModelEfficiency'] != -99 else 'N/A'])
    
    if summary_data:
        table = ax8.table(cellText=summary_data[1:], colLabels=summary_data[0],
                         cellLoc='center', loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        # Style header row
        for i in range(len(summary_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        ax8.set_title('h) Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # Overall title
    treatments_str = ', '.join([f'T{t}' for t in SELECTED_TREATMENTS_FOR_VISUALIZATION])
    fig.suptitle(f'Model Performance Evaluation - {EXPERIMENT_YEAR} Experiment\nSelected Treatments: {treatments_str}',
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = output_dir / f'{OUTPUT_PREFIX}_model_evaluation.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved model evaluation visualization: {output_path}")
    
    # Also save as PDF
    output_path_pdf = output_dir / f'{OUTPUT_PREFIX}_model_evaluation.pdf'
    fig.savefig(output_path_pdf, bbox_inches='tight')
    print(f"[OK] Saved model evaluation visualization (PDF): {output_path_pdf}")
    
    plt.close(fig)

def main():
    """Main execution function"""
    
    print("="*80)
    print(f"MODEL EVALUATION AND STATISTICAL ANALYSIS - {EXPERIMENT_YEAR}")
    print("="*80)
    print()
    print("Analyzing ALL 15 treatments for comprehensive model evaluation")
    print(f"Visualizations will show only selected treatments: {SELECTED_TREATMENTS_FOR_VISUALIZATION}")
    print()
    
    # Check required files
    required_files = ['Summary.OUT', 'PlantGro.OUT']
    missing = [f for f in required_files if not Path(f).exists()]
    
    if missing:
        print(f"[ERROR] Missing required files: {missing}")
        return 1
    
    # Create output directory
    output_dir = Path('Model_analysis')
    output_dir.mkdir(exist_ok=True)
    print(f"[OK] Output directory: {output_dir.absolute()}")
    
    # Parse all data
    print("[1/5] Parsing Summary.OUT for all treatments...")
    summary_data, stages, n_levels = parse_summary_all_treatments()
    if not summary_data:
        print("[ERROR] Failed to parse Summary.OUT!")
        return 1
    
    print("[2/5] Parsing PlantGro.OUT for all treatments...")
    plantgro_data = parse_plantgro_all_treatments()
    if not plantgro_data:
        print("[ERROR] Failed to parse PlantGro.OUT!")
        return 1
    
    print("[3/5] Parsing PlantN.OUT for all treatments...")
    plantn_data = parse_plantn_all_treatments()
    # This is optional, continue if missing
    
    print("[4/5] Parsing observed data for all treatments...")
    observed_data = parse_observed_all_treatments()
    if not observed_data:
        print("[WARNING] No observed data found - analysis will be limited to simulated data")
    
    # Create comparison dataframe
    print("[5/5] Creating comparison dataframe and calculating metrics...")
    comparison_df = create_comparison_dataframe(summary_data, plantgro_data, plantn_data, observed_data)
    
    # Calculate overall metrics
    overall_metrics = calculate_overall_metrics(comparison_df)
    
    # Save CSV files
    comparison_csv = output_dir / f'{OUTPUT_PREFIX}_comparison_all_treatments.csv'
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"[OK] Saved comparison data: {comparison_csv}")
    
    # Create metrics summary
    metrics_summary = []
    for var_name, metrics in overall_metrics.items():
        metrics_summary.append({
            'Variable': var_name,
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
            'R2': metrics['R2'],
            'ModelEfficiency': metrics['ModelEfficiency'],
            'Bias': metrics['Bias'],
            'n': metrics['n']
        })
    
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv = output_dir / f'{OUTPUT_PREFIX}_model_metrics_summary.csv'
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"[OK] Saved metrics summary: {metrics_csv}")
    
    # Create visualizations if observed data available
    if observed_data:
        print("[6/6] Creating statistical visualizations...")
        # Filter comparison_df for visualization (but keep full data for CSV)
        comparison_df_filtered = comparison_df[comparison_df['Treatment'].isin(SELECTED_TREATMENTS_FOR_VISUALIZATION)].copy()
        # Recalculate metrics for selected treatments only
        overall_metrics_filtered = calculate_overall_metrics(comparison_df_filtered)
        create_statistical_visualizations(comparison_df_filtered, overall_metrics_filtered, output_dir)
    else:
        print("[WARNING] Skipping visualizations - no observed data available")
    
    # Print summary
    print()
    print("="*80)
    print("MODEL EVALUATION SUMMARY")
    print("="*80)
    if overall_metrics['Yield']['n'] > 0:
        print(f"Yield Performance:")
        print(f"  R² = {overall_metrics['Yield']['R2']:.3f}")
        print(f"  RMSE = {overall_metrics['Yield']['RMSE']:.0f} kg/ha")
        print(f"  MAE = {overall_metrics['Yield']['MAE']:.0f} kg/ha")
        print(f"  Model Efficiency = {overall_metrics['Yield']['ModelEfficiency']:.3f}")
        print(f"  Bias = {overall_metrics['Yield']['Bias']:.0f} kg/ha")
        print(f"  n = {overall_metrics['Yield']['n']} treatments")
    
    print()
    print(f"Output files saved to: {output_dir.absolute()}")
    print("="*80)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

