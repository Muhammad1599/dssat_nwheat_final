#!/usr/bin/env python3
"""
Configuration for Duernast DSSAT Experiments

This module defines experiment configurations for different years (2015, 2017, etc.)
and provides functions to retrieve and list available experiments.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """Configuration for a single Duernast experiment"""
    year: int
    file_prefix: str
    output_prefix: str
    experiment_dir: str
    crop_type: str
    is_multi_year: bool = False
    normalize_das: bool = False
    additional_weather_file: Optional[str] = None


# Define all available experiments
EXPERIMENTS = {
    2015: ExperimentConfig(
        year=2015,
        file_prefix="TUDU1501",
        output_prefix="duernast_2015",
        experiment_dir="DUERNAST2015",
        crop_type="Spring Wheat",
        is_multi_year=False,
        normalize_das=False,
        additional_weather_file=None
    ),
    2017: ExperimentConfig(
        year=2017,
        file_prefix="TUDU1701",
        output_prefix="duernast_2017",
        experiment_dir="DUERNAST2017",
        crop_type="Winter Wheat",
        is_multi_year=True,
        normalize_das=True,
        additional_weather_file="TUDU1601.WTH"
    )
}


def get_config(year: int) -> ExperimentConfig:
    """
    Get configuration for a specific experiment year
    
    Args:
        year: Experiment year (e.g., 2015, 2017)
    
    Returns:
        ExperimentConfig object for the specified year
    
    Raises:
        ValueError: If the year is not configured
    """
    if year not in EXPERIMENTS:
        available = list(EXPERIMENTS.keys())
        raise ValueError(f"Experiment year {year} not found. Available years: {available}")
    
    return EXPERIMENTS[year]


def list_available_experiments() -> list:
    """
    List all available experiment years
    
    Returns:
        List of available experiment years (integers)
    """
    return sorted(EXPERIMENTS.keys())
