"""
File: fairness.py
Path: src/evaluation/fairness.py

Purpose:
  Implements fairness evaluation metrics for career recommendation systems.
  Monitors distribution of recommendations across RIASEC categories to ensure
  no systematic over-representation of specific career groups.

Original Author(s):
  - AI Assistant (GitHub Copilot)

AI Tools Used:
  - GitHub Copilot - Code generation and documentation

Editors:
  - AI Assistant (2026-03-23) — Initial implementation

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-03-23

Assumptions & Constraints:
  - Career categories are balanced in ground truth
  - Recommendations are DataFrame with 'Career Category' column
  - No demographic attributes used in evaluation

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

import pandas as pd
from typing import Dict, List
from collections import Counter


def compute_category_distribution(recommendations: List[pd.DataFrame],
                                category_names: List[str]) -> Dict[str, float]:
    """
    Name: compute_category_distribution

    Purpose:
      Computes the distribution of recommended career categories across all recommendations.

    Inputs:
      - recommendations: List[pd.DataFrame] — list of recommendation DataFrames
      - category_names: List[str] — all possible career category names

    Outputs:
      - Dict[str, float] — category distribution as percentages

    Raises / Errors:
      - ValueError: if recommendations is empty

    Notes:
      - Returns percentage of recommendations for each category
      - Includes all categories even if not recommended
    """
    if not recommendations:
        raise ValueError("Recommendations list cannot be empty")

    all_categories = []
    for rec_df in recommendations:
        if 'Career Category' not in rec_df.columns:
            raise ValueError("Recommendation DataFrame must contain 'Career Category' column")
        all_categories.extend(rec_df['Career Category'].tolist())

    total_recommendations = len(all_categories)
    if total_recommendations == 0:
        return {cat: 0.0 for cat in category_names}

    category_counts = Counter(all_categories)

    distribution = {}
    for cat in category_names:
        count = category_counts.get(cat, 0)
        distribution[cat] = (count / total_recommendations) * 100.0

    return distribution


def compute_fairness_score(distribution: Dict[str, float],
                          ideal_distribution: Dict[str, float]) -> float:
    """
    Name: compute_fairness_score

    Purpose:
      Computes fairness score based on deviation from ideal distribution.

    Inputs:
      - distribution: Dict[str, float] — actual category distribution
      - ideal_distribution: Dict[str, float] — ideal balanced distribution

    Outputs:
      - float — fairness score (0.0 = unfair, 1.0 = perfectly fair)

    Raises / Errors:
      - ValueError: if distributions have different categories

    Notes:
      - Score is 1.0 minus normalized deviation from ideal
      - Lower scores indicate more unfair distributions
    """
    if set(distribution.keys()) != set(ideal_distribution.keys()):
        raise ValueError("Distribution and ideal distribution must have same categories")

    total_deviation = 0.0
    for cat in distribution.keys():
        deviation = abs(distribution[cat] - ideal_distribution[cat])
        total_deviation += deviation

    # Normalize by maximum possible deviation (100% for each category)
    max_deviation = 100.0 * len(distribution)
    normalized_deviation = total_deviation / max_deviation

    fairness_score = 1.0 - normalized_deviation
    return max(0.0, fairness_score)  # Ensure non-negative


def detect_overrepresented_categories(distribution: Dict[str, float],
                                    ideal_distribution: Dict[str, float],
                                    threshold: float = 10.0) -> List[str]:
    """
    Name: detect_overrepresented_categories

    Purpose:
      Identifies career categories that are overrepresented compared to ideal distribution.

    Inputs:
      - distribution: Dict[str, float] — actual category distribution
      - ideal_distribution: Dict[str, float] — ideal balanced distribution
      - threshold: float — minimum percentage difference to flag as overrepresented

    Outputs:
      - List[str] — list of overrepresented category names

    Raises / Errors:
      - ValueError: if distributions have different categories

    Notes:
      - Returns categories where actual > ideal + threshold
    """
    if set(distribution.keys()) != set(ideal_distribution.keys()):
        raise ValueError("Distribution and ideal distribution must have same categories")

    overrepresented = []
    for cat in distribution.keys():
        if distribution[cat] > ideal_distribution[cat] + threshold:
            overrepresented.append(cat)

    return overrepresented


def detect_underrepresented_categories(distribution: Dict[str, float],
                                     ideal_distribution: Dict[str, float],
                                     threshold: float = 10.0) -> List[str]:
    """
    Name: detect_underrepresented_categories

    Purpose:
      Identifies career categories that are underrepresented compared to ideal distribution.

    Inputs:
      - distribution: Dict[str, float] — actual category distribution
      - ideal_distribution: Dict[str, float] — ideal balanced distribution
      - threshold: float — minimum percentage difference to flag as underrepresented

    Outputs:
      - List[str] — list of underrepresented category names

    Raises / Errors:
      - ValueError: if distributions have different categories

    Notes:
      - Returns categories where actual < ideal - threshold
    """
    if set(distribution.keys()) != set(ideal_distribution.keys()):
        raise ValueError("Distribution and ideal distribution must have same categories")

    underrepresented = []
    for cat in distribution.keys():
        if distribution[cat] < ideal_distribution[cat] - threshold:
            underrepresented.append(cat)

    return underrepresented