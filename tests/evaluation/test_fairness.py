"""
File: test_fairness.py
Path: tests/evaluation/test_fairness.py

Purpose:
  Unit tests for fairness evaluation functions.

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
  - Tests are deterministic
  - Category distributions are percentages
  - Fairness scores in [0, 1] range

Related Docs:
  - docs/src/evaluation/evaluation.md
"""

import unittest
import pandas as pd
from src.evaluation.fairness import (compute_category_distribution, compute_fairness_score,
                                   detect_overrepresented_categories, detect_underrepresented_categories)


class TestCategoryDistribution(unittest.TestCase):
    """Test category distribution computation."""

    def setUp(self):
        """Set up test recommendation data."""
        self.rec1 = pd.DataFrame({
            'Title': ['Job1', 'Job2', 'Job3'],
            'Career Category': ['Engineering', 'Science', 'Engineering'],
            'Match_Score': [0.9, 0.8, 0.7]
        })
        self.rec2 = pd.DataFrame({
            'Title': ['Job4', 'Job5'],
            'Career Category': ['Arts', 'Engineering'],
            'Match_Score': [0.6, 0.5]
        })
        self.category_names = ['Engineering', 'Science', 'Arts', 'Business']

    def test_compute_category_distribution(self):
        """Test basic category distribution computation."""
        distribution = compute_category_distribution([self.rec1, self.rec2], self.category_names)

        # Check all categories are present
        for cat in self.category_names:
            self.assertIn(cat, distribution)

        # Check Engineering: 3 out of 5 = 60%
        self.assertAlmostEqual(distribution['Engineering'], 60.0)
        # Check Science: 1 out of 5 = 20%
        self.assertAlmostEqual(distribution['Science'], 20.0)
        # Check Arts: 1 out of 5 = 20%
        self.assertAlmostEqual(distribution['Arts'], 20.0)
        # Check Business: 0 out of 5 = 0%
        self.assertAlmostEqual(distribution['Business'], 0.0)

    def test_empty_recommendations(self):
        """Test error handling for empty recommendations."""
        with self.assertRaises(ValueError):
            compute_category_distribution([], self.category_names)

    def test_missing_career_category_column(self):
        """Test error handling for missing Career Category column."""
        bad_rec = pd.DataFrame({
            'Title': ['Job1'],
            'Category': ['Engineering'],  # Wrong column name
            'Match_Score': [0.9]
        })
        with self.assertRaises(ValueError):
            compute_category_distribution([bad_rec], self.category_names)


class TestFairnessScore(unittest.TestCase):
    """Test fairness score computation."""

    def test_perfect_fairness(self):
        """Test fairness score for perfectly balanced distribution."""
        distribution = {'A': 25.0, 'B': 25.0, 'C': 25.0, 'D': 25.0}
        ideal = {'A': 25.0, 'B': 25.0, 'C': 25.0, 'D': 25.0}
        score = compute_fairness_score(distribution, ideal)
        self.assertAlmostEqual(score, 1.0)

    def test_no_fairness(self):
        """Test fairness score for completely unbalanced distribution."""
        distribution = {'A': 100.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
        ideal = {'A': 25.0, 'B': 25.0, 'C': 25.0, 'D': 25.0}
        score = compute_fairness_score(distribution, ideal)
        self.assertAlmostEqual(score, 0.625)

    def test_partial_fairness(self):
        """Test fairness score for partially balanced distribution."""
        distribution = {'A': 40.0, 'B': 20.0, 'C': 20.0, 'D': 20.0}
        ideal = {'A': 25.0, 'B': 25.0, 'C': 25.0, 'D': 25.0}
        score = compute_fairness_score(distribution, ideal)
        # Deviation: |40-25| + |20-25|*3 = 15 + 15 = 30
        # Max deviation: 100 * 4 = 400
        # Normalized deviation: 30/400 = 0.075
        # Fairness: 1 - 0.075 = 0.925
        self.assertAlmostEqual(score, 0.925, places=3)

    def test_mismatched_categories(self):
        """Test error handling for mismatched category sets."""
        distribution = {'A': 50.0, 'B': 50.0}
        ideal = {'A': 50.0, 'C': 50.0}
        with self.assertRaises(ValueError):
            compute_fairness_score(distribution, ideal)


class TestOverrepresentedCategories(unittest.TestCase):
    """Test detection of overrepresented categories."""

    def setUp(self):
        """Set up test distributions."""
        self.distribution = {'A': 40.0, 'B': 20.0, 'C': 20.0, 'D': 20.0}
        self.ideal = {'A': 25.0, 'B': 25.0, 'C': 25.0, 'D': 25.0}

    def test_detect_overrepresented_default_threshold(self):
        """Test overrepresentation detection with default threshold."""
        overrepresented = detect_overrepresented_categories(self.distribution, self.ideal)
        self.assertEqual(overrepresented, ['A'])  # A is 15% over ideal

    def test_detect_overrepresented_custom_threshold(self):
        """Test overrepresentation detection with custom threshold."""
        overrepresented = detect_overrepresented_categories(self.distribution, self.ideal, threshold=20.0)
        self.assertEqual(overrepresented, [])  # A is only 15% over, threshold is 20%

    def test_no_overrepresented(self):
        """Test when no categories are overrepresented."""
        balanced_dist = {'A': 26.0, 'B': 24.0, 'C': 25.0, 'D': 25.0}
        overrepresented = detect_overrepresented_categories(balanced_dist, self.ideal, threshold=5.0)
        self.assertEqual(overrepresented, [])

    def test_mismatched_categories(self):
        """Test error handling for mismatched category sets."""
        bad_ideal = {'A': 25.0, 'E': 25.0, 'F': 25.0, 'G': 25.0}
        with self.assertRaises(ValueError):
            detect_overrepresented_categories(self.distribution, bad_ideal)


class TestUnderrepresentedCategories(unittest.TestCase):
    """Test detection of underrepresented categories."""

    def setUp(self):
        """Set up test distributions."""
        self.distribution = {'A': 10.0, 'B': 30.0, 'C': 30.0, 'D': 30.0}
        self.ideal = {'A': 25.0, 'B': 25.0, 'C': 25.0, 'D': 25.0}

    def test_detect_underrepresented_default_threshold(self):
        """Test underrepresentation detection with default threshold."""
        underrepresented = detect_underrepresented_categories(self.distribution, self.ideal)
        self.assertEqual(underrepresented, ['A'])  # A is 15% under ideal

    def test_detect_underrepresented_custom_threshold(self):
        """Test underrepresentation detection with custom threshold."""
        underrepresented = detect_underrepresented_categories(self.distribution, self.ideal, threshold=20.0)
        self.assertEqual(underrepresented, [])  # A is only 15% under, threshold is 20%

    def test_no_underrepresented(self):
        """Test when no categories are underrepresented."""
        balanced_dist = {'A': 24.0, 'B': 26.0, 'C': 25.0, 'D': 25.0}
        underrepresented = detect_underrepresented_categories(balanced_dist, self.ideal, threshold=5.0)
        self.assertEqual(underrepresented, [])

    def test_mismatched_categories(self):
        """Test error handling for mismatched category sets."""
        bad_ideal = {'A': 25.0, 'E': 25.0, 'F': 25.0, 'G': 25.0}
        with self.assertRaises(ValueError):
            detect_underrepresented_categories(self.distribution, bad_ideal)


if __name__ == '__main__':
    unittest.main()