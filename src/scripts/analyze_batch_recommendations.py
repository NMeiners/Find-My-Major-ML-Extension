"""
File: analyze_batch_recommendations.py
Path: src/scripts/analyze_batch_recommendations.py

Purpose:
  CLI utility to analyze batch recommendations across a set of student profiles.
  Uses the champion Gradient Boosting model to generate and validate recommendations.

Original Author(s):
  - Nathan Meiners
  - AI Assistant

AI Tools Used:
  - GitHub Copilot - Initial implementation
  - Claude Sonnet 4.6 - Documentation

Editors:
  - AI Assistant (2026-04-20) — Added file header and relocated from scripts/ to src/scripts/

Last Editor:
  - AI Assistant

Last Edit Date:
  2026-04-20

Assumptions & Constraints:
  - Requires master training data and test data CSV files
  - Uses pre-configured Gradient Boosting model with fixed hyperparameters
  - Results compared against actual student major labels in test data
  - top_n_jobs controls number of recommendations (supports "Reject & Replace")

Related Docs:
  - docs/src/models/models.md
  - docs/evaluation_workflow.md
"""

import pandas as pd
from src.models.gradient_boosting import GradientBoostingModel


def analyze_batch(batch_size: int = 10) -> None:
    """
    Name: analyze_batch

    Purpose:
      Analyzes batch of student profiles using champion model and compares
      recommendations against ground truth majors.

    Inputs:
      - batch_size: int — number of students to analyze (default 10)

    Outputs:
      - Prints detailed recommendation analysis per student to console

    Raises / Errors:
      - FileNotFoundError: if data CSV files not found

    Notes:
      - Uses production-tuned model parameters
      - Validates recommendations against test data labels
    """
    # 1. Load the 6-feature master data and test dataset
    # These paths are defined in your project architecture
    master_df = pd.read_csv("docs/data/master_careers_riasec_categories.csv")
    test_df = pd.read_csv("docs/data/Kaggle_Cleaned_Mapped_Categories.csv")

    # 2. Define the 6 RIASEC features
    features = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]

    # 3. Initialize and train the champion model
    # We use the parameters from your tuned production config
    model = GradientBoostingModel(
        x_features=features,
        y_feature="Career Category",
        parameters={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 5},
        top_n_jobs=20,        # Support "Reject & Replace"
        top_n_categories=3
    )

    print("Training champion model...")
    model.train(master_df[features], master_df["Career Category"])

    # 4. Select a batch of users to analyze (e.g., first 10 users)

    for i in range(batch_size):
        student = test_df.iloc[[i]]
        actual_major = student['major'].values[0]

        print(f"\n=== Student {i+1} ===")
        print(f"Actual Major: {actual_major}")

        # Run inference
        recommendations = model.test(student[features], master_df)

        print("Top 3 Recommendations:")
        for idx, row in recommendations.head(3).iterrows():
            print(f"  {row['Title']}: {row['Match_Score']:.4f}")


if __name__ == "__main__":
    analyze_batch(batch_size=10)
