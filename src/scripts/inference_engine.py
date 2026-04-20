"""
File: inference_engine.py
Path: src/scripts/inference_engine.py

Purpose:
  Provides a unified interface for loading trained ONNX models and career databases,
  and generating ranked job recommendations via two-stage retrieval (category prediction
  + cosine similarity ranking).

Original Author(s):
  - Nathan Meiners (F1 Team)
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
  - ONNX model must be trained with 6 RIASEC input features
  - Jobs database JSON must contain Title, Career Category, and 6 RIASEC columns
  - Student scores are expected in [0.0, 1.0] range
  - Cosine similarity used for ranking

Related Docs:
  - docs/src/models/models.md
  - docs/evaluation_workflow.md
"""

import pandas as pd
import numpy as np
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity


class CareerRecommender:
    def __init__(self, model_path="riasec_model.onnx", db_path="riasec_jobs_db.json"):
        # 1. Load the "Brain" (ONNX Model)
        self.session = ort.InferenceSession(model_path)
        # 2. Load the "Library" (Jobs Database)
        self.db = pd.read_json(db_path)
        self.features = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]

    def get_all_recommendations(self, student_scores):
        """
        Name: get_all_recommendations

        Purpose:
          Takes 6 RIASEC scores [0.0 - 1.0] and returns ALL ~900 jobs ranked by similarity.

        Inputs:
          - student_scores: list[float] — 6 RIASEC scores in [0.0, 1.0]

        Outputs:
          - str — JSON string of ranked career recommendations with Match_Score

        Raises / Errors:
          - RuntimeError: if ONNX inference fails

        Notes:
          - Returns full database ranked by cosine similarity
          - Used primarily by frontend for "all jobs" view
        """
        # A. Predict the Top 3 Categories using the ONNX model
        input_name = self.session.get_inputs()[0].name
        input_data = np.array([student_scores], dtype=np.float32)
        raw_probs = self.session.run(None, {input_name: input_data})[1][0]
        
        # Sort categories by probability
        categories = sorted(raw_probs.items(), key=lambda x: x[1], reverse=True)
        top_3_categories = [cat[0] for cat in categories[:3]]

        # B. Filter the database to only those 3 categories
        candidates = self.db[self.db["Career Category"].isin(top_3_categories)].copy()

        # C. Calculate Cosine Similarity for every job in those categories
        student_vector = np.array([student_scores])
        job_vectors = candidates[self.features].values
        
        candidates["Match_Score"] = cosine_similarity(student_vector, job_vectors)[0]

        # D. Return the full list ranked by score
        sorted_candidates = candidates.sort_values(by="Match_Score", ascending=False)
        
        # Convert to a JSON string (orient="records" makes it a standard list of dictionaries)
        return sorted_candidates.to_json(orient="records")


# Example Usage for the F1 Team:
# engine = CareerRecommender()
# results = engine.get_all_recommendations([0.8, 0.2, 0.9, 0.5, 0.1, 0.4])
