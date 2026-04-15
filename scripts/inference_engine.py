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
        Takes 6 RIASEC scores [0.0 - 1.0] and returns ALL ~900 jobs ranked.
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
        return candidates.sort_values(by="Match_Score", ascending=False)

# Example Usage for the F1 Team:
# engine = CareerRecommender()
# results = engine.get_all_recommendations([0.8, 0.2, 0.9, 0.5, 0.1, 0.4])