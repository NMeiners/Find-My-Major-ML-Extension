import json
import pandas as pd
from scripts.inference_engine import CareerRecommender

def test_inference():
    print("--- Initializing Inference Engine ---")
    # 1. Initialize the engine (ensure .onnx and .json are in your root)
    engine = CareerRecommender(
        model_path="riasec_model.onnx", 
        db_path="riasec_jobs_db.json"
    )

    # 2. Define a "High Realistic / High Investigative" test student
    # Typical for STEM, Engineering, or Skilled Trades
    test_student = [
        0.9, # Realistic (High)
        0.8, # Investigative (High)
        0.1, # Artistic
        0.2, # Social
        0.3, # Enterprising
        0.4  # Conventional
    ]

    print(f"\n--- Running Inference for Realistic-Investigative Student ---")
    # 3. Run the engine to get ALL ~900 jobs ranked
    results = engine.get_all_recommendations(test_student)
    if isinstance(results, str):
        results = json.loads(results)
    results_df = pd.DataFrame(results)

    # 4. Display the results
    print(f"Total Careers Ranked: {len(results_df)}")
    print("\nTop 10 Recommendations:")
    print(results_df[['O*NET-SOC Code', 'Title', 'Career Category', 'Match_Score']].head(10))

    # 5. Simple Validation
    if not results_df.empty:
        print("\n✅ TEST SUCCESSFUL: The engine returned ranked results.")
    else:
        print("\n❌ TEST FAILED: No results returned.")

if __name__ == "__main__":
    test_inference()
