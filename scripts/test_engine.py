import json
<<<<<<< HEAD
=======
import pandas as pd
>>>>>>> 4415ceadd6dc08ea88c102ca8f8a9d1766d5b879
from scripts.inference_engine import CareerRecommender

def test_inference():
    print("--- Initializing Inference Engine ---")
    engine = CareerRecommender(
        model_path="riasec_model.onnx", 
        db_path="riasec_jobs_db.json"
    )

    # High Realistic / High Investigative test student
    test_student = [0.9, 0.8, 0.1, 0.2, 0.3, 0.4]

    print(f"\n--- Running Inference for Realistic-Investigative Student ---")
<<<<<<< HEAD
    # 1. Get the JSON string from the engine
    json_results = engine.get_all_recommendations(test_student)

    # 2. Parse the JSON string back into a Python list of dictionaries
    results_list = json.loads(json_results)

    # 3. Now we can correctly count the jobs
    print(f"Total Careers Ranked: {len(results_list)}")
    print("\nTop 10 Recommendations:")
    
    # 4. Loop through the first 10 items in the list and print them cleanly
    print(f"{'O*NET-SOC Code':<15} | {'Title':<45} | {'Match_Score':<10}")
    print("-" * 75)
    for job in results_list[:10]:
        code = job.get("O*NET-SOC Code", "N/A")
        title = job.get("Title", "N/A")[:43] # Truncate long titles for display
        score = round(job.get("Match_Score", 0), 6)
        print(f"{code:<15} | {title:<45} | {score:<10}")

    if len(results_list) > 0:
        print("\n✅ TEST SUCCESSFUL: The engine returned a valid JSON array.")
=======
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
>>>>>>> 4415ceadd6dc08ea88c102ca8f8a9d1766d5b879
    else:
        print("\n❌ TEST FAILED: No results returned.")

if __name__ == "__main__":
    test_inference()
