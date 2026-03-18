"""
Title: Baseline Random Forest Recommender
Description: Implements a hybrid recommendation engine using a Random Forest 
             classifier to predict career categories and Cosine Similarity 
             to rank specific O*NET occupations.
Inputs: 6-point RIASEC normalized vector (Pandas DataFrame)
Outputs: Top N specific job recommendations with match scores
Dependencies: pandas, numpy, joblib, sklearn
Author: Nate G. (and AI Assistant)
Date: 2024-03-15
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List, Any


def load_pipeline_artifacts(model_path: str, db_path: str) -> Tuple[Any, pd.DataFrame]:
    """
    Loads the trained Random Forest model and the O*NET career database.

    Args:
        model_path (str): Filepath to the saved .joblib ML model.
        db_path (str): Filepath to the mapped O*NET CSV database.

    Returns:
        Tuple[Any, pd.DataFrame]: A tuple containing (loaded_model, career_dataframe).
        
    Raises:
        FileNotFoundError: If either the model or database file cannot be found.
    """
    try:
        rf_model = joblib.load(model_path)
        onet_db = pd.read_csv(db_path)
        return rf_model, onet_db
    except FileNotFoundError as e:
        print(f"❌ Error loading artifacts: {e}")
        raise


def get_job_recommendations(
    student_vector: pd.DataFrame, 
    rf_model: Any, 
    onet_db: pd.DataFrame, 
    top_n_categories: int = 3, 
    top_n_jobs: int = 3
) -> pd.DataFrame:
    """
    Generates personalized job recommendations using a two-step hybrid approach.
    
    Step 1: Predicts the top probability career categories using Random Forest.
    Step 2: Ranks specific jobs within those categories using Cosine Similarity.

    Args:
        student_vector (pd.DataFrame): 1-row DataFrame containing 'R normalized', 
                                       'I normalized', etc.
        rf_model (Any): Trained Random Forest classifier.
        onet_db (pd.DataFrame): O*NET database with 'Career Category' and RIASEC scores.
        top_n_categories (int): Number of broad categories to pool. Defaults to 3.
        top_n_jobs (int): Number of final jobs to return. Defaults to 3.

    Returns:
        pd.DataFrame: Recommended Job Titles, Categories, and Match Scores.
    """
    # 1. Predict Category Probabilities
    # Probabilities allow us to pool the most likely "sections" of the job library
    probs = rf_model.predict_proba(student_vector)[0]
    
    # Get the names of the Top N highest probability categories
    top_indices = np.argsort(probs)[-top_n_categories:][::-1]
    top_categories = rf_model.classes_[top_indices]
    
    # 2. Filter the O*NET database for candidate generation
    filtered_jobs = onet_db[onet_db['Career Category'].isin(top_categories)].copy()
    
    if filtered_jobs.empty:
        return pd.DataFrame()
    
    # 3. Calculate Micro-Ranking via Cosine Similarity
    # Note: Column names must match the O*NET CSV structure
    riasec_cols = [
        'Realistic', 'Investigative', 'Artistic', 
        'Social', 'Enterprising', 'Conventional'
    ]
    
    job_vectors = filtered_jobs[riasec_cols].values
    student_array = student_vector.values # Converts DF to Numpy for the math
    
    similarities = cosine_similarity(student_array, job_vectors)[0]
    filtered_jobs['Match_Score'] = similarities
    
    # 4. Final Ranking
    best_jobs = filtered_jobs.sort_values(by='Match_Score', ascending=False).head(top_n_jobs)
    
    return best_jobs[['Title', 'Career Category', 'Match_Score']]