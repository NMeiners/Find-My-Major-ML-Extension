import pandas as pd
from src.models.gradient_boosting import GradientBoostingModel

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
batch_size = 10

for i in range(batch_size):
    student = test_df.iloc[[i]]
    actual_major = student['major'].values[0]
    
    # Preprocess student columns to match O*NET feature names
    student_input = student.rename(columns={
        "R normalized": "Realistic", "I normalized": "Investigative", 
        "A normalized": "Artistic", "S normalized": "Social", 
        "E normalized": "Enterprising", "C normalized": "Conventional"
    })
    
    # Run the ranking engine
    recommendations = model.test(student_input, master_df)
    
    print(f"\n============================================")
    print(f"USER #{i} | ACTUAL MAJOR: {actual_major.upper()}")
    print(f"============================================")
    # Display top 5 for readability, but model holds top 20
    print(recommendations[['O*NET-SOC Code', 'Title', 'Career Category', 'Match_Score']].head(5))