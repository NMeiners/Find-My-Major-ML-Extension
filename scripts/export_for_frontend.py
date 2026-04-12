import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 1. Load the pristine 6-Feature Master Data
print("Loading Master Data...")
df = pd.read_csv("docs/data/master_careers_riasec_categories.csv")
features = ["Realistic", "Investigative", "Artistic", "Social", "Enterprising", "Conventional"]

# 2. Train the Final Production Model 
print("Training Final Production Model...")
# Using your tuned parameters: depth 5, learning rate 0.1
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(df[features], df["Career Category"])

# 3. Convert and Save the Model to ONNX (For browser compatibility)
print("Exporting to ONNX...")
initial_type = [('float_input', FloatTensorType([None, 6]))]
onx = convert_sklearn(model, initial_types=initial_type)

with open("riasec_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())

# 4. Export the lightweight JSON database for the F1 Team
print("Exporting Jobs Database for Frontend...")
# Includes the O*NET-SOC Code for their MSU Major mapping
frontend_db = df[["O*NET-SOC Code", "Title", "Career Category"] + features]
frontend_db.to_json("riasec_jobs_db.json", orient="records")

print("\n✅ SUCCESS! You now have the two files for the F1 Team.")