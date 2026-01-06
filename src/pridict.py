import joblib
import pandas as pd

# Load trained model
model = joblib.load("../models/random_forest.pkl")

# Example: predict demand for future days
future_days = pd.DataFrame({"day": [11, 12, 13, 14, 15]})

predictions = model.predict(future_days)

result = future_days.copy()
result["predicted_sales"] = predictions

print(result)

# Save predictions
result.to_csv("../outputs/future_demand_prediction.csv", index=False)
