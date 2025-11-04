import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("project_dataset.csv")

# Features including Age
X = df[['Age', 'BloodPressureProblems', 'AnyTransplants', 'AnyChronicDiseases',
        'Height', 'Weight', 'KnownAllergies', 'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries']]

y = df['PremiumPrice']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("linear_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained with Age and saved successfully as 'linear_model.pkl'")
