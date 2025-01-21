import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Sample Data (Hypothetical dataset)
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Splitting dataset
X = df[["Experience"]]
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate R2 Score
r2 = r2_score(y_test, y_pred)

print(f"R2 Score: {r2}")
