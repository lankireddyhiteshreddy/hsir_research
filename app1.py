import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("Jobs_export.csv")

# Print dataset info
print("✅ Dataset loaded successfully!")
print(df.head())

# Check if 'UsedCPUTime' column exists
if 'UsedCPUTime' not in df.columns:
    print("❌ Error: 'UsedCPUTime' column not found in dataset!")
    exit()

# Drop non-numeric columns (or encode them)
for col in df.columns:
    if df[col].dtype == 'object':  # If column is categorical
        df[col] = LabelEncoder().fit_transform(df[col])  # Encode categorical data

# Split features and target
X = df.drop(columns=['UsedCPUTime'])  # Features
y = df['UsedCPUTime']  # Target

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Evaluation:")
print(f"📌 Mean Absolute Error (MAE): {mae:.4f}")
print(f"📌 R-Squared (R²) Score: {r2:.4f}")

# Save predictions to CSV
output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
output.to_csv("predicted_results.csv", index=False)
print("✅ Predictions saved to 'predicted_results.csv'")
