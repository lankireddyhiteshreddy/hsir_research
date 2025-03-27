import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ✅ Import ML models
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor

# ✅ Step 1: Load Dataset
df = pd.read_csv("Jobs_export.csv")  # Ensure the file exists in the working directory

# Drop non-numeric columns (if necessary)
df = df.select_dtypes(include=[np.number])

# ✅ Step 2: Define Features (X) and Target (y)
X = df.drop(columns=["UsedCPUTime"])  # Replace with actual target column
y = df["UsedCPUTime"]

# ✅ Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ✅ Step 4: Normalize the Data (For KNN, SVM, MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Step 5: Define Models (SVM Optimized)
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVM": SVR(kernel="linear", cache_size=1000),  # ✅ Faster SVM with linear kernel
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42),
    "REPtree": DecisionTreeRegressor(max_depth=10, random_state=42)
}

# ✅ Step 6: Train, Predict, and Evaluate
results = []
for name, model in models.items():
    print(f"\n🚀 Training {name} model...")

    # Special handling for SVM to speed up training
    if name == "SVM":
        sample_size = min(10000, len(X_train_scaled))  # Use 10,000 samples or full if smaller
        X_train_svm, y_train_svm = X_train_scaled[:sample_size], y_train[:sample_size]
        model.fit(X_train_svm, y_train_svm)  # Train SVM on limited data
        y_pred = model.predict(X_test_scaled)
    elif name in ["KNN", "MLP"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Compute Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store results
    results.append([name, mae, r2])

    print(f"✅ {name} - MAE: {mae:.4f}, R²: {r2:.4f}")

# ✅ Step 7: Save Results to CSV
results_df = pd.DataFrame(results, columns=["Model", "MAE", "R2_Score"])
results_df.to_csv("model_comparison_results.csv", index=False)

print("\n📊 ✅ Model comparison results saved to 'model_comparison_results.csv' 🎯")
