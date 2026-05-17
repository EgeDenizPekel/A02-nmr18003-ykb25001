import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train size: {X_train.shape[0]} samples")
print(f"Test size:  {X_test.shape[0]} samples")
print(f"Features:   {housing.feature_names}")


MLP_PARAMS = dict(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    max_iter=500,
    random_state=42,
)

# Baseline: unscaled MLP
mlp_unscaled = MLPRegressor(**MLP_PARAMS)
mlp_unscaled.fit(X_train, y_train)
r2_unscaled_train = r2_score(y_train, mlp_unscaled.predict(X_train))
r2_unscaled_test  = r2_score(y_test,  mlp_unscaled.predict(X_test))

# Scaled pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(**MLP_PARAMS)),
])
pipeline.fit(X_train, y_train)

# Train predictions
y_train_pred = pipeline.predict(X_train)
r2_scaled_train = r2_score(y_train, y_train_pred)

print(f"\nTrain Metrics:")
print(f"  R²:   {r2_score(y_train, y_train_pred):.4f}")
print(f"  MAE:  {mean_absolute_error(y_train, y_train_pred):.4f}")
print(f"  RMSE: {root_mean_squared_error(y_train, y_train_pred):.4f}")

# Plot: actual vs. predicted on training set
os.makedirs("figures", exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_train, y_train_pred, alpha=0.3, s=10, color="steelblue")
lims = [0, 5.5]
ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual Median House Value ($100k)")
ax.set_ylabel("Predicted Median House Value ($100k)")
ax.set_title("Training Set: Actual vs. Predicted")
ax.legend()
fig.tight_layout()
fig.savefig("figures/train_actual_vs_pred.png", dpi=150)
plt.close(fig)
print("Saved figures/train_actual_vs_pred.png")


# Test predictions
y_test_pred = pipeline.predict(X_test)
r2_scaled_test = r2_score(y_test, y_test_pred)

print(f"\nTest Metrics:")
print(f"  R²:   {r2_score(y_test, y_test_pred):.4f}")
print(f"  MAE:  {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"  RMSE: {root_mean_squared_error(y_test, y_test_pred):.4f}")

print(f"\nScaling impact (R²):")
print(f"  Train: {r2_unscaled_train:.4f} -> {r2_scaled_train:.4f}  ({r2_scaled_train - r2_unscaled_train:+.4f})")
print(f"  Test:  {r2_unscaled_test:.4f}  -> {r2_scaled_test:.4f}   ({r2_scaled_test - r2_unscaled_test:+.4f})")

# Plot: actual vs. predicted on test set
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_test_pred, alpha=0.3, s=10, color="darkorange")
lims = [0, 5.5]
ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Actual Median House Value ($100k)")
ax.set_ylabel("Predicted Median House Value ($100k)")
ax.set_title("Test Set: Actual vs. Predicted")
ax.legend()
fig.tight_layout()
fig.savefig("figures/test_actual_vs_pred.png", dpi=150)
plt.close(fig)
print("Saved figures/test_actual_vs_pred.png")
