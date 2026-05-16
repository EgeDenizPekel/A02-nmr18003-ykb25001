import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

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


mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    early_stopping=True,          # <-- enables early stopping
    validation_fraction=0.1,      # 10% of training data used as validation
    n_iter_no_change=10,          # stop if no improvement for 10 epochs
    max_iter=500,
    random_state=42
)

mlp.fit(X_train, y_train)

# Train predictions
y_train_pred = mlp.predict(X_train)

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
y_test_pred = mlp.predict(X_test)

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
