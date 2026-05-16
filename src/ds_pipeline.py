from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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
