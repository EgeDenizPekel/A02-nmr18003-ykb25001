# A02: Ping Pong - California Housing Regression

OPIM 5512 | University of Connecticut

**Partners:** Ege Deniz Pekel (ykb25001) & Noah Reed (nmr18003)

## Project Overview

This project trains a neural network regression model (MLPRegressor) on the California Housing dataset to predict median house values. The workflow follows a collaborative ping-pong Git branching strategy where both partners contribute via pull requests.

## Structure

```
src/
└── ds_pipeline.py   # Main pipeline: load, split, train, plot
figures/
├── train_actual_vs_pred.png
└── test_actual_vs_pred.png
requirements.txt
```

## How to Run

```bash
# Optional: set up a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
python src/ds_pipeline.py
```

## Output

Running the script generates two plots saved to `figures/`:
- `train_actual_vs_pred.png` - actual vs predicted on training set
- `test_actual_vs_pred.png` - actual vs predicted on test set

It also prints evaluation metrics (R², MAE, RMSE) for both sets and a before/after scaling comparison.

## Results

### Final model (with StandardScaler)

| Set   | R²     | MAE    | RMSE   |
|-------|--------|--------|--------|
| Train | 0.8060 | 0.3403 | 0.5092 |
| Test  | 0.7861 | 0.3563 | 0.5294 |

### Impact of feature scaling

Adding `StandardScaler` before the MLP improved test R² by **+0.31** - a large gain for a single change.

| Set   | Unscaled R² | Scaled R² | Change  |
|-------|-------------|-----------|---------|
| Train | 0.4879      | 0.8060    | +0.3181 |
| Test  | 0.4751      | 0.7861    | +0.3110 |

MLP gradient descent is sensitive to feature scale. The California Housing features span very different ranges (e.g. `AveRooms` vs `Population`), so normalizing them to zero mean and unit variance is critical for the optimizer to converge well.
