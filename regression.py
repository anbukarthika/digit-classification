# regression.py
import numpy as np
from utils import add_constant, fit_ols, print_summary, get_predictions

def main():
    # Sample data
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    y = np.array([2, 4, 5, 4, 5, 6, 5, 6])

    # Add constant for intercept term
    X_with_const = add_constant(X)

    # Fit the model
    model = fit_ols(X_with_const, y)

    # View the summary
    print_summary(model)

    # Predict
    y_pred = get_predictions(model, X_with_const)
    print("Predictions:", y_pred)

if __name__ == "__main__":
    main()