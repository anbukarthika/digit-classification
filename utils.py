# utils.py
import numpy as np
import statsmodels.api as sm

def add_constant(x):
    """Add a column of ones for intercept term."""
    return sm.add_constant(x)

def fit_ols(x, y):
    """Fit Ordinary Least Squares model."""
    # x should already include constant if needed
    model = sm.OLS(y, x).fit()
    return model

def print_summary(model):
    """Print the model summary."""
    print(model.summary())

def get_predictions(model, x):
    """Return predicted values."""
    return model.predict(x)