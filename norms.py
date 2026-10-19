import numpy as np
from scipy.integrate import simpson

def L2_norm(v, x):
    """Compute the L^2 norm using Simpson's rule."""
    return np.sqrt(simpson(np.abs(v)**2, x=x))

def H1_norm(v, x):
    """Compute the H^1 norm (includes spatial derivative)."""
    dx = x[1] - x[0]
    grad = np.gradient(v, dx)
    return np.sqrt(simpson(np.abs(v)**2 + np.abs(grad)**2, x=x))

def stepanov_norm(v, x, radius=0.1, sigma=0.05):
    """
    Compute the weighted Stepanov norm S_w^2.
    Uses a sliding window of radius radius/2 and Gaussian weight with width sigma.
    """
    dx = x[1] - x[0]
    max_val = 0.0
    for center in x:
        mask = np.abs(x - center) <= radius/2
        if not np.any(mask):
            continue
        weight = np.exp(-(x[mask] - center)**2 / sigma**2)
        integral = np.sum(weight * np.abs(v[mask])**2 * dx)
        max_val = max(max_val, integral)
    return np.sqrt(max_val)

def early_warning_indicator(S, t, window_length=0.2, eps=1e-12):
    """
    Compute the early-warning indicator H(t) = window_length * dS/dt / (S**2 + eps).
    """
    dS = np.gradient(S, t)
    return window_length * dS / (S**2 + eps)
