import numpy as np
import pandas as pd

def simulate_ou(theta=0.05, mu=0.0, sigma=0.3, n=2000, x0=0.0, seed=0):
    rng = np.random.default_rng(seed)
    a = np.exp(-theta)
    sd = sigma * np.sqrt((1 - np.exp(-2 * theta)) / (2 * theta))
    x = np.empty(n)
    x[0] = x0
    for t in range(n - 1):
        x[t + 1] = mu + (x[t] - mu) * a + sd * rng.standard_normal()
    return pd.Series(x, index=pd.bdate_range("2015-01-01", periods=n))
def closed_form_positions(spread, theta, mu, sigma, gamma=2.0, w_max=1.0):
    a = np.exp(-theta)
    m = (mu - spread) * (1 - a)
    v = sigma**2 * (1 - np.exp(-2 * theta)) / (2 * theta)
    w = m / (gamma * v)
    return w.clip(-w_max, w_max)