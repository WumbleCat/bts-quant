import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Union, Optional


def calculate_hurst_rs(series: np.ndarray, min_chunk: int = 10, max_chunk: Optional[int] = None) -> float:
    """
    Estimate the Hurst exponent using Rescaled Range (R/S) analysis.
    H < 0.5: Mean-reverting
    H = 0.5: Random walk (Brownian motion)
    H > 0.5: Trending / persistent
    """
    ts = np.asarray(series, dtype=float)
    ts = ts[~np.isnan(ts)]
    n = len(ts)
    
    if max_chunk is None:
        max_chunk = n // 2
    if n < min_chunk * 2:
        raise ValueError(f"Series length ({n}) is too short for R/S Hurst estimation.")

    chunk_sizes = np.unique(np.logspace(np.log10(min_chunk), np.log10(max_chunk), num=20).astype(int))
    chunk_sizes = chunk_sizes[chunk_sizes >= min_chunk]

    rs_values = []
    valid_sizes = []

    for size in chunk_sizes:
        num_chunks = n // size
        if num_chunks < 2:
            continue
            
        chunks = ts[:num_chunks * size].reshape(num_chunks, size)
        means = chunks.mean(axis=1, keepdims=True)
        deviates = np.cumsum(chunks - means, axis=1)
        
        r = deviates.max(axis=1) - deviates.min(axis=1)
        s = chunks.std(axis=1, ddof=1)
        
        valid = s > 0
        if np.any(valid):
            rs_values.append(np.mean(r[valid] / s[valid]))
            valid_sizes.append(size)

    hurst, _, _, _, _ = stats.linregress(np.log(valid_sizes), np.log(rs_values))
    return float(hurst)


def test_spread_dynamics(
    spread: Union[pd.Series, np.ndarray],
    adf_regression: str = "c",
    adf_autolag: str = "AIC",
    significance_level: float = 0.05
) -> Dict[str, Union[float, bool, str]]:
    """
    Run ADF, Hurst Exponent, and Jarque-Bera diagnostics on a spread series.
    """
    clean_spread = pd.Series(spread).dropna().astype(float)
    
    if len(clean_spread) < 30:
        raise ValueError("Spread series must have at least 30 non-NaN observations.")

    # 1. Augmented Dickey-Fuller (ADF) Test on spread levels
    adf_stat, adf_pval, adf_lags, nobs, crit_vals, _ = adfuller(
        clean_spread.values, regression=adf_regression, autolag=adf_autolag
    )

    # 2. Hurst Exponent on spread levels
    hurst_exp = calculate_hurst_rs(clean_spread.values)

    # 3. Jarque-Bera (JB) Test on spread increments
    increments = clean_spread.diff().dropna().values
    jb_stat, jb_pval = stats.jarque_bera(increments)
    skewness = float(stats.skew(increments))
    excess_kurtosis = float(stats.kurtosis(increments))

    return {
        "n_obs": int(nobs),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pval),
        "adf_lags": int(adf_lags),
        "adf_crit_5pct": float(crit_vals["5%"]),
        "is_stationary_adf": bool(adf_pval < significance_level),
        "hurst_exponent": round(hurst_exp, 4),
        "is_mean_reverting_hurst": bool(hurst_exp < 0.5),
        "jb_stat": float(jb_stat),
        "jb_pvalue": float(jb_pval),
        "increments_skew": round(skewness, 4),
        "increments_excess_kurtosis": round(excess_kurtosis, 4),
        "is_gaussian_increments": bool(jb_pval >= significance_level),
    }
