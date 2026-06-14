import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def garch_volatility(returns, omega=0.01, alpha=0.1, beta=0.85):
    """Compute GARCH(1,1) conditional volatility."""
    T = len(returns)
    vol = np.zeros(T)
    vol[0] = np.std(returns)
    for t in range(1, T):
        vol[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * vol[t-1]**2)
    return vol

def bayesian_inversion_score(returns, macro_df, prior_vol=0.1, obs_noise=0.05):
    """
    Compute score = relative change in volatility when macro is used to predict returns.
    Higher score = macro helps predict volatility → more macro‑driven.
    """
    if len(returns) < 20 or macro_df is None or len(macro_df) < 20:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    # GARCH volatility (baseline)
    vol_garch = garch_volatility(returns)
    # Predict volatility using macro (ridge regression)
    # Use lagged macro to predict current volatility
    X = macro_scaled[:-1]
    y = vol_garch[1:]
    if len(y) < 10:
        return 0.0
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    y_pred = ridge.predict(X)
    # Score = improvement in volatility prediction (R²)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1 - ss_res / ss_tot
    # r2 can be negative; clip to [0,1]
    return max(0.0, min(1.0, r2))
