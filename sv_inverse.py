import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def sv_likelihood(vol, returns, prior_vol=0.1, obs_noise=0.05):
    """
    Compute negative log-posterior for latent volatility.
    Model: returns = vol * epsilon, epsilon ~ N(0,1)
    Prior: log(vol) ~ N(0, prior_vol^2)
    """
    if vol <= 0:
        return 1e10
    # Observation likelihood
    ll = -0.5 * np.sum((returns / vol)**2 + np.log(2 * np.pi * vol**2))
    # Prior on log-volatility
    log_vol = np.log(vol)
    lp = -0.5 * (log_vol / prior_vol)**2 - 0.5 * np.log(2 * np.pi * prior_vol**2)
    return -(ll + lp)

def laplace_approx(returns, prior_vol=0.1, obs_noise=0.05):
    """
    Laplace approximation to the posterior of volatility.
    Returns: posterior mean (log-vol), posterior variance, MAP volatility.
    """
    # Find MAP estimate of log-volatility
    def neg_posterior(log_vol):
        vol = np.exp(log_vol)
        return sv_likelihood(vol, returns, prior_vol, obs_noise)
    # Initial guess: log of sample std
    init = np.log(max(np.std(returns), 0.01))
    res = minimize(neg_posterior, x0=init, method='L-BFGS-B')
    if not res.success:
        return 0.0, 0.0, 0.0
    log_vol_map = res.x[0]
    vol_map = np.exp(log_vol_map)
    # Approximate posterior variance via Hessian (finite difference)
    eps = 1e-4
    f0 = neg_posterior(log_vol_map)
    f1 = neg_posterior(log_vol_map + eps)
    f2 = neg_posterior(log_vol_map - eps)
    hess = (f1 + f2 - 2*f0) / (eps**2)
    posterior_var = 1.0 / (hess + 1e-8)
    return log_vol_map, posterior_var, vol_map

def bayesian_inversion_score(returns, macro_df, prior_vol=0.1, obs_noise=0.05):
    """
    Compute score = volatility shift: how much the posterior deviates from prior.
    Score = |log(vol_map) - log(prior_vol)| / log(prior_vol)
    """
    if len(returns) < 10:
        return 0.0
    # Adjust prior_vol using macro (as before)
    if macro_df is not None and len(macro_df) > 0:
        scaler = StandardScaler()
        macro_scaled = scaler.fit_transform(macro_df)
        pca = PCA(n_components=1)
        macro_factor = pca.fit_transform(macro_scaled).flatten()
        current_macro = macro_factor[-1]
        adjusted_prior = prior_vol * (1 + current_macro)
        adjusted_prior = max(0.05, min(0.5, adjusted_prior))
    else:
        adjusted_prior = prior_vol
    # Compute MAP volatility
    _, _, vol_map = laplace_approx(returns, adjusted_prior, obs_noise)
    if vol_map <= 0:
        return 0.0
    # Score = absolute relative deviation from prior
    prior_volatility = adjusted_prior
    if prior_volatility == 0:
        return 0.0
    score = abs(np.log(vol_map) - np.log(prior_volatility)) / abs(np.log(prior_volatility))
    # Clip to reasonable range
    return min(2.0, max(0.0, score))
