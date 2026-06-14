import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

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
    return -(ll + lp)  # negative for minimisation

def laplace_approx(returns, prior_vol=0.1, obs_noise=0.05):
    """
    Laplace approximation to the posterior of volatility.
    Returns: posterior mean, posterior variance (in log-vol space).
    """
    # Find MAP estimate of log-volatility
    def neg_posterior(log_vol):
        vol = np.exp(log_vol)
        return sv_likelihood(vol, returns, prior_vol, obs_noise)
    res = minimize(neg_posterior, x0=np.log(np.std(returns)), method='L-BFGS-B')
    log_vol_map = res.x[0]
    # Approximate posterior variance via Hessian (finite difference)
    eps = 1e-4
    f0 = neg_posterior(log_vol_map)
    f1 = neg_posterior(log_vol_map + eps)
    f2 = neg_posterior(log_vol_map - eps)
    hess = (f1 + f2 - 2*f0) / (eps**2)
    posterior_var = 1.0 / (hess + 1e-8)
    return log_vol_map, posterior_var

def bayesian_inversion_score(returns, macro_df, prior_vol=0.1, obs_noise=0.05):
    """
    Compute score = posterior variance of log-volatility.
    Higher variance = more uncertain regime.
    """
    if len(returns) < 10:
        return 0.0
    # Use macro to adjust prior_vol (higher macro -> higher prior uncertainty)
    if macro_df is not None and len(macro_df) > 0:
        # Composite macro factor
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        macro_scaled = scaler.fit_transform(macro_df)
        pca = PCA(n_components=1)
        macro_factor = pca.fit_transform(macro_scaled).flatten()
        current_macro = macro_factor[-1]
        # Scale prior_vol between 0.05 and 0.5
        adjusted_prior = prior_vol * (1 + current_macro)
        adjusted_prior = max(0.05, min(0.5, adjusted_prior))
    else:
        adjusted_prior = prior_vol
    _, posterior_var = laplace_approx(returns, adjusted_prior, obs_noise)
    return float(posterior_var)
