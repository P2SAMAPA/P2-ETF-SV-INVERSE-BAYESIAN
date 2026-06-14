# Stochastic Volatility Inverse Problem (Bayesian Inversion)

Treats latent volatility as an unknown field and uses Bayesian inversion to estimate it from observed returns and macro variables. Laplace approximation provides posterior variance of log‑volatility. High posterior variance indicates uncertain volatility regime – a potential signal for regime shifts or alpha.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Model: returns = vol × ε, ε ~ N(0,1)
- Prior on log(vol) ~ N(0, σ_prior²) with macro‑adapted σ_prior
- Laplace approximation for posterior variance
- Score = posterior variance (higher = more uncertain)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-sv-inverse-bayesian-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High uncertainty → volatile regime is unclear → potential for large moves (alpha or risk).
- Low uncertainty → volatility is well‑estimated.

## Requirements

See `requirements.txt`.
