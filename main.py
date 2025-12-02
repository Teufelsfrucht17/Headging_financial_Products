# [nicht in PDF]
import numpy as np
import pandas as pd

# ============================================================
# CONFIG — EDIT HERE (Exxon Scope 1, Markets, Ratios, Prices)
# ============================================================

# [nicht in PDF] Years to model
YEARS = list(range(2025, 2031))  # 2025–2030 inclusive

# [nicht in PDF] Exxon global Scope 1 (tCO2e) — choose your anchors
# These are *anchors*; the interpolation creates 2025–2030.
SCOPE1_ANCHOR_START_YEAR = 2024
SCOPE1_ANCHOR_START_T = 91_000_000   # tCO2e (global Scope 1)  [set to your chosen base]
SCOPE1_ANCHOR_END_YEAR = 2030
SCOPE1_ANCHOR_END_T = 87_000_000     # tCO2e (global Scope 1)  [set to your chosen target]

# [nicht in PDF] Biggest Exxon markets you want to model (rest can be ignored or kept as "Other")
MARKETS = ["USA", "Canada", "EU_UK"]

# [nicht in PDF] Market share of global Scope 1 (must sum <= 1.0; remainder is "Other" not modeled)
MARKET_SHARE = {
    "USA": 0.45,
    "Canada": 0.15,
    "EU_UK": 0.10,
}
# [nicht in PDF] If you want to explicitly track "Other", set include_other=True below.

# [nicht in PDF] Coverage ratios: share of Scope 1 that is effectively carbon-priced (ETS/tax)
# Linear interpolation from 2025 -> 2030 for each market
COVERAGE_2025 = {"USA": 0.40, "Canada": 0.60, "EU_UK": 0.80}
COVERAGE_2030 = {"USA": 0.60, "Canada": 0.80, "EU_UK": 0.90}

# [nicht in PDF] Carbon price anchors (local currency) per tCO2 (you can keep everything in USD if you want)
# We'll convert EU/UK to USD using FX below so costs are comparable.
PRICE_2025 = {"USA": 30.0, "Canada": 60.0, "EU_UK": 60.0}     # USA/Canada in USD, EU_UK in EUR (assumption)
PRICE_2030 = {"USA": 50.0, "Canada": 130.0, "EU_UK": 125.0}   # USA/Canada in USD, EU_UK in EUR (assumption)

# [nicht in PDF] FX assumptions to convert everything to USD (optional but recommended)
FX_EURUSD = 1.08  # 1 EUR = 1.08 USD (assumption)
# If you want EU_UK already in USD, set FX_EURUSD = 1.0

# [nicht in PDF] Futures fixed prices (USD/tCO2). If EU_UK is in EUR, convert here too.
# You can set these equal to PRICE_2025 * (1 + premium) or use actual quoted futures later.
FUTURES_PRICE_USD = {
    "USA": 35.0,
    "Canada": 70.0,
    "EU_UK": 80.0 * FX_EURUSD,  # converting assumed EUR future to USD
}

# [nicht in PDF] Price simulation parameters
N_SIMS = 20_000
SIGMA = {"USA": 0.20, "Canada": 0.30, "EU_UK": 0.25}  # annual vol assumptions
RANDOM_SEED = 42

# [nicht in PDF] Hedge ratios to test (global sweep; applied equally to all markets)
HEDGE_RATIOS_TO_TEST = [0.0, 0.3, 0.5, 0.7, 1.0]

# [nicht in PDF] Include "Other" bucket?
INCLUDE_OTHER = False  # keep False for "big markets only" approach

# ============================================================
# HELPERS
# ============================================================

# [nicht in PDF]
def linear_interp(years, x0_year, x0_value, x1_year, x1_value) -> dict:
    """Linear interpolation for yearly series."""
    years = list(years)
    out = {}
    for y in years:
        w = (y - x0_year) / (x1_year - x0_year)
        out[y] = (1 - w) * x0_value + w * x1_value
    return out

# [nicht in PDF]
def build_scope1_global(years) -> dict:
    """Build global Scope 1 series for modeled years."""
    return linear_interp(
        years,
        SCOPE1_ANCHOR_START_YEAR, SCOPE1_ANCHOR_START_T,
        SCOPE1_ANCHOR_END_YEAR, SCOPE1_ANCHOR_END_T
    )

# [nicht in PDF]
def build_scope1_by_market(scope1_global: dict, include_other=False) -> dict:
    """Allocate global Scope 1 to markets via market shares."""
    shares_sum = sum(MARKET_SHARE.values())
    if shares_sum > 1.0 + 1e-9:
        raise ValueError("MARKET_SHARE sums to > 1.0. Fix shares.")

    scope1_market = {y: {} for y in scope1_global.keys()}

    for y, total in scope1_global.items():
        for m in MARKETS:
            scope1_market[y][m] = total * MARKET_SHARE[m]

        if include_other:
            scope1_market[y]["Other"] = total * (1.0 - shares_sum)

    return scope1_market

# [nicht in PDF]
def build_coverage(years) -> dict:
    """Coverage ratio per year and market (linear 2025->2030)."""
    cov = {y: {} for y in years}
    for m in MARKETS:
        series = linear_interp(years, 2025, COVERAGE_2025[m], 2030, COVERAGE_2030[m])
        for y in years:
            cov[y][m] = float(series[y])
    return cov

# [nicht in PDF]
def to_usd_price(market: str, price_local: float) -> float:
    """Convert EU_UK EUR price into USD; USA/Canada assumed already USD."""
    if market == "EU_UK":
        return price_local * FX_EURUSD
    return price_local

# [nicht in PDF]
def build_price_anchors_usd() -> tuple[dict, dict]:
    """Convert 2025 and 2030 anchor prices into USD per market."""
    p25 = {m: to_usd_price(m, PRICE_2025[m]) for m in MARKETS}
    p30 = {m: to_usd_price(m, PRICE_2030[m]) for m in MARKETS}
    return p25, p30

# [nicht in PDF]
def simulate_prices_gbm_targeted(years, n_sims, p0_usd, pT_usd, sigma, seed=42) -> dict:
    """
    Simulate GBM price paths with drift chosen to hit pT in expectation.
    prices[m] shape: (n_sims, n_years)
    """
    np.random.seed(seed)
    n_years = len(years)
    T = n_years - 1  # steps from first to last year

    prices = {}

    for m in MARKETS:
        s0 = p0_usd[m]
        sT = pT_usd[m]
        vol = sigma[m]

        # Choose mu such that E[S_T] = S_0 * exp(mu*T) = sT  -> mu = ln(sT/s0)/T
        mu = np.log(sT / s0) / max(T, 1)

        paths = np.zeros((n_sims, n_years))
        paths[:, 0] = s0

        for t in range(1, n_years):
            dt = 1.0
            shock = np.random.normal(
                (mu - 0.5 * vol**2) * dt,
                vol * np.sqrt(dt),
                size=n_sims
            )
            paths[:, t] = paths[:, t-1] * np.exp(shock)

        prices[m] = paths

    return prices

# [nicht in PDF]
def compute_costs(prices_usd, scope1_by_market, coverage_by_year, hedge_ratio, futures_price_usd) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (total_cost_no_hedge, total_cost_with_hedge) for all sims.
    Costs in USD.
    """
    n_sims = next(iter(prices_usd.values())).shape[0]
    total_no = np.zeros(n_sims)
    total_hedged = np.zeros(n_sims)

    years = list(scope1_by_market.keys())

    for j, y in enumerate(years):
        for m in MARKETS:
            emissions_total = scope1_by_market[y][m]  # Scope 1 for market
            coverage = coverage_by_year[y][m]
            emissions_priced = emissions_total * coverage  # effective priced emissions

            spot = prices_usd[m][:, j]  # USD/tCO2
            total_no += emissions_priced * spot

            # Futures hedge (fixed price on hedge_ratio of priced emissions)
            hedged_vol = hedge_ratio * emissions_priced
            unhedged_vol = (1.0 - hedge_ratio) * emissions_priced
            K = futures_price_usd[m]

            total_hedged += hedged_vol * K + unhedged_vol * spot

    return total_no, total_hedged

# [nicht in PDF]
def summarize_costs(no_hedge, with_hedge) -> dict:
    """Return summary metrics."""
    saving = no_hedge - with_hedge
    return {
        "mean_no_hedge": float(no_hedge.mean()),
        "mean_with_hedge": float(with_hedge.mean()),
        "mean_saving": float(saving.mean()),
        "std_no_hedge": float(no_hedge.std()),
        "std_with_hedge": float(with_hedge.std()),
        "p95_no_hedge": float(np.quantile(no_hedge, 0.95)),
        "p95_with_hedge": float(np.quantile(with_hedge, 0.95)),
        "p05_no_hedge": float(np.quantile(no_hedge, 0.05)),
        "p05_with_hedge": float(np.quantile(with_hedge, 0.05)),
    }

# [nicht in PDF]
def run_hedge_ratio_sweep():
    # Build emissions & coverage
    scope1_global = build_scope1_global(YEARS)
    scope1_market = build_scope1_by_market(scope1_global, include_other=INCLUDE_OTHER)
    coverage = build_coverage(YEARS)

    # Build price anchors and simulate prices
    p25_usd, p30_usd = build_price_anchors_usd()
    prices_usd = simulate_prices_gbm_targeted(
        years=YEARS,
        n_sims=N_SIMS,
        p0_usd=p25_usd,
        pT_usd=p30_usd,
        sigma=SIGMA,
        seed=RANDOM_SEED
    )

    rows = []
    # No-hedge baseline computed once (hedge_ratio=0 equals no hedge in this setup)
    # But we compute baseline explicitly for clarity.
    base_no, _ = compute_costs(
        prices_usd=prices_usd,
        scope1_by_market=scope1_market,
        coverage_by_year=coverage,
        hedge_ratio=0.0,
        futures_price_usd=FUTURES_PRICE_USD
    )

    for hr in HEDGE_RATIOS_TO_TEST:
        no, hedged = compute_costs(
            prices_usd=prices_usd,
            scope1_by_market=scope1_market,
            coverage_by_year=coverage,
            hedge_ratio=hr,
            futures_price_usd=FUTURES_PRICE_USD
        )

        # no here == base_no; using it anyway for readability
        s = summarize_costs(no, hedged)
        s["hedge_ratio"] = hr
        rows.append(s)

    df = pd.DataFrame(rows).sort_values("hedge_ratio").reset_index(drop=True)

    # Add some interpretable deltas
    df["p95_reduction"] = df["p95_no_hedge"] - df["p95_with_hedge"]
    df["std_reduction"] = df["std_no_hedge"] - df["std_with_hedge"]

    return df


# ============================================================
# RUN
# ============================================================

# [nicht in PDF]
if __name__ == "__main__":
    sweep = run_hedge_ratio_sweep()

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("\n=== Exxon CO2 Hedge Ratio Sweep (USD) ===")
    print(sweep[[
        "hedge_ratio",
        "mean_no_hedge", "mean_with_hedge", "mean_saving",
        "p95_no_hedge", "p95_with_hedge", "p95_reduction",
        "std_no_hedge", "std_with_hedge", "std_reduction"
    ]])

    output_path = "hedge_ratio_sweep.xlsx"
    try:
        sweep.to_excel(output_path, index=False)
        print(f"\nExcel export written to {output_path}")
    except ImportError as exc:
        print(f"\nCould not export to Excel ({exc}). Install openpyxl or xlsxwriter to enable Excel export.")
