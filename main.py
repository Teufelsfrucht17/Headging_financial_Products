# [not in PDF]
import numpy as np
import pandas as pd

# ================================
# CONFIG – Exxon CO2 Hedging Model
# ================================

YEARS = list(range(2025, 2031))  # 2025–2030 inclusive

# Global Scope-1 anchor (ExxonMobil 2024 reported approx.)
SCOPE1_ANCHOR_START_YEAR = 2024
SCOPE1_ANCHOR_START_T    = 99_000_000   # tCO2e global Scope 1
SCOPE1_ANCHOR_END_YEAR   = 2030
SCOPE1_ANCHOR_END_T      = 82_000_000    # Target level after divestments (Gravenchon etc.)

# Modeled markets
MARKETS = ["USA", "Canada", "EU_UK"]

# Market shares of global Scope-1 emissions (gross emissions before regulation)
# Canada is high due to oil sands (Imperial Oil), EU share declines due to divestments.
MARKET_SHARE = {
    "USA":    0.20,  # ~36% der globalen Scope-1 Emissionen (Permian, Gulf Coast Refining)
    "Canada": 0.34,  # ~22% (Kearl, Cold Lake – hohe Intensität)
    "EU_UK":  0.26,  # ~17% (Fife, Antwerpen, UK/EU Raffinerien nach Gravenchon-Divestment)
}
# Rest (~25%) = Asia/LNG/Other (nicht explizit im Modell, CO2-Kosten ~0 in der Logik hier)

# "Effective Coverage": Welcher Prozentsatz der Emissionen fällt effektiv unter ein CO2-Preissystem?
# 2025 eher konservativ: noch viele Free Allowances / Benchmarks, Coverage steigt dann bis 2030 deutlich an.

COVERAGE_2025 = {
    "USA":    0.25,  # v.a. CA/WA + einzelne Programme, kein bundesweiter Carbon Price
    "Canada": 0.55,  # TIER/OBPS: signifikanter Teil über Benchmark, aber noch nicht "voll durchgereicht"
    "EU_UK":  0.65,  # ETS schon relativ "tight", aber noch Free Allocation
}

# 2030: "Risk / Bull Case" für CO2-Exposure:
# - USA: mehr Staaten + evtl. verschärfte Regionalprogramme -> höherer Anteil bepreist
# - Canada: Benchmarks strenger, mehr Volumen > Benchmark -> deutlich mehr taxable Volume
# - EU_UK: Free Allocation fast weg, CBAM voll ausgerollt -> Großteil der Emissionen preiswirksam
COVERAGE_2030 = {
    "USA":    0.50,  # ≈ 25% der US-Emissionen effektiv bepreist
    "Canada": 0.90,  # ≈ 66% der kanadischen Emissionen oberhalb Benchmark / voll im Pricing
    "EU_UK":  0.80,  # ≈ 85% der EU+UK-Emissionen unter ETS/UK ETS ohne nennenswerte Free Allocation
}


# Currency assumptions
FX_EURUSD = 1.05
FX_CADUSD = 0.71

# Spot price anchors (forecasts for drift calculation)
# USA/Canada converted into USD, EU in EUR (converted below)
PRICE_2025 = {
    "USA":    32.5,   # USD/t – Kalifornien / WCI Auktionen + RGGI (heute ~20–30 USD)
    "Canada": 67.5,   # USD/t – ~95 CAD/t TIER/OBPS Pfad (eingefrorener Korridor)
    "EU_UK":  81.5,   # EUR/t – in etwa aktuelles Dec25 EUA-Niveau
}

# Risk-scenario prices 2030 (bullish, aber noch im Rahmen der Studien/PDF)
# -> das ist das Szenario, GEGEN das wir hedgen wollen.
PRICE_2030 = {
    "USA":    55.0,   # USD/t – Oberes Ende 40–55 USD für CA ETS / RGGI
    "Canada": 90.0,   # USD/t – über dem risk-adjusted Mean (~85), unterhalb 170 CAD Pfad
    "EU_UK":  105.0,  # EUR/t – oberes Drittel der 75–105 EUR ETS-Projektionen
}

# Hedge price (K): what Exxon would effectively lock in TODAY for 2030
# EU: basiert grob auf Dec30-EUA-Futures (unterhalb Risk-Szenario),
# Canada/USA: abgeschätzt als moderat unterhalb des 2030-Riskszenarios.
FUTURES_PRICE_USD = {
    "USA":    45.0,              # USD/t – Forward unterhalb 55 USD Risk-Szenario
    "Canada": 75.0,              # USD/t – Forward etwas unter 90 USD Erwartung
    "EU_UK":  90.0 * FX_EURUSD,  # EUR 90 Dec30-Future (ca. Mitte der Terminkurve) in USD
}

# Volatility (sigma)
SIGMA = {
    "USA":    0.20,  # Floor-price mechanism dampens volatility
    "Canada": 0.30,  # High political uncertainty (Poilievre vs. Trudeau election)
    "EU_UK":  0.35,  # High volatility due to MSR interventions and gas prices
}

# Simulation settings
N_SIMS = 20_000
RANDOM_SEED = 42

# Hedge ratios you simulate (global, same proportion for all markets)
HEDGE_RATIOS_TO_TEST = [0.0, 0.25, 0.50, 0.75, 1.0]  # Exxon-specific steps

# "Other" not modeled explicitly
INCLUDE_OTHER = False
# ============================================================
# HELPERS
# ============================================================

# [not in PDF]
def linear_interp(years, x0_year, x0_value, x1_year, x1_value) -> dict:
    """Linear interpolation for yearly series."""
    years = list(years)
    out = {}
    for y in years:
        w = (y - x0_year) / (x1_year - x0_year)
        out[y] = (1 - w) * x0_value + w * x1_value
    return out

# [not in PDF]
def build_scope1_global(years) -> dict:
    """Build global Scope 1 series for modeled years."""
    return linear_interp(
        years,
        SCOPE1_ANCHOR_START_YEAR, SCOPE1_ANCHOR_START_T,
        SCOPE1_ANCHOR_END_YEAR, SCOPE1_ANCHOR_END_T
    )

# [not in PDF]
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

# [not in PDF]
def build_coverage(years) -> dict:
    """Coverage ratio per year and market (linear 2025->2030)."""
    cov = {y: {} for y in years}
    for m in MARKETS:
        series = linear_interp(years, 2025, COVERAGE_2025[m], 2030, COVERAGE_2030[m])
        for y in years:
            cov[y][m] = float(series[y])
    return cov

# [not in PDF]
def to_usd_price(market: str, price_local: float) -> float:
    """Convert EU_UK EUR price into USD; USA/Canada assumed already USD."""
    if market == "EU_UK":
        return price_local * FX_EURUSD
    return price_local

# [not in PDF]
def build_price_anchors_usd() -> tuple[dict, dict]:
    """Convert 2025 and 2030 anchor prices into USD per market."""
    p25 = {m: to_usd_price(m, PRICE_2025[m]) for m in MARKETS}
    p30 = {m: to_usd_price(m, PRICE_2030[m]) for m in MARKETS}
    return p25, p30

# [not in PDF]
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

# [not in PDF]
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


def compute_cost_paths(prices_usd, scope1_by_market, coverage_by_year, hedge_ratio, futures_price_usd) -> tuple[np.ndarray, np.ndarray]:
    years = list(scope1_by_market.keys())
    n_sims = next(iter(prices_usd.values())).shape[0]
    n_years = len(years)

    paths_no = np.zeros((n_sims, n_years))
    paths_hedged = np.zeros((n_sims, n_years))

    for j, y in enumerate(years):
        cost_no = np.zeros(n_sims)
        cost_hedged = np.zeros(n_sims)

        for m in MARKETS:
            emissions_total = scope1_by_market[y][m]
            coverage = coverage_by_year[y][m]
            emissions_priced = emissions_total * coverage

            spot = prices_usd[m][:, j]
            hedged_vol = hedge_ratio * emissions_priced
            unhedged_vol = (1.0 - hedge_ratio) * emissions_priced
            K = futures_price_usd[m]

            cost_no += emissions_priced * spot
            cost_hedged += hedged_vol * K + unhedged_vol * spot

        paths_no[:, j] = cost_no
        paths_hedged[:, j] = cost_hedged

    return paths_no, paths_hedged


def compute_mean_cost_by_market_per_year(prices_usd, scope1_by_market, coverage_by_year, hedge_ratio, futures_price_usd):
    years = list(scope1_by_market.keys())
    n_years = len(years)

    mean_no = {m: np.zeros(n_years) for m in MARKETS}
    mean_hedged = {m: np.zeros(n_years) for m in MARKETS}

    for j, y in enumerate(years):
        for m in MARKETS:
            emissions_total = scope1_by_market[y][m]
            coverage = coverage_by_year[y][m]
            emissions_priced = emissions_total * coverage

            spot = prices_usd[m][:, j]
            hedged_vol = hedge_ratio * emissions_priced
            unhedged_vol = (1.0 - hedge_ratio) * emissions_priced
            K = futures_price_usd[m]

            cost_no = emissions_priced * spot
            cost_hedged = hedged_vol * K + unhedged_vol * spot

            mean_no[m][j] = cost_no.mean()
            mean_hedged[m][j] = cost_hedged.mean()

    return years, mean_no, mean_hedged

# [not in PDF]
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

# [not in PDF]
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


def make_plots(sweep: pd.DataFrame):
    """Create quick diagnostic plots for hedge ratios."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Headless friendly backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plots were not generated: matplotlib not installed (pip install matplotlib).")
        return

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass  # style not available; continue with default

    hr = sweep["hedge_ratio"]
    saved = []

    # Expected costs (bn USD)
    fig1, ax1 = plt.subplots()
    ax1.plot(hr, sweep["mean_no_hedge"] / 1e9, marker="o", label="Mean without hedge")
    ax1.plot(hr, sweep["mean_with_hedge"] / 1e9, marker="o", label="Mean with hedge")
    ax1.set_xlabel("Hedge ratio")
    ax1.set_ylabel("Costs (bn USD)")
    ax1.set_title("Expected costs vs. hedge ratio")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fname = "plot_costs_mean.png"
    fig1.savefig(fname, dpi=150)
    saved.append(fname)
    plt.close(fig1)

    # Risk: 95th percentile (bn USD)
    fig2, ax2 = plt.subplots()
    ax2.plot(hr, sweep["p95_no_hedge"] / 1e9, marker="o", label="P95 without hedge")
    ax2.plot(hr, sweep["p95_with_hedge"] / 1e9, marker="o", label="P95 with hedge")
    ax2.set_xlabel("Hedge ratio")
    ax2.set_ylabel("95th percentile of costs (bn USD)")
    ax2.set_title("Risk (P95) vs. hedge ratio")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fname = "plot_risk_p95.png"
    fig2.savefig(fname, dpi=150)
    saved.append(fname)
    plt.close(fig2)

    # Expected savings (bn USD)
    fig3, ax3 = plt.subplots()
    ax3.bar(hr, sweep["mean_saving"] / 1e9, width=0.15)
    ax3.set_xlabel("Hedge ratio")
    ax3.set_ylabel("Expected savings (bn USD)")
    ax3.set_title("Savings from hedge vs. hedge ratio")
    ax3.grid(True, axis="y", alpha=0.3)
    fig3.tight_layout()
    fname = "plot_savings.png"
    fig3.savefig(fname, dpi=150)
    saved.append(fname)
    plt.close(fig3)

    # Risk-return (std vs. mean) for hedged costs
    fig4, ax4 = plt.subplots()
    x_std = sweep["std_with_hedge"] / 1e9
    y_mean = sweep["mean_with_hedge"] / 1e9
    scatter = ax4.scatter(x_std, y_mean, c=hr, cmap="viridis", s=80)
    for i, ratio in enumerate(hr):
        ax4.annotate(f"{ratio:.1f}", (x_std.iloc[i], y_mean.iloc[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    cbar = fig4.colorbar(scatter, ax=ax4)
    cbar.set_label("Hedge ratio")
    ax4.set_xlabel("Std dev of costs (bn USD)")
    ax4.set_ylabel("Expected costs (bn USD)")
    ax4.set_title("Risk-return of hedge ratios")
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    fname = "plot_risk_return.png"
    fig4.savefig(fname, dpi=150)
    saved.append(fname)
    plt.close(fig4)

    print("Plots saved:", ", ".join(saved))


def make_tradeoff_plot(sweep: pd.DataFrame):
    """Plot trade-off: risk reduction (P95 reduction) vs. expected additional cost."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Trade-off plot was not generated: matplotlib not installed (pip install matplotlib).")
        return

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass

    sweep_sorted = sweep.sort_values("hedge_ratio")
    hr = sweep_sorted["hedge_ratio"]

    # Risk reduction (P95 reduction) and expected additional cost from hedging in bn USD
    risk_reduction = sweep_sorted["p95_reduction"] / 1e9
    mean_delta = (sweep_sorted["mean_with_hedge"] - sweep_sorted["mean_no_hedge"]) / 1e9

    fig, ax = plt.subplots()
    scatter = ax.scatter(risk_reduction, mean_delta, c=hr, cmap="viridis", s=80)

    for i, ratio in enumerate(hr):
        ax.annotate(
            f"{ratio:.1f}",
            (risk_reduction.iloc[i], mean_delta.iloc[i]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    ax.axhline(0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlabel("Risk reduction P95 (bn USD)")
    ax.set_ylabel("Expected additional cost from hedging (bn USD)")
    ax.set_title("Trade-off: risk reduction vs. expected additional cost of the hedge")
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Hedge ratio")

    fig.tight_layout()
    fname = "plot_tradeoff_risk_vs_cost.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print("Trade-off plot saved:", fname)


def make_var_plot(hedge_ratio_for_plot: float = 0.7):
    """VaR histogram with overlaid normal distribution, without and with hedge."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("VaR plot was not generated: matplotlib not installed (pip install matplotlib).")
        return

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass

    scope1_global = build_scope1_global(YEARS)
    scope1_market = build_scope1_by_market(scope1_global, include_other=INCLUDE_OTHER)
    coverage = build_coverage(YEARS)

    p25_usd, p30_usd = build_price_anchors_usd()
    prices_usd = simulate_prices_gbm_targeted(
        years=YEARS,
        n_sims=N_SIMS,
        p0_usd=p25_usd,
        pT_usd=p30_usd,
        sigma=SIGMA,
        seed=RANDOM_SEED,
    )

    no_hedge, _ = compute_costs(
        prices_usd=prices_usd,
        scope1_by_market=scope1_market,
        coverage_by_year=coverage,
        hedge_ratio=0.0,
        futures_price_usd=FUTURES_PRICE_USD,
    )

    _, hedged = compute_costs(
        prices_usd=prices_usd,
        scope1_by_market=scope1_market,
        coverage_by_year=coverage,
        hedge_ratio=hedge_ratio_for_plot,
        futures_price_usd=FUTURES_PRICE_USD,
    )

    def _plot_var(ax, data, title: str):
        data_bn = data / 1e9
        mu = data_bn.mean()
        sigma = data_bn.std()
        var95 = float(np.quantile(data_bn, 0.95))

        ax.hist(data_bn, bins=60, density=True, alpha=0.5, color="tab:blue", label="Simulation")

        if sigma > 0:
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
            pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            ax.plot(x, pdf, color="tab:orange", label="Normal (same μ, σ)")

        ax.axvline(mu, color="black", linestyle="--", label=f"Mean ≈ {mu:.1f}")
        ax.axvline(var95, color="red", linestyle=":", label=f"P95 / VaR 95% ≈ {var95:.1f}")

        ax.set_xlabel("Total CO2 costs 2025–2030 (bn USD)")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    _plot_var(ax1, no_hedge, "Without hedge – VaR 95%")
    _plot_var(ax2, hedged, f"With hedge (HR={hedge_ratio_for_plot:.1f}) – VaR 95%")

    fig.suptitle("Distribution of total costs 2025–2030\nwith 95% VaR and normal approximation", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fname = f"plot_var_hist_normal_hr{hedge_ratio_for_plot:.1f}.png".replace('.', '_')
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print("VaR plot saved:", fname)


def make_market_timeseries_plots(hedge_ratio_for_plot: float = 0.7):
    """Time series of expected annual CO2 costs by market."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Market time-series plots were not generated: matplotlib not installed (pip install matplotlib).")
        return

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass

    scope1_global = build_scope1_global(YEARS)
    scope1_market = build_scope1_by_market(scope1_global, include_other=INCLUDE_OTHER)
    coverage = build_coverage(YEARS)

    p25_usd, p30_usd = build_price_anchors_usd()
    prices_usd = simulate_prices_gbm_targeted(
        years=YEARS,
        n_sims=N_SIMS,
        p0_usd=p25_usd,
        pT_usd=p30_usd,
        sigma=SIGMA,
        seed=RANDOM_SEED,
    )

    years, mean_no_by_market, _ = compute_mean_cost_by_market_per_year(
        prices_usd=prices_usd,
        scope1_by_market=scope1_market,
        coverage_by_year=coverage,
        hedge_ratio=0.0,
        futures_price_usd=FUTURES_PRICE_USD,
    )

    _, _, mean_hedged_by_market = compute_mean_cost_by_market_per_year(
        prices_usd=prices_usd,
        scope1_by_market=scope1_market,
        coverage_by_year=coverage,
        hedge_ratio=hedge_ratio_for_plot,
        futures_price_usd=FUTURES_PRICE_USD,
    )

    colors = {
        "USA": "tab:blue",
        "Canada": "tab:orange",
        "EU_UK": "tab:green",
    }

    # Without hedge – stacked areas by market
    fig1, ax1 = plt.subplots()
    data_no = [mean_no_by_market[m] / 1e9 for m in MARKETS]
    labels_no = ["USA", "Canada", "EU/UK"]
    colors_no = [colors[m] for m in MARKETS]
    ax1.stackplot(years, data_no, labels=labels_no, colors=colors_no, alpha=0.8)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Annual CO2 costs (bn USD)")
    ax1.set_title("Expected annual CO2 costs without hedge – by market")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fname1 = "plot_costs_by_market_no_hedge.png"
    fig1.savefig(fname1, dpi=150)
    plt.close(fig1)

    # With hedge – stacked areas by market
    fig2, ax2 = plt.subplots()
    data_h = [mean_hedged_by_market[m] / 1e9 for m in MARKETS]
    labels_h = [f"{name} (with hedge)" for name in ["USA", "Canada", "EU/UK"]]
    colors_h = [colors[m] for m in MARKETS]
    ax2.stackplot(years, data_h, labels=labels_h, colors=colors_h, alpha=0.8)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Annual CO2 costs (bn USD)")
    ax2.set_title(f"Expected annual CO2 costs with hedge – by market\n(Hedge ratio={hedge_ratio_for_plot:.1f})")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fname2 = "plot_costs_by_market_with_hedge.png"
    fig2.savefig(fname2, dpi=150)
    plt.close(fig2)

    print("Market time-series plots saved:", fname1, ",", fname2)


def make_timeseries_best_worst_plot(hedge_ratio_for_plot: float = 0.7):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Timeseries plot was not generated: matplotlib not installed (pip install matplotlib).")
        return

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass

    scope1_global = build_scope1_global(YEARS)
    scope1_market = build_scope1_by_market(scope1_global, include_other=INCLUDE_OTHER)
    coverage = build_coverage(YEARS)

    p25_usd, p30_usd = build_price_anchors_usd()
    prices_usd = simulate_prices_gbm_targeted(
        years=YEARS,
        n_sims=N_SIMS,
        p0_usd=p25_usd,
        pT_usd=p30_usd,
        sigma=SIGMA,
        seed=RANDOM_SEED
    )

    paths_no_unhedged, _ = compute_cost_paths(
        prices_usd=prices_usd,
        scope1_by_market=scope1_market,
        coverage_by_year=coverage,
        hedge_ratio=0.0,
        futures_price_usd=FUTURES_PRICE_USD
    )

    _, paths_hedged = compute_cost_paths(
        prices_usd=prices_usd,
        scope1_by_market=scope1_market,
        coverage_by_year=coverage,
        hedge_ratio=hedge_ratio_for_plot,
        futures_price_usd=FUTURES_PRICE_USD
    )

    # Means and uncertainty bands (P05–P95) in bn USD
    mean_no = paths_no_unhedged.mean(axis=0) / 1e9
    mean_h = paths_hedged.mean(axis=0) / 1e9
    q05_no = np.quantile(paths_no_unhedged, 0.05, axis=0) / 1e9
    q95_no = np.quantile(paths_no_unhedged, 0.95, axis=0) / 1e9
    q05_h = np.quantile(paths_hedged, 0.05, axis=0) / 1e9
    q95_h = np.quantile(paths_hedged, 0.95, axis=0) / 1e9

    years = YEARS

    fig, ax = plt.subplots()
    # Without hedge – mean + uncertainty band
    ax.plot(years, mean_no, color="tab:blue", label="Without hedge – mean")
    ax.fill_between(years, q05_no, q95_no, color="tab:blue", alpha=0.2)

    # With hedge – mean + uncertainty band
    ax.plot(
        years,
        mean_h,
        color="tab:orange",
        label=f"With hedge – mean (HR={hedge_ratio_for_plot:.1f})",
    )
    ax.fill_between(years, q05_h, q95_h, color="tab:orange", alpha=0.2)

    # Actual point for the start year on the unhedged mean line
    ist_year = years[0]
    ist_cost = mean_no[0]
    ax.scatter(
        [ist_year],
        [ist_cost],
        color="black",
        marker="o",
        zorder=5,
        label="Actual costs start year (without hedge)",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual CO2 costs (bn USD)")
    ax.set_title(
        "Development of annual CO2 costs through 2030\n"
        "Mean & uncertainty band (P05–P95) with/without hedging"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = "plot_timeseries_best_worst_hedge.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print("Timeseries plot saved:", fname)


def make_boxplot_costs_by_hedge_ratio():
    """Boxplots of total costs by hedge ratio (with hedged costs)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Boxplot was not generated: matplotlib not installed (pip install matplotlib).")
        return

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass

    scope1_global = build_scope1_global(YEARS)
    scope1_market = build_scope1_by_market(scope1_global, include_other=INCLUDE_OTHER)
    coverage = build_coverage(YEARS)

    p25_usd, p30_usd = build_price_anchors_usd()
    prices_usd = simulate_prices_gbm_targeted(
        years=YEARS,
        n_sims=N_SIMS,
        p0_usd=p25_usd,
        pT_usd=p30_usd,
        sigma=SIGMA,
        seed=RANDOM_SEED,
    )

    costs_by_hr = []
    labels = []
    for hr in HEDGE_RATIOS_TO_TEST:
        _, hedged = compute_costs(
            prices_usd=prices_usd,
            scope1_by_market=scope1_market,
            coverage_by_year=coverage,
            hedge_ratio=hr,
            futures_price_usd=FUTURES_PRICE_USD,
        )
        costs_by_hr.append(hedged / 1e9)
        labels.append(f"{hr:.1f}")

    fig, ax = plt.subplots()
    ax.boxplot(costs_by_hr, labels=labels, showfliers=False)
    ax.set_xlabel("Hedge ratio")
    ax.set_ylabel("Total CO2 costs 2025–2030 (bn USD)")
    ax.set_title("Distribution of total costs by hedge ratio (boxplots)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = "plot_boxplot_costs_by_hedge_ratio.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print("Boxplot saved:", fname)


def make_delta_plot(sweep: pd.DataFrame):
    """Line plot of deltas vs. no-hedge (mean, P95, std) by hedge ratio."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Delta plot was not generated: matplotlib not installed (pip install matplotlib).")
        return

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass

    sweep_sorted = sweep.sort_values("hedge_ratio")
    hr = sweep_sorted["hedge_ratio"]

    mean_delta = (sweep_sorted["mean_with_hedge"] - sweep_sorted["mean_no_hedge"]) / 1e9
    p95_delta = sweep_sorted["p95_reduction"] / 1e9
    std_delta = sweep_sorted["std_reduction"] / 1e9

    fig, ax = plt.subplots()
    ax.plot(hr, mean_delta, marker="o", label="Δ mean costs (bn USD)")
    ax.plot(hr, p95_delta, marker="o", label="P95 reduction (bn USD)")
    ax.plot(hr, std_delta, marker="o", label="Std reduction (bn USD)")

    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Hedge ratio")
    ax.set_ylabel("Difference vs. no hedge (bn USD)")
    ax.set_title("Deltas vs. no-hedge across hedge ratios")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    fname = "plot_deltas_vs_no_hedge.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print("Delta plot saved:", fname)


def make_pnl_vs_cost_plot(hedge_ratio_for_plot: float = 0.7):
    """Hedge PnL (savings) vs. no-hedge costs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("PnL plot was not generated: matplotlib not installed (pip install matplotlib).")
        return

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass

    scope1_global = build_scope1_global(YEARS)
    scope1_market = build_scope1_by_market(scope1_global, include_other=INCLUDE_OTHER)
    coverage = build_coverage(YEARS)

    p25_usd, p30_usd = build_price_anchors_usd()
    prices_usd = simulate_prices_gbm_targeted(
        years=YEARS,
        n_sims=N_SIMS,
        p0_usd=p25_usd,
        pT_usd=p30_usd,
        sigma=SIGMA,
        seed=RANDOM_SEED,
    )

    no_hedge, hedged = compute_costs(
        prices_usd=prices_usd,
        scope1_by_market=scope1_market,
        coverage_by_year=coverage,
        hedge_ratio=hedge_ratio_for_plot,
        futures_price_usd=FUTURES_PRICE_USD,
    )

    no_hedge_bn = no_hedge / 1e9
    hedged_bn = hedged / 1e9
    pnl_bn = no_hedge_bn - hedged_bn  # Savings from hedge

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram of hedge PnL
    ax1.hist(pnl_bn, bins=60, alpha=0.6, color="tab:blue")
    ax1.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax1.set_xlabel("Hedge PnL / savings (bn USD)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Hedge PnL distribution (HR={hedge_ratio_for_plot:.1f})")
    ax1.grid(True, alpha=0.3)

    # Scatter: PnL vs. no-hedge costs
    ax2.scatter(no_hedge_bn, pnl_bn, alpha=0.4, s=10, color="tab:orange")
    ax2.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax2.set_xlabel("No-hedge total costs 2025–2030 (bn USD)")
    ax2.set_ylabel("Hedge PnL / savings (bn USD)")
    ax2.set_title("Hedge PnL vs. no-hedge costs")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f"plot_pnl_vs_cost_hr{hedge_ratio_for_plot:.1f}.png".replace('.', '_')
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print("PnL plot saved:", fname)


# ============================================================
# RUN
# ============================================================

# [not in PDF]
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
    output_path_csv = "hedge_ratio_sweep.csv"
    try:
        excel_written = False # Flag to check if Excel export was successful
        try: # Try to write with decimal comma for Excel
            sweep.to_excel(output_path, index=False, decimal=",")
            excel_written = True
        except TypeError:
            sweep.to_excel(output_path, index=False)
            excel_written = True
            print("Note: this pandas version does not support decimal=','; Excel will use decimal point '.'.")

        sweep.to_csv(output_path_csv, index=False, sep=";", decimal=",", float_format="%.2f")

        if excel_written:
            print(f"\nExcel export written to {output_path}")
        print(f"Simpler CSV export written to {output_path_csv} (separator=';', decimal=',')")
    except ImportError as exc:
        print(f"\nCould not export to Excel ({exc}). Install openpyxl or xlsxwriter to enable Excel export.")

    make_plots(sweep)
    make_timeseries_best_worst_plot()
    make_tradeoff_plot(sweep)
    make_market_timeseries_plots()
    make_var_plot()       # Default: HR = 0.7
    make_var_plot(0.5)    # Additionally: HR = 0.5
    make_boxplot_costs_by_hedge_ratio()
    make_delta_plot(sweep)
    make_pnl_vs_cost_plot()
