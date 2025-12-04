# [nicht in PDF]
import numpy as np
import pandas as pd

# ================================
# CONFIG – Exxon CO2 Hedging Model
# ================================

YEARS = list(range(2025, 2031))  # 2025–2030 inkl.

# Globaler Scope-1-Anker (grob basierend auf Exxon-Daten, 2016–2024 Trend & 2030 Ziel)
SCOPE1_ANCHOR_START_YEAR = 2024
SCOPE1_ANCHOR_START_T    = 103_000_000   # tCO2e global Scope 1 (≈ Exxon 2024, gerundet)
SCOPE1_ANCHOR_END_YEAR   = 2030
SCOPE1_ANCHOR_END_T      = 90_000_000    # tCO2e 2030 (modellierte Zielgröße, leichter Rückgang)

# Modellierte größten Märkte
MARKETS = ["USA", "Canada", "EU_UK"]

# Marktanteile an globalen Scope-1-Emissionen (Approx. aus Produktionsanteilen abgeleitet)
MARKET_SHARE = {
    "USA":   0.45,   # ~45 % der Produktion in den USA
    "Canada":0.20,   # Kanada + weitere Amerika ≈ 20 % für Modell
    "EU_UK": 0.05,   # EU/UK sehr klein, 5 % als obere Schätzung
}
    # Rest (~30 %) = „Other“, wird im Modell ignoriert (kein Hedge)
# Anteil der Emissionen, die effektiv einem CO2-Preis unterliegen (Coverage)
# – konservativ, aber an Systemlogik orientiert
COVERAGE_2025 = {
    "USA":   0.30,   # nur Teile (Kalifornien, Washington, RGGI)
    "Canada":0.70,   # föderaler Mindestpreis + Provinzsysteme
    "EU_UK": 0.90,   # nahezu alle großen Anlagen im ETS/UK ETS
}
COVERAGE_2030 = {
    "USA":   0.50,   # mehr Staaten / stärkere Regulierung angenommen
    "Canada":0.90,   # fast vollständige Abdeckung der großen Emittenten
    "EU_UK": 1.00,   # vollständige Abdeckung angenommen
}

# CO2-Preisanker 2025/2030 (lokale Währung)
# USA & Canada hier bereits in USD, EU/UK in EUR (wird unten mit FX in USD umgerechnet)
PRICE_2025 = {
    "USA":   30.0,   # USD/t – angenähert an CCA/US-ETS-Größenordnung
    "Canada":69.0,   # USD/t – aus 95 CAD ≈ 69.38 USD (Bundes-Backstop 2025)  [oai_citation:0‡icapcarbonaction.com](https://icapcarbonaction.com/system/files/ets_pdfs/icap-etsmap-factsheet-135.pdf?utm_source=chatgpt.com)
    "EU_UK": 80.0,   # EUR/t – in der Nähe aktueller EUA-Dec25-Futures (~78–83 EUR)  [oai_citation:1‡Investing.com](https://www.investing.com/commodities/european-union-allowance-eua-year-futures-historical-data?utm_source=chatgpt.com)
}
PRICE_2030 = {
    "USA":   50.0,   # USD/t – moderates Wachstumsszenario
    "Canada":124.0,  # USD/t – aus 170 CAD ≈ 124.15 USD (Bundes-Backstop 2030)  [oai_citation:2‡icapcarbonaction.com](https://icapcarbonaction.com/system/files/ets_pdfs/icap-etsmap-factsheet-135.pdf?utm_source=chatgpt.com)
    "EU_UK": 90.0,   # EUR/t – leicht höheres EUA-Niveau in 2030 (Szenario)
}

# FX-Annahme für EUR → USD
FX_EURUSD = 1.08   # 1 EUR = 1.08 USD (grobe Annahme)

# Fixpreise der Futures (Hedge-Preis K) in USD/t
# – etwas über den 2025-Spots (Risk Premium)
FUTURES_PRICE_USD = {
    "USA":   35.0,                 # USD/t – CO2-Futures auf US-Systeme
    "Canada":75.0,                 # USD/t – leicht über 2025-Backstop
    "EU_UK": 85.0 * FX_EURUSD,     # EUR 85 → USD (EUA-Future-Niveau leicht über Spot)
}

# Volatilität (annualisierte sigma) pro Markt
SIGMA = {
    "USA":   0.20,   # 20 % – US-Carbonpreise (CCA/RGGI) historisch moderat volatil
    "Canada":0.25,   # 25 % – fragmentierter Markt, etwas höhere Unsicherheit  [oai_citation:3‡clearbluemarkets.com](https://www.clearbluemarkets.com/knowledge-base/navigating-canadas-carbon-markets-ahead-of-the-2026-federal-benchmark-review?utm_source=chatgpt.com)
    "EU_UK": 0.35,   # 35 % – EUAs historisch ~30–40 % Jahresvola  [oai_citation:4‡Investing.com](https://www.investing.com/commodities/european-union-allowance-eua-year-futures-historical-data?utm_source=chatgpt.com)
}

# Simulationseinstellungen
N_SIMS = 20_000     # Anzahl Monte-Carlo-Pfade
RANDOM_SEED = 42    # Reproduzierbarkeit

# Hedge-Ratios, die du durchsimulierst (global, gleiche Quote für alle Märkte)
HEDGE_RATIOS_TO_TEST = [0.0, 0.3, 0.5, 0.7, 1.0]

# „Other“ nicht explizit modellieren
INCLUDE_OTHER = False
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


def make_plots(sweep: pd.DataFrame):
    """Create quick diagnostic plots for hedge ratios."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Headless friendly backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plots wurden nicht erzeugt: matplotlib nicht installiert (pip install matplotlib).")
        return

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass  # style not available; continue with default

    hr = sweep["hedge_ratio"]
    saved = []

    # Erwartete Kosten (Mrd. USD)
    fig1, ax1 = plt.subplots()
    ax1.plot(hr, sweep["mean_no_hedge"] / 1e9, marker="o", label="Mean ohne Hedge")
    ax1.plot(hr, sweep["mean_with_hedge"] / 1e9, marker="o", label="Mean mit Hedge")
    ax1.set_xlabel("Hedge-Ratio")
    ax1.set_ylabel("Kosten (Mrd. USD)")
    ax1.set_title("Erwartete Kosten vs. Hedge-Ratio")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fname = "plot_costs_mean.png"
    fig1.savefig(fname, dpi=150)
    saved.append(fname)
    plt.close(fig1)

    # Risiko: 95. Perzentil (Mrd. USD)
    fig2, ax2 = plt.subplots()
    ax2.plot(hr, sweep["p95_no_hedge"] / 1e9, marker="o", label="P95 ohne Hedge")
    ax2.plot(hr, sweep["p95_with_hedge"] / 1e9, marker="o", label="P95 mit Hedge")
    ax2.set_xlabel("Hedge-Ratio")
    ax2.set_ylabel("95%-Perzentil Kosten (Mrd. USD)")
    ax2.set_title("Risiko (P95) vs. Hedge-Ratio")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fname = "plot_risk_p95.png"
    fig2.savefig(fname, dpi=150)
    saved.append(fname)
    plt.close(fig2)

    # Erwartete Einsparung (Mrd. USD)
    fig3, ax3 = plt.subplots()
    ax3.bar(hr, sweep["mean_saving"] / 1e9, width=0.15)
    ax3.set_xlabel("Hedge-Ratio")
    ax3.set_ylabel("Erwartete Einsparung (Mrd. USD)")
    ax3.set_title("Einsparung durch Hedge vs. Hedge-Ratio")
    ax3.grid(True, axis="y", alpha=0.3)
    fig3.tight_layout()
    fname = "plot_savings.png"
    fig3.savefig(fname, dpi=150)
    saved.append(fname)
    plt.close(fig3)

    # Risk-Return (Std vs. Mean) für Hedged-Kosten
    fig4, ax4 = plt.subplots()
    x_std = sweep["std_with_hedge"] / 1e9
    y_mean = sweep["mean_with_hedge"] / 1e9
    scatter = ax4.scatter(x_std, y_mean, c=hr, cmap="viridis", s=80)
    for i, ratio in enumerate(hr):
        ax4.annotate(f"{ratio:.1f}", (x_std.iloc[i], y_mean.iloc[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    cbar = fig4.colorbar(scatter, ax=ax4)
    cbar.set_label("Hedge-Ratio")
    ax4.set_xlabel("Std dev Kosten (Mrd. USD)")
    ax4.set_ylabel("Erwartete Kosten (Mrd. USD)")
    ax4.set_title("Risk-Return der Hedge-Ratios")
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    fname = "plot_risk_return.png"
    fig4.savefig(fname, dpi=150)
    saved.append(fname)
    plt.close(fig4)

    print("Plots gespeichert:", ", ".join(saved))


def make_tradeoff_plot(sweep: pd.DataFrame):
    """Plot Trade-off: Risikoreduktion (P95-Reduktion) vs. erwartete Mehrkosten."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Trade-off-Plot wurde nicht erzeugt: matplotlib nicht installiert (pip install matplotlib).")
        return

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass

    sweep_sorted = sweep.sort_values("hedge_ratio")
    hr = sweep_sorted["hedge_ratio"]

    # Risikoreduktion (P95-Reduktion) und erwartete Mehrkosten durch Hedge in Mrd. USD
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
    ax.set_xlabel("Risikoreduktion P95 (Mrd. USD)")
    ax.set_ylabel("Erwartete Mehrkosten durch Hedge (Mrd. USD)")
    ax.set_title("Trade-off: Risikoreduktion vs. erwartete Mehrkosten des Hedges")
    ax.grid(True, alpha=0.3)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Hedge-Ratio")

    fig.tight_layout()
    fname = "plot_tradeoff_risk_vs_cost.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print("Trade-off-Plot gespeichert:", fname)


def make_var_plot(hedge_ratio_for_plot: float = 0.7):
    """VaR-Histogramm mit überlagerter Normalverteilung, ohne und mit Hedge."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("VaR-Plot wurde nicht erzeugt: matplotlib nicht installiert (pip install matplotlib).")
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
            ax.plot(x, pdf, color="tab:orange", label="Normal (gleiches μ, σ)")

        ax.axvline(mu, color="black", linestyle="--", label=f"Mean ≈ {mu:.1f}")
        ax.axvline(var95, color="red", linestyle=":", label=f"P95 / VaR 95% ≈ {var95:.1f}")

        ax.set_xlabel("Gesamte CO2-Kosten 2025–2030 (Mrd. USD)")
        ax.set_ylabel("Dichte")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    _plot_var(ax1, no_hedge, "Ohne Hedge – VaR 95%")
    _plot_var(ax2, hedged, f"Mit Hedge (HR={hedge_ratio_for_plot:.1f}) – VaR 95%")

    fig.suptitle("Verteilung der Gesamtkosten 2025–2030\nmit VaR 95% und Normalapproximation", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    fname = f"plot_var_hist_normal_hr{hedge_ratio_for_plot:.1f}.png".replace('.', '_')
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print("VaR-Plot gespeichert:", fname)


def make_market_timeseries_plots(hedge_ratio_for_plot: float = 0.7):
    """Zeitreihen der erwarteten jährlichen CO2-Kosten nach Märkten."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Market-Zeitreihen-Plots wurden nicht erzeugt: matplotlib nicht installiert (pip install matplotlib).")
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

    # Ohne Hedge – gestapelte Flächen nach Märkten
    fig1, ax1 = plt.subplots()
    data_no = [mean_no_by_market[m] / 1e9 for m in MARKETS]
    labels_no = ["USA", "Kanada", "EU/UK"]
    colors_no = [colors[m] for m in MARKETS]
    ax1.stackplot(years, data_no, labels=labels_no, colors=colors_no, alpha=0.8)
    ax1.set_xlabel("Jahr")
    ax1.set_ylabel("Jährliche CO2-Kosten (Mrd. USD)")
    ax1.set_title("Erwartete jährliche CO2-Kosten ohne Hedge – nach Märkten")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fname1 = "plot_costs_by_market_no_hedge.png"
    fig1.savefig(fname1, dpi=150)
    plt.close(fig1)

    # Mit Hedge – gestapelte Flächen nach Märkten
    fig2, ax2 = plt.subplots()
    data_h = [mean_hedged_by_market[m] / 1e9 for m in MARKETS]
    labels_h = [f"{name} (mit Hedge)" for name in ["USA", "Kanada", "EU/UK"]]
    colors_h = [colors[m] for m in MARKETS]
    ax2.stackplot(years, data_h, labels=labels_h, colors=colors_h, alpha=0.8)
    ax2.set_xlabel("Jahr")
    ax2.set_ylabel("Jährliche CO2-Kosten (Mrd. USD)")
    ax2.set_title(f"Erwartete jährliche CO2-Kosten mit Hedge – nach Märkten\n(Hedge-Ratio={hedge_ratio_for_plot:.1f})")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fname2 = "plot_costs_by_market_with_hedge.png"
    fig2.savefig(fname2, dpi=150)
    plt.close(fig2)

    print("Market-Zeitreihen-Plots gespeichert:", fname1, ",", fname2)


def make_timeseries_best_worst_plot(hedge_ratio_for_plot: float = 0.7):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Timeseries-Plot wurde nicht erzeugt: matplotlib nicht installiert (pip install matplotlib).")
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

    # Mittelwerte und Unsicherheitsbänder (P05–P95) in Mrd. USD
    mean_no = paths_no_unhedged.mean(axis=0) / 1e9
    mean_h = paths_hedged.mean(axis=0) / 1e9
    q05_no = np.quantile(paths_no_unhedged, 0.05, axis=0) / 1e9
    q95_no = np.quantile(paths_no_unhedged, 0.95, axis=0) / 1e9
    q05_h = np.quantile(paths_hedged, 0.05, axis=0) / 1e9
    q95_h = np.quantile(paths_hedged, 0.95, axis=0) / 1e9

    years = YEARS

    fig, ax = plt.subplots()
    # Ohne Hedge – Mittelwert + Unsicherheitsband
    ax.plot(years, mean_no, color="tab:blue", label="Ohne Hedge – Mittelwert")
    ax.fill_between(years, q05_no, q95_no, color="tab:blue", alpha=0.2)

    # Mit Hedge – Mittelwert + Unsicherheitsband
    ax.plot(
        years,
        mean_h,
        color="tab:orange",
        label=f"Mit Hedge – Mittelwert (HR={hedge_ratio_for_plot:.1f})",
    )
    ax.fill_between(years, q05_h, q95_h, color="tab:orange", alpha=0.2)

    # Ist-Punkt zum Startjahr auf der unhedged-Mittelwert-Linie
    ist_year = years[0]
    ist_cost = mean_no[0]
    ax.scatter(
        [ist_year],
        [ist_cost],
        color="black",
        marker="o",
        zorder=5,
        label="Ist-Kosten Startjahr (ohne Hedge)",
    )

    ax.set_xlabel("Jahr")
    ax.set_ylabel("Jährliche CO2-Kosten (Mrd. USD)")
    ax.set_title(
        "Entwicklung der jährlichen CO2-Kosten bis 2030\n"
        "Mittelwert & Unsicherheitsband (P05–P95) mit/ohne Hedging"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = "plot_timeseries_best_worst_hedge.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print("Timeseries-Plot gespeichert:", fname)


def make_boxplot_costs_by_hedge_ratio():
    """Boxplots der Gesamtkosten je Hedge-Ratio (mit Hedge-Kosten)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Boxplot-Plot wurde nicht erzeugt: matplotlib nicht installiert (pip install matplotlib).")
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
    ax.set_xlabel("Hedge-Ratio")
    ax.set_ylabel("Gesamte CO2-Kosten 2025–2030 (Mrd. USD)")
    ax.set_title("Verteilung der Gesamtkosten je Hedge-Ratio (Boxplots)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = "plot_boxplot_costs_by_hedge_ratio.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print("Boxplot-Plot gespeichert:", fname)


def make_delta_plot(sweep: pd.DataFrame):
    """Linienplot der Deltas zu No-Hedge (Mean, P95, Std) je Hedge-Ratio."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Delta-Plot wurde nicht erzeugt: matplotlib nicht installiert (pip install matplotlib).")
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
    ax.plot(hr, mean_delta, marker="o", label="Δ Mean-Kosten (Mrd. USD)")
    ax.plot(hr, p95_delta, marker="o", label="P95-Reduktion (Mrd. USD)")
    ax.plot(hr, std_delta, marker="o", label="Std-Reduktion (Mrd. USD)")

    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Hedge-Ratio")
    ax.set_ylabel("Differenz zu No-Hedge (Mrd. USD)")
    ax.set_title("Deltas vs. No-Hedge über Hedge-Ratios")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    fname = "plot_deltas_vs_no_hedge.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print("Delta-Plot gespeichert:", fname)


def make_pnl_vs_cost_plot(hedge_ratio_for_plot: float = 0.7):
    """Hedge-PnL (Einsparung) vs. No-Hedge-Kosten."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("PnL-Plot wurde nicht erzeugt: matplotlib nicht installiert (pip install matplotlib).")
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
    pnl_bn = no_hedge_bn - hedged_bn  # Einsparung durch Hedge

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Histogramm des Hedge-PnL
    ax1.hist(pnl_bn, bins=60, alpha=0.6, color="tab:blue")
    ax1.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax1.set_xlabel("Hedge-PnL / Einsparung (Mrd. USD)")
    ax1.set_ylabel("Häufigkeit")
    ax1.set_title(f"Hedge-PnL-Verteilung (HR={hedge_ratio_for_plot:.1f})")
    ax1.grid(True, alpha=0.3)

    # Scatter: PnL vs. No-Hedge-Kosten
    ax2.scatter(no_hedge_bn, pnl_bn, alpha=0.4, s=10, color="tab:orange")
    ax2.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax2.set_xlabel("No-Hedge Gesamtkosten 2025–2030 (Mrd. USD)")
    ax2.set_ylabel("Hedge-PnL / Einsparung (Mrd. USD)")
    ax2.set_title("Hedge-PnL vs. No-Hedge-Kosten")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f"plot_pnl_vs_cost_hr{hedge_ratio_for_plot:.1f}.png".replace('.', '_')
    fig.savefig(fname, dpi=150)
    plt.close(fig)

    print("PnL-Plot gespeichert:", fname)


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
    output_path_csv = "hedge_ratio_sweep.csv"
    try:
        excel_written = False # Flag to check if Excel export was successful
        try: # Try to write with decimal comma for Excel
            sweep.to_excel(output_path, index=False, decimal=",")
            excel_written = True
        except TypeError:
            sweep.to_excel(output_path, index=False)
            excel_written = True
            print("Hinweis: pandas-Version unterstützt decimal=',' nicht; Excel nutzt Dezimalpunkt '.'.")

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
    make_var_plot()       # Standard: HR = 0.7
    make_var_plot(0.5)    # Zusätzlich: HR = 0.5
    make_boxplot_costs_by_hedge_ratio()
    make_delta_plot(sweep)
    make_pnl_vs_cost_plot()
